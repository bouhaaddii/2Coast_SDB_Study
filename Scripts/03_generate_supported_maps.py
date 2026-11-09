"""Fit site-specific full-data SDB models and map only locally supported pixels."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.transform import rowcol
from scipy.ndimage import distance_transform_edt
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'Config/experiment.json').read_text(encoding='utf-8'))
INV=json.loads((ROOT/'Data/Satellite/scene_inventory.json').read_text(encoding='utf-8'))
QUALITY=pd.read_csv(ROOT/'Results/Prepared/scene_quality.csv')
SAMPLES=pd.read_csv(ROOT/'Results/Prepared/pixel_samples.csv.gz')
SCALE=float(CFG['hls']['reflectance_scale_factor'])
BAD_QA_BITS=(1<<1)|(1<<2)|(1<<3)|(1<<4)
NODATA=-9999.0
MODELS=['StumpfLogRatio','RandomForest','HistGradientBoosting','SVR_RBF']


def read_raster(path):
    with rasterio.open(path) as ds:
        return ds.read(1),ds.profile.copy(),ds.transform,ds.crs,ds.nodata,ds.bounds


def local_tides(allow_uncorrected):
    tide=pd.read_csv(ROOT/'Config/tide_levels.csv',dtype=str).fillna('')
    missing=[]; heights={}
    column='tide_anomaly_m_relative_to_survey_day_mean'
    for row in QUALITY.itertuples(index=False):
        match=tide[(tide.site==row.site)&(tide.datetime_utc==row.datetime_utc)]
        if match.empty or column not in match.columns or not str(match.iloc[0][column]).strip():
            missing.append((row.site,row.datetime_utc))
        else:
            heights[row.item_id]=float(match.iloc[0][column])
    if missing and not allow_uncorrected:
        raise SystemExit(f'{len(missing)} model tide anomalies are missing. Run Scripts/05_estimate_tides.py first, or use --preliminary-uncorrected only for exploratory maps.')
    if missing:
        print(f'WARNING: {len(missing)} tide anomalies are missing; writing preliminary maps without model-tide adjustment.')
        for row in QUALITY.itertuples(index=False): heights.setdefault(row.item_id,0.0)
    return heights,not bool(missing)


def full_scene_composite(site_key,site,items,tide_by_id):
    if not items: raise RuntimeError(f'No completed HLS scenes for {site_key}')
    items=sorted(items,key=lambda item:item['datetime_utc'])
    qscore=QUALITY.set_index('item_id')['quality_score'].to_dict()
    ref=items[0]
    _,profile,transform,crs,_,bounds=read_raster(ROOT/ref['assets']['blue']['file'])
    h,w=int(profile['height']),int(profile['width'])
    band_sum=np.zeros((4,h,w),dtype=np.float64)
    denominator=np.zeros((h,w),dtype=np.float32)
    tide_sum=np.zeros((h,w),dtype=np.float32)
    clear_count=np.zeros((h,w),dtype=np.uint16)
    water_union=np.zeros((h,w),dtype=bool)

    for index,item in enumerate(items,1):
        iid=item['item_id']; bands=[]; bands_valid=np.ones((h,w),dtype=bool)
        for band in ['blue','green','red','nir']:
            arr,p,t,r,nodata,_=read_raster(ROOT/item['assets'][band]['file'])
            if arr.shape!=(h,w) or t!=transform or r!=crs:
                raise RuntimeError(f'{site_key}: misaligned HLS raster for {iid}/{band}')
            raw=arr.astype(np.float32); valid=np.isfinite(raw)
            if nodata is not None: valid &= raw!=nodata
            bands.append(np.where(valid,raw*SCALE,np.nan))
            bands_valid &= valid
        qa,qp,qt,qr,qnodata,_=read_raster(ROOT/item['assets']['qa']['file'])
        if qa.shape!=(h,w) or qt!=transform or qr!=crs:
            raise RuntimeError(f'{site_key}: misaligned Fmask raster for {iid}')
        q=qa.astype(np.uint16); qa_valid=q!=255
        if qnodata is not None: qa_valid &= q!=qnodata
        clear=qa_valid&((q&BAD_QA_BITS)==0)&bands_valid
        water=qa_valid&((q&(1<<5))!=0)
        water_union |= water
        weight=float(max(qscore.get(iid,0.0),1e-6))
        weighted_clear=clear.astype(np.float32)*weight
        denominator += weighted_clear
        tide_sum += weighted_clear*float(tide_by_id.get(iid,0.0))
        clear_count += clear.astype(np.uint16)
        for band_index,values in enumerate(bands):
            band_sum[band_index] += np.nan_to_num(values,nan=0.0)*weighted_clear
        if index%10==0 or index==len(items):
            print(f'    Composited {index}/{len(items)} scenes for {site_key}',flush=True)

    composite=np.divide(band_sum,denominator[None,:,:],out=np.full_like(band_sum,np.nan),where=denominator[None,:,:]>0)
    tide_map=np.divide(tide_sum,denominator,out=np.zeros_like(tide_sum),where=denominator>0)
    return composite.astype(np.float32),tide_map,denominator>0,water_union,clear_count,profile,transform,crs,bounds


def sample_composite(site_samples,scene_ids,tide_by_id):
    values=np.zeros((len(site_samples),4),dtype=np.float64)
    den=np.zeros(len(site_samples),dtype=np.float64)
    tide_num=np.zeros(len(site_samples),dtype=np.float64)
    qscore=QUALITY.set_index('item_id')['quality_score'].to_dict()
    for iid in scene_ids:
        x=np.column_stack([pd.to_numeric(site_samples[f'{iid}__{b}'],errors='coerce').to_numpy(dtype=float)
                           for b in ['blue','green','red','nir']])
        clear=np.isfinite(x).all(axis=1)
        weight=clear.astype(float)*max(float(qscore.get(iid,0.0)),1e-6)
        values += np.nan_to_num(x,nan=0.0)*weight[:,None]
        den += weight
        tide_num += weight*float(tide_by_id.get(iid,0.0))
    x=np.divide(values,den[:,None],out=np.full_like(values,np.nan),where=den[:,None]>0)
    h=np.divide(tide_num,den,out=np.zeros_like(tide_num),where=den>0)
    return x,h,den>0


def stumpf_ratio(x):
    b=x[:,0]; g=x[:,1]
    good=np.isfinite(b)&np.isfinite(g)&(b>0)&(g>0)
    ratio=np.full(len(x),np.nan,dtype=float)
    denominator=np.log(1000.0*g[good]); numerator=np.log(1000.0*b[good])
    stable=np.abs(denominator)>1e-8
    indices=np.flatnonzero(good)[stable]
    ratio[indices]=numerator[stable]/denominator[stable]
    return ratio[:,None]


def model_for(name,seed):
    if name=='StumpfLogRatio': return LinearRegression()
    if name=='RandomForest':
        return RandomForestRegressor(n_estimators=200,min_samples_leaf=2,max_features=1.0,
                                     n_jobs=-1,random_state=seed)
    if name=='HistGradientBoosting':
        return HistGradientBoostingRegressor(max_iter=200,max_leaf_nodes=15,l2_regularization=0.1,
                                             random_state=seed)
    if name=='SVR_RBF':
        return make_pipeline(StandardScaler(),SVR(kernel='rbf',C=10.0,epsilon=0.1,
                                                   gamma='scale',cache_size=500))
    raise ValueError(name)


def write_raster(path,array,profile,nodata=NODATA,tags=None,description=None):
    out_profile=profile.copy(); out_profile.update(driver='GTiff',count=1,dtype='float32',nodata=nodata,
                                                    compress='deflate',predictor=3)
    with rasterio.open(path,'w',**out_profile) as ds:
        ds.write(np.where(np.isfinite(array),array,nodata).astype(np.float32),1)
        if description: ds.set_band_description(1,description)
        if tags: ds.update_tags(**{k:str(v) for k,v in tags.items()})


def write_mask(path,array,profile,description,tags=None):
    p=profile.copy(); p.update(driver='GTiff',count=1,dtype='uint8',nodata=255,compress='deflate')
    with rasterio.open(path,'w',**p) as ds:
        ds.write(array.astype(np.uint8),1); ds.set_band_description(1,description)
        if tags: ds.update_tags(**{k:str(v) for k,v in tags.items()})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preliminary-uncorrected',action='store_true',help='Allow blank tide levels and mark all map outputs uncorrected.')
    args=parser.parse_args()
    tide_by_id,tide_model_anomaly_applied=local_tides(args.preliminary_uncorrected)
    outdir=ROOT/'Results/Predictions/Maps'; outdir.mkdir(parents=True,exist_ok=True)
    map_records=[]
    support_distance=float(CFG.get('prediction_support_distance_m',90))
    margin=float(CFG.get('prediction_depth_margin_m',1.0))
    seed=int(CFG['random_seed'])

    for site_key,site in CFG['sites'].items():
        print(f'Stage 7/9: generating supported full-data maps for {site["display_name"]}.')
        df=SAMPLES[SAMPLES.site==site_key].reset_index(drop=True)
        items=[item for item in INV['items'] if item['site']==site_key and item['status']=='complete']
        ids=[item['item_id'] for item in sorted(items,key=lambda item:item['datetime_utc'])]
        (cube,hmap,feature_available,water_union,clear_count,profile,transform,crs,bounds)=full_scene_composite(site_key,site,items,tide_by_id)
        xsample,hsample,sample_available=sample_composite(df,ids,tide_by_id)
        sample_depth=df['depth_m'].to_numpy(dtype=float)
        rows=df['row'].to_numpy(dtype=int); cols=df['col'].to_numpy(dtype=int)
        if np.any(rows<0) or np.any(rows>=profile['height']) or np.any(cols<0) or np.any(cols>=profile['width']):
            raise RuntimeError(f'{site_key}: prepared sample row/column lies outside the HLS grid')

        # Use 90 m support around sample pixels that actually have composite features.
        valid_train=np.isfinite(xsample).all(axis=1)&np.isfinite(sample_depth)&sample_available
        if int(valid_train.sum())<100: raise RuntimeError(f'{site_key}: fewer than 100 usable training pixels')
        support_pixels=np.zeros((profile['height'],profile['width']),dtype=bool)
        support_pixels[rows[valid_train],cols[valid_train]]=True
        resx=abs(transform.a); resy=abs(transform.e)
        nearest_distance=distance_transform_edt(~support_pixels,sampling=(resy,resx)).astype(np.float32)
        generic_support=(nearest_distance<=support_distance)&water_union&feature_available&np.isfinite(cube).all(axis=0)
        tags={'site':site['display_name'],'survey_date':site['survey_date'],'vertical_reference':site['vertical_reference'],
              'sensor_pool':'HLS S30 + L30, quality-weighted mean, primary ±30-day window',
              'prediction_support_distance_m':support_distance,'tide_model_anomaly_applied':tide_model_anomaly_applied,
              'prediction_type':'full-data fit for visualization; not validation; see spatial out-of-fold metrics',
              'depth_positive':'down','depth_unit':'m'}
        safe_key=site_key.replace('/','_')
        write_raster(outdir/f'{safe_key}_nearest_sample_distance_m.tif',nearest_distance,profile,
                     description='Distance to nearest usable sounding pixel (m)',tags=tags)
        write_raster(outdir/f'{safe_key}_valid_scene_count.tif',clear_count.astype(np.float32),profile,
                     description='Number of Fmask-clear scenes with four valid bands',tags=tags)
        support_raster=np.where(generic_support,1,0).astype(np.uint8)
        write_mask(outdir/f'{safe_key}_prediction_support_mask.tif',support_raster,profile,
                   'Supported mapped pixel: within spatial limit, Fmask water union, and valid composite',tags)

        for name in MODELS:
            train_x=stumpf_ratio(xsample) if name=='StumpfLogRatio' else xsample
            map_x=np.moveaxis(cube,0,-1).reshape(-1,4)
            map_x=stumpf_ratio(map_x) if name=='StumpfLogRatio' else map_x
            train_instantaneous=sample_depth+hsample
            train_valid=np.isfinite(train_x).all(axis=1)&np.isfinite(train_instantaneous)&sample_available
            support=np.zeros_like(support_pixels); support[rows[train_valid],cols[train_valid]]=True
            distance=distance_transform_edt(~support,sampling=(resy,resx))
            allowed=(distance<=support_distance)&water_union&feature_available&np.isfinite(cube).all(axis=0)
            # Stumpf's logarithmic ratio is undefined for non-positive blue or
            # green reflectance. Keep those pixels outside its model-specific map.
            allowed &= np.isfinite(map_x).all(axis=1).reshape(profile['height'],profile['width'])
            model=model_for(name,seed); model.fit(train_x[train_valid],train_instantaneous[train_valid])
            selected=np.flatnonzero(allowed.ravel())
            if not len(selected): raise RuntimeError(f'{site_key}/{name}: no water pixels lie within the declared sample-support distance')
            instantaneous=model.predict(map_x[selected])
            chart_depth=instantaneous-hmap.ravel()[selected]
            train_chart=sample_depth[train_valid]
            domain=(chart_depth>=max(0.0,float(np.nanmin(train_chart))-margin)) & (chart_depth<=float(np.nanmax(train_chart))+margin)
            full=np.full(profile['height']*profile['width'],np.nan,dtype=np.float32)
            full[selected[domain]]=chart_depth[domain].astype(np.float32)
            full=full.reshape(profile['height'],profile['width'])
            map_path=outdir/f'{safe_key}_{name}_supported_depth.tif'
            model_tags={**tags,'model':name,'training_pixels':int(train_valid.sum()),
                        'depth_domain_min_m':max(0.0,float(np.nanmin(train_chart))-margin),
                        'depth_domain_max_m':float(np.nanmax(train_chart))+margin}
            write_raster(map_path,full,profile,tags=model_tags,
                         description='Full-data predicted depth below local hydrographic zero (m; positive down)')
            count=int(np.isfinite(full).sum())
            map_records.append({'site':site_key,'display_name':site['display_name'],'model':name,
                                'prediction_file':str(map_path.relative_to(ROOT)).replace('\\','/'),
                                'tide_model_anomaly_applied':tide_model_anomaly_applied,'n_training_pixels':int(train_valid.sum()),
                                'n_supported_prediction_pixels':count,'support_distance_limit_m':support_distance,
                                'predicted_depth_min_m':float(np.nanmin(full)) if count else np.nan,
                                'predicted_depth_max_m':float(np.nanmax(full)) if count else np.nan,
                                'vertical_reference':site['vertical_reference']})
            print(f'  {name}: {count:,} supported map pixels written to {map_path.relative_to(ROOT)}')

    pd.DataFrame(map_records).to_csv(outdir/'map_summary.csv',index=False)
    print(f'Stage 3 complete. Full-data maps are limited to {support_distance:g} m from valid survey pixels.')
    print('These maps visualize fitted bathymetry; use out-of-fold metrics from Stage 2 for accuracy claims.')


if __name__=='__main__': main()
