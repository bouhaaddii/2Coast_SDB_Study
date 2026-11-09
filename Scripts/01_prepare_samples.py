"""Build one row per submerged 30 m survey pixel with Fmask-screened HLS features."""
from __future__ import annotations
import csv, json, re
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.transform import rowcol

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'Config/experiment.json').read_text(encoding='utf-8'))
INV=json.loads((ROOT/'Data/Satellite/scene_inventory.json').read_text(encoding='utf-8'))
SCALE=float(CFG['hls']['reflectance_scale_factor'])
BAD_QA_BITS=(1<<1)|(1<<2)|(1<<3)|(1<<4)


def read_survey(site_key,site):
    path=ROOT/site['survey_points']
    if path.suffix.lower()=='.csv':
        df=pd.read_csv(path)
        zcol='z_source'
    else:
        df=pd.read_csv(path,sep=r'\s+',header=None,names=['point_id','x','y','z_source'])
        zcol='z_source'
    for col in ['x','y',zcol]: df[col]=pd.to_numeric(df[col],errors='coerce')
    df=df.dropna(subset=['x','y',zcol]).copy()
    raw_valid_xyz_records=len(df)
    exact_xyz_duplicates_removed=int(df.duplicated(['x','y',zcol]).sum())
    # CAD exports can repeat the same XYZ observation in multiple entities.
    # Remove exact copies so they do not receive extra weight in the per-pixel
    # median. Near-coincident points and same-XY points with different Z remain.
    df=df.drop_duplicates(subset=['x','y',zcol],keep='first').copy()
    # Z is elevation relative to the site's own hydrographic zero; submerged depth is positive down.
    df=df[df[zcol]<=0].copy()
    df['depth_m']=-df[zcol]
    tr=Transformer.from_crs(site['survey_crs'],site['image_crs'],always_xy=True)
    df['x_img'],df['y_img']=tr.transform(df['x'].to_numpy(),df['y'].to_numpy())
    return df,raw_valid_xyz_records,exact_xyz_duplicates_removed


def load_raster(path):
    with rasterio.open(path) as ds:
        return ds.read(1),ds.profile.copy(),ds.transform,ds.crs,ds.nodata


def process_site(site_key,site,items):
    if not items: raise RuntimeError(f'No completed imagery for {site_key}')
    items=sorted(items,key=lambda r:r['datetime_utc'])
    ref_path=ROOT/items[0]['assets']['blue']['file']
    _,ref_profile,transform,crs,_=load_raster(ref_path)
    if str(crs)!=site['image_crs']:
        raise RuntimeError(f'{site_key}: image CRS {crs} differs from configured {site["image_crs"]}')
    survey,raw_valid_xyz_records,exact_xyz_duplicates_removed=read_survey(site_key,site)
    rows,cols=rowcol(transform,survey['x_img'].to_numpy(),survey['y_img'].to_numpy())
    survey['row']=np.asarray(rows,dtype=int); survey['col']=np.asarray(cols,dtype=int)
    inside=(survey['row']>=0)&(survey['row']<ref_profile['height'])&(survey['col']>=0)&(survey['col']<ref_profile['width'])
    survey=survey.loc[inside].copy()
    if survey.empty: raise RuntimeError(f'{site_key}: no submerged survey points fall inside the HLS chips')
    grouped=(survey.groupby(['row','col'],as_index=False)
             .agg(depth_m=('depth_m','median'),n_soundings=('depth_m','size'),
                  z_median=('z_source','median')))
    xy=[rasterio.transform.xy(transform,int(r),int(c),offset='center') for r,c in grouped[['row','col']].to_numpy()]
    grouped['x_m']=[p[0] for p in xy]; grouped['y_m']=[p[1] for p in xy]
    grouped['site']=site_key; grouped['vertical_reference']=site['vertical_reference']
    grouped['image_crs']=site['image_crs']
    rows_i=grouped['row'].to_numpy(); cols_i=grouped['col'].to_numpy()
    # Rank scenes using image-only QA over a fixed water mask. Do not derive
    # scene weights from the locations of the bathymetric labels: doing so
    # would let validation-label sampling influence the input selection.
    image_data={}; fixed_water_mask=np.zeros((ref_profile['height'],ref_profile['width']),dtype=bool)
    for item in items:
        iid=item['item_id']; bands={}; bands_valid=np.ones(fixed_water_mask.shape,dtype=bool)
        for band in ['blue','green','red','nir']:
            arr,profile,t,band_crs,nodata=load_raster(ROOT/item['assets'][band]['file'])
            if arr.shape!=(ref_profile['height'],ref_profile['width']) or t!=transform or band_crs!=crs:
                raise RuntimeError(f'{site_key}: misaligned input grid for {iid}/{band}')
            raw=arr.astype(np.float32)
            valid=np.isfinite(raw)
            if nodata is not None: valid &= raw!=nodata
            bands[band]=np.where(valid,raw*SCALE,np.nan)
            bands_valid &= valid
        qa,qa_profile,qa_transform,qa_crs,qa_nodata=load_raster(ROOT/item['assets']['qa']['file'])
        if qa.shape!=(ref_profile['height'],ref_profile['width']) or qa_transform!=transform or qa_crs!=crs:
            raise RuntimeError(f'{site_key}: misaligned Fmask grid for {iid}')
        q=qa.astype(np.uint16)
        qa_valid=(q!=255)
        if qa_nodata is not None: qa_valid &= q!=qa_nodata
        clear=qa_valid&((q&BAD_QA_BITS)==0)&bands_valid
        water=qa_valid&((q&(1<<5))!=0)
        fixed_water_mask |= water
        image_data[iid]={'bands':bands,'qa_valid':qa_valid,'clear':clear,'water':water,'qa':q}

    if not fixed_water_mask.any():
        raise RuntimeError(f'{site_key}: no Fmask water pixels found in the image chips; cannot compute image-only scene quality.')

    quality=[]; scene_ids=[]; feature_columns={}
    water_pixel_count=int(fixed_water_mask.sum())
    for item in items:
        iid=item['item_id']; product=item['product']; scene_ids.append(iid)
        data=image_data[iid]
        q=data['qa'][rows_i,cols_i]
        qa_valid=data['qa_valid'][rows_i,cols_i]
        clear=data['clear'][rows_i,cols_i]
        water_flag=data['water'][rows_i,cols_i]
        aot=((q>>6)&3).astype(float)
        for band,v in data['bands'].items():
            sample_values=v[rows_i,cols_i]
            feature_columns[f'{iid}__{band}']=np.where(clear,sample_values,np.nan)
        feature_columns[f'{iid}__clear']=clear.astype(np.uint8)
        feature_columns[f'{iid}__water_flag']=water_flag.astype(np.uint8)
        feature_columns[f'{iid}__aot']=np.where(qa_valid,aot,np.nan)
        dt=item['datetime_utc']
        survey_day=date.fromisoformat(site['survey_date'])
        image_day=date.fromisoformat(dt[:10])
        quality.append({'site':site_key,'item_id':iid,'product':product,'datetime_utc':dt,
                        'date_offset_days':(image_day-survey_day).days,
                        'cloud_cover_tile_percent':item.get('cloud_cover_tile_percent'),
                        'clear_fraction_at_survey_pixels':float(np.mean(clear)),
                        'water_flag_fraction_at_survey_pixels':float(np.mean(water_flag)),
                        'clear_fraction_over_fixed_water_mask':float(np.mean(data['clear'][fixed_water_mask])),
                        'clear_pixel_count':int(np.sum(data['clear']&fixed_water_mask)),
                        'fixed_water_mask_pixel_count':water_pixel_count,
                        'survey_pixel_count':int(len(grouped)),
                        'quality_score':float(np.mean(data['clear'][fixed_water_mask]))})
    grouped=pd.concat([grouped,pd.DataFrame(feature_columns,index=grouped.index)],axis=1)
    point_cleaning={
        'raw_valid_xyz_records':int(raw_valid_xyz_records),
        'exact_xyz_duplicates_removed':int(exact_xyz_duplicates_removed),
        'unique_xyz_records_after_deduplication':int(raw_valid_xyz_records-exact_xyz_duplicates_removed),
        'submerged_unique_soundings_inside_hls_chip':int(len(survey)),
    }
    return grouped,quality,scene_ids,point_cleaning


def main():
    print('Stage 1/4: reading local bathymetric soundings and clipped HLS inputs.')
    all_samples=[]; all_quality=[]; point_cleaning_by_site={}
    for site_key,site in CFG['sites'].items():
        items=[r for r in INV['items'] if r['site']==site_key and r['status']=='complete']
        samples,quality,_,point_cleaning=process_site(site_key,site,items)
        all_samples.append(samples); all_quality.extend(quality)
        point_cleaning_by_site[site_key]=point_cleaning
        print(f'  {site_key}: {len(samples):,} submerged 30 m pixels from {sum(samples.n_soundings):,} unique soundings; '
              f'removed {point_cleaning["exact_xyz_duplicates_removed"]:,} exact XYZ duplicate rows; scenes={len(items)}')
    out=ROOT/'Results/Prepared'; out.mkdir(parents=True,exist_ok=True)
    samples=pd.concat(all_samples,ignore_index=True,sort=False)
    samples.to_csv(out/'pixel_samples.csv.gz',index=False,compression='gzip')
    quality=pd.DataFrame(all_quality).sort_values(['site','datetime_utc'])
    quality.to_csv(out/'scene_quality.csv',index=False)
    summary={'pixel_samples':len(samples),'sites':{},'scene_count':len(quality),
             'fmask_bad_bits':{'cloud':1,'adjacent_cloud_shadow':2,'cloud_shadow':3,'snow_ice':4,'water_flag':5,'aerosol': '6-7'},
             'reflectance_scale_factor':SCALE,'training_depth_rule':'depth_m = -z_source; z_source <= 0 only',
             'pixel_aggregation':'median depth per HLS 30 m grid cell before validation',
             'survey_point_cleaning':'Exact duplicate XYZ records are removed before submerged-point filtering and pixel aggregation; same-XY points with different Z are retained.',
             'scene_quality_definition':'Fraction of pixels in the fixed union of Fmask water pixels across candidate acquisitions that have valid four-band reflectance and no cloud, adjacent-cloud/shadow, cloud-shadow, or snow/ice flags. Computed over the image chip, independently of survey sample locations and depth labels.'}
    for site_key,df in samples.groupby('site'):
        summary['sites'][site_key]={'pixel_count':int(len(df)),'sounding_count':int(df['n_soundings'].sum()),
                                    'depth_min_m':float(df['depth_m'].min()),'depth_max_m':float(df['depth_m'].max()),
                                    'local_vertical_reference':str(df['vertical_reference'].iloc[0]),
                                    **point_cleaning_by_site[site_key]}
    (out/'preparation_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('  Wrote Results/Prepared/pixel_samples.csv.gz')
    print('  Wrote Results/Prepared/scene_quality.csv')
    print('Stage 1 complete. No model has been trained and no predictions were made.')

if __name__=='__main__': main()
