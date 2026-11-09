"""Run spatially blocked, buffered validation and scene-count experiments."""
from __future__ import annotations
import argparse, hashlib, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'Config/experiment.json').read_text(encoding='utf-8'))
QUALITY=pd.read_csv(ROOT/'Results/Prepared/scene_quality.csv')
SAMPLES=pd.read_csv(ROOT/'Results/Prepared/pixel_samples.csv.gz')
SEED=int(CFG['random_seed'])
N_SPLITS=int(CFG.get('spatial_cv_folds',5))
BLOCK_M=float(CFG.get('spatial_block_size_m',500))
BUFFER_M=float(CFG.get('spatial_buffer_m',90))
SUBSET_REPEATS=int(CFG.get('random_scene_subset_repeats',3))


def load_tide_anomalies(allow_uncorrected):
    path=ROOT/'Config/tide_levels.csv'
    tide=pd.read_csv(path,dtype=str).fillna('')
    column='tide_anomaly_m_relative_to_survey_day_mean'
    missing=[]; heights={}
    for row in QUALITY.itertuples(index=False):
        match=tide[(tide.site==row.site)&(tide.datetime_utc==row.datetime_utc)]
        if match.empty or column not in tide.columns or not str(match.iloc[0][column]).strip():
            missing.append((row.site,row.datetime_utc))
        else:
            heights[row.item_id]=float(match.iloc[0][column])
    if missing and not allow_uncorrected:
        preview='; '.join(f'{s} {t}' for s,t in missing[:4])
        raise SystemExit(f'Model tide-anomaly input is incomplete for {len(missing)} satellite acquisitions (for example: {preview}). Run Scripts/05_estimate_tides.py first, or use --preliminary-uncorrected only for exploratory outputs.')
    if missing:
        print(f'WARNING: {len(missing)} model tide anomalies are missing. Setting them to 0 m. Outputs have no model-tide adjustment.')
        for row in QUALITY.itertuples(index=False): heights.setdefault(row.item_id,0.0)
    return heights,not bool(missing)


def read_composite(df, scene_ids, scene_quality, tide_by_id, aggregation):
    bands=['blue','green','red','nir']
    cubes=[]; clear=[]; qweights=[]; tides=[]
    qmap=scene_quality.set_index('item_id')['quality_score'].to_dict()
    for sid in scene_ids:
        a=np.column_stack([pd.to_numeric(df[f'{sid}__{b}'],errors='coerce').to_numpy(dtype=float) for b in bands])
        cubes.append(a)
        clear.append(np.isfinite(a).all(axis=1))
        qweights.append(max(float(qmap.get(sid,0.0)),1e-6))
        tides.append(float(tide_by_id.get(sid,0.0)))
    cube=np.stack(cubes,axis=0)       # scene x sample x band
    valid=np.stack(clear,axis=0)      # scene x sample
    q=np.asarray(qweights,dtype=float)
    stage=np.asarray(tides,dtype=float)
    if aggregation=='single':
        X=cube[0].copy()
        h=np.full(len(df),stage[0],dtype=float)
        return X,h
    if aggregation=='quality_weighted_mean':
        w=valid*q[:,None]
        den=w.sum(axis=0)
        X=np.divide(np.nansum(cube*w[:,:,None],axis=0),den[:,None],out=np.full((len(df),4),np.nan),where=den[:,None]>0)
        h=np.divide((w*stage[:,None]).sum(axis=0),den,out=np.zeros(len(df)),where=den>0)
    elif aggregation=='unweighted_median':
        # Samples with no clear scene remain NaN and are excluded later by cv_fit.
        # Exclude those all-missing columns before nanmedian so NumPy does not
        # emit an "All-NaN slice" warning for every such sample.
        has_clear=valid.any(axis=0)
        X=np.full((cube.shape[1],cube.shape[2]),np.nan,dtype=float)
        h=np.full(cube.shape[1],np.nan,dtype=float)
        if has_clear.any():
            X[has_clear]=np.nanmedian(
                np.where(valid[:,has_clear,None],cube[:,has_clear,:],np.nan),axis=0
            )
            h[has_clear]=np.nanmedian(
                np.where(valid[:,has_clear],stage[:,None],np.nan),axis=0
            )
        h=np.where(np.isfinite(h),h,0.0)
    else: raise ValueError(aggregation)
    return X,h


def stumpf_ratio(X):
    b=X[:,0]; g=X[:,1]
    good=np.isfinite(b)&np.isfinite(g)&(b>0)&(g>0)
    ratio=np.full(len(X),np.nan,dtype=float)
    den=np.log(1000.0*g[good]); num=np.log(1000.0*b[good])
    stable=np.abs(den)>1e-8
    idx=np.flatnonzero(good)[stable]
    ratio[idx]=num[stable]/den[stable]
    return ratio[:,None]


def new_model(name):
    if name=='StumpfLogRatio': return LinearRegression()
    if name=='RandomForest':
        return RandomForestRegressor(n_estimators=200,min_samples_leaf=2,max_features=1.0,
                                     n_jobs=-1,random_state=SEED)
    if name=='HistGradientBoosting':
        return HistGradientBoostingRegressor(max_iter=200,max_leaf_nodes=15,l2_regularization=0.1,
                                             random_state=SEED)
    if name=='SVR_RBF':
        # Fixed a priori as a distinct kernel-regression benchmark. Scaling is
        # fitted within each outer fold through this pipeline.
        return make_pipeline(StandardScaler(),SVR(kernel='rbf',C=10.0,epsilon=0.1,
                                                   gamma='scale',cache_size=500))
    raise ValueError(name)


def cv_fit(X,depth,h,coords,groups,model_name,save_predictions=False):
    if model_name=='StumpfLogRatio': X=stumpf_ratio(X)
    y=np.asarray(depth,dtype=float)+np.asarray(h,dtype=float)
    valid=np.isfinite(y)&np.isfinite(X).all(axis=1)
    original=np.flatnonzero(valid)
    if len(original)<N_SPLITS*10: return [],[]
    xx=X[valid]; yy=y[valid]; hh=np.asarray(h)[valid]; xy=np.asarray(coords)[valid]; gg=np.asarray(groups)[valid]
    # Assign folds from the complete set of spatial blocks, before filtering
    # rows for scene-specific reflectance availability. This keeps the same
    # geographic fold boundaries across models and scene composites.
    all_groups=np.unique(np.asarray(groups))
    if len(all_groups)<N_SPLITS: return [],[]
    splitter=GroupKFold(n_splits=N_SPLITS)
    metric_rows=[]; pred_rows=[]
    group_splits=splitter.split(np.zeros(len(all_groups)),groups=all_groups)
    for fold,(train_group_ix,test_group_ix) in enumerate(group_splits,start=1):
        train_group_ids=all_groups[train_group_ix]
        test_group_ids=all_groups[test_group_ix]
        train0=np.flatnonzero(np.isin(gg,train_group_ids))
        test0=np.flatnonzero(np.isin(gg,test_group_ids))
        if len(train0)==0 or len(test0)==0: continue
        if BUFFER_M>0:
            # Keep only training samples farther than the exclusion buffer
            # from every held-out sample. Querying train points against a
            # tree of test points makes the boolean mask align with train0.
            tree=cKDTree(xy[test0]); distance,_=tree.query(xy[train0],k=1)
            keep=distance>BUFFER_M
            train=train0[keep]
        else: train=train0
        if len(train)<30 or len(test0)<10: continue
        model=new_model(model_name); model.fit(xx[train],yy[train])
        predicted_inst=model.predict(xx[test0])
        predicted_chart=predicted_inst-hh[test0]
        observed_chart=yy[test0]-hh[test0]
        err=predicted_chart-observed_chart
        metric_rows.append({'fold':fold,'n_train':int(len(train)),'n_test':int(len(test0)),
                            'rmse_m':float(np.sqrt(mean_squared_error(observed_chart,predicted_chart))),
                            'mae_m':float(mean_absolute_error(observed_chart,predicted_chart)),
                            'bias_m':float(np.mean(err)),'r2':float(r2_score(observed_chart,predicted_chart))})
        if save_predictions:
            valid_indices=original[test0]
            for j,ix in enumerate(valid_indices):
                pred_rows.append({'sample_index':int(ix),'fold':fold,'observed_depth_m':float(observed_chart[j]),
                                  'predicted_depth_m':float(predicted_chart[j]),'residual_m':float(err[j]),
                                  'tide_anomaly_m_relative_to_survey_day_mean':float(hh[test0][j])})
    return metric_rows,pred_rows


def make_scenarios(site, pool_rows):
    pool_rows=pool_rows.sort_values('datetime_utc').copy()
    pool_rows['absolute_date_offset_days']=pool_rows['date_offset_days'].abs()
    scene_ids=pool_rows.item_id.tolist()
    if not scene_ids: return []
    quality_sorted=(pool_rows.sort_values(['quality_score','absolute_date_offset_days','datetime_utc'],
                      ascending=[False,True,True]).item_id.tolist())
    survey_day=pd.Timestamp(site['survey_date'])
    s30=pool_rows[pool_rows['product']=='S30'].copy()
    nearest=None
    if not s30.empty:
        s30['abs_date_offset']=(pd.to_datetime(s30['datetime_utc']).dt.tz_localize(None)-survey_day).abs().dt.total_seconds()
        nearest=s30.sort_values(['abs_date_offset','quality_score'],ascending=[True,False]).iloc[0]['item_id']
    scenarios=[]
    # Every single scene is assessed with the established Stumpf log-ratio baseline.
    for sid in scene_ids:
        scenarios.append({'ids':[sid],'n':1,'selection':'single_scene','repeat':0,'aggregation':'single','models':['StumpfLogRatio']})
    # The quality-ranked best scene and survey-nearest S30 scene are model-comparison references.
    references=[('quality_best',quality_sorted[0]),('nearest_s30',nearest)]
    seen=set()
    for label,sid in references:
        if sid is None or (label,sid) in seen: continue
        seen.add((label,sid))
        scenarios.append({'ids':[sid],'n':1,'selection':label,'repeat':0,'aggregation':'single',
                          'models':['StumpfLogRatio','RandomForest','HistGradientBoosting']})
    requested=CFG.get('scene_count_levels',[2,4,8,12,16,'all'])
    counts=sorted(set(int(n) for n in requested if isinstance(n,int) and 1<n<=len(scene_ids)) | {len(scene_ids)})
    rng=np.random.default_rng(SEED+sum(ord(c) for c in str(site['survey_date']))+len(scene_ids))
    for n in counts:
        scenarios.append({'ids':quality_sorted[:n],'n':n,'selection':'quality_ranked','repeat':0,
                          'aggregation':'quality_weighted_mean','models':['StumpfLogRatio','RandomForest','HistGradientBoosting']})
        scenarios.append({'ids':quality_sorted[:n],'n':n,'selection':'quality_ranked','repeat':0,
                          'aggregation':'unweighted_median','models':['StumpfLogRatio','RandomForest','HistGradientBoosting']})
        if n<len(scene_ids):
            for rep in range(1,SUBSET_REPEATS+1):
                choose=sorted(rng.choice(scene_ids,size=n,replace=False).tolist())
                scenarios.append({'ids':choose,'n':n,'selection':'random_subset','repeat':rep,
                                  'aggregation':'quality_weighted_mean','models':['StumpfLogRatio','RandomForest','HistGradientBoosting']})
                scenarios.append({'ids':choose,'n':n,'selection':'random_subset','repeat':rep,
                                  'aggregation':'unweighted_median','models':['StumpfLogRatio','RandomForest','HistGradientBoosting']})
        if n==len(scene_ids):
            # Explicit all-scenes descriptor is the same fixed input as the quality-ranked full list.
            scenarios.append({'ids':scene_ids,'n':n,'selection':'all_scenes','repeat':0,
                              'aggregation':'quality_weighted_mean','models':['StumpfLogRatio','RandomForest','HistGradientBoosting']})
    return scenarios


def checkpoint_signature():
    """Bind resumable Stage 2 records to the inputs and code that produced them."""
    digest=hashlib.sha256()
    paths=[ROOT/'Config/experiment.json',ROOT/'Config/tide_levels.csv',
           ROOT/'Results/Prepared/scene_quality.csv',ROOT/'Results/Prepared/pixel_samples.csv.gz',
           Path(__file__).resolve()]
    for path in paths:
        digest.update(path.name.encode('utf-8'))
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):
                digest.update(chunk)
    return digest.hexdigest()


def checkpoint_key(site_key,pool_name,scenario):
    return json.dumps([site_key,pool_name,scenario['selection'],scenario['aggregation'],
                       int(scenario['n']),int(scenario['repeat']),scenario['ids']],
                      separators=(',',':'),ensure_ascii=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--preliminary-uncorrected',action='store_true',help='Allow missing model tide anomalies for exploratory outputs only.')
    parser.add_argument('--resume',action='store_true',help='Resume completed composites from a previous interrupted Stage 2 run.')
    args=parser.parse_args()
    tide_by_id,tide_model_anomaly_applied=load_tide_anomalies(args.preliminary_uncorrected)
    outm=ROOT/'Results/Metrics'; outp=ROOT/'Results/Predictions'; outl=ROOT/'Results/Logs'
    for p in [outm,outp,outl]: p.mkdir(parents=True,exist_ok=True)
    checkpoint_path=outl/'stage2_checkpoint.jsonl'
    signature=checkpoint_signature()
    checkpoint_records=[]; completed=set(); metrics=[]; saved=[]
    if args.resume and checkpoint_path.exists():
        with checkpoint_path.open(encoding='utf-8') as stream:
            for line_number,line in enumerate(stream,1):
                if not line.strip(): continue
                record=json.loads(line)
                if record.get('signature')!=signature:
                    raise SystemExit(f'Stage 2 checkpoint inputs/code differ at line {line_number}. Start a fresh run without --resume or restore the matching inputs.')
                checkpoint_records.append(record)
                completed.add(record['key'])
                metrics.extend(record.get('metrics',[]))
                saved.extend(record.get('predictions',[]))
        print(f'Resuming Stage 2: {len(completed)} completed composites loaded from checkpoint.')
    else:
        if checkpoint_path.exists(): checkpoint_path.unlink()
        if args.resume: print('No Stage 2 checkpoint found; starting a new run.')
    total_scenarios=0
    models=['StumpfLogRatio','RandomForest','HistGradientBoosting']
    print('Stage 3/9: buffered spatial block validation; held-out depth values are not used for model selection.')
    for site_key,site in CFG['sites'].items():
        df=SAMPLES[SAMPLES.site==site_key].reset_index(drop=True)
        qsite=QUALITY[QUALITY.site==site_key].copy()
        pools=[('S30_only',qsite[qsite['product']=='S30']),('HLS_Mixed',qsite)]
        xy=df[['x_m','y_m']].to_numpy(dtype=float)
        groups=(np.floor(xy[:,0]/BLOCK_M).astype(int).astype(str)+'_'+np.floor(xy[:,1]/BLOCK_M).astype(int).astype(str))
        depth=df['depth_m'].to_numpy(dtype=float)
        for pool_name,pool in pools:
            scenarios=make_scenarios(site,pool)
            print(f'  {site_key} / {pool_name}: {len(pool)} scenes, {len(scenarios)} composite evaluations')
            for s_index,scenario in enumerate(scenarios,start=1):
                key=checkpoint_key(site_key,pool_name,scenario)
                if key in completed:
                    total_scenarios+=1
                    continue
                X,h=read_composite(df,scenario['ids'],pool,tide_by_id,scenario['aggregation'])
                scene_signature='|'.join(scenario['ids'])
                scenario_metrics=[]; scenario_predictions=[]
                for model_name in scenario['models']:
                    folds,pred=cv_fit(X,depth,h,xy,groups,model_name,save_predictions=(
                        pool_name=='HLS_Mixed' and scenario['selection']=='all_scenes' and
                        scenario['aggregation']=='quality_weighted_mean'))
                    base={'site':site_key,'scene_pool':pool_name,'aggregation':scenario['aggregation'],
                          'scene_selection':scenario['selection'],'n_scenes':scenario['n'],'replicate':scenario['repeat'],
                          'model':model_name,'scene_ids':scene_signature,
                          'tide_model_anomaly_applied':tide_model_anomaly_applied}
                    for fr in folds: scenario_metrics.append({**base,**fr})
                    if pred:
                        for pr in pred:
                            srow=df.iloc[pr['sample_index']]
                            scenario_predictions.append({**base,**pr,'x_m':float(srow.x_m),'y_m':float(srow.y_m),
                                                         'n_soundings':int(srow.n_soundings),'vertical_reference':srow.vertical_reference})
                record={'signature':signature,'key':key,'site':site_key,'scene_pool':pool_name,
                        'metrics':scenario_metrics,'predictions':scenario_predictions}
                with checkpoint_path.open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(record,ensure_ascii=False)+'\n')
                checkpoint_records.append(record); completed.add(key)
                metrics.extend(scenario_metrics); saved.extend(scenario_predictions)
                total_scenarios+=1
                if s_index%10==0 or s_index==len(scenarios):
                    print(f'    {s_index}/{len(scenarios)} composites complete',flush=True)
        short_window=int(CFG.get('sensitivity_temporal_window_days',14))
        near=qsite[qsite.date_offset_days.abs()<=short_window]
        sensitivity_pools=[('S30_only_14d',near[near['product']=='S30']),('HLS_Mixed_14d',near)]
        for pool_name,pool in sensitivity_pools:
            if pool.empty: continue
            ids=pool.sort_values('datetime_utc').item_id.tolist()
            print(f'  {site_key} / {pool_name}: {len(ids)} scenes; full-window sensitivity')
            for aggregation in ['quality_weighted_mean','unweighted_median']:
                scenario={'ids':ids,'n':len(ids),'selection':'all_scenes','repeat':0,'aggregation':aggregation}
                key=checkpoint_key(site_key,pool_name,scenario)
                if key in completed:
                    total_scenarios+=1
                    continue
                X,h=read_composite(df,ids,pool,tide_by_id,aggregation)
                scene_signature='|'.join(ids)
                scenario_metrics=[]
                for model_name in models:
                    folds,_=cv_fit(X,depth,h,xy,groups,model_name,save_predictions=False)
                    base={'site':site_key,'scene_pool':pool_name,'aggregation':aggregation,
                          'scene_selection':'all_scenes','n_scenes':len(ids),'replicate':0,
                          'model':model_name,'scene_ids':scene_signature,
                          'tide_model_anomaly_applied':tide_model_anomaly_applied}
                    for fr in folds: scenario_metrics.append({**base,**fr})
                record={'signature':signature,'key':key,'site':site_key,'scene_pool':pool_name,
                        'metrics':scenario_metrics,'predictions':[]}
                with checkpoint_path.open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(record,ensure_ascii=False)+'\n')
                checkpoint_records.append(record); completed.add(key); metrics.extend(scenario_metrics)
                total_scenarios+=1
    mdf=pd.DataFrame(metrics)
    if mdf.empty: raise SystemExit('No valid folds completed; inspect sample coverage and spatial block settings.')
    mdf.to_csv(outm/'cv_metrics.csv',index=False)
    grouped=(mdf.groupby(['site','scene_pool','aggregation','scene_selection','n_scenes','model','tide_model_anomaly_applied'],dropna=False)
             .agg(rmse_mean_m=('rmse_m','mean'),rmse_sd_m=('rmse_m','std'),mae_mean_m=('mae_m','mean'),
                  bias_mean_m=('bias_m','mean'),r2_mean=('r2','mean'),folds=('fold','nunique'),
                  replicates=('replicate','nunique')).reset_index())
    grouped.to_csv(outm/'scene_count_summary.csv',index=False)
    if saved:
        pd.DataFrame(saved).to_csv(outp/'oof_all_scenes_quality_weighted.csv.gz',index=False,compression='gzip')
    runmeta={'tide_model_anomaly_applied':tide_model_anomaly_applied,
             'preliminary_uncorrected':not tide_model_anomaly_applied,
             'tide_model':'NASA GSFC GOT5.6 via pyTMD',
             'tide_input':'Astronomical tide anomaly relative to the modeled mean over the local survey date (m)',
             'survey_day_reference':'Local civil-day GOT5.6 mean; field collection intervals are documented, but individual sounding timestamps are unavailable.',
             'survey_collection_windows_local':{site:values.get('survey_collection_window_local',{}) for site,values in CFG['sites'].items()},
             'datum_offset_applied':False,
             'datum_offset_note':'Subtracting each site’s modeled survey-day mean from its acquisition predictions removes the arbitrary constant GOT5.6 reference. This provides a relative tidal adjustment without asserting an absolute tie to the local hydrographic zero. Outputs remain in each survey’s local hydrographic-zero frame.',
             'nontidal_water_level_residual_applied':False,
             'tide_model_limitations':['Global 1/8-degree harmonic model; coastal wet-node extrapolation limited to 10 km.',
                                       'No local-port gauge validation; no meteorological surge or wave setup correction.'],
             'spatial_folds':N_SPLITS,'block_size_m':BLOCK_M,'training_exclusion_buffer_m':BUFFER_M,
             'scene_subset_repeats':SUBSET_REPEATS,'models':models,'composite_scenario_count':total_scenarios,
             'depth_target':'positive-down depth below the site-specific hydrographic zero',
             'warnings':([] if tide_model_anomaly_applied else ['No GOT5.6 tide anomalies were supplied; all anomalies set to zero.'])}
    (outl/'run_metadata.json').write_text(json.dumps(runmeta,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('  Wrote Results/Metrics/cv_metrics.csv and scene_count_summary.csv')
    if saved: print('  Wrote Results/Predictions/oof_all_scenes_quality_weighted.csv.gz')
    print('Stage 2 complete. These are spatially held-out predictions; inspect run_metadata.json before interpretation.')

if __name__=='__main__': main()
