"""Test cross-coast transfer with leakage-free local vertical-offset calibration.

Because the sites use different hydrographic zeros, raw zero-shot absolute-depth
scores are marked non-comparable. Target-site calibration uses only target points
outside the spatially held-out fold and outside its exclusion buffer.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import GroupKFold

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'Scripts'))
spec=importlib.util.spec_from_file_location('validation_stage',ROOT/'Scripts/02_run_validation.py')
V=importlib.util.module_from_spec(spec); spec.loader.exec_module(V)
MODELS=['StumpfLogRatio','RandomForest','HistGradientBoosting','SVR_RBF']
CALIBRATION_FRACTIONS=V.CFG.get('cross_site_calibration_fractions',[0.0,0.01,0.05,0.10])
CALIBRATION_REPEATS=int(V.CFG.get('cross_site_calibration_repeats',5))


def model_features(X,name):
    return V.stumpf_ratio(X) if name=='StumpfLogRatio' else X


def metrics_for(observed,predicted):
    residual=predicted-observed
    return {'rmse_m':float(np.sqrt(mean_squared_error(observed,predicted))),
            'mae_m':float(mean_absolute_error(observed,predicted)),
            'bias_m':float(np.mean(residual)),
            'r2':float(r2_score(observed,predicted))}


def main():
    tides,available=V.load_tide_anomalies(False)
    if not available: raise SystemExit('Tide-anomaly table is incomplete. Run Scripts/05_estimate_tides.py first.')
    sites=list(V.CFG['sites'].items())
    site_data={}
    for site_key,_ in sites:
        df=V.SAMPLES[V.SAMPLES.site==site_key].reset_index(drop=True)
        quality=V.QUALITY[V.QUALITY.site==site_key].copy()
        ids=quality.sort_values('datetime_utc').item_id.tolist()
        X,h=V.read_composite(df,ids,quality,tides,'quality_weighted_mean')
        depth=df.depth_m.to_numpy(dtype=float)
        coords=df[['x_m','y_m']].to_numpy(dtype=float)
        groups=(np.floor(coords[:,0]/V.BLOCK_M).astype(int).astype(str)+'_'+
                np.floor(coords[:,1]/V.BLOCK_M).astype(int).astype(str))
        valid=np.isfinite(X).all(axis=1)&np.isfinite(depth)&np.isfinite(h)
        site_data[site_key]={'df':df,'X':X,'h':h,'depth':depth,'coords':coords,
                             'groups':groups,'valid':valid,'scene_ids':ids}

    result_rows=[]
    print('Stage 5/9: leave-one-coast-out transfer with target-only offset calibration.')
    for target_key,_ in sites:
        source_key=next(key for key,_ in sites if key!=target_key)
        src=site_data[source_key]; tgt=site_data[target_key]
        source_valid=src['valid']
        target_original=np.flatnonzero(tgt['valid'])
        if source_valid.sum()<30 or len(target_original)<V.N_SPLITS*10:
            print(f'  Skipping {source_key} -> {target_key}: insufficient complete samples.')
            continue
        target_groups=np.unique(tgt['groups'])
        for model_name in MODELS:
            xsource=model_features(src['X'][source_valid],model_name)
            ysource=src['depth'][source_valid]+src['h'][source_valid]
            fitmask=np.isfinite(xsource).all(axis=1)&np.isfinite(ysource)
            model=V.new_model(model_name); model.fit(xsource[fitmask],ysource[fitmask])
            xtarget_all=model_features(tgt['X'][target_original],model_name)
            target_model_valid=np.isfinite(xtarget_all).all(axis=1)
            target_original_model=target_original[target_model_valid]
            xtarget=xtarget_all[target_model_valid]
            target_h=tgt['h'][target_original_model]
            target_depth=tgt['depth'][target_original_model]
            target_xy=tgt['coords'][target_original_model]
            target_group=tgt['groups'][target_original_model]
            prediction=(model.predict(xtarget)-target_h)
            all_group_ids=np.unique(tgt['groups'])
            # The split definitions depend only on target block geometry and are
            # reused unchanged for every model and calibration budget.
            fold_list=list(GroupKFold(n_splits=V.N_SPLITS).split(
                np.zeros(len(all_group_ids)),groups=all_group_ids))
            for fold,(train_group_ix,test_group_ix) in enumerate(fold_list,start=1):
                train_group_ids=all_group_ids[train_group_ix]
                test_group_ids=all_group_ids[test_group_ix]
                train0=np.flatnonzero(np.isin(target_group,train_group_ids))
                test0=np.flatnonzero(np.isin(target_group,test_group_ids))
                if len(train0)==0 or len(test0)<10: continue
                if V.BUFFER_M>0:
                    tree=cKDTree(target_xy[test0]); distances,_=tree.query(target_xy[train0],k=1)
                    calibration_pool=train0[distances>V.BUFFER_M]
                else:
                    calibration_pool=train0
                if len(calibration_pool)<5: continue
                seed_base=int(V.SEED+fold*1009+sum(ord(ch) for ch in target_key+source_key+model_name))
                draws={}
                for fraction in CALIBRATION_FRACTIONS:
                    if fraction==0:
                        draws[fraction]=[np.array([],dtype=int)]
                        continue
                    count=min(len(calibration_pool),max(5,int(np.ceil(fraction*len(calibration_pool)))))
                    rng=np.random.default_rng(seed_base+int(fraction*100000))
                    draws[fraction]=[np.sort(rng.choice(calibration_pool,size=count,replace=False))
                                     for _ in range(CALIBRATION_REPEATS)]
                for fraction,subsets in draws.items():
                    for repeat,cal_ix in enumerate(subsets,1):
                        if not len(cal_ix):
                            calibrations=[('zero_shot_source_only',prediction[test0],0.0,1.0)]
                        else:
                            offset=float(np.median(target_depth[cal_ix]-prediction[cal_ix]))
                            additive=prediction[test0]+offset
                            affine=LinearRegression().fit(prediction[cal_ix,None],target_depth[cal_ix])
                            affine_prediction=affine.predict(prediction[test0,None])
                            calibrations=[('median_additive_offset',additive,offset,1.0),
                                          ('affine_linear',affine_prediction,
                                           float(affine.intercept_),float(affine.coef_[0]))]
                        for calibration_method,calibrated,intercept,slope in calibrations:
                            score=metrics_for(target_depth[test0],calibrated)
                            result_rows.append({'source_site':source_key,'target_site':target_key,
                                                'model':model_name,'target_fold':fold,
                                                'calibration_fraction':fraction,
                                                'calibration_points_n':int(len(cal_ix)),
                                                'calibration_repeat':repeat,
                                                'calibration_method':calibration_method,
                                                'calibration_type':calibration_method,
                                                'calibration_intercept_m':intercept,
                                                'calibration_slope':slope,
                                                'absolute_datum_comparable':bool(len(cal_ix)>0),
                                                'source_training_points_n':int(fitmask.sum()),
                                                'target_test_points_n':int(len(test0)),
                                                'target_exclusion_buffer_m':float(V.BUFFER_M),
                                                'target_scene_ids':'|'.join(tgt['scene_ids']),**score})
        print(f'  Completed {source_key} -> {target_key}.')
    if not result_rows: raise SystemExit('No cross-site transfer folds completed.')
    out=ROOT/'Results/Additional_Analyses'; out.mkdir(parents=True,exist_ok=True)
    raw=pd.DataFrame(result_rows)
    # Preserve the original additive-offset transfer outputs as a baseline.
    raw.to_csv(out/'cross_site_transfer_calibration_comparison_fold_metrics.csv',index=False)
    valid=raw[raw.absolute_datum_comparable].copy()
    summary=(valid.groupby(['source_site','target_site','model','calibration_method','calibration_fraction'],as_index=False)
             .agg(calibration_points_mean=('calibration_points_n','mean'),
                  rmse_mean_m=('rmse_m','mean'),rmse_sd_across_folds_repeats_m=('rmse_m','std'),
                  mae_mean_m=('mae_m','mean'),bias_mean_m=('bias_m','mean'),r2_mean=('r2','mean'),
                  folds=('target_fold','nunique'),repeats=('calibration_repeat','nunique')))
    summary.to_csv(out/'cross_site_transfer_calibration_comparison_summary.csv',index=False)
    print(f'Wrote transfer metrics to {out.relative_to(ROOT)}')
    print('Zero-shot scores are retained in the fold table for diagnostics and are excluded from the absolute-depth summary because the two local hydrographic zeros are not tied.')
if __name__=='__main__': main()
