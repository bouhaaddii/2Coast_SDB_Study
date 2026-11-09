"""Select temporal composite size inside spatial CV, then test it on outer blocks.

This nested comparison checks whether a smaller quality-ranked scene set can
beat the all-scene composite without selecting the scene count on the reported
outer test blocks. Candidate counts are 4, 8, 12, 16, and all available scenes.
The inner selection and outer test both use buffered spatial folds.
"""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import GroupKFold

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'Scripts'))
spec=importlib.util.spec_from_file_location('validation_stage',ROOT/'Scripts/02_run_validation.py')
V=importlib.util.module_from_spec(spec); spec.loader.exec_module(V)
MODELS=['RandomForest','HistGradientBoosting','SVR_RBF']
INNER_FOLDS=3
CANDIDATE_COUNTS=[4,8,12,16]


def blocked_score(X,depth,h,xy,train_indices,test_indices,model_name):
    """Fit a model after buffering training points away from this validation block."""
    y=np.asarray(depth,dtype=float)+np.asarray(h,dtype=float)
    valid=np.isfinite(y)&np.isfinite(X).all(axis=1)&np.isfinite(h)
    train=np.asarray(train_indices,dtype=int)
    test=np.asarray(test_indices,dtype=int)
    train=train[valid[train]]; test=test[valid[test]]
    if len(train)==0 or len(test)<10: return None
    if V.BUFFER_M>0:
        tree=cKDTree(xy[test]); distance,_=tree.query(xy[train],k=1)
        train=train[distance>V.BUFFER_M]
    if len(train)<30 or len(test)<10: return None
    model=V.new_model(model_name)
    model.fit(X[train],y[train])
    predicted=model.predict(X[test])-h[test]
    observed=depth[test]
    residual=predicted-observed
    result={'rmse_m':float(np.sqrt(mean_squared_error(observed,predicted))),
            'mae_m':float(mean_absolute_error(observed,predicted)),
            'bias_m':float(np.mean(residual)),
            'r2':float(r2_score(observed,predicted)),
            'n_train':int(len(train)),'n_test':int(len(test))}
    return result,test,predicted,residual


def main():
    tide_by_id,available=V.load_tide_anomalies(False)
    if not available: raise SystemExit('Tide-anomaly input is incomplete. Run Scripts/05_estimate_tides.py first.')
    metric_rows=[]; prediction_rows=[]; inner_rows=[]
    print('Nested quality-ranked scene-count selection with buffered spatial folds.')
    print(f'  Candidate models={MODELS}; inner folds={INNER_FOLDS}; outer folds={V.N_SPLITS}.')

    for site_key,site in V.CFG['sites'].items():
        df=V.SAMPLES[V.SAMPLES.site==site_key].reset_index(drop=True)
        quality=V.QUALITY[V.QUALITY.site==site_key].copy()
        xy=df[['x_m','y_m']].to_numpy(dtype=float)
        groups=(np.floor(xy[:,0]/V.BLOCK_M).astype(int).astype(str)+'_'+
                np.floor(xy[:,1]/V.BLOCK_M).astype(int).astype(str))
        depth=df.depth_m.to_numpy(dtype=float)
        pools=[('S30_only',quality[quality['product']=='S30']),('HLS_Mixed',quality)]
        for pool_name,pool in pools:
            pool=pool.copy()
            pool['abs_offset']=pool.date_offset_days.abs()
            ranked=pool.sort_values(['quality_score','abs_offset','datetime_utc'],
                                    ascending=[False,True,True]).item_id.tolist()
            if not ranked: continue
            counts=sorted(set(min(n,len(ranked)) for n in CANDIDATE_COUNTS)|{len(ranked)})
            candidates=[]
            for n in counts:
                ids=ranked[:n]
                X,h=V.read_composite(df,ids,pool,tide_by_id,'quality_weighted_mean')
                candidates.append({'n':n,'ids':ids,'X':X,'h':h})
            # A common sample mask makes the outer comparison use the same test
            # cells for every candidate scene count.
            common=np.isfinite(depth)
            for candidate in candidates:
                common &= np.isfinite(candidate['h'])&np.isfinite(candidate['X']).all(axis=1)
            sample_n=int(common.sum())
            all_groups=np.unique(groups)
            if len(all_groups)<V.N_SPLITS:
                print(f'  {site_key}/{pool_name}: too few spatial groups; skipped.')
                continue
            outer=GroupKFold(n_splits=V.N_SPLITS)
            split_iter=outer.split(np.zeros(len(all_groups)),groups=all_groups)
            print(f'  {site_key}/{pool_name}: scenes={len(ranked)}, common pixels={sample_n}, candidates={counts}.',flush=True)

            for fold,(outer_train_group_ix,outer_test_group_ix) in enumerate(split_iter,start=1):
                print(f'    {site_key}/{pool_name}: outer fold {fold}/{V.N_SPLITS} — inner scene selection.',flush=True)
                train_group_ids=all_groups[outer_train_group_ix]
                test_group_ids=all_groups[outer_test_group_ix]
                outer_train=np.flatnonzero(common&np.isin(groups,train_group_ids))
                outer_test=np.flatnonzero(common&np.isin(groups,test_group_ids))
                if len(outer_train)==0 or len(outer_test)<10: continue
                # Remove the declared buffer once before the inner model-selection pool.
                if V.BUFFER_M>0:
                    tree=cKDTree(xy[outer_test]); distance,_=tree.query(xy[outer_train],k=1)
                    outer_train=outer_train[distance>V.BUFFER_M]
                inner_groups=np.unique(groups[outer_train])
                n_inner=min(INNER_FOLDS,len(inner_groups))
                if n_inner<2: continue
                inner=GroupKFold(n_splits=n_inner)
                inner_splits=list(inner.split(np.zeros(len(inner_groups)),groups=inner_groups))
                scores_by_candidate={}
                for candidate in candidates:
                    for model_name in MODELS:
                        inner_rmse=[]
                        for inner_train_group_ix,inner_val_group_ix in inner_splits:
                            inner_train_ids=inner_groups[inner_train_group_ix]
                            inner_val_ids=inner_groups[inner_val_group_ix]
                            # Restrict both sets to the current outer training pool.
                            inner_train=outer_train[np.isin(groups[outer_train],inner_train_ids)]
                            inner_val=outer_train[np.isin(groups[outer_train],inner_val_ids)]
                            result=blocked_score(candidate['X'],depth,candidate['h'],xy,
                                                 inner_train,inner_val,model_name)
                            if result is not None: inner_rmse.append(result[0]['rmse_m'])
                        mean_inner=float(np.mean(inner_rmse)) if len(inner_rmse)>=2 else np.inf
                        scores_by_candidate[(candidate['n'],model_name)]=mean_inner
                        inner_rows.append({'site':site_key,'scene_pool':pool_name,'outer_fold':fold,
                                           'candidate_scene_count':candidate['n'],'model':model_name,
                                           'inner_rmse_mean_m':mean_inner,'inner_folds_used':len(inner_rmse),
                                           'common_pixels_n':sample_n})
                for model_name in MODELS:
                    eligible=[c for c in candidates if np.isfinite(scores_by_candidate[(c['n'],model_name)])]
                    if not eligible: continue
                    selected=min(eligible,key=lambda c:scores_by_candidate[(c['n'],model_name)])
                    baseline=next(c for c in candidates if c['n']==len(ranked))
                    selected_eval=blocked_score(selected['X'],depth,selected['h'],xy,
                                                outer_train,outer_test,model_name)
                    baseline_eval=(selected_eval if selected['n']==baseline['n'] else
                                   blocked_score(baseline['X'],depth,baseline['h'],xy,
                                                 outer_train,outer_test,model_name))
                    if selected_eval is None or baseline_eval is None: continue
                    sm,si,sp,sr=selected_eval; bm,bi,bp,br=baseline_eval
                    metric_rows.append({'site':site_key,'scene_pool':pool_name,'model':model_name,
                                        'outer_fold':fold,'selected_scene_count':selected['n'],
                                        'all_scene_count':len(ranked),
                                        'selected_inner_rmse_m':scores_by_candidate[(selected['n'],model_name)],
                                        'selected_rmse_m':sm['rmse_m'],'selected_mae_m':sm['mae_m'],
                                        'selected_bias_m':sm['bias_m'],'selected_r2':sm['r2'],
                                        'all_scenes_rmse_m':bm['rmse_m'],'all_scenes_mae_m':bm['mae_m'],
                                        'all_scenes_bias_m':bm['bias_m'],'all_scenes_r2':bm['r2'],
                                        'delta_rmse_selected_minus_all_m':sm['rmse_m']-bm['rmse_m'],
                                        'selected_test_pixels_n':sm['n_test'],'all_scenes_test_pixels_n':bm['n_test'],
                                        'selected_train_pixels_n':sm['n_train'],'all_scenes_train_pixels_n':bm['n_train'],
                                        'spatial_buffer_m':V.BUFFER_M,
                                        'selected_scene_ids':'|'.join(selected['ids'])})
                    print(f'      {model_name}: selected {selected["n"]}/{len(ranked)} scenes; '
                          f'held-out ΔRMSE={sm["rmse_m"]-bm["rmse_m"]:+.3f} m.',flush=True)
                    for strategy,ev in [('nested_selected',selected_eval),('all_scenes',baseline_eval)]:
                        mm,idx,pred,resid=ev
                        for j,sample_index in enumerate(idx):
                            sample=df.iloc[int(sample_index)]
                            prediction_rows.append({'site':site_key,'scene_pool':pool_name,'model':model_name,
                                                    'outer_fold':fold,'strategy':strategy,
                                                    'selected_scene_count':selected['n'],
                                                    'sample_index':int(sample_index),
                                                    'x_m':float(sample.x_m),'y_m':float(sample.y_m),
                                                    'block_id':str(groups[sample_index]),
                                                    'observed_depth_m':float(depth[sample_index]),
                                                    'predicted_depth_m':float(pred[j]),
                                                    'residual_m':float(resid[j]),
                                                    'vertical_reference':sample.vertical_reference})
                print(f'    Outer fold {fold}/{V.N_SPLITS} complete.',flush=True)

    if not metric_rows: raise SystemExit('No nested scene-selection folds completed; inspect spatial block coverage.')
    out=ROOT/'Results/Additional_Analyses'; out.mkdir(parents=True,exist_ok=True)
    metrics=pd.DataFrame(metric_rows)
    metrics.to_csv(out/'nested_scene_selection_outer_metrics.csv',index=False)
    pd.DataFrame(inner_rows).to_csv(out/'nested_scene_selection_inner_scores.csv',index=False)
    pd.DataFrame(prediction_rows).to_csv(out/'nested_scene_selection_oof_predictions.csv.gz',index=False,compression='gzip')
    settings={'selection':'min mean inner spatial-fold RMSE among quality-ranked scene counts',
              'candidate_scene_counts':'4, 8, 12, 16, and all available scenes (duplicates removed)',
              'common_test_mask':'intersection of valid composite pixels across all candidate scene counts',
              'outer_spatial_folds':V.N_SPLITS,'inner_spatial_folds':INNER_FOLDS,
              'block_size_m':V.BLOCK_M,'buffer_m':V.BUFFER_M,'models':MODELS,
              'tide':'GOT5.6 anomalies relative to site survey-day mean'}
    settings['random_seed']=V.SEED
    settings['quality_ranking']='image-only scene_quality.csv score, then absolute date offset, then acquisition time'
    (out/'nested_scene_selection_settings.json').write_text(json.dumps(settings,indent=2)+'\n',encoding='utf-8')
    print(f'Wrote nested scene-count results to {out.relative_to(ROOT)}')
    print('Use selected-versus-all outer metrics for improvement claims; inner scores only select scene count.')


if __name__=='__main__': main()
