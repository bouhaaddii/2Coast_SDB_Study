"""Evaluate an RBF support-vector regressor on the primary SDB composites.

The SVR uses fixed, stated settings and the same buffered spatial folds as
Scripts/02_run_validation.py. It is restricted to the primary 30-day,
quality-weighted S30-only and HLS-mixed composites to avoid repeating the
model over every exploratory scene-count scenario.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'Scripts'))
spec=importlib.util.spec_from_file_location('validation_stage',ROOT/'Scripts/02_run_validation.py')
V=importlib.util.module_from_spec(spec); spec.loader.exec_module(V)
MODEL='SVR_RBF'
TIDE='GOT5.6_anomaly'
NO_TIDE='No_tide_adjustment'


def depth_bin(depth):
    if depth<=2: return '0–2 m'
    if depth<=5: return '>2–5 m'
    if depth<=10: return '>5–10 m'
    return '>10 m'


def main():
    tide_by_id,available=V.load_tide_anomalies(False)
    if not available:
        raise SystemExit('Tide-anomaly input is incomplete. Run Scripts/05_estimate_tides.py first.')
    metric_rows=[]; prediction_rows=[]
    print('SVR extension: same primary composites and buffered spatial folds; fixed RBF settings.')
    print('  StandardScaler + SVR(kernel=rbf, C=10, epsilon=0.1 m, gamma=scale); no parameter search.')

    for site_key,site in V.CFG['sites'].items():
        df=V.SAMPLES[V.SAMPLES.site==site_key].reset_index(drop=True)
        qsite=V.QUALITY[V.QUALITY.site==site_key].copy()
        xy=df[['x_m','y_m']].to_numpy(dtype=float)
        groups=(np.floor(xy[:,0]/V.BLOCK_M).astype(int).astype(str)+'_'+
                np.floor(xy[:,1]/V.BLOCK_M).astype(int).astype(str))
        depth=df['depth_m'].to_numpy(dtype=float)
        pools=[('S30_only',qsite[qsite['product']=='S30']),('HLS_Mixed',qsite)]
        for pool_name,pool in pools:
            ids=pool.sort_values('datetime_utc').item_id.tolist()
            if not ids:
                print(f'  {site_key}/{pool_name}: no scenes; skipped.')
                continue
            X,h_tide=V.read_composite(df,ids,pool,tide_by_id,'quality_weighted_mean')
            for tide_case,h in [(TIDE,h_tide),(NO_TIDE,np.zeros(len(df),dtype=float))]:
                folds,pred=V.cv_fit(X,depth,h,xy,groups,MODEL,save_predictions=True)
                base={'site':site_key,'scene_pool':pool_name,'aggregation':'quality_weighted_mean',
                      'scene_selection':'all_scenes','n_scenes':len(ids),'model':MODEL,
                      'tide_case':tide_case,'tide_model_anomaly_applied':tide_case==TIDE,
                      'scene_ids':'|'.join(ids),
                      'model_settings':'StandardScaler + RBF SVR; C=10; epsilon=0.1 m; gamma=scale; fixed a priori'}
                metric_rows.extend({**base,**row} for row in folds)
                for row in pred:
                    ix=int(row['sample_index']); sample=df.iloc[ix]
                    prediction_rows.append({**base,**row,'x_m':float(sample.x_m),'y_m':float(sample.y_m),
                                            'block_id':str(groups[ix]),'depth_bin_m':depth_bin(float(depth[ix])),
                                            'n_soundings':int(sample.n_soundings),
                                            'vertical_reference':sample.vertical_reference})
                print(f'  {site_key}/{pool_name}/{tide_case}: {len(folds)} spatial folds complete.',flush=True)

    if not metric_rows:
        raise SystemExit('No SVR folds completed; inspect prepared samples and spatial-fold settings.')
    out=ROOT/'Results/Additional_Analyses'; out.mkdir(parents=True,exist_ok=True)
    metrics=pd.DataFrame(metric_rows)
    metrics.to_csv(out/'svr_extension_fold_metrics.csv',index=False)
    predictions=pd.DataFrame(prediction_rows)
    predictions.to_csv(out/'svr_tide_ablation_oof_predictions.csv.gz',index=False,compression='gzip')
    summary=(metrics.groupby(['site','scene_pool','model','tide_case'],as_index=False)
             .agg(rmse_mean_m=('rmse_m','mean'),rmse_sd_across_folds_m=('rmse_m','std'),
                  mae_mean_m=('mae_m','mean'),bias_mean_m=('bias_m','mean'),r2_mean=('r2','mean'),
                  folds=('fold','nunique')))
    summary.to_csv(out/'svr_extension_summary.csv',index=False)
    paired=(metrics.pivot_table(index=['site','scene_pool','fold'],columns='tide_case',
                                values=['rmse_m','mae_m','bias_m'],aggfunc='first').reset_index())
    paired.columns=['_'.join(str(part) for part in col if str(part)!='') if isinstance(col,tuple) else col
                    for col in paired.columns]
    for metric in ['rmse_m','mae_m','bias_m']:
        corrected=f'{metric}_{TIDE}'; uncorrected=f'{metric}_{NO_TIDE}'
        if corrected in paired and uncorrected in paired:
            paired[f'delta_{metric}_tide_minus_none']=paired[corrected]-paired[uncorrected]
    paired.to_csv(out/'svr_paired_tide_fold_differences.csv',index=False)
    print(f'Wrote the SVR fold metrics, OOF predictions, and paired tide comparison to {out.relative_to(ROOT)}')
    print('The RBF-SVR is an additional benchmark; it is not asserted to be a new algorithm in SDB.')


if __name__=='__main__': main()
