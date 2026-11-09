"""Paired spatial-CV ablation: GOT5.6 tide anomaly versus zero adjustment."""
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
MODELS=['StumpfLogRatio','RandomForest','HistGradientBoosting']


def main():
    tide_by_id,available=V.load_tide_anomalies(False)
    if not available:
        raise SystemExit('Tide-anomaly table is incomplete. Run Scripts/05_estimate_tides.py first.')
    metric_rows=[]; prediction_rows=[]
    print('Stage 4/9: paired tide ablation with identical samples, scene composites, and spatial folds.')
    for site_key,site in V.CFG['sites'].items():
        df=V.SAMPLES[V.SAMPLES.site==site_key].reset_index(drop=True)
        qsite=V.QUALITY[V.QUALITY.site==site_key].copy()
        pools=[('S30_only',qsite[qsite['product']=='S30']),('HLS_Mixed',qsite)]
        xy=df[['x_m','y_m']].to_numpy(dtype=float)
        groups=(np.floor(xy[:,0]/V.BLOCK_M).astype(int).astype(str)+'_'+
                np.floor(xy[:,1]/V.BLOCK_M).astype(int).astype(str))
        depth=df['depth_m'].to_numpy(dtype=float)
        for pool_name,pool in pools:
            ids=pool.sort_values('datetime_utc').item_id.tolist()
            if not ids: continue
            X,h_corrected=V.read_composite(df,ids,pool,tide_by_id,'quality_weighted_mean')
            print(f'  {site_key}/{pool_name}: {len(ids)} scenes; {len(MODELS)} models × 2 tide cases')
            for tide_case,h in [('GOT5.6_anomaly',h_corrected),('No_tide_adjustment',np.zeros(len(df),dtype=float))]:
                for model_name in MODELS:
                    folds,pred=V.cv_fit(X,depth,h,xy,groups,model_name,save_predictions=True)
                    base={'site':site_key,'scene_pool':pool_name,'aggregation':'quality_weighted_mean',
                          'scene_selection':'all_scenes','n_scenes':len(ids),'model':model_name,
                          'tide_case':tide_case,'scene_ids':'|'.join(ids)}
                    metric_rows.extend({**base,**row} for row in folds)
                    for row in pred:
                        sample=df.iloc[row['sample_index']]
                        prediction_rows.append({**base,**row,'x_m':float(sample.x_m),'y_m':float(sample.y_m),
                                                'depth_bin_m':depth_bin(float(sample.depth_m)),
                                                'block_id':str(groups[row['sample_index']]),
                                                'vertical_reference':sample.vertical_reference})
    if not metric_rows: raise SystemExit('No tide-ablation folds completed; inspect sample coverage and fold settings.')
    out=ROOT/'Results/Additional_Analyses'; out.mkdir(parents=True,exist_ok=True)
    metrics=pd.DataFrame(metric_rows)
    metrics.to_csv(out/'tide_ablation_fold_metrics.csv',index=False)
    predictions=pd.DataFrame(prediction_rows)
    predictions.to_csv(out/'tide_ablation_oof_predictions.csv.gz',index=False,compression='gzip')
    summary=(metrics.groupby(['site','scene_pool','model','tide_case'],as_index=False)
             .agg(rmse_mean_m=('rmse_m','mean'),rmse_sd_across_folds_m=('rmse_m','std'),
                  mae_mean_m=('mae_m','mean'),bias_mean_m=('bias_m','mean'),r2_mean=('r2','mean'),
                  folds=('fold','nunique')))
    summary.to_csv(out/'tide_ablation_summary.csv',index=False)
    paired=(metrics.pivot_table(index=['site','scene_pool','model','fold'],columns='tide_case',
                                values=['rmse_m','mae_m','bias_m'],aggfunc='first').reset_index())
    paired.columns=['_'.join(str(part) for part in col if str(part)!='') if isinstance(col,tuple) else col
                    for col in paired.columns]
    for metric in ['rmse_m','mae_m','bias_m']:
        a=f'{metric}_GOT5.6_anomaly'; b=f'{metric}_No_tide_adjustment'
        if a in paired and b in paired: paired[f'delta_{metric}_tide_minus_none']=paired[a]-paired[b]
    paired.to_csv(out/'tide_ablation_paired_fold_differences.csv',index=False)
    print(f'Wrote paired fold metrics, OOF predictions, and summaries to {out.relative_to(ROOT)}')
    print('Interpretation: negative tide-minus-none RMSE/MAE deltas favor the modeled anomaly adjustment.')


def depth_bin(depth):
    if depth<=2: return '0–2 m'
    if depth<=5: return '>2–5 m'
    if depth<=10: return '>5–10 m'
    return '>10 m'


if __name__=='__main__': main()
