"""Depth-stratified errors and spatial-block bootstrap intervals for OOF SDB."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'Config/experiment.json').read_text(encoding='utf-8'))
PRED=ROOT/'Results/Additional_Analyses/tide_ablation_oof_predictions.csv.gz'
N_BOOT=int(CFG.get('spatial_bootstrap_resamples',500))
TIDE='GOT5.6_anomaly'
NO_TIDE='No_tide_adjustment'


def score(observed,residual):
    observed=np.asarray(observed,dtype=float); residual=np.asarray(residual,dtype=float)
    return {'rmse_m':float(np.sqrt(np.mean(residual**2))),
            'mae_m':float(np.mean(np.abs(residual))),
            'bias_m':float(np.mean(residual))}


def bootstrap_rows(df,group_key,rng):
    groups=list(df[group_key].astype(str).unique())
    if not groups: return []
    by_group={str(g):part for g,part in df.groupby(group_key,sort=False)}
    scores=[]
    for _ in range(N_BOOT):
        sampled=rng.choice(groups,size=len(groups),replace=True)
        boot=pd.concat([by_group[str(g)] for g in sampled],ignore_index=True)
        scores.append(score(boot.observed_depth_m,boot.residual_m))
    return scores


def main():
    if not PRED.exists(): raise SystemExit('Run Scripts/06_tide_ablation.py first.')
    data=pd.read_csv(PRED)
    svr_pred=ROOT/'Results/Additional_Analyses/svr_tide_ablation_oof_predictions.csv.gz'
    if svr_pred.exists():
        data=pd.concat([data,pd.read_csv(svr_pred)],ignore_index=True)
    out=ROOT/'Results/Additional_Analyses'; out.mkdir(parents=True,exist_ok=True)
    strata=[]; intervals=[]; paired_intervals=[]
    rng_seed=int(CFG['random_seed'])
    group_cols=['site','scene_pool','model']

    for keys,subset in data.groupby(group_cols+['tide_case'],sort=True):
        site,pool,model,tide_case=keys
        for depth_bin,part in subset.groupby('depth_bin_m',sort=False):
            result=score(part.observed_depth_m,part.residual_m)
            strata.append({'site':site,'scene_pool':pool,'model':model,'tide_case':tide_case,
                           'depth_bin_m':depth_bin,'n_test_pixels':len(part),**result})
        overall=score(subset.observed_depth_m,subset.residual_m)
        local_rng=np.random.default_rng(rng_seed+sum(ord(c) for c in '|'.join(keys)))
        boots=bootstrap_rows(subset,'block_id',local_rng)
        row={'site':site,'scene_pool':pool,'model':model,'tide_case':tide_case,
             'n_test_pixels':len(subset),'spatial_blocks_n':subset.block_id.nunique(),**overall}
        for metric in ['rmse_m','mae_m','bias_m']:
            vals=np.asarray([b[metric] for b in boots],dtype=float)
            row[f'{metric}_ci95_low']=float(np.nanpercentile(vals,2.5))
            row[f'{metric}_ci95_high']=float(np.nanpercentile(vals,97.5))
        intervals.append(row)

    strata_df=pd.DataFrame(strata)
    strata_df.to_csv(out/'depth_stratified_error_metrics.csv',index=False)
    pd.DataFrame(intervals).to_csv(out/'spatial_block_bootstrap_intervals.csv',index=False)

    for keys,subset in data.groupby(group_cols,sort=True):
        site,pool,model=keys
        if TIDE not in set(subset.tide_case) or NO_TIDE not in set(subset.tide_case): continue
        tide=subset[subset.tide_case==TIDE][['sample_index','fold','block_id','observed_depth_m','residual_m']]
        no_tide=subset[subset.tide_case==NO_TIDE][['sample_index','fold','block_id','observed_depth_m','residual_m']]
        pair=tide.merge(no_tide,on=['sample_index','fold','block_id'],suffixes=('_tide','_none'))
        if pair.empty: continue
        pair['observed_depth_m']=pair.observed_depth_m_tide
        blocks=list(pair.block_id.astype(str).unique())
        by_block={str(g):part for g,part in pair.groupby('block_id',sort=False)}
        local_rng=np.random.default_rng(rng_seed+98765+sum(ord(c) for c in f'{site}|{pool}|{model}'))
        boot=[]
        for _ in range(N_BOOT):
            sampled=local_rng.choice(blocks,size=len(blocks),replace=True)
            b=pd.concat([by_block[str(g)] for g in sampled],ignore_index=True)
            st=score(b.observed_depth_m,b.residual_m_tide)
            sn=score(b.observed_depth_m,b.residual_m_none)
            boot.append({key:st[key]-sn[key] for key in ['rmse_m','mae_m','bias_m']})
        st=score(pair.observed_depth_m,pair.residual_m_tide)
        sn=score(pair.observed_depth_m,pair.residual_m_none)
        result={'site':site,'scene_pool':pool,'model':model,'paired_test_pixels_n':len(pair),
                'spatial_blocks_n':len(blocks)}
        for metric in ['rmse_m','mae_m','bias_m']:
            delta=st[metric]-sn[metric]
            vals=np.asarray([item[metric] for item in boot],dtype=float)
            result[f'delta_{metric}_tide_minus_none']=delta
            result[f'delta_{metric}_ci95_low']=float(np.nanpercentile(vals,2.5))
            result[f'delta_{metric}_ci95_high']=float(np.nanpercentile(vals,97.5))
        paired_intervals.append(result)
    pd.DataFrame(paired_intervals).to_csv(out/'paired_tide_effect_block_bootstrap.csv',index=False)
    cv_path=ROOT/'Results/Metrics/cv_metrics.csv'
    if cv_path.exists():
        cv=pd.read_csv(cv_path)
        random=cv[cv.scene_selection=='random_subset'].copy()
        if not random.empty:
            per_subset=(random.groupby(['site','scene_pool','aggregation','n_scenes','model','replicate'],as_index=False)
                        .agg(subset_rmse_m=('rmse_m','mean'),subset_mae_m=('mae_m','mean'),
                             subset_bias_m=('bias_m','mean')))
            scene_uncertainty=(per_subset.groupby(['site','scene_pool','aggregation','n_scenes','model'],as_index=False)
                               .agg(random_subset_repeats=('replicate','nunique'),
                                    rmse_mean_m=('subset_rmse_m','mean'),
                                    rmse_sd_across_subsets_m=('subset_rmse_m','std'),
                                    rmse_p05_m=('subset_rmse_m',lambda x:float(np.nanpercentile(x,5))),
                                    rmse_p95_m=('subset_rmse_m',lambda x:float(np.nanpercentile(x,95))),
                                    mae_mean_m=('subset_mae_m','mean'),bias_mean_m=('subset_bias_m','mean')))
            scene_uncertainty.to_csv(out/'random_scene_count_uncertainty.csv',index=False)
    print(f'Stage 6/9: wrote depth-stratified metrics and block-bootstrap intervals to {out.relative_to(ROOT)}')
    print(f'Bootstrap resamples: {N_BOOT} per comparison. Intervals resample whole held-out spatial blocks, not individual pixels.')


if __name__=='__main__': main()
