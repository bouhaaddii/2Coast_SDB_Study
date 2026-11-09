"""Create additional publication-resolution figures and table extracts."""
from __future__ import annotations
import json,math
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.ticker import FuncFormatter

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'Config/experiment.json').read_text(encoding='utf-8'))
OUT=ROOT/'Results/Figures'
INK='#1E293B'; NAVY='#183B56'; TEAL='#168C8C'; CORAL='#E76F51'; GOLD='#D6A23A'; PALE='#DCE5EA'
MODEL_COLORS={'StumpfLogRatio':NAVY,'RandomForest':TEAL,'HistGradientBoosting':CORAL,'SVR_RBF':'#8F5BCB'}
MODEL_LABELS={'StumpfLogRatio':'Stumpf log-ratio','RandomForest':'Random Forest',
              'HistGradientBoosting':'Histogram gradient boosting','SVR_RBF':'RBF support vector regression'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.labelsize':8.5,
                     'xtick.labelsize':7.5,'ytick.labelsize':7.5,'axes.edgecolor':'#73808C',
                     'axes.linewidth':0.75,'savefig.facecolor':'white','figure.facecolor':'white',
                     'pdf.fonttype':42,'ps.fonttype':42})


def save(fig,name):
    OUT.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUT/(name+'.png'),dpi=450,bbox_inches='tight')
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)


def panel(ax,label):
    ax.text(.018,.98,label,transform=ax.transAxes,ha='left',va='top',fontweight='bold',
            color=INK,bbox={'facecolor':'white','edgecolor':'none','alpha':.85,'pad':1.7})


def compass(ax):
    x,y,size=.91,.87,.045
    points=[]
    for i in range(16):
        angle=math.radians(90+i*22.5); radius=size if i%2==0 else size*.32
        points.append((x+radius*math.cos(angle),y+radius*math.sin(angle)))
    ax.add_patch(Polygon(points,closed=True,facecolor=NAVY,edgecolor='white',lw=.4,
                         transform=ax.transAxes,zorder=20))
    ax.text(x,y+size*1.3,'N',transform=ax.transAxes,ha='center',va='bottom',weight='bold',color=NAVY,fontsize=7)
    ax.text(x+size*1.35,y,'E',transform=ax.transAxes,ha='left',va='center',color=NAVY,fontsize=6)
    ax.text(x-size*1.35,y,'W',transform=ax.transAxes,ha='right',va='center',color=NAVY,fontsize=6)
    ax.text(x,y-size*1.3,'S',transform=ax.transAxes,ha='center',va='top',color=NAVY,fontsize=6)


def scale_bar(ax,bounds,length_m=1000):
    left,bottom,right,top=bounds
    x=left+.07*(right-left); y=bottom+.07*(top-bottom)
    ax.plot([x,x+length_m],[y,y],color='white',lw=4.2,solid_capstyle='butt',zorder=15)
    ax.plot([x,x+length_m/2],[y,y],color=NAVY,lw=3.2,solid_capstyle='butt',zorder=16)
    ax.plot([x+length_m/2,x+length_m],[y,y],color=TEAL,lw=3.2,solid_capstyle='butt',zorder=16)
    ax.text(x+length_m/2,y+.025*(top-bottom),f'{length_m/1000:g} km',ha='center',va='bottom',
            fontsize=7,color=NAVY,weight='bold',bbox={'facecolor':'white','edgecolor':'none','alpha':.82,'pad':1})


def plot_tide_ablation():
    path=ROOT/'Results/Additional_Analyses/paired_tide_effect_block_bootstrap.csv'
    if not path.exists(): raise SystemExit('Run Scripts/08_uncertainty_analysis.py first.')
    data=pd.read_csv(path)
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.0),layout='constrained',sharey=True)
    pool_markers={'S30_only':'o','HLS_Mixed':'s'}
    offsets={'S30_only':-.14,'HLS_Mixed':.14}
    for ax,(site_key,site) in zip(axes,CFG['sites'].items()):
        subset=data[data.site==site_key]
        for ix,model in enumerate(MODEL_COLORS):
            for pool,color in [('S30_only',TEAL),('HLS_Mixed',NAVY)]:
                row=subset[(subset.model==model)&(subset.scene_pool==pool)]
                if row.empty: continue
                row=row.iloc[0]; y=float(row.delta_rmse_m_tide_minus_none)
                low=float(row.delta_rmse_m_ci95_low); high=float(row.delta_rmse_m_ci95_high)
                ax.errorbar(ix+offsets[pool],y,yerr=[[y-low],[high-y]],fmt=pool_markers[pool],
                            ms=6,color=color,ecolor=color,capsize=2.8,lw=1.25,
                            markeredgecolor='white',markeredgewidth=.5,zorder=4)
        ax.axhline(0,color=INK,lw=.9,ls=(0,(3,2)))
        ax.set_xticks(range(len(MODEL_COLORS)),[MODEL_LABELS[m] for m in MODEL_COLORS],rotation=12,ha='right')
        ax.set_ylabel('Δ spatially held-out RMSE (m)\nGOT5.6 anomaly − no tide adjustment')
        ax.grid(axis='y',color=PALE,lw=.65)
        panel(ax,'A' if site_key==next(iter(CFG['sites'])) else 'B')
        ax.text(.98,.04,site['display_name'],transform=ax.transAxes,ha='right',va='bottom',fontsize=8,color=INK)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([0],[0],marker='o',color=TEAL,linestyle='none',label='S30 only'),
                        Line2D([0],[0],marker='s',color=NAVY,linestyle='none',label='HLS mixed')],
               loc='lower center',bbox_to_anchor=(.5,-.04),ncol=2,frameon=False,fontsize=8)
    save(fig,'Figure12_paired_tide_adjustment_effect')


def plot_scene_count():
    path=ROOT/'Results/Metrics/cv_metrics.csv'
    if not path.exists(): raise SystemExit('Run Scripts/02_run_validation.py first.')
    data=pd.read_csv(path)
    d=data[(data.model=='RandomForest')&(data.aggregation=='quality_weighted_mean')&
           (data.scene_pool.isin(['S30_only','HLS_Mixed']))&
           (data.scene_selection.isin(['quality_ranked','random_subset','all_scenes']))].copy()
    if d.empty: raise SystemExit('No scene-count scenarios found in cv_metrics.csv.')
    per_run=(d.groupby(['site','scene_pool','scene_selection','n_scenes','replicate'],as_index=False)
             .agg(rmse_m=('rmse_m','mean')))
    fig,axes=plt.subplots(1,2,figsize=(11.0,4.4),layout='constrained',sharey=True)
    colors={'S30_only':TEAL,'HLS_Mixed':NAVY}
    labels={'S30_only':'Sentinel-2 only','HLS_Mixed':'Sentinel-2 + Landsat'}
    for ax,(site_key,site) in zip(axes,CFG['sites'].items()):
        one=per_run[per_run.site==site_key]
        for pool,color in colors.items():
            z=one[one.scene_pool==pool]
            q=z[z.scene_selection.isin(['quality_ranked','all_scenes'])].groupby('n_scenes').rmse_m.mean().sort_index()
            if not q.empty:
                ax.plot(q.index,q.values,color=color,marker='o',ms=4,lw=1.8,label=f'{labels[pool]} · quality ranked')
            r=z[z.scene_selection=='random_subset'].groupby('n_scenes').rmse_m
            if not r.count().empty:
                means=r.mean().sort_index(); lo=r.quantile(.05).reindex(means.index); hi=r.quantile(.95).reindex(means.index)
                ax.plot(means.index,means.values,color=color,marker='.',ms=3,lw=1.0,ls='--',alpha=.8,
                        label=f'{labels[pool]} · random subsets')
                ax.fill_between(means.index,lo,hi,color=color,alpha=.12,linewidth=0)
        ax.set_xlabel('Number of scenes in composite')
        ax.set_ylabel('Mean spatial-fold RMSE (m)')
        ax.grid(color=PALE,lw=.65)
        panel(ax,'A' if site_key==next(iter(CFG['sites'])) else 'B')
        ax.text(.98,.04,site['display_name'],transform=ax.transAxes,ha='right',va='bottom',fontsize=8,color=INK)
    handles,labels0=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels0,loc='lower center',bbox_to_anchor=(.5,-.04),ncol=2,frameon=False,fontsize=7.4)
    save(fig,'Figure13_scene_count_and_sensor_mix')


def plot_cross_site_transfer():
    path=ROOT/'Results/Additional_Analyses/cross_site_transfer_summary.csv'
    if not path.exists(): raise SystemExit('Run Scripts/07_cross_site_transfer.py first.')
    data=pd.read_csv(path)
    # Newer transfer tables include both methods; Figure 14 remains the
    # established additive-offset comparison for continuity with prior outputs.
    if 'calibration_method' in data.columns:
        data=data[data.calibration_method=='median_additive_offset']
    if data.empty: raise SystemExit('No target-calibrated transfer results are available.')
    directions=list(data[['source_site','target_site']].drop_duplicates().itertuples(index=False,name=None))
    fig,axes=plt.subplots(1,len(directions),figsize=(10.8,4.2),layout='constrained',sharey=True,squeeze=False)
    for ax,(source,target) in zip(axes[0],directions):
        sub=data[(data.source_site==source)&(data.target_site==target)]
        for model,color in MODEL_COLORS.items():
            z=sub[sub.model==model].sort_values('calibration_fraction')
            if z.empty: continue
            x=100*z.calibration_fraction.to_numpy(dtype=float)
            y=z.rmse_mean_m.to_numpy(dtype=float)
            spread=z.rmse_sd_across_folds_repeats_m.fillna(0).to_numpy(dtype=float)
            ax.plot(x,y,color=color,marker='o',ms=4,lw=1.7,label=MODEL_LABELS[model])
            ax.fill_between(x,np.maximum(0,y-spread),y+spread,color=color,alpha=.11,linewidth=0)
        ax.set_xlabel('Target-site calibration sample (% of buffered training pool)')
        ax.set_ylabel('Target-site spatial holdout RMSE (m)')
        ax.set_xticks([1,5,10],['1%','5%','10%'])
        ax.grid(color=PALE,lw=.65)
        panel(ax,'A' if source==directions[0][0] else 'B')
        src_name=CFG['sites'][source]['display_name'].split(',')[0]
        dst_name=CFG['sites'][target]['display_name'].split(',')[0]
        ax.text(.98,.04,f'{src_name} → {dst_name}',transform=ax.transAxes,ha='right',va='bottom',fontsize=8,color=INK)
    handles,labels0=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels0,loc='lower center',bbox_to_anchor=(.5,-.04),ncol=3,frameon=False,fontsize=7.5)
    save(fig,'Figure14_cross_site_low_shot_transfer')


def plot_depth_strata():
    path=ROOT/'Results/Additional_Analyses/depth_stratified_error_metrics.csv'
    if not path.exists(): raise SystemExit('Run Scripts/08_uncertainty_analysis.py first.')
    data=pd.read_csv(path)
    d=data[(data.model=='RandomForest')&(data.scene_pool=='HLS_Mixed')]
    order=['0–2 m','>2–5 m','>5–10 m','>10 m']
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.0),layout='constrained',sharey=True)
    colors={'GOT5.6_anomaly':TEAL,'No_tide_adjustment':CORAL}
    labels={'GOT5.6_anomaly':'GOT5.6 anomaly','No_tide_adjustment':'No tide adjustment'}
    for ax,(site_key,site) in zip(axes,CFG['sites'].items()):
        sub=d[d.site==site_key]
        for case,color in colors.items():
            z=sub[sub.tide_case==case].set_index('depth_bin_m').reindex(order)
            ax.plot(np.arange(len(order)),z.rmse_m,color=color,marker='o',ms=4,lw=1.7,label=labels[case])
            for i,n in enumerate(z.n_test_pixels.fillna(0)):
                if n: ax.annotate(f'n={int(n):,}',(i,z.rmse_m.iloc[i]),xytext=(0,6),textcoords='offset points',
                                  ha='center',fontsize=6.2,color=color)
        ax.set_xticks(np.arange(len(order)),order)
        ax.set_xlabel('Observed sounding depth stratum')
        ax.set_ylabel('Out-of-fold RMSE (m)')
        ax.grid(axis='y',color=PALE,lw=.65)
        panel(ax,'A' if site_key==next(iter(CFG['sites'])) else 'B')
        ax.text(.98,.04,site['display_name'],transform=ax.transAxes,ha='right',va='bottom',fontsize=8,color=INK)
    handles,labels0=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels0,loc='lower center',bbox_to_anchor=(.5,-.04),ncol=2,frameon=False,fontsize=7.5)
    save(fig,'Figure15_depth_stratified_validation')


def plot_primary_model_comparison():
    """Compare the RBF-SVR extension with existing models on matched folds."""
    base_path=ROOT/'Results/Metrics/cv_metrics.csv'
    svr_path=ROOT/'Results/Additional_Analyses/svr_extension_fold_metrics.csv'
    if not base_path.exists() or not svr_path.exists():
        raise SystemExit('Run Scripts/10_run_svr_extension.py first.')
    base=pd.read_csv(base_path)
    base=base[(base.scene_selection=='all_scenes')&
              (base.aggregation=='quality_weighted_mean')&
              (base.scene_pool.isin(['S30_only','HLS_Mixed']))&
              (base.tide_model_anomaly_applied.astype(str).str.lower().isin(['true','1']))]
    base=base[['site','scene_pool','model','fold','rmse_m']]
    svr=pd.read_csv(svr_path)
    svr=svr[(svr.tide_case=='GOT5.6_anomaly')&
            (svr.aggregation=='quality_weighted_mean')]
    svr=svr[['site','scene_pool','model','fold','rmse_m']]
    data=pd.concat([base,svr],ignore_index=True)
    model_order=['StumpfLogRatio','RandomForest','HistGradientBoosting','SVR_RBF']
    fig,axes=plt.subplots(2,2,figsize=(11.2,7.2),layout='constrained',sharey=False)
    pool_order=['S30_only','HLS_Mixed']
    pool_label={'S30_only':'Sentinel-2 only','HLS_Mixed':'Sentinel-2 + Landsat'}
    letters='ABCD'
    panel_index=0
    for row,(site_key,site) in enumerate(CFG['sites'].items()):
        for col,pool in enumerate(pool_order):
            ax=axes[row,col]
            subset=data[(data.site==site_key)&(data.scene_pool==pool)]
            for ix,model in enumerate(model_order):
                vals=subset[subset.model==model].sort_values('fold').rmse_m.dropna().to_numpy(dtype=float)
                if not len(vals): continue
                jitter=np.linspace(-.075,.075,len(vals)) if len(vals)>1 else np.array([0.0])
                ax.scatter(ix+jitter,vals,s=19,color=MODEL_COLORS[model],alpha=.72,
                           edgecolors='white',linewidths=.35,zorder=3)
                mean=float(np.mean(vals)); spread=float(np.std(vals,ddof=1)) if len(vals)>1 else 0.0
                ax.errorbar(ix,mean,yerr=spread,fmt='D',ms=4.5,color=MODEL_COLORS[model],
                            ecolor=MODEL_COLORS[model],capsize=2.5,lw=1.1,zorder=4)
            ax.set_xticks(range(len(model_order)),[MODEL_LABELS[m] for m in model_order],rotation=17,ha='right')
            ax.set_ylabel('Spatial-fold RMSE (m)')
            ax.set_xlabel(pool_label[pool])
            ax.grid(axis='y',color=PALE,lw=.65)
            panel(ax,letters[panel_index]); panel_index+=1
            ax.text(.98,.96,site['display_name'].split(',')[0],transform=ax.transAxes,
                    ha='right',va='top',fontsize=8,color=INK)
    fig.text(.5,-.015,'Points: individual spatial folds; diamonds and whiskers: mean ± SD across folds.',
             ha='center',va='top',fontsize=7.5,color=INK)
    save(fig,'Figure17_svr_spatial_model_comparison')


def plot_nested_scene_selection():
    """Show paired outer-fold change after selecting scene count internally."""
    path=ROOT/'Results/Additional_Analyses/nested_scene_selection_outer_metrics.csv'
    if not path.exists():
        print('Figure 18 skipped: run Scripts/11_nested_scene_selection.py first.')
        return
    data=pd.read_csv(path)
    if data.empty:
        print('Figure 18 skipped: nested scene-selection table is empty.')
        return
    pool_order=['S30_only','HLS_Mixed']
    pool_labels={'S30_only':'Sentinel-2 only','HLS_Mixed':'Sentinel-2 + Landsat'}
    models=['RandomForest','HistGradientBoosting','SVR_RBF']
    fig,axes=plt.subplots(2,2,figsize=(10.8,7.2),layout='constrained',sharey=True)
    for row,(site_key,site) in enumerate(CFG['sites'].items()):
        for col,pool in enumerate(pool_order):
            ax=axes[row,col]
            sub=data[(data.site==site_key)&(data.scene_pool==pool)]
            for ix,model in enumerate(models):
                z=sub[sub.model==model].sort_values('outer_fold')
                delta=z.delta_rmse_selected_minus_all_m.to_numpy(dtype=float)
                folds=z.outer_fold.to_numpy(dtype=int)
                if not len(delta): continue
                jitter=np.linspace(-.075,.075,len(delta)) if len(delta)>1 else np.array([0.0])
                ax.scatter(ix+jitter,delta,s=25,color=MODEL_COLORS[model],alpha=.78,
                           edgecolors='white',linewidths=.45,zorder=3)
                ax.plot([ix-.11,ix+.11],[np.mean(delta)]*2,color=MODEL_COLORS[model],lw=2.3,zorder=4)
                for dx,val,fold in zip(jitter,delta,folds):
                    ax.annotate(str(fold), (ix+dx,val), xytext=(0,4),
                                textcoords='offset points', ha='center', fontsize=5.4,
                                color=INK, alpha=.8)
            ax.axhline(0,color=INK,lw=.9,ls=(0,(3,2)))
            ax.set_xticks(range(len(models)),['RF','HGB','RBF-SVR'])
            ax.set_ylabel('Selected scenes − all scenes\nouter-fold RMSE (m)')
            ax.set_xlabel(pool_labels[pool])
            ax.grid(axis='y',color=PALE,lw=.65)
            panel(ax,'ABCD'[row*2+col])
            ax.text(.98,.97,site['display_name'].split(',')[0],transform=ax.transAxes,
                    ha='right',va='top',fontsize=8,color=INK)
    fig.text(.5,-.01,'RF = Random Forest; HGB = histogram gradient boosting. Each point is one held-out fold (numbered); values below zero favor nested selection.',
             ha='center',va='top',fontsize=7.5,color=INK)
    save(fig,'Figure18_nested_scene_count_selection')


def plot_transfer_calibration_comparison():
    """Compare additive and affine target calibration on matching held-out folds."""
    path=ROOT/'Results/Additional_Analyses/cross_site_transfer_calibration_comparison_summary.csv'
    if not path.exists():
        print('Figure 19 skipped: run Scripts/07_cross_site_transfer.py first.')
        return
    data=pd.read_csv(path)
    data=data[data.calibration_method.isin(['median_additive_offset','affine_linear'])]
    if data.empty:
        print('Figure 19 skipped: no calibrated transfer rows are available.')
        return
    directions=list(data[['source_site','target_site']].drop_duplicates().itertuples(index=False,name=None))
    fig,axes=plt.subplots(1,len(directions),figsize=(11.2,4.6),layout='constrained',sharey=True,squeeze=False)
    styles={'median_additive_offset':('-', 'o'),'affine_linear':('--','^')}
    for ax,(source,target) in zip(axes[0],directions):
        sub=data[(data.source_site==source)&(data.target_site==target)]
        for model,color in MODEL_COLORS.items():
            for method,(linestyle,marker) in styles.items():
                z=sub[(sub.model==model)&(sub.calibration_method==method)].sort_values('calibration_fraction')
                if z.empty: continue
                x=100*z.calibration_fraction.to_numpy(dtype=float)
                y=z.rmse_mean_m.to_numpy(dtype=float)
                spread=z.rmse_sd_across_folds_repeats_m.fillna(0).to_numpy(dtype=float)
                ax.errorbar(x,y,yerr=spread,fmt=marker,color=color,linestyle=linestyle,
                            markersize=3.4,linewidth=1.2,capsize=1.8,alpha=.88)
        ax.set_xticks([1,5,10],['1%','5%','10%'])
        ax.set_xlabel('Target-site calibration samples')
        ax.set_ylabel('Held-out target RMSE (m)')
        ax.grid(color=PALE,lw=.65)
        panel(ax,'A' if source==directions[0][0] else 'B')
        src_name=CFG['sites'][source]['display_name'].split(',')[0]
        dst_name=CFG['sites'][target]['display_name'].split(',')[0]
        ax.text(.98,.97,f'{src_name} → {dst_name}',transform=ax.transAxes,ha='right',va='top',fontsize=8,color=INK)
    from matplotlib.lines import Line2D
    model_handles=[Line2D([0],[0],color=MODEL_COLORS[m],lw=1.7,label=MODEL_LABELS[m])
                   for m in MODEL_COLORS]
    method_handles=[Line2D([0],[0],color=INK,linestyle=styles[k][0],marker=styles[k][1],
                           markersize=4,lw=1.2,label=v)
                    for k,v in [('median_additive_offset','Median offset'),('affine_linear','Affine calibration')]]
    fig.legend(handles=model_handles+method_handles,loc='lower center',
               bbox_to_anchor=(.5,-.08),ncol=3,frameon=False,fontsize=7.2)
    fig.text(.5,-.15,'Error bars show SD across spatial folds and calibration repeats; calibration points are outside each test block and its buffer.',
             ha='center',va='top',fontsize=7.2,color=INK)
    save(fig,'Figure19_transfer_calibration_methods')


def plot_model_disagreement_maps():
    from matplotlib.colors import Normalize
    fig,axes=plt.subplots(1,2,figsize=(11.1,5.8),layout='constrained')
    for panel_index,(ax,(site_key,site)) in enumerate(zip(axes,CFG['sites'].items())):
        paths=[ROOT/'Results/Predictions/Maps'/f'{site_key}_{model}_supported_depth.tif'
               for model in MODEL_COLORS]
        missing=[p for p in paths if not p.exists()]
        if missing: raise SystemExit('Run Scripts/03_generate_supported_maps.py first; missing '+', '.join(p.name for p in missing))
        arrays=[]; profile=None; transform=None; bounds=None
        for path in paths:
            with rasterio.open(path) as ds:
                arr=ds.read(1).astype(float)
                if ds.nodata is not None: arr[arr==ds.nodata]=np.nan
                arrays.append(arr); profile=ds.profile.copy(); transform=ds.transform; bounds=ds.bounds
        stack=np.stack(arrays)
        valid_count=np.isfinite(stack).sum(axis=0)
        has_any=valid_count>0
        median=np.full(stack.shape[1:],np.nan,dtype=float)
        median[has_any]=np.nanmedian(stack[:,has_any],axis=0)
        deviation=np.abs(stack-median[None,:,:])
        disagreement=np.full(stack.shape[1:],np.nan,dtype=float)
        has_multiple=valid_count>=2
        disagreement[has_multiple]=np.nanmedian(deviation[:,has_multiple],axis=0)
        finite=disagreement[np.isfinite(disagreement)]
        vmax=max(float(np.nanpercentile(finite,98)),.05) if len(finite) else .05
        im=ax.imshow(np.ma.masked_invalid(disagreement),extent=(bounds.left,bounds.right,bounds.bottom,bounds.top),
                     origin='upper',cmap='magma',vmin=0,vmax=vmax,interpolation='nearest',zorder=1)
        x=transform.c+(np.arange(median.shape[1])+.5)*transform.a
        y=transform.f+(np.arange(median.shape[0])+.5)*transform.e
        finite_depth=median[np.isfinite(median)]
        if finite_depth.size:
            step=1.0 if np.nanmax(finite_depth)>6 else .5
            levels=np.arange(max(step,step*np.ceil(np.nanmin(finite_depth)/step)),np.nanmax(finite_depth),step)
            if len(levels):
                contour=ax.contour(x,y,np.ma.masked_invalid(median),levels=levels,colors='white',linewidths=.45,alpha=.85,zorder=3)
                ax.clabel(contour,fmt=lambda val:f'{val:g}',fontsize=5.5,inline=True)
        ax.set_xlim(bounds.left,bounds.right); ax.set_ylim(bounds.bottom,bounds.top); ax.set_aspect('equal',adjustable='box')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v/1000:.2f}'))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v/1000:.2f}'))
        ax.set_xlabel(f'Easting X (km; {site["image_crs"]})'); ax.set_ylabel(f'Northing Y (km; {site["image_crs"]})')
        panel(ax,'A' if panel_index==0 else 'B'); compass(ax); scale_bar(ax,(bounds.left,bounds.bottom,bounds.right,bounds.top),1000)
        fig.colorbar(im,ax=ax,shrink=.82,pad=.025,label='Across-model median absolute deviation (m)')
    save(fig,'Figure16_spatial_model_disagreement')


def main():
    print('Stage 9/9: creating extended publication figures from completed analysis outputs.')
    plot_tide_ablation()
    plot_scene_count()
    plot_cross_site_transfer()
    plot_depth_strata()
    plot_model_disagreement_maps()
    plot_primary_model_comparison()
    plot_nested_scene_selection()
    plot_transfer_calibration_comparison()
    print(f'Additional journal-style PNG/PDF figures saved to {OUT.relative_to(ROOT)}')
    print('Figure 16 is between-model disagreement, a diagnostic only; it is not a calibrated prediction interval.')


if __name__=='__main__': main()
