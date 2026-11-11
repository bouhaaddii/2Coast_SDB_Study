"""Create journal-style maps and validation figures from completed pipeline outputs."""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.dates as mdates
from matplotlib.patches import Polygon, Rectangle, Patch
from matplotlib.ticker import FuncFormatter

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'Config/experiment.json').read_text(encoding='utf-8'))
INV=json.loads((ROOT/'Data/Satellite/scene_inventory.json').read_text(encoding='utf-8'))
OUT=ROOT/'Results/Figures'; OUT.mkdir(parents=True,exist_ok=True)
NAVY='#183B56'; TEAL='#168C8C'; CORAL='#E76F51'; GOLD='#E9B44C'; INK='#20313B'; PALE='#E9F0F3'
MODELS=['StumpfLogRatio','RandomForest','HistGradientBoosting']
MODEL_LABELS={'StumpfLogRatio':'Stumpf log-ratio','RandomForest':'Random Forest','HistGradientBoosting':'Histogram gradient boosting'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,'axes.titlesize':11,
                     'axes.edgecolor':'#708090','axes.linewidth':0.8,'axes.grid':False,
                     'savefig.facecolor':'white','figure.facecolor':'white'})


def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=350,bbox_inches='tight')
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)


def add_compass_rose(ax,x=0.91,y=0.87,size=0.055):
    # Eight-point alternating compass rose in axes coordinates.
    pts=[]
    for i in range(16):
        angle=math.radians(90+i*22.5)
        radius=size if i%2==0 else size*0.34
        pts.append((x+radius*math.cos(angle),y+radius*math.sin(angle)))
    rose=Polygon(pts,closed=True,facecolor=NAVY,edgecolor='white',linewidth=0.45,transform=ax.transAxes,zorder=20)
    ax.add_patch(rose)
    ax.add_patch(plt.Circle((x,y),size*0.12,transform=ax.transAxes,facecolor='white',edgecolor=NAVY,lw=0.5,zorder=21))
    ax.text(x,y+size*1.35,'N',transform=ax.transAxes,ha='center',va='bottom',weight='bold',color=NAVY,fontsize=8,zorder=22)
    ax.text(x+size*1.5,y,'E',transform=ax.transAxes,ha='left',va='center',color=NAVY,fontsize=6,zorder=22)
    ax.text(x-size*1.5,y,'W',transform=ax.transAxes,ha='right',va='center',color=NAVY,fontsize=6,zorder=22)
    ax.text(x,y-size*1.35,'S',transform=ax.transAxes,ha='center',va='top',color=NAVY,fontsize=6,zorder=22)


def add_scale_bar(ax,bounds,length_m=1000):
    left,bottom,right,top=bounds
    x0=left+0.08*(right-left); y0=bottom+0.07*(top-bottom)
    half=length_m/2
    ax.plot([x0,x0+half],[y0,y0],color='white',lw=4,solid_capstyle='butt',zorder=18)
    ax.plot([x0+half,x0+length_m],[y0,y0],color=NAVY,lw=4,solid_capstyle='butt',zorder=19)
    ax.plot([x0,x0+length_m],[y0,y0],color=NAVY,lw=0.7,zorder=20)
    ax.text(x0+length_m/2,y0+0.025*(top-bottom),f'{length_m/1000:g} km',ha='center',va='bottom',
            color=NAVY,weight='bold',fontsize=7,zorder=20,
            bbox={'facecolor':'white','edgecolor':'none','alpha':0.75,'pad':1.3})


def add_panel_label(ax,label):
    ax.text(0.015,0.985,label,transform=ax.transAxes,ha='left',va='top',weight='bold',
            fontsize=10,color=INK,bbox={'facecolor':'white','edgecolor':'none','alpha':0.8,'pad':1.5})


def masked_triangulation(x,y,max_edge_m=90.0):
    """Triangulate points but leave gaps larger than three 30 m cells empty."""
    tri=mtri.Triangulation(np.asarray(x,dtype=float),np.asarray(y,dtype=float))
    xy=np.column_stack([x,y])
    vertices=xy[tri.triangles]
    a=np.linalg.norm(vertices[:,0]-vertices[:,1],axis=1)
    b=np.linalg.norm(vertices[:,1]-vertices[:,2],axis=1)
    c=np.linalg.norm(vertices[:,2]-vertices[:,0],axis=1)
    tri.set_mask(np.maximum.reduce([a,b,c])>max_edge_m)
    return tri


def read_chip(site_key,scene_id,band):
    item=next(r for r in INV['items'] if r['site']==site_key and r['item_id']==scene_id)
    p=ROOT/item['assets'][band]['file']
    with rasterio.open(p) as ds:
        a=ds.read(1).astype(float)
        if ds.nodata is not None: a[a==ds.nodata]=np.nan
        a*=float(CFG['hls']['reflectance_scale_factor'])
        return a,ds.bounds,ds.crs


def stretch_rgb(r,g,b):
    vals=[]
    for a in [r,g,b]:
        ok=np.isfinite(a)
        if not ok.any(): vals.append(np.zeros_like(a)); continue
        lo,hi=np.nanpercentile(a[ok],[2,98])
        if not np.isfinite(hi-lo) or hi<=lo: hi=lo+1e-6
        vals.append(np.clip((a-lo)/(hi-lo),0,1))
    rgb=np.stack(vals,axis=-1)
    rgb[~np.isfinite(rgb).all(axis=-1)]=0.82
    return rgb


def write_figure1():
    samples=pd.read_csv(ROOT/'Results/Prepared/pixel_samples.csv.gz')
    fig,axes=plt.subplots(1,2,figsize=(12.2,6.4),layout='constrained')
    depths=[]
    for ax,(site_key,site) in zip(axes,CFG['sites'].items()):
        sid=site.get('nearest_date_s30_reference')
        r,bounds,_=read_chip(site_key,sid,'red'); g,_,_=read_chip(site_key,sid,'green'); b,_,_=read_chip(site_key,sid,'blue')
        rgb=stretch_rgb(r,g,b)
        ax.imshow(rgb,extent=(bounds.left,bounds.right,bounds.bottom,bounds.top),origin='upper',interpolation='nearest')
        pts=samples[samples.site==site_key]
        sc=ax.scatter(pts.x_m,pts.y_m,c=pts.depth_m,s=7,cmap='viridis',vmin=0,
                      vmax=np.nanpercentile(samples.depth_m,98),edgecolors='none',alpha=0.92,zorder=5)
        add_panel_label(ax,'A' if site_key==next(iter(CFG['sites'])) else 'B')
        ax.set_xlabel(f"Easting X (km; {site['image_crs']})")
        ax.set_ylabel(f"Northing Y (km; {site['image_crs']})")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v/1000:.1f}'))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v/1000:.1f}'))
        ax.tick_params(labelsize=8)
        add_compass_rose(ax); add_scale_bar(ax,bounds)
        depths.append(sc)
    cb=fig.colorbar(depths[-1],ax=axes,location='right',shrink=0.78,pad=0.02)
    cb.set_label('Survey depth below local hydrographic zero (m)')
    save(fig,'Figure1_sites_and_survey_pixels')


def write_figure2():
    q=pd.read_csv(ROOT/'Results/Prepared/scene_quality.csv',parse_dates=['datetime_utc'])
    fig,axes=plt.subplots(2,1,figsize=(11.5,6.2),sharex=False,layout='constrained')
    for ax,(site_key,site) in zip(axes,CFG['sites'].items()):
        d=q[q.site==site_key].copy()
        for product,color in [('S30',TEAL),('L30',CORAL)]:
            sub=d[d['product']==product]
            ax.scatter(sub.datetime_utc,[1 if product=='S30' else 0]*len(sub),
                       c=sub.clear_fraction_over_fixed_water_mask,cmap='viridis',vmin=0,vmax=1,
                       s=35+110*sub.clear_fraction_over_fixed_water_mask,marker='o' if product=='S30' else 's',
                       edgecolors=color,linewidths=0.65,label=f'{product} ({len(sub)})',zorder=4)
        ax.axvline(pd.Timestamp(site['survey_date']),color=INK,lw=1.2,ls='--',label='Survey date')
        add_panel_label(ax,'A' if site_key==next(iter(CFG['sites'])) else 'B')
        ax.set_yticks([0,1],['HLS L30 · Landsat','HLS S30 · Sentinel-2'])
        ax.set_ylim(-0.65,1.65); ax.grid(axis='x',color='#DCE5EA',lw=0.65)
        ax.set_ylabel('Input sensor')
        ax.legend(frameon=False,ncol=3,loc='upper right',fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    axes[-1].set_xlabel('Acquisition date (UTC)')
    sm=plt.cm.ScalarMappable(cmap='viridis',norm=plt.Normalize(0,1)); sm.set_array([])
    cb=fig.colorbar(sm,ax=axes,location='right',shrink=0.82,pad=0.02); cb.set_label('Clear fraction over fixed image-only water mask')
    save(fig,'Figure2_acquisition_timeline_quality')


def load_outputs():
    metrics=pd.read_csv(ROOT/'Results/Metrics/cv_metrics.csv')
    summary=pd.read_csv(ROOT/'Results/Metrics/scene_count_summary.csv')
    oof=pd.read_csv(ROOT/'Results/Predictions/oof_all_scenes_quality_weighted.csv.gz')
    metadata=json.loads((ROOT/'Results/Logs/run_metadata.json').read_text(encoding='utf-8'))
    if not metadata.get('tide_model_anomaly_applied',False):
        print('WARNING: figures are based on preliminary predictions without the model-tide anomaly adjustment.')
    return metrics,summary,oof,metadata


def diagnostic_model():
    model=CFG.get('primary_diagnostic_model','RandomForest')
    if model not in MODELS: raise ValueError(f'Unknown predeclared diagnostic model: {model}')
    return model


def write_figure3(metrics,summary,oof,metadata):
    fold_colors=[plt.get_cmap('tab10')(i) for i in range(5)]
    fig=plt.figure(figsize=(12,5.9),layout='constrained')
    gs=fig.add_gridspec(2,2,height_ratios=[1,.10])
    axes=[fig.add_subplot(gs[0,col]) for col in range(2)]
    legend_ax=fig.add_subplot(gs[1,:]); legend_ax.set_axis_off()
    for ax,(site_key,site) in zip(axes,CFG['sites'].items()):
        model=diagnostic_model()
        d=oof[(oof.site==site_key)&(oof.model==model)]
        if d.empty: continue
        d=d.drop_duplicates(['sample_index'])
        folds=pd.to_numeric(d.fold,errors='coerce').fillna(0).astype(int)
        point_colors=[fold_colors[f-1] if 1<=f<=5 else '#9AA7B0' for f in folds]
        ax.scatter(d.x_m,d.y_m,c=point_colors,s=8,alpha=0.82,edgecolors='none')
        # Show the fixed 500 m spatial blocks assigned to each held-out fold.
        block=d.assign(bx=np.floor(d.x_m/500).astype(int),by=np.floor(d.y_m/500).astype(int)).groupby(['bx','by']).fold.agg(lambda x:x.mode().iloc[0])
        for (bx,by),fold in block.items():
            fold=int(fold)
            color=fold_colors[fold-1] if 1<=fold<=5 else '#9AA7B0'
            ax.add_patch(Rectangle((bx*500,by*500),500,500,facecolor='none',edgecolor=color,lw=0.9,alpha=0.85))
        ax.set_aspect('equal'); ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v/1000:.1f}'))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v/1000:.1f}'))
        ax.set_xlabel('Easting X (km)'); ax.set_ylabel('Northing Y (km)')
        add_panel_label(ax,'A' if site_key==next(iter(CFG['sites'])) else 'B')
        ax.grid(color='#E4EBEF',lw=0.5)
    fold_handles=[Patch(facecolor='white',edgecolor=fold_colors[i],linewidth=1.5,label=f'Fold {i+1}') for i in range(5)]
    legend_ax.legend(handles=fold_handles,loc='center',ncol=5,frameon=False,fontsize=8,
                     handlelength=1.5,columnspacing=1.8,title='Held-out spatial fold')
    save(fig,'Figure3_spatial_validation_blocks')


def write_figure4(metrics,summary):
    from matplotlib.lines import Line2D
    fig,axes=plt.subplots(3,2,figsize=(13,12))
    palette={'quality_ranked':TEAL,'random_subset':CORAL,'all_scenes':NAVY}
    panel=0
    for row,model in enumerate(MODELS):
        for col,(site_key,site) in enumerate(CFG['sites'].items()):
            ax=axes[row,col]
            d=metrics[(metrics.site==site_key)&(metrics.model==model)&(metrics.scene_pool=='HLS_Mixed')]
            for selection in ['quality_ranked','random_subset','all_scenes']:
                sub=d[d.scene_selection==selection]
                if sub.empty: continue
                # Average spatial folds per random subset, then show the empirical min–max range.
                reps=(sub.groupby(['aggregation','n_scenes','replicate'],as_index=False).rmse_m.mean())
                agg=reps.groupby(['aggregation','n_scenes']).rmse_m.agg(['mean','min','max']).reset_index()
                for aggregation,ls in [('quality_weighted_mean','-'),('unweighted_median','--')]:
                    a=agg[agg.aggregation==aggregation].sort_values('n_scenes')
                    if a.empty: continue
                    ax.plot(a.n_scenes,a['mean'],marker='o',ms=3.5,lw=1.6,ls=ls,color=palette[selection])
                    if selection=='random_subset':
                        ax.fill_between(a.n_scenes,a['min'],a['max'],color=palette[selection],alpha=0.12,linewidth=0)
            if model=='StumpfLogRatio':
                single=d[d.scene_selection=='single_scene']
                if not single.empty:
                    one=single.groupby('scene_ids').rmse_m.mean().to_numpy()
                    if one.size: ax.scatter(np.ones(one.size),one,s=12,c='#7A8B96',alpha=0.52,edgecolors='none')
            ax.set_xlabel('Number of scenes')
            ax.set_ylabel('Spatially held-out RMSE (m)')
            add_panel_label(ax,chr(ord('A')+panel)); panel+=1
            ax.grid(color='#E4EBEF',lw=0.65)
    handles=[Line2D([0],[0],color=TEAL,lw=1.8,label='Quality-ranked scenes'),
             Line2D([0],[0],color=CORAL,lw=1.8,label='Random scene subsets'),
             Line2D([0],[0],color=NAVY,lw=1.8,label='All scenes'),
             Line2D([0],[0],color=INK,lw=1.8,ls='-',label='Quality-weighted mean'),
             Line2D([0],[0],color=INK,lw=1.8,ls='--',label='Unweighted median'),
             Patch(facecolor=CORAL,alpha=0.12,edgecolor='none',label='Random-subset min–max range'),
             Line2D([0],[0],marker='o',color='none',markerfacecolor='#7A8B96',markersize=4,label='Individual single-scene Stumpf runs')]
    fig.subplots_adjust(left=0.075,right=0.99,bottom=0.08,top=0.88,hspace=0.28,wspace=0.14)
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(0.5,0.98),ncol=4,frameon=False,fontsize=8,
              columnspacing=1.35,handletextpad=0.5,labelspacing=0.6)
    save(fig,'Figure4_scene_count_convergence')


def write_figure5(metrics,summary):
    q=pd.read_csv(ROOT/'Results/Prepared/scene_quality.csv')
    d=metrics[(metrics.scene_pool=='HLS_Mixed')&(metrics.scene_selection=='single_scene')&
              (metrics.aggregation=='single')&(metrics.model=='StumpfLogRatio')].copy()
    if d.empty:return
    d=d.groupby(['site','scene_ids'],as_index=False).rmse_m.mean().rename(columns={'scene_ids':'item_id','rmse_m':'single_scene_rmse_m'})
    d=d.merge(q,on=['site','item_id'],how='left')
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.7),layout='constrained')
    for ax,(site_key,site) in zip(axes,CFG['sites'].items()):
        s=d[d.site==site_key]
        for product,color,marker in [('S30',TEAL,'o'),('L30',CORAL,'s')]:
            z=s[s['product']==product]
            ax.scatter(z.clear_fraction_at_survey_pixels,z.single_scene_rmse_m,c=z.date_offset_days,
                       cmap='coolwarm',vmin=-30,vmax=30,marker=marker,s=45,edgecolors='white',linewidths=0.5,
                       label=product)
        ax.set_xlabel('Clear fraction at submerged sounding pixels')
        ax.set_ylabel('Single-scene Stumpf RMSE (m)')
        add_panel_label(ax,'A' if site_key==next(iter(CFG['sites'])) else 'B')
        ax.grid(color='#E4EBEF',lw=0.7); ax.legend(frameon=False)
    sm=plt.cm.ScalarMappable(cmap='coolwarm',norm=plt.Normalize(-30,30)); sm.set_array([])
    cb=fig.colorbar(sm,ax=axes,location='right',shrink=0.84,pad=0.02); cb.set_label('Scene offset from survey (days)')
    save(fig,'Figure5_scene_quality_vs_single_scene_error')


def write_figure6(summary,oof,metadata):
    from matplotlib.lines import Line2D
    fig=plt.figure(figsize=(12,8.7),layout='constrained')
    gs=fig.add_gridspec(4,2,height_ratios=[1,1,.10,.12])
    axes=np.array([[fig.add_subplot(gs[row,col]) for col in range(2)] for row in range(2)])
    colorbar_ax=fig.add_subplot(gs[2,:])
    legend_ax=fig.add_subplot(gs[3,:]); legend_ax.set_axis_off()
    residual_scatter=None
    for col,(site_key,site) in enumerate(CFG['sites'].items()):
        model=diagnostic_model()
        d=oof[(oof.site==site_key)&(oof.model==model)].drop_duplicates('sample_index')
        ax=axes[0,col]
        vmax=max(float(np.nanpercentile(np.r_[d.observed_depth_m,d.predicted_depth_m],98)),1)
        hb=ax.hexbin(d.observed_depth_m,d.predicted_depth_m,gridsize=32,mincnt=1,bins='log',cmap='cividis')
        ax.plot([0,vmax],[0,vmax],color=CORAL,ls='--',lw=1.2,label='1:1')
        ax.set_xlim(0,vmax); ax.set_ylim(0,vmax); ax.set_aspect('equal',adjustable='box')
        ax.set_xlabel('Survey depth below local zero (m)'); ax.set_ylabel('Out-of-fold predicted depth (m)')
        add_panel_label(ax,'A' if col==0 else 'B')
        fig.colorbar(hb,ax=ax,fraction=0.046,pad=0.02,label='Pixel count (log scale)')
        ax2=axes[1,col]
        residual_scatter=ax2.scatter(d.observed_depth_m,d.residual_m,c=d.residual_m,cmap='RdBu_r',vmin=-2,vmax=2,s=10,alpha=0.55,edgecolors='none')
        ax2.axhline(0,color=INK,ls='--',lw=0.9)
        qbins=pd.qcut(d.observed_depth_m,q=min(10,max(3,d.observed_depth_m.nunique()//10)),duplicates='drop')
        trend=d.groupby(qbins,observed=False).residual_m.median()
        centers=d.groupby(qbins,observed=False).observed_depth_m.median()
        ax2.plot(centers,trend,color=GOLD,lw=2.1,marker='o',ms=3.5,label='Median by depth bin')
        ax2.set_xlabel('Survey depth below local zero (m)'); ax2.set_ylabel('Prediction − survey (m)')
        add_panel_label(ax2,'C' if col==0 else 'D')
        ax2.grid(color='#E4EBEF',lw=0.65)
    cbar=fig.colorbar(residual_scatter,cax=colorbar_ax,orientation='horizontal',ticks=[-2,-1,0,1,2])
    cbar.set_label('Out-of-fold residual (prediction − survey, m)',labelpad=2)
    cbar.ax.tick_params(labelsize=7,pad=1)
    shared_handles=[Line2D([0],[0],color=CORAL,ls='--',lw=1.2,label='1:1'),
                    Line2D([0],[0],color=GOLD,lw=2.1,marker='o',ms=3.5,label='Median by depth bin')]
    legend_ax.legend(handles=shared_handles,loc='center',ncol=2,frameon=False,fontsize=7.5)
    save(fig,'Figure6_heldout_accuracy_and_residuals')


def write_figure7(summary,oof):
    fig,axes=plt.subplots(1,2,figsize=(12,5.7),layout='constrained')
    for ax,(site_key,site) in zip(axes,CFG['sites'].items()):
        model=diagnostic_model()
        d=oof[(oof.site==site_key)&(oof.model==model)].drop_duplicates('sample_index')
        ax.set_aspect('equal')
        if len(d)>3:
            tri=masked_triangulation(d.x_m.to_numpy(),d.y_m.to_numpy(),max_edge_m=90.0)
            levels=np.arange(0,max(2,math.ceil(float(d.observed_depth_m.max())/2)*2)+1,2)
            cf=ax.tricontourf(tri,d.observed_depth_m,levels=levels,cmap='viridis',extend='max')
            ax.tricontour(tri,d.observed_depth_m,levels=levels[::2],colors='white',linewidths=0.45,alpha=0.75)
            # The Ras El Ma panel has denser points at the same map scale; enlarge
            # its markers to preserve point visibility without changing the data.
            marker_size=6 if site_key=='Anza_Agadir' else 28
            marker_edge='none' if site_key=='Anza_Agadir' else 'white'
            marker_line=0 if site_key=='Anza_Agadir' else 0.2
            sc=ax.scatter(d.x_m,d.y_m,c=d.residual_m,cmap='RdBu_r',vmin=-2,vmax=2,
                          s=marker_size,edgecolors=marker_edge,linewidths=marker_line,alpha=0.92)
            cb=fig.colorbar(cf,ax=ax,shrink=0.82,pad=0.02); cb.set_label('Survey depth below local zero (m)')
            cb2=fig.colorbar(sc,ax=ax,shrink=0.82,pad=0.09); cb2.set_label('Out-of-fold residual (m)')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v/1000:.2f}'))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v/1000:.2f}'))
        ax.set_xlabel(f"Easting X (km; {site['image_crs']})"); ax.set_ylabel(f"Northing Y (km; {site['image_crs']})")
        add_panel_label(ax,'A' if site_key==next(iter(CFG['sites'])) else 'B')
        ax.grid(color='#E4EBEF',lw=0.5)
        add_compass_rose(ax); add_scale_bar(ax,(d.x_m.min(),d.y_m.min(),d.x_m.max(),d.y_m.max()),1000)
    save(fig,'Figure7_spatial_residual_maps')


def write_figure8(summary,oof):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig=plt.figure(figsize=(13,9),layout='constrained')
    for col,(site_key,site) in enumerate(CFG['sites'].items()):
        model=diagnostic_model()
        d=oof[(oof.site==site_key)&(oof.model==model)].drop_duplicates('sample_index').copy()
        if len(d)<10: continue
        x0=float(d.x_m.min()); y0=float(d.y_m.min())
        x=(d.x_m.to_numpy()-x0)/1000.0; y=(d.y_m.to_numpy()-y0)/1000.0
        depth_limit=max(float(np.nanmax(d[['observed_depth_m','predicted_depth_m']].to_numpy(dtype=float))),1.0)
        for row,field in enumerate(['observed_depth_m','predicted_depth_m']):
            ax=fig.add_subplot(2,2,row*2+col+1,projection='3d')
            depth=d[field].to_numpy(dtype=float)
            bed=-depth  # chart-zero elevation, negative below datum
            zkm=bed/1000.0; ve=12.0
            tri=masked_triangulation(x,y,max_edge_m=0.09)
            surf=ax.plot_trisurf(tri,zkm*ve,cmap='viridis',linewidth=0.08,edgecolor=(1,1,1,0.15),antialiased=True,alpha=0.97)
            # Color encodes physical depth in metres; only the plotted Z geometry
            # is vertically exaggerated. Use unmasked triangle means so the
            # colorbar cannot inherit the scaled Z coordinates from plot_trisurf.
            faces=tri.get_masked_triangles()
            surf.set_array(np.nanmean(depth[faces],axis=1))
            surf.set_clim(0.0,depth_limit)
            ax.set_xlabel('Easting X (km; UTM)',labelpad=6); ax.set_ylabel('Northing Y (km; UTM)',labelpad=6)
            ax.set_zlabel(f"Bed elevation Z (m; local ZH; VE {ve:g}×)",labelpad=5)
            panel={(0,0):'A',(0,1):'B',(1,0):'C',(1,1):'D'}[(row,col)]
            ax.text2D(0.015,0.985,panel,transform=ax.transAxes,ha='left',va='top',weight='bold',fontsize=10,color=INK)
            ax.view_init(elev=27,azim=-132); ax.set_box_aspect((max(np.ptp(x),0.1),max(np.ptp(y),0.1),max(np.ptp(zkm)*ve,0.015)))
            ticks=np.linspace(float(np.nanmin(bed)),float(np.nanmax(bed)),5)
            ax.set_zticks(ticks/1000*ve); ax.set_zticklabels([f'{v:.1f}' for v in ticks])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{x0/1000+v:.2f}'))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{y0/1000+v:.2f}'))
            ax.tick_params(axis='both',labelsize=7,pad=1); ax.tick_params(axis='z',labelsize=7,pad=1)
            fig.colorbar(surf,ax=ax,shrink=0.58,pad=0.04,aspect=18,
                         label='Depth below local hydrographic zero (m)')
    save(fig,'Figure8_xyz_bathymetry_surfaces')


def write_figure9_open_grid_context():
    """Map public grid values and GEBCO source classes without scoring them as truth."""
    from matplotlib.colors import TwoSlopeNorm, ListedColormap, BoundaryNorm
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch as LegendPatch
    from pyproj import Transformer

    grid_files={
        'Anza_Agadir':[
            ROOT/'Data/Open_Bathymetry/EMODnet/EMODnet_Anza_Agadir_release_2022_mean_depth.tif',
            ROOT/'Data/Open_Bathymetry/EMODnet/EMODnet_Anza_Agadir_current_mean_depth.tif',
            ROOT/'Data/Open_Bathymetry/GEBCO/gebco_2026_n30.49_s30.41_w-9.71_e-9.64_geotiff.tif',
            ROOT/'Data/Open_Bathymetry/GEBCO/gebco_2026_tid_n30.49_s30.41_w-9.71_e-9.64_geotiff.tif'],
        'Ras_El_Ma_Ras_Kebdana':[
            ROOT/'Data/Open_Bathymetry/EMODnet/EMODnet_Ras_El_Ma_release_2022_mean_depth.tif',
            ROOT/'Data/Open_Bathymetry/EMODnet/EMODnet_Ras_El_Ma_current_mean_depth.tif',
            ROOT/'Data/Open_Bathymetry/GEBCO/gebco_2026_n35.19_s35.1_w-2.47_e-2.35_geotiff.tif',
            ROOT/'Data/Open_Bathymetry/GEBCO/gebco_2026_tid_n35.19_s35.1_w-2.47_e-2.35_geotiff.tif']}
    missing=[str(p) for group in grid_files.values() for p in group if not p.exists()]
    if missing: raise SystemExit('Missing contextual bathymetry input(s): '+', '.join(missing))

    sample_path=ROOT/'Results/Prepared/pixel_samples.csv.gz'
    samples=pd.read_csv(sample_path,usecols=['site','x_m','y_m'])
    projectors={key:Transformer.from_crs(site['image_crs'],'EPSG:4326',always_xy=True)
                for key,site in CFG['sites'].items()}
    fig=plt.figure(figsize=(15,8.3),layout='constrained')
    gs=fig.add_gridspec(3,4,height_ratios=[1,1,0.22])
    axes=np.array([[fig.add_subplot(gs[row,col]) for col in range(4)] for row in range(2)])
    legend_ax=fig.add_subplot(gs[2,:]); legend_ax.set_axis_off()
    grid_cmap='RdBu_r'
    tid_codes=[0,16,40,44]
    tid_names={0:'Land',16:'Optical',40:'Altimetry prediction',44:'Sounding-constrained grid'}
    tid_colors=['#D7E0E5','#168C8C','#E9B44C','#665191']
    tid_cmap=ListedColormap(tid_colors)
    tid_norm=BoundaryNorm(np.arange(-0.5,len(tid_codes)+0.5),tid_cmap.N)
    panel=0

    for row,(site_key,site) in enumerate(CFG['sites'].items()):
        files=grid_files[site_key]
        rasters=[]
        for path in files:
            with rasterio.open(path) as ds:
                arr=ds.read(1).astype(float)
                if ds.nodata is not None: arr[arr==ds.nodata]=np.nan
                rasters.append((arr,ds.bounds,ds.nodata))
        finite=np.concatenate([np.asarray(a)[np.isfinite(a)] for a,_,_ in rasters[:3]])
        lower=float(np.nanpercentile(finite,3)); upper=float(np.nanpercentile(finite,97))
        if lower>=0: lower=-1.0
        if upper<=0: upper=1.0
        norm=TwoSlopeNorm(vmin=lower,vcenter=0.0,vmax=upper)

        pts=samples[samples.site==site_key]
        if len(pts)>1600: pts=pts.iloc[::int(math.ceil(len(pts)/1600))]
        lon,lat=projectors[site_key].transform(pts.x_m.to_numpy(),pts.y_m.to_numpy())
        for col,(ax,(arr,bounds,nodata)) in enumerate(zip(axes[row],rasters)):
            extent=(bounds.left,bounds.right,bounds.bottom,bounds.top)
            if col<3:
                im=ax.imshow(np.ma.masked_invalid(arr),extent=extent,origin='upper',interpolation='nearest',
                             cmap=grid_cmap,norm=norm,zorder=1)
            else:
                code_to_index={code:i for i,code in enumerate(tid_codes)}
                idx=np.full(arr.shape,np.nan,dtype=float)
                for code,index in code_to_index.items(): idx[arr==code]=index
                im=ax.imshow(np.ma.masked_invalid(idx),extent=extent,origin='upper',interpolation='nearest',
                             cmap=tid_cmap,norm=tid_norm,zorder=1)
            ax.scatter(lon,lat,s=2.2,c='#17212B',alpha=0.52,linewidths=0,zorder=3)
            ax.set_xlabel('Longitude (°)'); ax.set_ylabel('Latitude (°)')
            ax.set_aspect(1.0/np.cos(np.deg2rad((bounds.bottom+bounds.top)/2)))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:.2f}'))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:.2f}'))
            ax.tick_params(labelsize=7)
            add_panel_label(ax,chr(ord('A')+panel)); panel+=1
            ax.grid(color='white',lw=0.5,alpha=0.4)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=grid_cmap),ax=axes[row,:3],shrink=0.75,pad=0.015,
                     label='Source-grid value (m; native sign)')

    tid_legend=[LegendPatch(facecolor=color,edgecolor='white',label=f'{code} · {tid_names[code]}')
                for code,color in zip(tid_codes,tid_colors)]
    survey_handle=Line2D([0],[0],marker='o',linestyle='none',markersize=3,
                         markerfacecolor='#17212B',markeredgewidth=0,label='Survey sample pixels')
    legend_ax.legend(handles=[survey_handle,*tid_legend],loc='center',ncol=3,frameon=False,
                     fontsize=7,title='Survey overlay and GEBCO TID classes',title_fontsize=7.5)
    save(fig,'Figure9_open_bathymetry_and_source_context')


def write_figure10_sensor_window_sensitivity(metrics):
    from matplotlib.lines import Line2D
    pools=[('S30_only','#168C8C','S30 only · ±30 d'),
           ('HLS_Mixed','#183B56','S30 + L30 · ±30 d'),
           ('S30_only_14d','#6DBBB8','S30 only · ±14 d'),
           ('HLS_Mixed_14d','#E9A26A','S30 + L30 · ±14 d')]
    d=metrics[(metrics.scene_selection=='all_scenes')&
              (metrics.aggregation=='quality_weighted_mean')&
              (metrics.scene_pool.isin([p[0] for p in pools]))].copy()
    if d.empty: return
    fig,axes=plt.subplots(1,2,figsize=(12,5.2),layout='constrained')
    offsets=np.linspace(-0.27,0.27,len(pools))
    for ax,(site_key,site) in zip(axes,CFG['sites'].items()):
        site_data=d[d.site==site_key]
        for ix,(pool_name,color,label) in enumerate(pools):
            subset=site_data[site_data.scene_pool==pool_name]
            for mi,model in enumerate(MODELS):
                vals=subset[subset.model==model].rmse_m.dropna().to_numpy(dtype=float)
                if not len(vals): continue
                x=mi+offsets[ix]
                ax.scatter(np.full(len(vals),x),vals,s=16,color=color,alpha=0.42,edgecolors='white',linewidths=0.35,zorder=2)
                ax.errorbar(x,np.mean(vals),yerr=np.std(vals,ddof=1) if len(vals)>1 else 0.0,
                            fmt='o',markersize=5,color=color,ecolor=color,capsize=2.5,lw=1.1,zorder=4)
        ax.set_xticks(range(len(MODELS)),[MODEL_LABELS[m] for m in MODELS],rotation=15,ha='right')
        ax.set_ylabel('Spatially held-out RMSE (m)')
        ax.set_xlabel('Model')
        ax.grid(axis='y',color='#E4EBEF',lw=0.65)
        add_panel_label(ax,'A' if site_key==next(iter(CFG['sites'])) else 'B')
        ax.legend(handles=[Line2D([0],[0],marker='o',linestyle='none',color=color,label=label,markersize=5)
                           for _,color,label in pools],frameon=False,fontsize=7,loc='best')
    save(fig,'Figure10_sensor_mix_and_window_sensitivity')


def write_figure11_supported_depth_maps(metadata):
    """Draw full-fit depth maps only inside the predefined survey-support mask."""
    map_dir=ROOT/'Results/Predictions/Maps'
    required=[map_dir/f'{site_key}_RandomForest_supported_depth.tif'
              for site_key in CFG['sites']]
    missing=[str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit('Run Scripts/03_generate_supported_maps.py first; missing map(s): '+', '.join(missing))
    summary_path=map_dir/'map_summary.csv'
    if not summary_path.exists():
        raise SystemExit('Missing map_summary.csv; rerun Scripts/03_generate_supported_maps.py.')
    map_status=pd.read_csv(summary_path)['tide_model_anomaly_applied'].astype(str).str.lower().isin(['true','1']).all()
    if bool(map_status)!=bool(metadata.get('tide_model_anomaly_applied',False)):
        raise SystemExit('The map and validation tide-anomaly states differ. Rerun Stages 2 and 3 with the same tide table and flags.')

    samples=pd.read_csv(ROOT/'Results/Prepared/pixel_samples.csv.gz',usecols=['site','x_m','y_m'])
    fig,axes=plt.subplots(1,2,figsize=(12.5,6.1),layout='constrained')
    for panel,(ax,(site_key,site),path) in enumerate(zip(axes,CFG['sites'].items(),required)):
        with rasterio.open(path) as ds:
            depth=ds.read(1).astype(float)
            if ds.nodata is not None: depth[depth==ds.nodata]=np.nan
            bounds=ds.bounds; transform=ds.transform; crs=ds.crs
            rowcol_count=int(np.isfinite(depth).sum())
        if rowcol_count==0:
            raise SystemExit(f'No supported prediction pixels in {path}; inspect sample coverage and image QA.')
        extent=(bounds.left,bounds.right,bounds.bottom,bounds.top)
        finite=depth[np.isfinite(depth)]
        vmax=max(float(np.nanpercentile(finite,99)),0.5)
        vmax=math.ceil(vmax*2)/2
        im=ax.imshow(np.ma.masked_invalid(depth),extent=extent,origin='upper',interpolation='nearest',
                     cmap='mako' if 'mako' in plt.colormaps() else 'viridis',vmin=0,vmax=vmax,zorder=1)
        step=0.5 if vmax<=8 else 1.0
        levels=np.arange(step, vmax, step)
        if len(levels):
            xcenters=transform.c+(np.arange(depth.shape[1])+0.5)*transform.a
            ycenters=transform.f+(np.arange(depth.shape[0])+0.5)*transform.e
            contour=ax.contour(xcenters,ycenters,np.ma.masked_invalid(depth),levels=levels,
                               colors='#17324D',linewidths=0.55,alpha=0.75,zorder=2,corner_mask=False)
            ax.clabel(contour,fmt=lambda val:f'{val:g} m',fontsize=6,inline=True,inline_spacing=2)
        pts=samples[samples.site==site_key]
        if len(pts)>3500: pts=pts.iloc[::int(math.ceil(len(pts)/3500))]
        ax.scatter(pts.x_m,pts.y_m,s=1.0,c='white',alpha=0.65,linewidths=0,zorder=3)
        ax.set_xlim(bounds.left,bounds.right); ax.set_ylim(bounds.bottom,bounds.top)
        ax.set_aspect('equal',adjustable='box')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v/1000:.2f}'))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v/1000:.2f}'))
        ax.set_xlabel(f"Easting X (km; {site['image_crs']})")
        ax.set_ylabel(f"Northing Y (km; {site['image_crs']})")
        add_panel_label(ax,'A' if panel==0 else 'B')
        ax.grid(color='white',lw=0.45,alpha=0.45,zorder=4)
        add_compass_rose(ax,x=0.91,y=0.87,size=0.045)
        add_scale_bar(ax,(bounds.left,bounds.bottom,bounds.right,bounds.top),1000)
        fig.colorbar(im,ax=ax,shrink=0.82,pad=0.025,label='Predicted depth below local hydrographic zero (m)')
    save(fig,'Figure11_supported_full_data_depth_maps')


def main():
    required=[ROOT/'Results/Prepared/pixel_samples.csv.gz',ROOT/'Results/Metrics/cv_metrics.csv',
              ROOT/'Results/Metrics/scene_count_summary.csv',ROOT/'Results/Predictions/oof_all_scenes_quality_weighted.csv.gz',
              ROOT/'Results/Logs/run_metadata.json']
    missing=[str(p) for p in required if not p.exists()]
    if missing: raise SystemExit('Run Scripts/01_prepare_samples.py and Scripts/02_run_validation.py first; missing: '+', '.join(missing))
    print('Stage 8/9: drawing base figures from saved validation results and supported prediction maps.')
    metrics,summary,oof,metadata=load_outputs()
    write_figure1(); print('  Figure 1: study maps')
    write_figure2(); print('  Figure 2: acquisition quality')
    write_figure3(metrics,summary,oof,metadata); print('  Figure 3: spatial validation blocks')
    write_figure4(metrics,summary); print('  Figure 4: scene-count response')
    write_figure5(metrics,summary); print('  Figure 5: scene-quality/error relationship')
    write_figure6(summary,oof,metadata); print('  Figure 6: validation diagnostics')
    write_figure7(summary,oof); print('  Figure 7: spatial residual maps')
    write_figure8(summary,oof); print('  Figure 8: 3D XYZ surfaces')
    write_figure9_open_grid_context(); print('  Figure 9: open bathymetry and source context')
    write_figure10_sensor_window_sensitivity(metrics); print('  Figure 10: sensor-mix and temporal-window sensitivity')
    write_figure11_supported_depth_maps(metadata); print('  Figure 11: supported full-data depth maps (not validation maps)')
    print(f'All figures saved as 350 dpi PNG and vector PDF in {OUT}')

if __name__=='__main__': main()
