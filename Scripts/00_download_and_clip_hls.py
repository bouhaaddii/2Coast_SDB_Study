#!/usr/bin/env python
"""Download compact HLS v2.0 input chips; this does not run bathymetry models."""
from __future__ import annotations
import csv, json, math, sys, time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
import requests
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.windows import from_bounds

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'Config/experiment.json').read_text(encoding='utf-8'))
API=CFG['hls']['stac_api']
SAS_ENDPOINT='https://planetarycomputer.microsoft.com/api/sas/v1/token/hls2euwest/hls2'
BAND_MAP=CFG['hls']['band_map']
HEADERS={'User-Agent':'ECSS-Morocco-SDB-Research/1.0'}


def point_bounds(path: Path, site: dict):
    xs=[]; ys=[]
    if path.suffix.lower()=='.csv':
        with path.open(newline='',encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                xs.append(float(row['x'])); ys.append(float(row['y']))
    else:
        with path.open(encoding='utf-8-sig') as f:
            for line in f:
                parts=line.split()
                if len(parts)>=4:
                    try: xs.append(float(parts[1])); ys.append(float(parts[2]))
                    except ValueError: continue
    if not xs: raise RuntimeError(f'No valid survey coordinates found: {path}')
    trans=Transformer.from_crs(site['survey_crs'],site['image_crs'],always_xy=True)
    X,Y=trans.transform(np.asarray(xs),np.asarray(ys))
    b=CFG['roi_buffer_m']
    return min(X)-b,min(Y)-b,max(X)+b,max(Y)+b


def get_items(site_name: str, site: dict, collection: str):
    survey=date.fromisoformat(site['survey_date'])
    lo=(survey-timedelta(days=CFG['primary_temporal_window_days'])).isoformat()+'T00:00:00Z'
    hi=(survey+timedelta(days=CFG['primary_temporal_window_days'])).isoformat()+'T23:59:59Z'
    payload={'collections':[collection],'bbox':site['query_bbox_wgs84'],'datetime':f'{lo}/{hi}','limit':500}
    r=requests.post(API+'/search',json=payload,headers=HEADERS,timeout=90); r.raise_for_status()
    data=r.json(); features=data.get('features',[])
    features=[f for f in features if f.get('id','').split('.')[2]==site['hls_tile']]
    features.sort(key=lambda f:f.get('properties',{}).get('datetime',''))
    return features


def token_string():
    r=requests.get(SAS_ENDPOINT,headers=HEADERS,timeout=45); r.raise_for_status()
    token=r.json().get('token')
    if not token: raise RuntimeError('Planetary Computer did not return a data-access token.')
    return token.lstrip('?')


def sign_href(href: str, token: str):
    clean=href.split('?',1)[0]
    return clean+'?'+token


def write_chip(url: str, out: Path, bounds, tags):
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',GDAL_HTTP_MULTIRANGE='YES'):
        with rasterio.open(url) as ds:
            if ds.crs is None: raise RuntimeError('HLS source asset has no CRS')
            win=from_bounds(*bounds,transform=ds.transform).round_offsets().round_lengths()
            col0=max(0,int(win.col_off)); row0=max(0,int(win.row_off))
            col1=min(ds.width,col0+int(win.width)); row1=min(ds.height,row0+int(win.height))
            if col1<=col0 or row1<=row0: raise RuntimeError('Study window does not intersect the HLS granule')
            win=rasterio.windows.Window(col0,row0,col1-col0,row1-row0)
            arr=ds.read(1,window=win)
            profile=ds.profile.copy()
            profile.update(driver='GTiff',width=arr.shape[1],height=arr.shape[0],count=1,
                           transform=ds.window_transform(win),compress='DEFLATE',predictor=2,
                           tiled=False,bigTIFF='IF_SAFER')
            out.parent.mkdir(parents=True,exist_ok=True)
            with rasterio.open(out,'w',**profile) as target:
                target.write(arr,1)
                safe={k:str(v) for k,v in tags.items() if v is not None}
                target.update_tags(**safe)
            return {'shape':[int(arr.shape[0]),int(arr.shape[1])],
                    'crs':str(ds.crs),'transform':list(ds.window_transform(win)),
                    'valid_pixels':int(np.count_nonzero(arr!=ds.nodata)) if ds.nodata is not None else int(arr.size)}


def main():
    session=requests.Session(); session.headers.update(HEADERS)
    sas=token_string()
    inventory=[]; failures=[]
    for site_name,site in CFG['sites'].items():
        bounds=point_bounds(ROOT/site['survey_points'],site)
        for collection,product in [(CFG['hls']['collection_s30'],'S30'),(CFG['hls']['collection_l30'],'L30')]:
            features=get_items(site_name,site,collection)
            print(f'{site_name}: {product} candidate scenes={len(features)}',flush=True)
            for item in features:
                iid=item['id']; props=item.get('properties',{})
                dt=props.get('datetime','')
                out_dir=ROOT/'Data'/'Satellite'/site_name/(dt[:10] if dt else 'date_unknown')/iid
                layers=BAND_MAP[product]
                row={'site':site_name,'display_name':site['display_name'],'product':product,'item_id':iid,
                     'datetime_utc':dt,'cloud_cover_tile_percent':props.get('eo:cloud_cover'),
                     'tile':site['hls_tile'],'collection':collection,'source':'NASA/USGS HLS v2.0 via Microsoft Planetary Computer',
                     'data_doi':('10.5067/HLS/HLSS30.002' if product=='S30' else '10.5067/HLS/HLSL30.002'),
                     'roi_buffer_m':CFG['roi_buffer_m'],'assets':{},'status':'pending'}
                try:
                    for logical,asset_name in layers.items():
                        asset=item.get('assets',{}).get(asset_name)
                        if not asset or not asset.get('href'):
                            raise RuntimeError(f'Missing HLS layer {asset_name}')
                        out=out_dir/f'{logical}.tif'
                        tags={'site':site_name,'stac_item_id':iid,'acquisition_utc':dt,
                              'hls_product':product,'logical_band':logical,'original_layer':asset_name,
                              'source_doi':row['data_doi'],'source_href':asset['href'].split('?',1)[0]}
                        if out.exists():
                            row['assets'][logical]={'file':str(out.relative_to(ROOT)),'status':'already_present'}
                            continue
                        stats=write_chip(sign_href(asset['href'],sas),out,bounds,tags)
                        row['assets'][logical]={'file':str(out.relative_to(ROOT)),'status':'downloaded',**stats}
                    row['status']='complete'
                    print('  saved',iid,flush=True)
                except Exception as e:
                    row['status']='failed'; row['error']=f'{type(e).__name__}: {e}'
                    failures.append({'site':site_name,'item_id':iid,'error':row['error']})
                    print('  FAILED',iid,row['error'],file=sys.stderr,flush=True)
                inventory.append(row)
                time.sleep(0.05)
    out_json=ROOT/'Data'/'Satellite'/'scene_inventory.json'
    out_json.write_text(json.dumps({'source':'NASA/USGS HLS v2.0','downloaded_utc':datetime.now(timezone.utc).isoformat(),
                                    'temporal_window_days':CFG['primary_temporal_window_days'],
                                    'roi_buffer_m':CFG['roi_buffer_m'],'items':inventory,'failures':failures},
                                   indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    csv_path=ROOT/'Data'/'Satellite'/'scene_inventory.csv'
    fields=['site','display_name','product','item_id','datetime_utc','cloud_cover_tile_percent','tile','collection','data_doi','status','error']
    with csv_path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for row in inventory: w.writerow({k:row.get(k,'') for k in fields})
    print(f'Input-image acquisition finished. Items={len(inventory)}; complete={sum(r["status"]=="complete" for r in inventory)}; failed={len(failures)}')
    print('No depth models, predictions, validation summaries, or publication figures were run.')
    if failures: raise SystemExit(2)

if __name__=='__main__': main()
