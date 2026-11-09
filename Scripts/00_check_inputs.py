"""Inventory copied inputs and report the datasets that are still absent."""
from __future__ import annotations
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'Config/experiment.json').read_text(encoding='utf-8'))


def main():
    lines=['Two-coast SDB input audit','='*28]
    missing=[]
    for site_key,site in CFG['sites'].items():
        path=ROOT/site['survey_points']
        ok=path.is_file()
        lines.append(f"Survey points | {site_key} | {'FOUND' if ok else 'MISSING'} | {path.relative_to(ROOT)}")
        if not ok: missing.append(str(path.relative_to(ROOT)))

    inventory_path=ROOT/'Data/Satellite/scene_inventory.json'
    if inventory_path.exists():
        inventory=json.loads(inventory_path.read_text(encoding='utf-8'))
        items=inventory.get('items',[])
        complete=0; broken=[]
        for item in items:
            asset_paths=[ROOT/a.get('file','') for a in item.get('assets',{}).values() if a.get('file')]
            assets_ok=(item.get('status')=='complete' and len(asset_paths)>=5 and all(p.is_file() for p in asset_paths))
            complete+=int(assets_ok)
            if not assets_ok: broken.append(item.get('item_id','unknown'))
        lines.append(f'HLS acquisitions | {complete}/{len(items)} complete local chips')
        if broken:
            lines.append('HLS acquisitions needing download/recovery: '+', '.join(broken))
            lines.append('Command: .\\.venv\\Scripts\\python.exe .\\Scripts\\00_download_and_clip_hls.py')
    else:
        lines.append('HLS acquisitions | inventory missing')
        missing.append('Data/Satellite/scene_inventory.json')
        lines.append('Command: .\\.venv\\Scripts\\python.exe .\\Scripts\\00_download_and_clip_hls.py')

    tide_files=list((ROOT/'Data/Tide_Model').rglob('*.nc'))
    open_grids=list((ROOT/'Data/Open_Bathymetry').rglob('*.tif'))
    atl24_files=list((ROOT/'Data/ICESat2_ATL24').rglob('*.h5')) if (ROOT/'Data/ICESat2_ATL24').exists() else []
    lines.append(f'GOT5 tide-model NetCDF files | {len(tide_files)} found')
    lines.append(f'EMODnet/GEBCO context rasters | {len(open_grids)} found')
    lines.append(f'ICESat-2 ATL24 external-check granules | {len(atl24_files)} found')
    lines.append('Local port/ANP tide-gauge observations | not included in the copied data')
    lines.append('The pipeline uses modeled relative anomalies without claiming locally observed or absolute port-zero water levels.')
    if not atl24_files:
        lines.append('Optional input not downloaded: ATL24 granules near the survey dates. Search/download command:')
        lines.append('  .\\.venv\\Scripts\\python.exe .\\Scripts\\00_download_atl24.py')
        lines.append('ATL24 is optional. Check track overlap, quality flags, time, and vertical reference before using it as an independent comparison.')
    if missing:
        lines.append('Required items still missing: '+', '.join(missing))
    else:
        lines.append('Required copied inputs are present. HLS download command can be rerun to fill missing chips.')

    out=ROOT/'Results/Logs/input_audit.txt'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))
    print(f'\nAudit saved to {out.relative_to(ROOT)}')


if __name__=='__main__': main()
