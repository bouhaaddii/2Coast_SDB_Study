"""Optionally search and download ICESat-2 ATL24 granules near survey dates.

ATL24 is supplementary: a spatial intersection does not guarantee usable
bottom photons or a directly compatible vertical reference.
"""
from __future__ import annotations
import json
from datetime import date,timedelta
from pathlib import Path

import earthaccess

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'Config/experiment.json').read_text(encoding='utf-8'))
WINDOW=int(CFG.get('atl24_optional_search_window_days',30))
VERSION=str(CFG.get('atl24_version','002'))


def main():
    out_root=ROOT/'Data/ICESat2_ATL24'
    out_root.mkdir(parents=True,exist_ok=True)
    manifest={'product':'ATL24','version':VERSION,'doi':'10.5067/ATLAS/ATL24.002',
              'window_days_each_side':WINDOW,'sites':[]}
    all_results=[]
    for site_key,site in CFG['sites'].items():
        survey=date.fromisoformat(site['survey_date'])
        start=(survey-timedelta(days=WINDOW)).isoformat()
        end=(survey+timedelta(days=WINDOW)).isoformat()
        west,south,east,north=site['query_bbox_wgs84']
        results=earthaccess.search_data(short_name='ATL24',version=VERSION,
                                        bounding_box=(west,south,east,north),
                                        temporal=(start,end))
        manifest['sites'].append({'site':site_key,'survey_date':survey.isoformat(),
                                  'temporal_start':start,'temporal_end':end,
                                  'bounding_box_wgs84':[west,south,east,north],
                                  'granules_found':len(results)})
        print(f'{site_key}: ATL24 granules found={len(results)} for {start} to {end}',flush=True)
        all_results.extend((site_key,g) for g in results)

    if not all_results:
        manifest['downloaded_files']=[]
        (out_root/'atl24_search_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
        print('No ATL24 granules intersect the configured windows. The core SDB workflow can continue without ATL24.')
        return

    print('Earthdata Login is required to download the selected NASA files. The login prompt appears in this terminal.')
    earthaccess.login()
    downloaded=[]
    for site_key,granule in all_results:
        site_dir=out_root/site_key
        site_dir.mkdir(parents=True,exist_ok=True)
        paths=earthaccess.download([granule],local_path=str(site_dir))
        downloaded.extend(str(Path(p).resolve().relative_to(ROOT)) for p in paths)
    manifest['downloaded_files']=downloaded
    (out_root/'atl24_search_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(f'ATL24 download attempt complete. Files reported by earthaccess: {len(downloaded)}')
    print('Before using ATL24, inspect granule quality, overpass time, spatial track intersection, and vertical reference.')


if __name__=='__main__': main()
