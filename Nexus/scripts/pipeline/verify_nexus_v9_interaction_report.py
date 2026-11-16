"""Verify local links, all asset hashes and every frame of delivered videos."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    stem='nexus_3dgs_interaction_v9';manifest=json.loads((a.report/(stem+'_manifest.json')).read_text());assets=a.report/'assets'/stem
    page=(a.report/(stem+'.html')).read_text();links=sorted(set(re.findall(r'(?:src|href)="([^"]+)"',page)))
    for link in links:assert (a.report/link).is_file(),link
    for name,digest in manifest['assets'].items():assert hashlib.sha256((assets/name).read_bytes()).hexdigest()==digest,name
    import imageio.v2 as imageio
    videos={}
    for path in sorted(assets.glob('*.mp4')):
        with imageio.get_reader(path) as reader:
            meta=reader.get_meta_data();count=sum(1 for _ in reader)
        assert count==80 and meta['fps']==10,path
        videos[path.name]={'decoded_frames':count,'fps':meta['fps']}
    result={'status':'passed','links_verified':len(links),'asset_hashes_verified':len(manifest['assets']),'videos':videos}
    (a.report/(stem+'_delivery_checks.json')).write_text(json.dumps(result,indent=2)+'\n')
    print('INTERACTION_DELIVERY_VERIFIED',len(links),'links',len(manifest['assets']),'hashes',len(videos),'videos',flush=True)


if __name__=='__main__':main()
