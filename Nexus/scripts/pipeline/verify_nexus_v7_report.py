"""Verify exported experiment links, asset hashes, and every video frame."""
import argparse,hashlib,json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote,urlsplit


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True);p.add_argument('--version',choices=['v7','v8','v9'],default='v7');a=p.parse_args();root=a.report.resolve();stem='nexus_3dgs_'+a.version
    import imageio.v2 as imageio
    import numpy as np
    manifest=json.loads((root/(stem+'_manifest.json')).read_text());assert manifest['status']=='complete'
    for item in manifest['assets']:
        path=root/item['file'];assert path.stat().st_size==item['bytes'];assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
    links=[]
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for key,value in attrs:
                if key in ['src','href','poster'] and value:links.append(value)
    parser=Links();parser.feed((root/(stem+'.html')).read_text())
    for value in links:
        url=urlsplit(value)
        if not url.scheme and url.path:assert (root/unquote(url.path)).is_file(),value
    videos=[]
    for item in manifest['assets']:
        path=root/item['file']
        if path.suffix!='.mp4':continue
        with imageio.get_reader(path,format='FFMPEG') as reader:
            meta=reader.get_meta_data();assert abs(meta['fps']-10)<1e-6
            count=0;first=None;last=None
            for frame in reader:
                if first is None:first=frame.copy()
                last=frame;count+=1
            assert count==80,(path,count)
            delta=float(np.abs(last.astype(float)-first).mean());assert delta>1.,(path,delta)
        videos.append({'file':str(path.relative_to(root)),'fully_decoded_frames':count,'fps':meta['fps'],'first_last_mean_abs_pixel_difference':delta})
    if a.version=='v8':assert len(videos)==5
    elif a.version=='v9':assert len(videos)==4
    else:assert len(videos)>=1
    result={'status':'passed','links_checked':len(links),'asset_hashes_checked':len(manifest['assets']),'videos':videos}
    (root/(stem+'_delivery_checks.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
