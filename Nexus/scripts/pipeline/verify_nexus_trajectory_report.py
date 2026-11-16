"""Check local report links, hashes and all six decoded comparison videos."""
import argparse,hashlib,json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote,urlsplit
import imageio.v2 as imageio


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True);a=p.parse_args();root=a.report.resolve()
    manifest=json.loads((root/'nexus_real_trajectory_manifest.json').read_text());assert manifest['status']=='complete'
    for item in manifest['assets']:
        path=root/item['file'];assert path.stat().st_size==item['bytes'];assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
    links=[]
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for key,value in attrs:
                if key in ('src','href','poster') and value:links.append(value)
    parser=Links();parser.feed((root/'nexus_real_trajectory_comparison.html').read_text())
    for value in links:
        url=urlsplit(value)
        if not url.scheme and url.path:assert (root/unquote(url.path)).is_file(),value
    videos=[]
    for item in manifest['assets']:
        path=root/item['file']
        if path.suffix!='.mp4':continue
        assert path.read_bytes()[4:8]==b'ftyp',path
        with imageio.get_reader(path,format='FFMPEG') as reader:
            meta=reader.get_meta_data();assert abs(meta['fps']-10)<1e-6
            count=sum(1 for _ in reader);assert count==80,(path,count)
        videos.append({'file':str(path.relative_to(root)),'fully_decoded_frames':count,'fps':meta['fps']})
    assert len(videos)==6
    result={'status':'passed','checked_links':len(links),'checked_asset_hashes':len(manifest['assets']),'fully_decoded_videos':videos}
    (root/'nexus_real_trajectory_delivery_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
