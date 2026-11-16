"""Verify exported report links, hashes, and fully decoded video frame counts."""
import argparse,hashlib,json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote,urlsplit
import imageio.v2 as imageio


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    root=a.report.resolve();manifest=json.loads((root/'nexus_3dgs_motion_v6_manifest.json').read_text())
    assert manifest['result']['full_scene_motion_ghosting_solved'] is False
    for item in manifest['assets']:
        path=root/item['file'];assert path.stat().st_size==item['bytes'];assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
    links=[]
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for key,value in attrs:
                if key in ('src','href','poster') and value:links.append(value)
    parser=Links();parser.feed((root/'nexus_3dgs_motion_v6.html').read_text())
    for value in links:
        parts=urlsplit(value)
        if not parts.scheme and parts.path:assert (root/unquote(parts.path)).is_file(),value
    result={'status':'passed','report_links_checked':len(links),'asset_hashes_checked':len(manifest['assets']),'videos':[]}
    for name,frames in [('comparison.mp4',160),('candidate_go.mp4',80),('candidate_stop.mp4',80)]:
        path=root/'assets/nexus_3dgs_motion_v6'/name
        assert path.read_bytes()[4:8]==b'ftyp',name
        with imageio.get_reader(path,format='FFMPEG') as reader:
            metadata=reader.get_meta_data();assert abs(metadata['fps']-10)<1e-6
            count=sum(1 for _ in reader);assert count==frames,(name,count,frames)
        result['videos'].append({'file':name,'decoded_frames':count,'fps':metadata['fps']})
    (root/'nexus_3dgs_motion_v6_delivery_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
