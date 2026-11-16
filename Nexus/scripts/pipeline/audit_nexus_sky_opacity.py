"""Check finite background opacity inside rotation-warped confident sky cores."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('old','new','reference','scene','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();cv2.setNumThreads(4)
    def records(directory):
        r=json.loads((directory/'run.json').read_text())
        return {f['frame_idx']:i for f in r['frames'] for i in f['images'] if i['camera_id']==0}
    old,new,ref=map(records,(a.old,a.new,a.reference));values={'old':[],'new':[]};rows=[]
    for f in sorted(old):
        source=ref[f]['scene_metadata'];target=old[f]['scene_metadata']
        K0,K1=np.array(source['intrinsics']),np.array(target['intrinsics'])
        R0,R1=np.array(source['camera_world'])[:3,:3],np.array(target['camera_world'])[:3,:3]
        with np.load(a.scene/'scene_semantics'/f'{f:03d}_0.npz') as masks:mask=masks['sky'].astype(np.uint8)
        mask=cv2.erode(mask,np.ones((31,31),np.uint8))
        H=K1@R1.T@R0@np.linalg.inv(K0)
        mask=cv2.warpPerspective(mask,H,(target['width'],target['height']),flags=cv2.INTER_NEAREST)>0
        row={'frame':f,'sky_core_pixels':int(mask.sum())}
        for label,directory in [('old',a.old),('new',a.new)]:
            np.testing.assert_allclose(old[f]['scene_metadata']['camera_world'],new[f]['scene_metadata']['camera_world'],atol=1e-8,rtol=0)
            with np.load(directory/'geometry'/f'{f:03d}_0.npz') as g:opacity=g['opacity'][mask].astype(np.float32)
            values[label].append(opacity);row[label]={'mean_background_alpha':float(opacity.mean()),'alpha_over_005_fraction':float((opacity>.05).mean())}
        rows.append(row)
    summary={label:{'mean_background_alpha':float(np.concatenate(items).mean()),'alpha_over_005_fraction':float((np.concatenate(items)>.05).mean())} for label,items in values.items()}
    result={'status':'complete','scope':'predicted confident sky eroded 15 pixels, rotation-only infinity warp; semantic and occlusion uncertainty remains; not sky ground truth',
            'summary':summary,'frames_detail':rows}
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
