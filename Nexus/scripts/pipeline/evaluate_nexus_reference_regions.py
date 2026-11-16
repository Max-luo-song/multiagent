"""Compare observed-camera reconstructions on fixed full-image/vehicle ROIs.

Requires reference rendering without ground-camera substitution. These are
training views, not ground truth for generated trajectories.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old',type=Path,required=True);p.add_argument('--new',type=Path,required=True)
    p.add_argument('--scene',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frames',type=int,nargs='+',help='Compare an explicitly reported diagnostic subset')
    a=p.parse_args();cv2.setNumThreads(4)
    runs=[json.loads((d/'run.json').read_text()) for d in (a.old,a.new)]
    for r in runs:
        if r['status']!='complete' or not r['reference'] or r['camera_mode']!='logged' or r['ground_registration'] is not None:
            raise ValueError('Expected logged reference render without substituted camera registration')
    def records(r):return {(f['frame_idx'],i['timestamp_us']):i['file'] for f in r['frames'] for i in f['images'] if i['camera_id']==0}
    left,right=map(records,runs)
    if a.frames:
        left={k:v for k,v in left.items() if k[0] in a.frames};right={k:v for k,v in right.items() if k[0] in a.frames}
        if {k[0] for k in left}!=set(a.frames):raise ValueError('Requested frames absent from baseline')
    if left.keys()!=right.keys():raise ValueError('Compared timestamps differ')
    rois={'left_van':[225,235,310,300],'right_vehicles':[720,225,960,320]}
    intr=np.loadtxt(a.scene/'intrinsics/0.txt');rows=[]
    for key in sorted(left):
        index,_=key;images=[np.array(Image.open(d/items[key]).convert('RGB')) for d,items in [(a.old,left),(a.new,right)]]
        h,w=images[0].shape[:2]
        source=Image.open(a.scene/'images'/f'{index:03d}_0.jpg').convert('RGB')
        sx,sy=w/source.width,h/source.height
        K=np.array([[intr[0]*sx,0,intr[2]*sx],[0,intr[1]*sy,intr[3]*sy],[0,0,1.]])
        gt=cv2.undistort(np.array(source.resize((w,h),Image.Resampling.BILINEAR)),K,intr[4:]).astype(np.float32)/255
        for region,(x0,y0,x1,y1) in {'full':[0,0,w,h],**rois}.items():
            target=gt[y0:y1,x0:x1]
            row={'frame':index,'region':region}
            for label,image in zip(('old','new'),images):
                patch=image[y0:y1,x0:x1].astype(np.float32)/255
                row[label]={'psnr':float(-10*np.log10(max(float(np.mean((patch-target)**2)),1e-12))),
                            'ssim':float(structural_similarity(target,patch,channel_axis=-1,data_range=1.))}
            rows.append(row)
    summary={region:{label:{metric:float(np.mean([r[label][metric] for r in rows if r['region']==region]))
              for metric in ('psnr','ssim')} for label in ('old','new')} for region in ('full',*rois)}
    result={'status':'complete','frames':len(left),'scope':'same logged training camera reconstruction, fixed image-space ROIs; not generated-view ground truth',
            'roi_xyxy':rois,'summary':summary,'frames_detail':rows}
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
