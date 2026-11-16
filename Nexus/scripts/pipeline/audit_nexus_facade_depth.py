"""Check facade depth against sparse three-view support in logged cameras.

Low-parallax triangulation has substantial uncertainty; this detects large
depth disagreement and does not establish metric facade ground truth.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,nargs='+',required=True)
    for key in ('scene','support','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--frame',type=int,default=99);a=p.parse_args();cv2.setNumThreads(4)
    first=json.loads((a.scene/'nexus_scene.json').read_text())['frames'][0]['ego_to_world'];S=np.linalg.inv(first)
    points=np.load(a.support)['points'];summary={}
    with np.load(a.scene/'scene_semantics'/f'{a.frame:03d}_0.npz') as masks:
        building=cv2.erode((np.isin(masks['labels'],[1,25,48])&(masks['confidence']>=204)).astype(np.uint8),np.ones((5,5),np.uint8))>0
    for directory in a.reference:
        run=json.loads((directory/'run.json').read_text())
        if not run['reference'] or run['camera_mode']!='logged' or run['ground_registration'] is not None:
            raise ValueError('Expected actual logged reference camera')
        records=[i for f in run['frames'] if f['frame_idx']==a.frame for i in f['images'] if i['camera_id']==0]
        m=records[0]['scene_metadata'];C=S@np.asarray(m['camera_world']);K=np.array(m['intrinsics'])
        xyz=(points-C[:3,3])@C[:3,:3];uv=np.rint(xyz[:,:2]/np.maximum(xyz[:,2,None],.1)@K[:2,:2].T+K[:2,2]).astype(int)
        valid=(xyz[:,2]>60)&(uv[:,0]>=0)&(uv[:,0]<960)&(uv[:,1]>=0)&(uv[:,1]<540)
        ids=np.flatnonzero(valid);u=uv[ids];ids=ids[building[u[:,1],u[:,0]]];u=uv[ids]
        with np.load(directory/'geometry'/f'{a.frame:03d}_0.npz') as g:
            keep=(g['dynamic_opacity'][u[:,1],u[:,0]]<.05)&(g['opacity'][u[:,1],u[:,0]]>.9)
            ids=ids[keep];u=uv[ids];depth=g['depth'][u[:,1],u[:,0]]
        target=xyz[ids,2];error=abs(depth-target)
        summary[str(directory.resolve())]={'count':len(ids),'sparse_support_depth_median_m':float(np.median(target)),
            'rendered_depth_median_m':float(np.median(depth)),'depth_ratio_median':float(np.median(depth/target)),
            'absolute_depth_disagreement_median_m':float(np.median(error)),'absolute_depth_disagreement_p90_m':float(np.quantile(error,.9))}
    report={'status':'complete','frame':a.frame,'scope':'actual logged camera, confident building core, sparse three-view support, vehicle opacity excluded; triangulation is uncertain and is not ground truth','summary':summary}
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
