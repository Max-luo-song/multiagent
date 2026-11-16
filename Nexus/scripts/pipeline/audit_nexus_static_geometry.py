"""Compare static geometry at identical generated cameras against measured LiDAR.

Sparse depth checks are geometry diagnostics, not novel-view RGB ground truth.
Report common opaque coverage as well as errors so deleting geometry cannot
masquerade as an improvement.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter


def stats(values):
    if not len(values):
        return {'count':0}
    return {'count':len(values),'mean_m':float(np.mean(values)),
            'median_m':float(np.median(values)),'p95_m':float(np.quantile(values,.95))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('old','new','cloud','ground','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--allow-asset-calibration-delta',action='store_true',help='Independent vehicle assets, same Nexus prediction and actual cameras')
    a=p.parse_args();cv2.setNumThreads(4)
    runs=[json.loads((d/'run.json').read_text()) for d in (a.old,a.new)]
    if runs[0]['bundle_sha256']!=runs[1]['bundle_sha256']:raise ValueError('Prediction bundles differ')
    def records(r):
        return {f['frame_idx']:i for f in r['frames'] for i in f['images'] if i['camera_id']==0}
    old,new=map(records,runs)
    if old.keys()!=new.keys():raise ValueError('Compared frame sets differ')
    ground=json.loads(a.ground.read_text());W=np.array(ground['world_from_scene'])
    with np.load(a.cloud) as cloud:
        points=cloud['points'].astype(float);normals=cloud['normals'];planar=cloud['planar']
    x,y,z=ground['coefficients_z_ax_by_c']
    is_road=(np.abs(normals[:,2])>.95)&planar&(np.abs(points[:,2]-x*points[:,0]-y*points[:,1]-z)<.2)
    errors={region:{k:[] for k in ('old','new')} for region in ('all_static','road','other_static')}
    rows=[];actors_identical=True
    for f in sorted(old):
        m0,m1=[r[f]['scene_metadata'] for r in (old,new)]
        for key in ('camera_world','intrinsics'):
            np.testing.assert_allclose(m0[key],m1[key],atol=1e-8,rtol=0)
        physical_keys=('model_index','agent_index','nexus_slot','track_token','valid','center_world','yaw_world_rad','size_lwh')
        physical=lambda m:[{k:actor[k] for k in physical_keys} for actor in m['actors']]
        same=physical(m0)==physical(m1);actors_identical &= same
        if not a.allow_asset_calibration_delta:assert same, 'Actual vehicle poses differ'
        C=np.linalg.inv(W)@np.asarray(m0['camera_world']);K=np.array(m0['intrinsics'])
        width,height=m0['width'],m0['height']
        xyz=(points-C[:3,3])@C[:3,:3];depth=xyz[:,2]
        uv=xyz[:,:2]/np.maximum(depth[:,None],.1)@K[:2,:2].T+K[:2,2]
        ix,iy=np.rint(uv).astype(int).T
        valid=(depth>1)&(depth<150)&(ix>=0)&(ix<width)&(iy>=0)&(iy<height)
        ids=np.flatnonzero(valid);pixels=iy[ids]*width+ix[ids]
        order=np.argsort(depth[ids],kind='stable');_,first=np.unique(pixels[order],return_index=True)
        ids=ids[order[first]]
        measured=np.full((height,width),np.inf);road=np.zeros((height,width),bool)
        measured[iy[ids],ix[ids]]=depth[ids];road[iy[ids],ix[ids]]=is_road[ids]
        low=minimum_filter(measured,size=5)
        high=maximum_filter(np.where(np.isfinite(measured),measured,-np.inf),size=5)
        stable=np.isfinite(measured)&((high-low)<np.maximum(.6,.04*measured))
        layers=[]
        for directory in (a.old,a.new):
            with np.load(directory/'geometry'/f'{f:03d}_0.npz') as g:
                layers.append({k:g[k] for k in g.files})
        common=stable.copy();coverage={}
        for label,g in zip(('old','new'),layers):
            visible=stable&(g['opacity']>.9)&(g['dynamic_opacity']<.05)&np.isfinite(g['depth'])
            coverage[label]=int(visible.sum());common &= visible
        detail={'frame':f,'measured_stable_pixels':int(stable.sum()),'opaque_static_pixels':coverage,'common_pixels':int(common.sum()),'regions':{}}
        for region,region_mask in [('all_static',np.ones_like(common)),('road',road),('other_static',~road)]:
            mask=common&region_mask
            detail['regions'][region]={}
            for label,g in zip(('old','new'),layers):
                error=np.abs(g['depth'][mask]-measured[mask])
                errors[region][label].append(error.astype(np.float32))
                detail['regions'][region][label]=stats(error)
        rows.append(detail)
    summary={r:{k:stats(np.concatenate(v)) for k,v in by_label.items()} for r,by_label in errors.items()}
    result={'status':'complete','frames':len(rows),'actual_cameras_identical':True,'actual_actor_poses_identical':bool(actors_identical),'prediction_bundles_identical':True,
            'scope':'static real-LiDAR sparse depth reprojection, common opaque pixels, edges and vehicle opacity excluded; not novel-view RGB ground truth',
            'summary':summary,'frames_detail':rows}
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
