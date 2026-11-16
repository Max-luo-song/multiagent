"""Derive conservative vehicle-interior supervision from visible real LiDAR.

These are eroded projected point hulls, not semantic ground-truth silhouettes.
Only vehicles with at least 12 visible box-assigned returns supply supervision.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',type=Path,required=True)
    a=p.parse_args();cv2.setNumThreads(4)
    scene=json.loads((a.scene/'nexus_scene.json').read_text())
    info=json.loads((a.scene/'instances/instances_info.json').read_text())
    objects={}
    for item in info.values():
        if item['class_name']!='vehicle':continue
        ann=item['frame_annotations']
        for f,pose,size in zip(ann['frame_idx'],ann['obj_to_world'],ann['box_size']):
            objects.setdefault(int(f),[]).append((np.array(pose),np.array(size)))
    out=a.scene/'vehicle_interior_masks';out.mkdir(exist_ok=True)
    report={'status':'preparing','scene_identity':scene['identity'],
            'kind':'visible real LiDAR assigned to annotated 3D vehicle boxes; convex hull eroded 2 pixels',
            'is_semantic_ground_truth':False,'min_points':12,'resolution_wh':[960,540],'frames':[]}
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    for frame in scene['frames']:
        index=frame['frame_idx'];raw=np.fromfile(a.scene/'lidar'/f'{index:03d}.bin',np.float32).reshape(-1,6)[:,:3]
        ego=np.array(frame['ego_to_world']);points=raw@ego[:3,:3].T+ego[:3,3]
        assignments=[]
        for pose,size in objects.get(index,[]):
            local=(points-pose[:3,3])@pose[:3,:3]
            assignments.append((np.abs(local)<size/2).all(axis=-1))
        for camera in frame['images']:
            cam_id=camera['camera_id'];ext=np.loadtxt(a.scene/'extrinsics'/f'{cam_id}.txt')
            c2w=np.array(camera['ego_to_world'])@ext
            c=(points-c2w[:3,3])@c2w[:3,:3]
            k=np.loadtxt(a.scene/'intrinsics'/f'{cam_id}.txt')[:4]/2
            z=c[:,2];uv=c[:,:2]/np.maximum(z[:,None],1e-8)*k[:2]+k[2:]
            in_view=(z>0)&np.isfinite(uv).all(axis=-1)&(uv[:,0]>=0)&(uv[:,0]<960)&(uv[:,1]>=0)&(uv[:,1]<540)
            ids=np.flatnonzero(in_view);pixels=uv[ids].astype(int)
            nearest=np.full(960*540,np.inf);flat=pixels[:,1]*960+pixels[:,0]
            np.minimum.at(nearest,flat,z[ids]);visible=np.zeros(len(points),bool)
            visible[ids]=z[ids]<=nearest[flat]+.2
            mask=np.zeros((540,960),np.uint8);hulls=0
            for assignment in assignments:
                select=np.flatnonzero(assignment&visible)
                if len(select)<12:continue
                hull=cv2.convexHull(uv[select].astype(np.float32)).round().astype(np.int32)
                if cv2.contourArea(hull)<12:continue
                part=np.zeros_like(mask);cv2.fillConvexPoly(part,hull,255)
                part=cv2.erode(part,np.ones((5,5),np.uint8))
                mask|=part;hulls+=1
            cv2.imwrite(str(out/f'{index:03d}_{cam_id}.png'),mask)
            report['frames'].append({'frame':index,'camera_id':cam_id,'hulls':hulls,'pixels':int((mask>0).sum())})
    report['status']='complete';(out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':'complete','masks':len(report['frames']),'observed_pixels':sum(f['pixels'] for f in report['frames'])}))


if __name__=='__main__':main()
