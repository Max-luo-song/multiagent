"""Refine real-LiDAR vehicle seeds using image colors inside projected boxes.

Masks are image/3D-box-derived supervision, not semantic ground truth. A new
scene of symlinks preserves all original images, point clouds and annotations.
"""
import argparse
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True);a=p.parse_args();cv2.setNumThreads(4)
    source=a.scene.resolve();out=a.output_root.resolve()/source.name
    out.mkdir(parents=True,exist_ok=False)
    for name in ('images','lidar','ego_pose','camera_ego_pose','intrinsics','extrinsics','instances'):
        (out/name).symlink_to(source/name,target_is_directory=True)
    shutil.copyfile(source/'nexus_scene.json',out/'nexus_scene.json')
    scene=json.loads((source/'nexus_scene.json').read_text());info=json.loads((source/'instances/instances_info.json').read_text())
    objects={}
    for item in info.values():
        if item['class_name']!='vehicle':continue
        ann=item['frame_annotations'];z=float(np.median(np.array(ann['obj_to_world'])[:,2,3]))
        for f,pose,size in zip(ann['frame_idx'],ann['obj_to_world'],ann['box_size']):
            stable=np.array(pose);stable[2,3]=z
            objects.setdefault(int(f),[]).append((item['id'],np.array(pose),stable,np.array(size)))
    masks=out/'vehicle_interior_masks';masks.mkdir();preview=out/'mask_previews';preview.mkdir()
    report={'status':'preparing','scene_identity':scene['identity'],'is_semantic_ground_truth':False,
            'kind':'GrabCut from real visible LiDAR seeds, confined to stable-height projected vehicle boxes; eroded boundary',
            'source_scene':str(source),'resolution_wh':[960,540],'frames':[]}
    (masks/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    corner_sign=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
    for frame in scene['frames']:
        index=frame['frame_idx'];raw=np.fromfile(source/'lidar'/f'{index:03d}.bin',np.float32).reshape(-1,6)[:,:3]
        ego=np.array(frame['ego_to_world']);points=raw@ego[:3,:3].T+ego[:3,3]
        for cam in frame['images']:
            camera=cam['camera_id'];values=np.loadtxt(source/'intrinsics'/f'{camera}.txt');k=values[:4]/2
            K=np.array([[k[0],0,k[2]],[0,k[1],k[3]],[0,0,1.]])
            image=np.array(Image.open(source/'images'/f'{index:03d}_{camera}.jpg').convert('RGB').resize((960,540),Image.Resampling.BILINEAR))
            image=cv2.undistort(image,K,values[4:]);c2w=np.array(cam['ego_to_world'])@np.loadtxt(source/'extrinsics'/f'{camera}.txt')
            c=(points-c2w[:3,3])@c2w[:3,:3];depth=c[:,2]
            uv=c[:,:2]/np.maximum(depth[:,None],1e-8)*k[:2]+k[2:]
            in_view=(depth>0)&np.isfinite(uv).all(-1)&(uv[:,0]>=0)&(uv[:,0]<960)&(uv[:,1]>=0)&(uv[:,1]<540)
            ids=np.flatnonzero(in_view);pixels=uv[ids].astype(int);flat=pixels[:,1]*960+pixels[:,0]
            nearest=np.full(960*540,np.inf);np.minimum.at(nearest,flat,depth[ids]);visible=np.zeros(len(points),bool)
            visible[ids]=depth[ids]<=nearest[flat]+.2
            union=np.zeros((540,960),np.uint8);seeds_union=np.zeros_like(union);actors=[]
            for token,pose,stable,size in objects.get(index,[]):
                local=(points-pose[:3,3])@pose[:3,:3]
                inside=(np.abs(local)<size/2).all(-1)&(local[:,2]>-size[2]/2+.15)
                select=np.flatnonzero(inside&visible)
                if len(select)<12:continue
                corners=(corner_sign*size/2)@stable[:3,:3].T+stable[:3,3]
                pc=(corners-c2w[:3,3])@c2w[:3,:3]
                if pc[:,2].min()<=.1:continue
                bounds=pc[:,:2]/pc[:,2,None]*k[:2]+k[2:]
                polygon=cv2.convexHull(bounds.astype(np.float32)).round().astype(np.int32)
                box=np.zeros_like(union);cv2.fillConvexPoly(box,polygon,1)
                seed=np.zeros_like(union);cv2.fillConvexPoly(seed,cv2.convexHull(uv[select].astype(np.float32)).round().astype(np.int32),1)
                seed=cv2.erode(seed,np.ones((3,3),np.uint8));seed &= box
                if int(seed.sum())<8 or not box.any():continue
                ys,xs=np.nonzero(box);x0=max(int(xs.min())-6,0);x1=min(int(xs.max())+7,960);y0=max(int(ys.min())-6,0);y1=min(int(ys.max())+7,540)
                mask=np.full((y1-y0,x1-x0),cv2.GC_BGD,np.uint8)
                mask[box[y0:y1,x0:x1]>0]=cv2.GC_PR_FGD
                mask[seed[y0:y1,x0:x1]>0]=cv2.GC_FGD
                mask[[0,-1],:]=cv2.GC_BGD;mask[:,[0,-1]]=cv2.GC_BGD
                try:cv2.grabCut(image[y0:y1,x0:x1].copy(),mask,None,np.zeros((1,65)),np.zeros((1,65)),3,cv2.GC_INIT_WITH_MASK)
                except cv2.error:continue
                foreground=((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD)).astype(np.uint8)
                foreground &= box[y0:y1,x0:x1]
                foreground=cv2.erode(foreground,np.ones((3,3),np.uint8))
                union[y0:y1,x0:x1]|=foreground;seeds_union|=seed
                actors.append({'track_token':token,'seed_pixels':int(seed.sum()),'mask_pixels':int(foreground.sum())})
            cv2.imwrite(str(masks/f'{index:03d}_{camera}.png'),union*255)
            report['frames'].append({'frame':index,'camera_id':camera,'pixels':int(union.sum()),'actors':actors})
            if index in (20,60,99):
                overlay=image.copy();overlay[union>0]=(overlay[union>0]*.55+np.array([40,220,60])*.45).astype(np.uint8)
                Image.fromarray(overlay).save(preview/f'{index:03d}_{camera}.png')
    report['status']='complete';(masks/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':'complete','scene':str(out),'masks':len(report['frames']),'pixels':sum(f['pixels'] for f in report['frames'])}),flush=True)


if __name__=='__main__':main()
