"""Predict uncertainty-masked sky/road labels in a separate linked scene."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from nexus_3dgs.segformer_inference import SegformerInference


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,required=True);p.add_argument('--vehicle-masks',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--protect-lidar',action='store_true',help='Keep real projected LiDAR hits out of sky-only supervision')
    a=p.parse_args();cv2.setNumThreads(4);out=a.output_root/a.scene.name;out.mkdir(parents=True,exist_ok=False)
    for name in ('images','lidar','ego_pose','camera_ego_pose','intrinsics','extrinsics','instances'):
        (out/name).symlink_to((a.scene/name).resolve(),target_is_directory=True)
    (out/'vehicle_interior_masks').symlink_to(a.vehicle_masks.resolve(),target_is_directory=True)
    shutil.copyfile(a.scene/'nexus_scene.json',out/'nexus_scene.json')
    masks=out/'scene_semantics';masks.mkdir();preview=out/'semantic_previews';preview.mkdir()
    scene=json.loads((out/'nexus_scene.json').read_text());model=SegformerInference(a.model)
    report={'status':'predicting','is_semantic_ground_truth':False,'model_source':json.loads((a.model/'source.json').read_text()),
            'lens':'undistorted_pinhole','resolution_wh':[960,540],'sky_confidence':.85,'other_confidence':.75,
            'boundary_erosion_pixels':4,'lidar_hit_protection_dilation_px':3 if a.protect_lidar else None,'frames':[]}
    for frame in scene['frames']:
        f=frame['frame_idx']
        if a.protect_lidar:
            raw=np.fromfile(out/'lidar'/f'{f:03d}.bin',np.float32).reshape(-1,6)[:,:3]
            ego=np.asarray(frame['ego_to_world']);world=raw@ego[:3,:3].T+ego[:3,3]
        for camera in frame['images']:
            c=camera['camera_id'];intr=np.loadtxt(out/'intrinsics'/f'{c}.txt')
            K=np.array([[intr[0]/2,0,intr[2]/2],[0,intr[1]/2,intr[3]/2],[0,0,1.]])
            image=np.array(Image.open(out/'images'/f'{f:03d}_{c}.jpg').convert('RGB').resize((960,540),Image.Resampling.BILINEAR))
            image=cv2.undistort(image,K,intr[4:]);labels,confidence,sky_probability=model(image)
            kernel=np.ones((9,9),np.uint8)
            sky=cv2.erode(((labels==2)&(confidence>=.85)).astype(np.uint8),kernel)
            if a.protect_lidar:
                C=np.asarray(camera['ego_to_world'])@np.loadtxt(out/'extrinsics'/f'{c}.txt')
                xyz=(world-C[:3,3])@C[:3,:3];z=xyz[:,2]
                uv=xyz[:,:2]/np.maximum(z[:,None],.1)@K[:2,:2].T+K[:2,2]
                valid=(z>.1)&(uv[:,0]>=0)&(uv[:,0]<960)&(uv[:,1]>=0)&(uv[:,1]<540)
                pix=uv[valid].astype(int);hits=np.zeros((540,960),np.uint8);hits[pix[:,1],pix[:,0]]=1
                sky[cv2.dilate(hits,np.ones((7,7),np.uint8))>0]=0
            occupied=cv2.erode(((labels!=2)&(confidence>=.75)).astype(np.uint8),kernel)
            road=cv2.erode(((labels==6)&(confidence>=.8)).astype(np.uint8),kernel)
            structure=cv2.erode((np.isin(labels,[0,1,6,11,25,48])&(confidence>=.8)).astype(np.uint8),kernel)
            np.savez_compressed(masks/f'{f:03d}_{c}.npz',sky=sky,occupied=occupied,road=road,structure=structure,
                                labels=labels,confidence=np.rint(confidence*255).astype(np.uint8))
            report['frames'].append({'frame':f,'camera_id':c,'sky_pixels':int(sky.sum()),'road_pixels':int(road.sum()),
                                     'structure_pixels':int(structure.sum())})
            if f in (20,60,99):
                overlay=image.copy();overlay[sky>0]=(image[sky>0]*.55+np.array([60,150,255])*.45).astype(np.uint8)
                overlay[road>0]=(image[road>0]*.55+np.array([210,80,230])*.45).astype(np.uint8)
                Image.fromarray(overlay).save(preview/f'{f:03d}_{c}.png')
        if f%10==0:print('SEMANTIC_FRAME '+str(f),flush=True)
    report['status']='complete';(masks/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print('SEMANTICS_OK '+str(out.resolve()),flush=True)


if __name__=='__main__':main()
