"""Build local non-vehicle rings, retaining uncertain edges and unseeded cars."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',type=Path,required=True)
    a=p.parse_args();cv2.setNumThreads(4)
    root=a.scene/'vehicle_exterior_masks';root.mkdir(exist_ok=False)
    scene=json.loads((a.scene/'nexus_scene.json').read_text())
    interior=json.loads((a.scene/'vehicle_interior_masks/manifest.json').read_text())
    known={(r['frame'],r['camera_id']):{x['track_token'] for x in r['actors']} for r in interior['frames']}
    objects={}
    for item in json.loads((a.scene/'instances/instances_info.json').read_text()).values():
        if item['class_name']!='vehicle':continue
        ann=item['frame_annotations'];z=np.median(np.asarray(ann['obj_to_world'])[:,2,3])
        for index,pose,size in zip(ann['frame_idx'],ann['obj_to_world'],ann['box_size']):
            pose=np.array(pose);pose[2,3]=z
            objects.setdefault(index,[]).append((item['id'],pose,np.asarray(size)))
    signs=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
    report={'status':'preparing','is_semantic_ground_truth':False,'source':'real-LiDAR-seeded GrabCut vehicle masks',
            'uncertain_edge_px':5,'local_outer_radius_px':40,'unseeded_vehicle_boxes_protected':True,'frames':[]}
    for frame in scene['frames']:
        i=frame['frame_idx']
        for cam in frame['images']:
            k=cam['camera_id'];mask=cv2.imread(str(a.scene/'vehicle_interior_masks'/f'{i:03d}_{k}.png'),0)>0
            inner=cv2.dilate(mask.astype(np.uint8),np.ones((11,11),np.uint8))
            outer=cv2.dilate(mask.astype(np.uint8),np.ones((81,81),np.uint8))
            negative=outer&~inner
            intr=np.loadtxt(a.scene/'intrinsics'/f'{k}.txt')[:4]/2
            camera=np.array(cam['ego_to_world'])@np.loadtxt(a.scene/'extrinsics'/f'{k}.txt')
            for token,pose,size in objects.get(i,[]):
                if token in known[i,k]:continue
                points=(signs*size/2)@pose[:3,:3].T+pose[:3,3]
                local=(points-camera[:3,3])@camera[:3,:3]
                if local[:,2].min()<=.1:continue
                uv=local[:,:2]/local[:,2,None]*intr[:2]+intr[2:]
                protect=np.zeros_like(negative)
                cv2.fillConvexPoly(protect,cv2.convexHull(uv.astype(np.float32)).round().astype(np.int32),1)
                protect=cv2.dilate(protect,np.ones((7,7),np.uint8))
                negative[protect>0]=0
            cv2.imwrite(str(root/f'{i:03d}_{k}.png'),negative*255)
            report['frames'].append({'frame':i,'camera_id':k,'pixels':int(negative.sum())})
            if i==60 and k==0:
                image=np.array(Image.open(a.scene/'mask_previews/060_0.png'))
                image[negative>0]=(image[negative>0]*.6+np.array([235,65,65])*.4).astype(np.uint8)
                Image.fromarray(image).save(root/'preview_060_0.png')
    report['status']='complete';(root/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print('EXTERIOR_MASKS_OK '+str(len(report['frames'])),flush=True)


if __name__=='__main__':main()
