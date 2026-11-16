"""Prepare real RGB/semantic/occlusion observations without changing old scenes."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'Omniscenegraph')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['scene','reference','output','model']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--frames',type=int,nargs='+',default=[20,30,40,50,60,70,80,90,99])
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    import cv2,numpy as np,torch
    from PIL import Image
    from nexus_3dgs.segformer_inference import SegformerInference
    cv2.setNumThreads(4);torch.set_num_threads(8);model=SegformerInference(a.model)
    meta=json.loads((a.scene/'nexus_scene.json').read_text());S=np.linalg.inv(meta['frames'][0]['ego_to_world']);info=json.loads((a.scene/'instances/instances_info.json').read_text())
    records={r['frame_idx']:r['images'][0]['scene_metadata'] for r in json.loads(a.reference.read_text())['frames']}
    boxes={}
    for item in info.values():
        ann=item['frame_annotations']
        for f,B,size in zip(ann['frame_idx'],ann['obj_to_world'],ann['box_size']):boxes.setdefault(f,[]).append((np.array(B),np.array(size)))
    intr=np.loadtxt(a.scene/'intrinsics/0.txt');sign=np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]])
    report={'status':'complete','source_scene':str(a.scene.resolve()),'reference':str(a.reference.resolve()),'masks_are_predictions_not_ground_truth':True,'frames':[]}
    for f in a.frames:
        C=S@np.asarray(records[f]['camera_world']);K=np.array(records[f]['intrinsics']);W,H=records[f]['width'],records[f]['height']
        image=cv2.undistort(np.array(Image.open(a.scene/'images'/f'{f:03d}_0.jpg').convert('RGB').resize((W,H),Image.Resampling.BILINEAR)),K,intr[4:]);labels,conf,_=model(image)
        vehicle=np.zeros((H,W),np.uint8)
        for B,size in boxes.get(f,[]):
            P=(sign*size/2)@B[:3,:3].T+B[:3,3]
            P=(P-(np.array(records[f]['camera_world'])[:3,3]))@np.array(records[f]['camera_world'])[:3,:3]
            if P[:,2].min()<=.1:continue
            uv=P[:,:2]/P[:,2,None]@K[:2,:2].T+K[:2,2];poly=cv2.convexHull(np.rint(uv).astype(np.int32));cv2.fillConvexPoly(vehicle,poly,1)
        vehicle=cv2.dilate(vehicle,np.ones((9,9),np.uint8))>0
        sky=cv2.erode(((labels==2)&(conf>.97)).astype(np.uint8),np.ones((15,15),np.uint8))>0
        road=cv2.erode(((labels==6)&(conf>.85)).astype(np.uint8),np.ones((7,7),np.uint8))>0;road &= ~vehicle
        raw=np.fromfile(a.scene/'lidar'/f'{f:03d}.bin',np.float32).reshape(-1,6)[:,:3];E=np.asarray(meta['frames'][f]['ego_to_world']);world=raw@E[:3,:3].T+E[:3,3];local=(world-np.array(records[f]['camera_world'])[:3,3])@np.array(records[f]['camera_world'])[:3,:3];z=local[:,2];uv=local[:,:2]/np.maximum(z[:,None],.1)@K[:2,:2].T+K[:2,2];ix,iy=np.rint(uv).astype(int).T;valid=(z>.1)&(ix>=0)&(ix<W)&(iy>=0)&(iy<H);hits=np.zeros((H,W),np.uint8);hits[iy[valid],ix[valid]]=1;sky &= cv2.dilate(hits,np.ones((7,7),np.uint8))==0
        Image.fromarray(image).save(a.output/f'{f:03d}.png');np.savez_compressed(a.output/f'{f:03d}.npz',road=road,sky=sky,vehicle=vehicle,labels=labels,confidence=conf,C=C,K=K)
        report['frames'].append({'frame':f,'road_pixels':int(road.sum()),'sky_pixels':int(sky.sum())});print('SURFACE_OBSERVATION '+str(f),flush=True)
    (a.output/'observations.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
