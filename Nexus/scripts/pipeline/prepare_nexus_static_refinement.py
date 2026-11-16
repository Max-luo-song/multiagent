"""Prepare observed traffic exclusion masks and fixed-layout road anchors."""
import argparse,json,sys
from pathlib import Path
import cv2,numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('scene','checkpoint','plane-checkpoint','road-indices','reference','output'):
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(4)
    scene=json.loads((a.scene/'nexus_scene.json').read_text())
    boxes={};info=json.loads((a.scene/'instances/instances_info.json').read_text())
    for item in info.values():
        ann=item['frame_annotations']
        for f,T,size in zip(ann['frame_idx'],ann['obj_to_world'],ann['box_size']):
            boxes.setdefault(f,[]).append((np.asarray(T),np.asarray(size)))
    signs=np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]])
    masks=a.output/'traffic_masks';masks.mkdir();count=0
    for frame in scene['frames']:
        f=frame['frame_idx']
        for camera in frame['images']:
            c=camera['camera_id'];intr=np.loadtxt(a.scene/'intrinsics'/f'{c}.txt')
            K=np.array([[intr[0]/2,0,intr[2]/2],[0,intr[1]/2,intr[3]/2],[0,0,1.]])
            C=np.asarray(camera['ego_to_world'])@np.loadtxt(a.scene/'extrinsics'/f'{c}.txt')
            mask=np.zeros((540,960),np.uint8)
            for B,size in boxes.get(f,[]):
                xyz=((signs*size/2)@B[:3,:3].T+B[:3,3]-C[:3,3])@C[:3,:3]
                # Clip all twelve box edges to the near plane before projection.
                points=[v for v in xyz if v[2]>.1]
                for i in range(8):
                    for j in range(i+1,8):
                        if np.count_nonzero(signs[i]!=signs[j])!=1:continue
                        u,v=xyz[i],xyz[j]
                        if (u[2]>.1)!=(v[2]>.1):points.append(u+(v-u)*((.10001-u[2])/(v[2]-u[2])))
                if not points:continue
                q=np.asarray(points);uv=q[:,:2]/q[:,2,None]@K[:2,:2].T+K[:2,2]
                lo=uv.min(0);hi=uv.max(0)
                if hi[0]<0 or hi[1]<0 or lo[0]>=960 or lo[1]>=540:continue
                height=min(540.,hi[1]-lo[1]);width=min(960.,hi[0]-lo[0])
                # Exclude measured traffic plus a conservative observed shadow halo.
                left=max(0,int(lo[0]-max(4,.08*width)));right=min(960,int(hi[0]+max(4,.08*width)+1))
                top=max(0,int(lo[1]-4));bottom=min(540,int(hi[1]+max(6,.3*height)+1))
                mask[top:bottom,left:right]=255
            cv2.imwrite(str(masks/f'{f:03d}_{c}.png'),mask);count+=1
    (masks/'manifest.json').write_text(json.dumps({'status':'complete','is_semantic_ground_truth':False,'use':'static-training exclusion only; these are conservative projected annotation boxes, not observed vehicle interior RGB supervision','images':count},indent=2)+'\n')
    import torch
    old=torch.load(a.checkpoint,map_location='cpu',weights_only=False)['models']['Background']['_means'].numpy()
    plane=torch.load(a.plane_checkpoint,map_location='cpu',weights_only=False)['models']['Background']['_means'].numpy()
    ids=np.load(a.road_indices);valid=np.zeros(len(old),bool);valid[ids]=True
    anchors=old.copy();anchors[ids]=plane[ids]
    ground=json.loads(a.reference.read_text())['ground_registration']['ground_model']
    slope=np.asarray(ground['coefficients_z_ax_by_c']);n=np.array([-slope[0],-slope[1],1.]);n/=np.linalg.norm(n)
    normals=np.broadcast_to(n,old.shape).astype(np.float32).copy()
    np.savez_compressed(a.output/'anchors.npz',valid=valid,anchors=anchors,normals=normals)
    (a.output/'preparation.json').write_text(json.dumps({'status':'complete','observed_images':count,'supported_road_gaussians':int(valid.sum()),'mask_policy':'GT 3D traffic boxes with near-plane clipping and conservative image shadow halo; all annotations, including future-only actors','source_checkpoint':str(a.checkpoint.resolve())},indent=2)+'\n')
    print('STATIC_PREPARATION_OK '+str(a.output.resolve()),flush=True)


if __name__=='__main__':main()
