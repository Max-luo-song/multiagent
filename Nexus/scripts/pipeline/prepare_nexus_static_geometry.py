"""Anchor static Gaussian geometry to real LiDAR and separate observed sky."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'Omniscenegraph')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(4)
    scene=json.loads((a.scene/'nexus_scene.json').read_text());info=json.loads((a.scene/'instances/instances_info.json').read_text())
    by_frame={}
    for item in info.values():
        ann=item['frame_annotations']
        for f,pose,size in zip(ann['frame_idx'],ann['obj_to_world'],ann['box_size']):
            by_frame.setdefault(f,[]).append((np.array(pose),np.asarray(size)+.3))
    first=np.asarray(scene['frames'][0]['ego_to_world']);scene_from_world=np.linalg.inv(first)
    support=[];protected=0
    for frame in scene['frames']:
        f=frame['frame_idx'];raw=np.fromfile(a.scene/'lidar'/f'{f:03d}.bin',np.float32).reshape(-1,6)[:,:3]
        ego=np.asarray(frame['ego_to_world']);world=raw@ego[:3,:3].T+ego[:3,3]
        # A LiDAR hit is observed geometry, even when segmentation misses a thin pole.
        for camera in frame['images']:
            c=camera['camera_id'];intr=np.loadtxt(a.scene/'intrinsics'/f'{c}.txt')[:4]/2
            C=np.array(camera['ego_to_world'])@np.loadtxt(a.scene/'extrinsics'/f'{c}.txt')
            xyz=(world-C[:3,3])@C[:3,:3];z=xyz[:,2]
            uv=xyz[:,:2]/np.maximum(z[:,None],.1)*intr[:2]+intr[2:]
            valid=(z>.1)&(uv[:,0]>=0)&(uv[:,0]<960)&(uv[:,1]>=0)&(uv[:,1]<540)
            pix=uv[valid].astype(int);hits=np.zeros((540,960),np.uint8);hits[pix[:,1],pix[:,0]]=1
            hits=cv2.dilate(hits,np.ones((7,7),np.uint8))>0
            path=a.scene/'scene_semantics'/f'{f:03d}_{c}.npz'
            with np.load(path) as arrays:values={k:arrays[k].copy() for k in arrays.files}
            protected+=int(((values['sky']>0)&hits).sum());values['sky'][hits]=0
            np.savez_compressed(path,**values)
        if f%3==0:
            keep=np.ones(len(world),bool)
            for pose,size in by_frame.get(f,[]):
                extent=np.abs(pose[:3,:3])@(size/2)
                candidates=np.flatnonzero(keep&(np.abs(world-pose[:3,3])<=extent).all(-1))
                local=(world[candidates]-pose[:3,3])@pose[:3,:3]
                keep[candidates[(np.abs(local)<=size/2).all(-1)]]=False
            points=(world[keep]-first[:3,3])@first[:3,:3]
            support.append(points.astype(np.float32))
        if f%20==0:print('STATIC_LIDAR_FRAME '+str(f),flush=True)
    points=np.concatenate(support);voxel=np.floor(points/.15).astype(np.int32)
    _,ids=np.unique(voxel,axis=0,return_index=True);points=points[ids]
    tree=cKDTree(points);distance,neighbors=tree.query(points,k=20,workers=8)
    group=points[neighbors];center=group.mean(1);offset=group-center[:,None]
    covariance=np.einsum('nki,nkj->nij',offset,offset)/20
    eigenvalue,eigenvector=np.linalg.eigh(covariance);normals=eigenvector[:,:,0]
    planar=(eigenvalue[:,0]/np.maximum(eigenvalue.sum(-1),1e-9)<.02)&(distance[:,-1]<1.5)
    np.savez_compressed(a.output/'static_surface_cloud.npz',points=points,normals=normals,planar=planar)
    print('STATIC_SURFACES '+str(len(points)),flush=True)
    import torch
    import torch.nn.functional as F
    from models.modules import CameraOptModule
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False);bg=state['models']['Background'];means=bg['_means'].numpy();N=len(means)
    nearest,anchor_id=tree.query(means,k=1,workers=8);anchors=points[anchor_id];normal=normals[anchor_id]
    ids=torch.arange(N,device='cuda');mu=torch.from_numpy(means).to('cuda');obs=torch.zeros(N,dtype=torch.int16,device='cuda')
    sky_vote=torch.zeros_like(obs);road_vote=torch.zeros_like(obs);structure_vote=torch.zeros_like(obs)
    cs=state['models']['CamPose'];cam=CameraOptModule('CamPose',len(cs['embeds.weight']),device=torch.device('cuda')).cuda();cam.load_state_dict(cs)
    cameras=[i['camera_id'] for i in scene['frames'][0]['images']]
    with torch.no_grad():
        for frame in scene['frames'][::5]:
            f=frame['frame_idx']
            for position,camera in enumerate(frame['images']):
                c=camera['camera_id'];intr=np.loadtxt(a.scene/'intrinsics'/f'{c}.txt')[:4]/2
                C=scene_from_world@np.array(camera['ego_to_world'])@np.loadtxt(a.scene/'extrinsics'/f'{c}.txt')
                C=cam(torch.tensor(C[None],device='cuda',dtype=torch.float32),torch.tensor([f*len(cameras)+position],device='cuda'))[0]
                xyz=(mu-C[:3,3])@C[:3,:3];z=xyz[:,2];uv=xyz[:,:2]/z.clamp_min(.1)[:,None]
                uv=uv*torch.tensor(intr[:2],device='cuda')+torch.tensor(intr[2:],device='cuda')
                valid=(z>.1)&(uv[:,0]>=0)&(uv[:,0]<960)&(uv[:,1]>=0)&(uv[:,1]<540)
                index=ids[valid];pix=uv[valid].long()
                with np.load(a.scene/'scene_semantics'/f'{f:03d}_{c}.npz') as masks:
                    obs[index]+=1
                    for key,count in [('sky',sky_vote),('road',road_vote),('structure',structure_vote)]:
                        mask=torch.from_numpy(masks[key]).to('cuda');count[index]+=mask[pix[:,1],pix[:,0]].to(torch.int16)
    obs=obs.cpu().numpy();sky_vote=sky_vote.cpu().numpy();road_vote=road_vote.cpu().numpy();structure_vote=structure_vote.cpu().numpy()
    visible=obs>=4
    sky=(visible&(sky_vote/np.maximum(obs,1)>.85)&(nearest>.3))|((obs==0)&(nearest>1.5))
    structure=visible&(structure_vote/np.maximum(obs,1)>.5)&planar[anchor_id]&(nearest<.6)
    road=visible&(road_vote/np.maximum(obs,1)>.5)&planar[anchor_id]&(np.abs(normal[:,2])>.95)&(nearest<1.5)
    valid=structure|road
    delta=((means-anchors)*normal).sum(-1)
    means[valid]-=normal[valid]*delta[valid,None]
    bg['_means']=torch.from_numpy(means.copy())
    scale=np.exp(bg['_scales'].numpy());selected=np.flatnonzero(valid)
    n=normal[selected].copy();n[n[:,2]<0]*=-1
    tangent=np.tile(np.array([1.,0,0]),(len(selected),1));tangent-=n*(tangent*n).sum(-1,keepdims=True)
    bad=np.linalg.norm(tangent,axis=-1)<.1;tangent[bad]=np.cross(n[bad],np.array([0.,1,0]));tangent/=np.linalg.norm(tangent,axis=-1,keepdims=True)
    R=np.stack((tangent,np.cross(n,tangent),n),axis=-1);q=Rotation.from_matrix(R).as_quat()[:,[3,0,1,2]]
    sorted_scale=np.sort(scale[selected],axis=-1)
    new_scale=np.stack((np.clip(sorted_scale[:,2],.04,.3),np.clip(sorted_scale[:,1],.04,.3),np.clip(sorted_scale[:,0],.005,.025)),axis=-1)
    bg['_quats'][selected]=torch.from_numpy(q).to(bg['_quats'].dtype);bg['_scales'][selected]=torch.from_numpy(np.log(new_scale)).to(bg['_scales'].dtype)
    bg['_opacities'][torch.from_numpy(sky)]=float(np.log(.0001/(1-.0001)))
    state['models']['Sky']['base']=F.interpolate(state['models']['Sky']['base'].permute(0,3,1,2),size=(128,128),mode='bilinear',align_corners=False).permute(0,2,3,1).contiguous()
    checkpoint=a.output/'checkpoint_final.pth';torch.save(state,checkpoint)
    import yaml
    cfg=yaml.safe_load((a.checkpoint.parent/'config.yaml').read_text());cfg['data']['data_root']=str(a.scene.parent);cfg['data']['pixel_source']['load_scene_semantics']=True;cfg['model']['Sky']['params']['resolution']=128
    (a.output/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    np.savez_compressed(a.output/'anchors.npz',anchors=anchors.astype(np.float32),normals=normal.astype(np.float32),valid=valid)
    report={'status':'complete','static_lidar_voxels':len(points),'gaussians':N,'lidar_supported_planar_gaussians':int(valid.sum()),
            'road_gaussians':int(road.sum()),'sky_or_unobserved_unsupported_gaussians_suppressed':int(sky.sum()),
            'normal_snap_median_m':float(np.median(np.abs(delta[valid]))),'normal_snap_p95_m':float(np.quantile(np.abs(delta[valid]),.95)),
            'lidar_protected_sky_pixels':protected,'rigid_and_camera_parameters_modified':False,'masks_are_predictions':True}
    (a.output/'geometry.json').write_text(json.dumps(report,indent=2)+'\n')
    semantic_manifest=a.scene/'scene_semantics/manifest.json';manifest=json.loads(semantic_manifest.read_text());manifest['lidar_hit_protection_dilation_px']=3;manifest['lidar_protected_sky_pixels']=protected;semantic_manifest.write_text(json.dumps(manifest,indent=2)+'\n')
    print('STATIC_GEOMETRY_OK '+json.dumps(report),flush=True)


if __name__=='__main__':main()
