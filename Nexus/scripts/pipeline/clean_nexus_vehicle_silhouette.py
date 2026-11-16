"""Conservative multi-view Gaussian support cleanup of one original vehicle."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'Omniscenegraph')]
from nexus_3dgs.geometry import quaternion_matrix
from models.modules import CameraOptModule


def rotations(quaternions):
    q=np.asarray(quaternions);q=q/np.linalg.norm(q,axis=-1,keepdims=True)
    w,x,y,z=q.T
    return np.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
                     2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                     2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),axis=-1).reshape(-1,3,3)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--mask-scene',type=Path,required=True)
    p.add_argument('--track-token',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--margin-px',type=int,default=8);p.add_argument('--outside-fraction',type=float,default=.9)
    p.add_argument('--min-views',type=int,default=30);p.add_argument('--minimum-scale-ratio',type=float,default=.4)
    p.add_argument('--region',choices=('all','roof'),default='all')
    a=p.parse_args();cv2.setNumThreads(4);a.output.mkdir(parents=True,exist_ok=False)
    cfg=yaml.safe_load((a.checkpoint.parent/'config.yaml').read_text())
    scene=Path(cfg['data']['data_root'])/cfg['data']['scene_idx'];meta=json.loads((scene/'nexus_scene.json').read_text())
    assert meta['identity']==json.loads((a.mask_scene/'nexus_scene.json').read_text())['identity']
    info=json.loads((scene/'instances/instances_info.json').read_text())
    identity=next(int(k) for k,v in info.items() if v['id']==a.track_token)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False);rigid=state['models']['RigidNodes']
    index=int(torch.nonzero(rigid['dataset_instance_ids']==identity,as_tuple=True)[0].item())
    ids=rigid['points_ids'].reshape(-1);selected=ids==index;model_ids=np.flatnonzero(selected.numpy())
    means=rigid['_means'][selected].numpy();N=len(means)
    scales=np.exp(rigid['_scales'][selected].numpy());ctrl=cfg['model']['RigidNodes']['ctrl']
    scales=np.minimum(scales,ctrl['physical_max_std_m'])
    rot=rotations(rigid['_quats'][selected].numpy())
    remaining=rigid['instances_size'][index].numpy()/2+.08-np.abs(means)
    axis_std=np.sqrt(np.maximum((rot**2*scales[:,None,:]**2).sum(-1),1e-16))
    factor=np.minimum(1,np.maximum(remaining,0)/(ctrl['physical_support_sigma']*axis_std)).min(-1)
    scales=np.maximum(scales*factor[:,None],1e-6)
    covariance=(rot*scales[:,None,:]**2)@rot.transpose(0,2,1)
    camera=None
    if 'CamPose' in state['models']:
        cs=state['models']['CamPose'];camera=CameraOptModule('CamPose',len(cs['embeds.weight']),device=torch.device('cpu'))
        camera.load_state_dict(cs);camera.eval()
    if 'CamPosePerturb' in state['models']:raise ValueError('Unhandled additional camera refinement')
    start=int(cfg['data']['start_timestep']);scene_from_world=np.linalg.inv(np.array(meta['frames'][start]['ego_to_world']))
    intr=np.loadtxt(scene/'intrinsics/0.txt')[:4]/2
    extr=np.loadtxt(scene/'extrinsics/0.txt');C=len(cfg['data']['pixel_source']['cameras'])
    position=cfg['data']['pixel_source']['cameras'].index(0)
    votes=np.zeros(N,np.int32);outside=np.zeros(N,np.int32);ratios=[]
    for f in range(len(rigid['instances_fv'])):
        if not bool(rigid['instances_fv'][f,index]):continue
        record=meta['frames'][start+f];image=next(i for i in record['images'] if i['camera_id']==0)
        c2w=scene_from_world@np.array(image['ego_to_world'])@extr
        if camera is not None:
            with torch.no_grad():c2w=camera(torch.tensor(c2w[None],dtype=torch.float32),torch.tensor([f*C+position]))[0].numpy()
        R=quaternion_matrix(rigid['instances_quats'][f,index].numpy());t=rigid['instances_trans'][f,index].numpy()
        local=((means@R.T+t)-c2w[:3,3])@c2w[:3,:3]
        z=np.maximum(local[:,2],.1);uv=local[:,:2]/z[:,None]*intr[:2]+intr[2:]
        mask=cv2.imread(str(a.mask_scene/'vehicle_interior_masks'/f'{record["frame_idx"]:03d}_0.png'),0)
        if mask is None:raise ValueError('Missing observed silhouette')
        mask=cv2.dilate((mask>0).astype(np.uint8),np.ones((2*a.margin_px+1,)*2,np.uint8))
        distance=cv2.distanceTransform(mask,cv2.DIST_L2,5)
        valid=(local[:,2]>.1)&(uv[:,0]>=0)&(uv[:,0]<mask.shape[1])&(uv[:,1]>=0)&(uv[:,1]<mask.shape[0])
        x=np.clip(uv[:,0].astype(int),0,mask.shape[1]-1);y=np.clip(uv[:,1].astype(int),0,mask.shape[0]-1)
        d=distance[y,x];votes+=valid;outside+=valid&(d==0)
        to_camera=c2w[:3,:3].T@R
        cov=to_camera[None]@covariance@to_camera.T[None]
        J=np.zeros((N,2,3));J[:,0,0]=intr[0]/z;J[:,1,1]=intr[1]/z
        J[:,0,2]=-intr[0]*local[:,0]/z**2;J[:,1,2]=-intr[1]*local[:,1]/z**2
        screen=J@cov@J.transpose(0,2,1)
        trace=screen[:,0,0]+screen[:,1,1]
        disc=np.sqrt(np.maximum((screen[:,0,0]-screen[:,1,1])**2+4*screen[:,0,1]**2,0))
        sigma=np.sqrt(np.maximum((trace+disc)/2,1e-12))
        ratio=np.ones(N,np.float32);use=valid&(d>0)
        ratio[use]=np.clip(d[use]/(3*sigma[use]),a.minimum_scale_ratio,1)
        ratios.append(ratio)
    observed=votes>=a.min_views
    remove=observed&(outside/np.maximum(votes,1)>=a.outside_fraction)
    region=means[:,2]>float(rigid['instances_size'][index,2])*.25 if a.region=='roof' else np.ones(N,bool)
    remove &= region
    if remove.mean()>.3:raise ValueError('Silhouette cleanup would remove too much of the vehicle')
    ratio=np.quantile(np.array(ratios),.2,axis=0).astype(np.float32);ratio[~observed]=1
    ratio[~region]=1
    shrink=(ratio<.995)&~remove
    # Preserve original interior colour/opacity/pose parameters; only boundary
    # support changes, using effective physically bounded standard deviations.
    updates=model_ids[shrink]
    rigid['_scales'][updates]=torch.from_numpy(np.log(scales[shrink]*ratio[shrink,None])).to(rigid['_scales'].dtype)
    keep=torch.ones(len(ids),dtype=torch.bool);keep[torch.from_numpy(model_ids[remove])]=False
    count=len(ids)
    for key,value in list(rigid.items()):
        if torch.is_tensor(value) and value.ndim and value.shape[0]==count:rigid[key]=value[keep].clone()
    output=a.output/'checkpoint_final.pth';torch.save(state,output)
    shutil.copyfile(a.checkpoint.parent/'config.yaml',a.output/'config.yaml')
    # All non-target scene parameters and target appearance/pose are protected.
    original=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    written=torch.load(output,map_location='cpu',weights_only=False)
    from scripts.pipeline.merge_nexus_vehicle_asset import equal
    for name,value in original['models'].items():
        if name!='RigidNodes':equal(value,written['models'][name])
    old=original['models']['RigidNodes'];new=written['models']['RigidNodes']
    for key,value in old.items():
        if torch.is_tensor(value) and value.ndim and value.shape[0]==count:
            equal(value[ids!=index],new[key][new['points_ids'].reshape(-1)!=index])
            if key!='_scales':equal(value[keep],new[key])
        else:equal(value,new[key])
    report={'status':'passed','method':'original asset: multi-view silhouette-supported centre culling and boundary support contraction',
            'track_token':a.track_token,'model_index':index,'vehicle_points_before':N,'removed_unsupported_centres':int(remove.sum()),
            'contracted_boundary_gaussians':int(shrink.sum()),'vehicle_points_after':N-int(remove.sum()),
            'mask_margin_px':a.margin_px,'required_outside_fraction':a.outside_fraction,'minimum_views':a.min_views,
            'minimum_scale_ratio':a.minimum_scale_ratio,'observation_frames':len(ratios),'is_semantic_ground_truth':False,
            'region':a.region,'roof_local_z_min_fraction_of_height':.25 if a.region=='roof' else None,
            'other_assets_and_poses_unchanged':True,'target_colours_opacities_and_poses_unchanged':True,
            'output_sha256':hashlib.sha256(output.read_bytes()).hexdigest()}
    (a.output/'silhouette_verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
