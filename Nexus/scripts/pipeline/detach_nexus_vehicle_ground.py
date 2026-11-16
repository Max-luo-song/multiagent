"""Reassign observed road Gaussians from a vehicle to the static background.

Only low, horizontal, neutral-colored points repeatedly observed as road are
eligible. Reference-pose world position/covariance/color/opacity are retained.
"""
import argparse,copy,json,shutil,sys
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'Omniscenegraph')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','render','reference','semantic-scene','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--track-token',default='441830199c6752a5')
    p.add_argument('--max-height',type=float,default=.18)
    p.add_argument('--project-to-ground',action='store_true')
    p.add_argument('--audit-only',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import numpy as np,torch,yaml
    from models.gaussians.basics import quat_to_rotmat,quat_mult,spherical_harmonics
    from models.gaussians.support import confine_gaussian_support
    from nexus_3dgs.runtime import TrajectoryController
    from nexus_3dgs.grounding import GroundedRegistration
    from scripts.pipeline.merge_nexus_vehicle_asset import equal
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False);cfg=yaml.safe_load((a.checkpoint.parent/'config.yaml').read_text());w=state['models']['RigidNodes']
    run=json.loads((a.render/'run.json').read_text());g=run['ground_registration'];scene=Path(g['ground_model']['scene']);info=json.loads((scene/'instances/instances_info.json').read_text());dataset_id=next(int(k) for k,v in info.items() if v['id']==a.track_token);model=int(torch.nonzero(w['dataset_instance_ids']==dataset_id)[0])
    point_ids=w['points_ids'].reshape(-1);bus_ids=torch.nonzero(point_ids==model)[:,0];mu=w['_means'][bus_ids];q_local=torch.nn.functional.normalize(w['_quats'][bus_ids],dim=-1)
    rigid=SimpleNamespace(instance_ids=w['dataset_instance_ids'],instances_trans=w['instances_trans'],instances_quats=w['instances_quats'],instances_fv=w['instances_fv']);rigid.num_frames,rigid.num_instances=rigid.instances_fv.shape
    controller=TrajectoryController(run['bundle'],scene,rigid);ground_path=a.output/'ground.json';ground_path.write_text(json.dumps(g['ground_model']));registration=GroundedRegistration(controller,ground_path,stabilize_all=g['stabilize_all_rigid_instances']);registration.object_mounts={int(k):np.array(v) for k,v in g['object_mounts'].items()};registration.flat_mounts={int(k):np.array(v) for k,v in g.get('flat_track_mounts',{}).items()};controller.grounding=registration
    refs={f['frame_idx']:next(i for i in f['images'] if i['camera_id']==0) for f in json.loads((a.reference/'run.json').read_text())['frames']};votes=np.zeros(len(mu),int);observed=np.zeros(len(mu),int)
    for f in [20,35,55,70,85,99]:
        image=refs[f];C=controller.scene_from_world@np.array(image['scene_metadata']['camera_world']);K=np.array(image['scene_metadata']['intrinsics']);R=quat_to_rotmat(torch.nn.functional.normalize(w['instances_quats'][f,model][None],dim=-1))[0]
        world=mu@R.T+w['instances_trans'][f,model];xyz=(world.numpy()-C[:3,3])@C[:3,:3];uv=xyz[:,:2]/np.maximum(xyz[:,2,None],.1)@K[:2,:2].T+K[:2,2];visible=(xyz[:,2]>.1)&(uv[:,0]>=0)&(uv[:,0]<960)&(uv[:,1]>=0)&(uv[:,1]<540);ids=np.flatnonzero(visible);pixels=uv[ids].astype(int)
        with np.load(a.semantic_scene/'scene_semantics'/f'{f:03d}_0.npz') as masks:road=masks['road']
        observed[ids]+=1;votes[ids]+=(road[pixels[:,1],pixels[:,0]]>0)
    image=run['frames'][0]['images'][0];frame=run['frames'][0]['frame_idx'];C=torch.tensor(controller.scene_from_world@np.array(image['scene_metadata']['camera_world']),dtype=torch.float32)
    with controller.edit_frame(frame,image['timestamp_us']):
        q=torch.nn.functional.normalize(rigid.instances_quats[frame,model],dim=-1);R=quat_to_rotmat(q[None])[0];world=mu@R.T+rigid.instances_trans[frame,model];world_q=quat_mult(q.expand(len(mu),-1),q_local)
    scales=w['_scales'][bus_ids].exp();ctrl=cfg['model']['RigidNodes']['ctrl'];scales=scales.clamp(max=ctrl.get('physical_max_std_m',1e9));scales,valid=confine_gaussian_support(mu,scales,quat_to_rotmat(q_local),w['instances_size'][model].expand(len(mu),-1),sigma=ctrl.get('physical_support_sigma',3.))
    aa,bb,cc=g['ground_model']['coefficients_z_ax_by_c'];height=world[:,2]-(aa*world[:,0]+bb*world[:,1]+cc)
    rotation=quat_to_rotmat(world_q);axis=scales.argmin(dim=-1);normal=rotation[torch.arange(len(mu)),:,axis];up=torch.tensor([-aa,-bb,1.]);up/=up.norm();alignment=(normal@up).abs()
    coeff=torch.cat((w['_features_dc'][bus_ids,None,:],w['_features_rest'][bus_ids]),dim=1).cuda();direction=torch.nn.functional.normalize(world-C[:3,3],dim=-1).cuda()
    with torch.no_grad():rgb=(spherical_harmonics(3,direction,coeff)+.5).clamp(0,1).cpu()
    low=(height>-.08)&(height<a.max_height);road=torch.from_numpy((observed>=3)&(votes/np.maximum(observed,1)>=.8));horizontal=alignment>.65;neutral=(rgb.max(-1).values-rgb.min(-1).values)<.18;bright=rgb.mean(-1)>.10
    take=valid&low&road&horizontal&neutral&bright
    np.save(a.output/'repeated_road_indices.npy',bus_ids[road].numpy())
    report={'status':'candidate','track_token':a.track_token,'reference_frame':frame,'bus_gaussians':len(mu),'candidates_by_rule':{'low':int(low.sum()),'repeated_road':int(road.sum()),'horizontal':int(horizontal.sum()),'all':int(take.sum())},'height_quantiles_m':torch.quantile(height,torch.tensor([0.,.1,.5,.9,1.])).tolist()}
    if a.audit_only:
        report['status']='diagnostic_only';(a.output/'ground_assignment.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True);return
    if not take.any():
        report['status']='no_supported_points';(a.output/'ground_assignment.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True);return
    global_take=torch.zeros(len(point_ids),dtype=torch.bool);global_take[bus_ids[take]]=True;bg=state['models']['Background'];count=len(point_ids)
    static_means=world[take].clone();static_scales=scales[take].clone()
    if a.project_to_ground:
        normal=torch.tensor([-aa,-bb,1.]);rays=static_means-C[:3,3]
        ratio=-(C[:3,3]@normal-cc)/(rays@normal)
        if not ((ratio>.1)&(ratio<5)&torch.isfinite(ratio)).all():raise ValueError('Road ray intersection is unreliable')
        static_means=C[:3,3]+rays*ratio[:,None];static_scales*=ratio[:,None]
        bounds=torch.tensor(g['ground_model']['supported_xy_bounds_scene'])
        if not ((static_means[:,:2]>=bounds[0]).all(-1)&(static_means[:,:2]<=bounds[1]).all(-1)).all():raise ValueError('Projected road leaves validated coverage')
    additions={'_means':static_means,'_quats':world_q[take],'_scales':static_scales.log(),'_features_dc':w['_features_dc'][bus_ids[take]],'_features_rest':w['_features_rest'][bus_ids[take]],'_opacities':w['_opacities'][bus_ids[take]]}
    for key,value in additions.items():bg[key]=torch.cat((bg[key],value))
    for key,value in list(w.items()):
        if torch.is_tensor(value) and value.ndim and value.shape[0]==count:w[key]=value[~global_take].clone()
    checkpoint=a.output/'checkpoint_final.pth';torch.save(state,checkpoint);shutil.copyfile(a.checkpoint.parent/'config.yaml',a.output/'config.yaml')
    source=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    for key in ['instances_trans','instances_quats','instances_fv','instances_size','dataset_instance_ids']:equal(source['models']['RigidNodes'][key],w[key])
    for name in ['CamPose','Affine','Sky']:equal(source['models'][name],state['models'][name])
    for key,value in source['models']['Background'].items():equal(value,bg[key][:len(value)])
    report.update(transferred=int(take.sum()),maximum_original_height_m=a.max_height,projected_to_ground=a.project_to_ground,original_background_preserved=True,all_vehicle_poses_sizes_visibility_preserved=True,method='semantic+height+normal+color corroborated ownership reassignment, native Gaussians fixed at reference world pose; no pixel repainting')
    (a.output/'ground_assignment.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
