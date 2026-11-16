"""Controlled rear-surface geometry experiments with frozen traffic and cameras."""
import argparse,copy,json,shutil,sys
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'Omniscenegraph')]
from nexus_3dgs.surface_lifting import rectify_gaussians


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','render','layers','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--project-centres',action='store_true',help='Also place the rear-panel points on the measured rear plane')
    p.add_argument('--local-surface',type=Path,help='Use locally fitted real LiDAR surface planes instead of one flat rear plane')
    p.add_argument('--exclude-indices',type=Path,help='Original point indices excluded by semantic evidence')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import numpy as np,torch,yaml
    from models.gaussians.basics import quat_to_rotmat
    from models.gaussians.support import confine_gaussian_support
    from nexus_3dgs.runtime import TrajectoryController
    from nexus_3dgs.grounding import GroundedRegistration
    from scripts.pipeline.merge_nexus_vehicle_asset import equal
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False);w=state['models']['RigidNodes'];cfg=yaml.safe_load((a.checkpoint.parent/'config.yaml').read_text());layer=json.loads(a.layers.read_text());model=layer['model_index'];rear=layer['rear_plane_x_m']
    run=json.loads((a.render/'run.json').read_text());g=run['ground_registration'];rigid=SimpleNamespace(instance_ids=w['dataset_instance_ids'],instances_trans=w['instances_trans'],instances_quats=w['instances_quats'],instances_fv=w['instances_fv']);rigid.num_frames,rigid.num_instances=rigid.instances_fv.shape
    controller=TrajectoryController(run['bundle'],Path(g['ground_model']['scene']),rigid);ground=a.output/'ground.json';ground.write_text(json.dumps(g['ground_model']));reg=GroundedRegistration(controller,ground,stabilize_all=g['stabilize_all_rigid_instances']);reg.object_mounts={int(k):np.array(v) for k,v in g['object_mounts'].items()};reg.flat_mounts={int(k):np.array(v) for k,v in g.get('flat_track_mounts',{}).items()};controller.grounding=reg
    record=run['frames'][0];image=record['images'][0];f=record['frame_idx'];C=controller.scene_from_world@np.array(image['scene_metadata']['camera_world']);K=np.array(image['scene_metadata']['intrinsics'])
    with controller.edit_frame(f,image['timestamp_us']):
        R=quat_to_rotmat(torch.nn.functional.normalize(rigid.instances_quats[f,model][None],dim=-1))[0];B=np.eye(4);B[:3,:3]=R.numpy();B[:3,3]=rigid.instances_trans[f,model].numpy()
    local_camera=np.linalg.inv(B)@C;index=torch.nonzero(w['points_ids'].reshape(-1)==model)[:,0];mu=w['_means'][index];q=torch.nn.functional.normalize(w['_quats'][index],dim=-1)
    (a.output/'canonical_pose.json').write_text(json.dumps({'frame':f,'body_to_scene':B.tolist(),'camera_to_scene':C.tolist()},indent=2)+'\n')
    ctrl=cfg['model']['RigidNodes']['ctrl'];maximum=ctrl.get('physical_max_std_m',1e9);sigma=ctrl.get('physical_support_sigma',3.);size=w['instances_size'][model].expand(len(mu),-1)
    scales=w['_scales'][index].exp().clamp(max=maximum);scales,valid=confine_gaussian_support(mu,scales,quat_to_rotmat(q),size,sigma=sigma)
    covariance=(quat_to_rotmat(q)*scales[:,None,:]**2)@quat_to_rotmat(q).transpose(-1,-2);thickness=covariance[:,0,0].sqrt()
    world=mu.numpy()@B[:3,:3].T+B[:3,3];aa,bb,cc=g['ground_model']['coefficients_z_ax_by_c'];height=world[:,2]-(aa*world[:,0]+bb*world[:,1]+cc)
    selected=valid&((mu[:,0]-rear).abs()<.35)&torch.from_numpy(height>.45)
    if not a.project_centres:selected &= thickness>.04
    if a.exclude_indices:
        excluded=torch.zeros(len(w['_means']),dtype=torch.bool);excluded[torch.from_numpy(np.load(a.exclude_indices))]=True;selected &= ~excluded[index]
    local_ids=torch.nonzero(selected)[:,0];points=mu[local_ids].numpy();normals=np.tile([1.,0,0],(len(points),1));anchors=points.copy();anchors[:,0]=rear
    if a.local_surface:
        from scipy.spatial import cKDTree
        P=np.load(a.local_surface)['points'];distance,neighbors=cKDTree(P).query(points,k=48,workers=8);groups=P[neighbors];center=groups.mean(axis=1);offset=groups-center[:,None]
        covariance=np.einsum('nki,nkj->nij',offset,offset)/48;eigenvalue,eigenvector=np.linalg.eigh(covariance)
        normals=eigenvector[:,:,0];supported=(distance[:,0]<.25)&(distance[:,-1]<.7)&(eigenvalue[:,0]/np.maximum(eigenvalue.sum(-1),1e-12)<.08)&(np.abs(normals[:,0])>.7)
        local_ids=local_ids[torch.from_numpy(supported)];points=points[supported];normals=normals[supported];anchors=center[supported]
    xyz=(points-local_camera[:3,3])@local_camera[:3,:3]
    def projected_covariance(scales,quats,locations=None):
        R=quat_to_rotmat(torch.nn.functional.normalize(quats,dim=-1)).numpy();cov=(R*scales.numpy()[:,None,:]**2)@R.transpose(0,2,1)
        p=xyz if locations is None else (locations-local_camera[:3,3])@local_camera[:3,:3]
        J=np.zeros((len(p),2,3));J[:,0,0]=K[0,0]/p[:,2];J[:,1,1]=K[1,1]/p[:,2];J[:,0,2]=-K[0,0]*p[:,0]/p[:,2]**2;J[:,1,2]=-K[1,1]*p[:,1]/p[:,2]**2;J=J@local_camera[:3,:3].T
        return J@cov@J.transpose(0,2,1)
    old_cov=projected_covariance(scales[local_ids],q[local_ids]);reports=[]
    for name,normal_std,preserve in [('thin_01',.01,True),('thin_03',.03,True),('thin_01_all',.01,False)]:
        depth=xyz[:,2]
        if a.project_centres:depth=depth*((anchors-local_camera[:3,3])*normals).sum(-1)/((points-local_camera[:3,3])*normals).sum(-1)
        new_points,logs,quats=rectify_gaussians(points,np.log(scales[local_ids].numpy()),q[local_ids].numpy(),K,local_camera,depth,normals,normal_std=normal_std)
        if not a.project_centres:new_points=points
        new_mu=torch.tensor(new_points,dtype=torch.float32);new_scale=torch.tensor(np.exp(logs),dtype=torch.float32).clamp(max=maximum);new_q=torch.tensor(quats,dtype=torch.float32);new_scale,new_valid=confine_gaussian_support(new_mu,new_scale,quat_to_rotmat(new_q),size[local_ids],sigma=sigma)
        new_cov=projected_covariance(new_scale,new_q,new_points);error=np.linalg.norm(new_cov-old_cov,axis=(1,2))/np.maximum(np.linalg.norm(old_cov,axis=(1,2)),1e-9)
        keep=torch.from_numpy(error<.01)&new_valid if preserve else new_valid
        chosen=index[local_ids[keep]];candidate=copy.deepcopy(w);candidate['_scales'][chosen]=new_scale[keep].log();candidate['_quats'][chosen]=new_q[keep]
        if a.project_centres:candidate['_means'][chosen]=new_mu[keep]
        directory=a.output/name;directory.mkdir();state['models']['RigidNodes']=candidate;torch.save(state,directory/'checkpoint_final.pth');np.save(directory/'selected_indices.npy',chosen.numpy());shutil.copyfile(a.checkpoint.parent/'config.yaml',directory/'config.yaml')
        for key in w:
            if key not in (['_means','_scales','_quats'] if a.project_centres else ['_scales','_quats']):equal(w[key],candidate[key])
        untouched=torch.ones(len(w['_means']),dtype=torch.bool);untouched[chosen]=False
        for key in ['_means','_scales','_quats']:equal(w[key][untouched],candidate[key][untouched])
        reports.append({'name':name,'modified_gaussians':len(chosen),'normal_std_m':normal_std,'preserves_reference_covariance_within_one_percent':preserve,'source_covariance_relative_error_p90':float(np.quantile(error[keep.numpy()],.9)) if keep.any() else None,'old_normal_std_median_m':float(thickness[local_ids[keep]].median()) if keep.any() else None})
    result={'status':'candidates','rear_plane_from_lidar_x_m':rear,'eligible_gaussians':len(points),'projected_centres':a.project_centres,'local_surface':str(a.local_surface) if a.local_surface else None,'colors_opacity_poses_and_all_other_models_unchanged':True,'variants':reports,'scope':'rear skin covariance normal compression using real-LiDAR face direction and source pixel footprints; not an accepted improvement until rendered'}
    (a.output/'surface.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
