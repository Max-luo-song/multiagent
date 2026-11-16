"""Measure single-vehicle depth layering and actual rear-surface opacity."""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'Omniscenegraph')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','render','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--track-token',default='441830199c6752a5');p.add_argument('--points-only',action='store_true');p.add_argument('--grounded-points',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import cv2,numpy as np,torch,yaml
    from gsplat import rasterization
    from models.gaussians.basics import quat_to_rotmat,quat_mult
    from models.gaussians.support import confine_gaussian_support
    from nexus_3dgs.runtime import TrajectoryController
    from nexus_3dgs.grounding import GroundedRegistration
    from nexus_3dgs.geometry import quaternion_matrix
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False);w=state['models']['RigidNodes'];cfg=yaml.safe_load((a.checkpoint.parent/'config.yaml').read_text())
    run=json.loads((a.render/'run.json').read_text());g=run['ground_registration'];scene=Path(g['ground_model']['scene']);metadata=json.loads((scene/'nexus_scene.json').read_text());info=json.loads((scene/'instances/instances_info.json').read_text())
    dataset_id=next(int(k) for k,v in info.items() if v['id']==a.track_token);model=int(torch.nonzero(w['dataset_instance_ids']==dataset_id)[0])
    selected=w['points_ids'].reshape(-1)==model
    rigid=SimpleNamespace(instance_ids=w['dataset_instance_ids'],instances_trans=w['instances_trans'],instances_quats=w['instances_quats'],instances_fv=w['instances_fv'])
    rigid.num_frames,rigid.num_instances=rigid.instances_fv.shape
    controller=TrajectoryController(run['bundle'],scene,rigid);ground=a.output/'ground.json';ground.write_text(json.dumps(g['ground_model']))
    registration=GroundedRegistration(controller,ground,stabilize_all=g['stabilize_all_rigid_instances']);registration.object_mounts={int(k):np.array(v) for k,v in g['object_mounts'].items()};registration.flat_mounts={int(k):np.array(v) for k,v in g.get('flat_track_mounts',{}).items()};controller.grounding=registration
    ann=info[str(dataset_id)]['frame_annotations'];byframe=dict(zip(ann['frame_idx'],zip(ann['obj_to_world'],ann['box_size'])));local_points=[]
    for f in [0,20,40,60,80,99]:
        if f not in byframe:continue
        record=metadata['frames'][f];raw=np.fromfile(scene/'lidar'/f'{f:03d}.bin',np.float32).reshape(-1,6)[:,:3]
        ego=np.array(record['ego_to_world']);world=raw@ego[:3,:3].T+ego[:3,3];pose,size=map(np.array,byframe[f]);q=(world-pose[:3,3])@pose[:3,:3]
        valid=(np.abs(q)<size/2+.02).all(-1)&(q[:,2]>-size[2]/2+.15)
        pts=(world[valid]-np.linalg.inv(controller.scene_from_world)[:3,3])@np.linalg.inv(controller.scene_from_world)[:3,:3]
        if a.grounded_points:
            body=registration.object_scene_pose(model,pose);local=(pts-body[:3,3])@body[:3,:3]
        else:
            R=quaternion_matrix(w['instances_quats'][f,model].numpy());local=(pts-w['instances_trans'][f,model].numpy())@R
        local_points.append(local)
    P=np.concatenate(local_points);rear_candidates=P[P[:,0]<np.quantile(P[:,0],.3),0];edges=np.arange(rear_candidates.min()-.02,rear_candidates.max()+.04,.02);hist,_=np.histogram(rear_candidates,edges);binid=hist.argmax();rear=float(np.median(rear_candidates[(rear_candidates>=edges[binid]-.04)&(rear_candidates<=edges[binid+1]+.04)]))
    np.savez_compressed(a.output/'local_surface_points.npz',points=P)
    if a.points_only:
        (a.output/'surface_points.json').write_text(json.dumps({'status':'complete','model_index':model,'track_token':a.track_token,'points':len(P),'rear_plane_x_m':rear,'fixed_renderer_body_coordinates':a.grounded_points},indent=2)+'\n')
        print('LOCAL_SURFACE_POINTS '+str(len(P)),flush=True);return
    local=w['_means'][selected].cuda();q_local=torch.nn.functional.normalize(w['_quats'][selected].cuda(),dim=-1);scales=w['_scales'][selected].exp().cuda();opacity=w['_opacities'][selected].sigmoid().cuda()
    ctrl=cfg['model']['RigidNodes']['ctrl'];scales=scales.clamp(max=ctrl.get('physical_max_std_m',1e9));scales,valid=confine_gaussian_support(local,scales,quat_to_rotmat(q_local),w['instances_size'][model].cuda().expand(len(local),-1),sigma=ctrl.get('physical_support_sigma',3.))
    opacity*=valid[:,None];rear_mask=(local[:,0]-rear).abs()<.15
    records={f['frame_idx']:next(i for i in f['images'] if i['camera_id']==0) for f in run['frames']};rows=[]
    for f in [20,55,70,85,99]:
        image=records[f];C=torch.tensor(controller.scene_from_world@np.array(image['scene_metadata']['camera_world']),dtype=torch.float32,device='cuda');K=torch.tensor(image['scene_metadata']['intrinsics'],dtype=torch.float32,device='cuda')
        with controller.edit_frame(f,image['timestamp_us']):
            q=torch.nn.functional.normalize(rigid.instances_quats[f,model].cuda(),dim=-1);R=quat_to_rotmat(q[None])[0];mu=local@R.T+rigid.instances_trans[f,model].cuda();rotation=quat_mult(q.expand(len(local),-1),q_local)
        z=((mu-C[:3,3])@C[:3,:3])[:,2];normalizer=z.max().clamp_min(1);color=torch.stack((z/normalizer,(z/normalizer)**2,torch.ones_like(z)),dim=-1)
        def render(mask):
            values,alpha,_=rasterization(means=mu,quats=rotation,scales=scales,opacities=opacity[:,0]*mask,colors=color,viewmats=torch.linalg.inv(C)[None],Ks=K[None],width=960,height=540,near_plane=.1,far_plane=1e10,packed=False,render_mode='RGB',rasterize_mode='antialiased' if cfg['trainer']['render']['antialiased'] else 'classic')
            return values[0],alpha[0,:,:,0]
        with torch.no_grad():moments,alpha=render(torch.ones_like(z));_,rear_alpha=render(rear_mask)
        mean=moments[:,:,0]/alpha.clamp_min(1e-6)*normalizer;second=moments[:,:,1]/alpha.clamp_min(1e-6)*normalizer**2;std=(second-mean**2).clamp_min(0).sqrt()
        core=cv2.erode((alpha.cpu().numpy()>.95).astype(np.uint8),np.ones((9,9),np.uint8))>0;core=torch.from_numpy(core).cuda()
        row={'frame':f,'opaque_core_pixels':int(core.sum()),'depth_std_median_m':float(std[core].median()),'depth_std_p90_m':float(torch.quantile(std[core],.9)),'rear_surface_alpha_median':float(rear_alpha[core].median()),'rear_surface_alpha_mean':float(rear_alpha[core].mean())};rows.append(row)
        np.savez_compressed(a.output/f'{f:03d}.npz',depth_std=std.cpu().numpy(),alpha=alpha.cpu().numpy(),rear_alpha=rear_alpha.cpu().numpy(),core=core.cpu().numpy())
    report={'status':'complete','track_token':a.track_token,'model_index':model,'gaussians':len(local),'local_lidar_points':len(P),'rear_plane_x_m':rear,'rear_gaussians':int(rear_mask.sum()),'gaussian_x_quantiles':np.quantile(local[:,0].cpu().numpy(),[0,.1,.5,.9,1]).tolist(),'frames':rows,'scope':'opacity-weighted Gaussian-centre depth variance of the isolated bus, excluding within-Gaussian depth variance; rear-plane support estimated from real LiDAR'}
    (a.output/'layers.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
