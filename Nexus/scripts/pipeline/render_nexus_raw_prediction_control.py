"""Append saved Nexus inference outputs without the later offline interventions."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'Omniscenegraph')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['experiment','scene','registration-source','free-bundle','goal-bundle']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();v=a.experiment;print('RAW_CONTROL loading unchanged baseline checkpoint',flush=True)
    import imageio.v2 as imageio,numpy as np,torch
    from nexus_3dgs.bundle import load_bundle
    from nexus_3dgs.geometry import matrix_quaternion,quaternion_matrix,resample_poses
    from nexus_3dgs.replay import RecordedStates,camera_inputs,load_trainer
    from nexus_3dgs.trajectory_control import planar,register_measured_actor
    torch.set_num_threads(8);digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    control=json.loads((v/'native_rr/run.json').read_text());audit=json.loads((v/'audit.json').read_text());assert control['status']=='complete' and audit['status']=='passed'
    records={r['frame_idx']:r for r in control['frames']};frames=sorted(records);checkpoint=Path(control['checkpoint']);checkpoint_sha=digest(checkpoint);assert checkpoint_sha==audit['checkpoint_sha256']
    trainer,cfg=load_trainer(checkpoint);rigid=trainer.models['RigidNodes'];originals={k:getattr(rigid,k).clone() for k in ['instances_trans','instances_quats','instances_fv']}
    immutable=[(p,p._version) for name,model in trainer.models.items() for key,p in model.named_parameters() if not(name=='RigidNodes' and key in ('instances_trans','instances_quats'))]
    states=RecordedStates(a.registration_source,a.scene);S=states.S;worldS=np.linalg.inv(S);scene=states.scene;info=json.loads((a.scene/'instances/instances_info.json').read_text());ids=rigid.instance_ids.cpu().tolist();bytoken={info[str(k)]['id']:m for m,k in enumerate(ids)}
    ground=states.run['ground_registration']['ground_model'];gs=np.linalg.inv(ground.get('world_from_scene',scene['frames'][0]['ego_to_world']));aa,bb,cc=ground['coefficients_z_ax_by_c'];plane=gs.T@np.array([-aa,-bb,1.,-cc]);targets=[records[f]['timestamp_us'] for f in frames]
    bundles={};bundle_report={};motion={}
    for name,path in [('free',a.free_bundle),('goal',a.goal_bundle)]:
        arrays,manifest=load_bundle(path)
        assert manifest['scenario_token']==scene['identity']['scenario_token'] and Path(manifest['database']).resolve()==Path(scene['identity']['database']).resolve()
        poses,valid=resample_poses(arrays['timestamps_us'],arrays['predicted_poses_world'],arrays['valid'],targets)
        assert valid[:,0].all();bindings={i:bytoken[str(token)] for i,token in enumerate(arrays['track_tokens']) if i and str(token) in bytoken}
        assert len(bindings)==len(arrays['track_tokens'])-1,'Unmodeled raw prediction tracks'
        bundles[name]=(poses,valid,bindings);bundle_report[name]={'bundle':str(path.resolve()),'bundle_sha256':manifest['bundle_sha256'],'task':manifest['task'],'source_prediction_sha256':manifest['prediction_sha256'],'later_offline_trajectory_edits':False,'controlled_other_vehicles':len(bindings)}
        motion[name]=[{'frame':f,'timestamp_us':records[f]['timestamp_us'],'ego_world':poses[j,0].tolist(),'travel_m':float(np.linalg.norm(poses[j,0,:2,3]-poses[0,0,:2,3]))} for j,f in enumerate(frames)]
    variants=[('raw_free_nn','free',True,True),('raw_goal_nn','goal',True,True),('raw_goal_nr','goal',True,False),('raw_goal_rn','goal',False,True)]
    max_gt_center_error=0.;max_gt_rotation_error=0.
    try:
        for name,bundle,generated_camera,generated_actors in variants:
            directory=v/name;directory.mkdir(exist_ok=False);poses,valid,bindings=bundles[bundle]
            run={'status':'started','checkpoint':str(checkpoint),'checkpoint_sha256':checkpoint_sha,'calibration':'native','camera_motion':('raw_'+bundle) if generated_camera else 'real','actor_motion':('raw_'+bundle) if generated_actors else 'real','raw_prediction_source':bundle_report[bundle],'fixed_exposure_frame':20,'frames':[]}
            with imageio.get_writer(str(directory/'camera_0.mp4'),fps=10,macro_block_size=2,ffmpeg_params=['-threads','2','-crf','18']) as writer:
                try:
                    for j,f in enumerate(frames):
                        record=records[f];realC=np.array(record['camera_world']);E=np.array(next(i for i in scene['frames'][f]['images'] if i['camera_id']==0)['ego_to_world']);Cworld=planar(poses[j,0])@np.linalg.inv(planar(E))@realC if generated_camera else realC
                        C=torch.tensor(S@Cworld,dtype=torch.float32,device='cuda');K=torch.tensor(record['intrinsics'],dtype=torch.float32,device='cuda')
                        with torch.no_grad():
                            for body in record['actors']:
                                m=body['model_index'];T=S@np.array(body['pose_world']);rigid.instances_fv[f,m]=body['valid'];rigid.instances_trans[f,m].copy_(torch.tensor(T[:3,3],dtype=torch.float32,device='cuda'));rigid.instances_quats[f,m].copy_(torch.tensor(matrix_quaternion(T[:3,:3]),dtype=torch.float32,device='cuda'))
                            if generated_actors:
                                for slot,m in bindings.items():
                                    rigid.instances_fv[f,m]=bool(valid[j,slot])
                                    if not valid[j,slot]:continue
                                    key=str(m);flat=key in states.flat;mount=np.array(states.flat[key] if flat else states.mounts[key]);T=S@register_measured_actor(poses[j,slot],mount,plane,flat)
                                    rigid.instances_trans[f,m].copy_(torch.tensor(T[:3,3],dtype=torch.float32,device='cuda'));rigid.instances_quats[f,m].copy_(torch.tensor(matrix_quaternion(T[:3,:3]),dtype=torch.float32,device='cuda'))
                            image,camera=camera_inputs(C,K,f,540,960,rigid.num_frames,len(cfg.data.pixel_source.cameras));image['img_idx'].fill_(20*len(cfg.data.pixel_source.cameras));result=trainer(image,camera,novel_view=True);outputs=result[0] if isinstance(result,tuple) else result
                            rgb=(outputs['rgb'].clamp(0,1).cpu().numpy()*255).round().astype(np.uint8);imageio.imwrite(directory/f'{f:03d}_0.png',rgb);writer.append_data(rgb)
                            bodies=[]
                            for body in record['actors']:
                                b=dict(body);m=b['model_index'];T=np.eye(4);T[:3,:3]=quaternion_matrix(rigid.instances_quats[f,m].detach().cpu().numpy());T[:3,3]=rigid.instances_trans[f,m].detach().cpu().numpy();T=worldS@T;b.update(pose_world=T.tolist(),valid=bool(rigid.instances_fv[f,m]));bodies.append(b)
                                if not generated_actors:
                                    max_gt_center_error=max(max_gt_center_error,float(np.linalg.norm(T[:3,3]-np.array(body['pose_world'])[:3,3])));max_gt_rotation_error=max(max_gt_rotation_error,float(np.abs(T[:3,:3]-np.array(body['pose_world'])[:3,:3]).max()))
                            actualC=worldS@C.cpu().numpy()
                            if not generated_camera:np.testing.assert_allclose(actualC,realC,atol=1e-8,rtol=0)
                            run['frames'].append({'frame_idx':f,'timestamp_us':record['timestamp_us'],'file':f'{f:03d}_0.png','camera_world':actualC.tolist(),'intrinsics':record['intrinsics'],'actors':bodies})
                        if j%10==0:print(f'RAW_CONTROL {name} {j+1}/{len(frames)} frame={f}',flush=True)
                    run['status']='complete'
                finally:(directory/'run.json').write_text(json.dumps(run,indent=2)+'\n')
        assert all(p._version==version for p,version in immutable)
    finally:
        with torch.no_grad():
            for key,value in originals.items():getattr(rigid,key).copy_(value)
        for key,value in originals.items():assert torch.equal(getattr(rigid,key),value)
    assert digest(checkpoint)==checkpoint_sha;assert max_gt_center_error<1e-6 and max_gt_rotation_error<1e-6
    result={'status':'passed','conditions':[v[0] for v in variants],'frames_per_condition':len(frames),'checkpoint_sha256':checkpoint_sha,'checkpoint_unchanged':True,'appearance_parameters_unchanged':True,'temporary_pose_parameters_restored_exactly':True,'max_measured_actor_center_roundtrip_error_m':max_gt_center_error,'max_measured_actor_rotation_matrix_roundtrip_error':max_gt_rotation_error,'raw_prediction_sources':bundle_report,'motion':motion,'scope':'saved original inference outputs replayed without this project later offline heading, clearance, or interaction edits; original inference recipe constraints still apply'}
    (v/'raw_predictions_audit.json').write_text(json.dumps(result,indent=2)+'\n');print('RAW_CONTROL_COMPLETE',flush=True)


if __name__=='__main__':main()
