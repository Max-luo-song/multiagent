"""Render measured/generated motion with one immutable scene and paired cameras."""
import argparse,copy,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'Omniscenegraph')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','source','scene','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--frames',type=int,nargs='+',help='Optional smoke-test subset; default all 80 source exposures')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    print('TRAJECTORY_CONTROL loading unchanged baseline checkpoint',flush=True)
    import cv2,imageio.v2 as imageio,numpy as np,torch
    from PIL import Image
    from nexus_3dgs.geometry import matrix_quaternion,quaternion_matrix,resample_poses
    from nexus_3dgs.replay import RecordedStates,camera_inputs,load_trainer
    from nexus_3dgs.trajectory_control import paired_cameras,recover_generated_ego,register_measured_actor
    cv2.setNumThreads(4);torch.set_num_threads(8)
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    states=RecordedStates(a.source,a.scene);scene=states.scene;frames=sorted(states.records) if a.frames is None else a.frames
    checkpoint_sha=digest(a.checkpoint);trainer,cfg=load_trainer(a.checkpoint);rigid=trainer.models['RigidNodes'];cams=len(cfg.data.pixel_source.cameras)
    originals={k:getattr(rigid,k).clone() for k in ('instances_trans','instances_quats','instances_fv')}
    immutable=[(p,p._version) for name,model in trainer.models.items() for key,p in model.named_parameters() if not(name=='RigidNodes' and key in ('instances_trans','instances_quats'))]
    with torch.no_grad():
        transforms=trainer.models['Affine']({'img_idx':torch.arange(rigid.num_frames*cams,device='cuda')})
        assert torch.equal(transforms,transforms[:1].expand_as(transforms)),'Use verified constant-exposure baseline'
    info=json.loads((a.scene/'instances/instances_info.json').read_text());ids=rigid.instance_ids.cpu().tolist();M=len(ids)
    annotations=np.broadcast_to(np.eye(4),(len(scene['frames']),M,4,4)).copy();valid=np.zeros(annotations.shape[:2],bool)
    for m,index in enumerate(ids):
        item=info[str(index)];cached=next(x for x in states.records[frames[0]]['scene_metadata']['actors'] if x['model_index']==m)
        assert item['id']==cached['track_token']
        ann=item['frame_annotations']
        for f,T in zip(ann['frame_idx'],ann['obj_to_world']):annotations[f,m]=T;valid[f,m]=True
    times=[f['timestamp_us'] for f in scene['frames']];targets=[states.records[f]['timestamp_us'] for f in frames]
    measured,measured_valid=resample_poses(times,annotations,valid,targets)
    ground=states.run['ground_registration'];g=ground['ground_model'];groundS=np.linalg.inv(g.get('world_from_scene',scene['frames'][0]['ego_to_world']));aa,bb,cc=g['coefficients_z_ax_by_c'];plane=groundS.T@np.array([-aa,-bb,1.,-cc])
    mount=np.array(ground['camera_mounts']['0']);intr=np.loadtxt(a.scene/'intrinsics/0.txt');extrinsic=np.loadtxt(a.scene/'extrinsics/0.txt');S=states.S;W,H=960,540
    K=np.array([[intr[0]/2,0,intr[2]/2],[0,intr[1]/2,intr[3]/2],[0,0,1.]])
    Ktensor=torch.tensor(K,dtype=torch.float32,device='cuda')
    variants=[('factory_rr','factory','real','real'),('factory_nr','factory','nexus','real'),('factory_rn','factory','real','nexus'),('factory_nn','factory','nexus','nexus'),('native_rr','native','real','real'),('native_nn','native','nexus','nexus'),('native_fitted','native','real','fitted')]
    exposure_records={};motion=[];gt_world={};camera_pairs={};original_dir=a.output/'original';original_dir.mkdir()
    with imageio.get_writer(str(original_dir/'camera_0.mp4'),fps=10,macro_block_size=2,ffmpeg_params=['-threads','2','-crf','18']) as writer:
        for j,f in enumerate(frames):
            row=scene['frames'][f];rawcam=next(i for i in row['images'] if i['camera_id']==0);E=np.array(rawcam['ego_to_world']);record=states.records[f];meta=record['scene_metadata'];assert rawcam['timestamp_us']==record['timestamp_us']
            ego=next(x for x in meta['actors'] if x['model_index']==-1);pred=recover_generated_ego(np.array(meta['camera_world']),mount,ego['yaw_world_rad'])
            centre_check=np.array(ego['center_world'])[:2]-1.461*np.array([np.cos(ego['yaw_world_rad']),np.sin(ego['yaw_world_rad'])]);assert np.linalg.norm(pred[:2,3]-centre_check)<.005
            with torch.no_grad():D=trainer.models['CamPose'](torch.eye(4,device='cuda'),torch.tensor(f*cams,device='cuda')).cpu().numpy().astype(float)
            for calibration,correction in [('factory',None),('native',D)]:camera_pairs[calibration,f]=paired_cameras(E,extrinsic,pred,correction)
            photo_path=a.scene/'images'/f'{f:03d}_0.jpg';photo_path=photo_path if photo_path.is_file() else Path(rawcam['source'])
            pixels=cv2.undistort(np.array(Image.open(photo_path).convert('RGB').resize((W,H),Image.Resampling.BILINEAR)),K,intr[4:]);imageio.imwrite(original_dir/f'{f:03d}_0.png',pixels);writer.append_data(pixels)
            gt_world[f]={}
            for m in range(M):
                key=str(m)
                if measured_valid[j,m]:
                    flat=key in states.flat;asset=np.array(states.flat[key] if flat else states.mounts[key]);gt_world[f][m]=register_measured_actor(measured[j,m],asset,plane,flat)
            yawE=np.arctan2(E[1,0],E[0,0]);dyaw=np.arctan2(np.sin(ego['yaw_world_rad']-yawE),np.cos(ego['yaw_world_rad']-yawE))
            motion.append({'frame':f,'timestamp_us':record['timestamp_us'],'measured_ego_world':E.tolist(),'nexus_ego_planar_world':pred.tolist(),'same_time_ego_xy_difference_m':float(np.linalg.norm(pred[:2,3]-E[:2,3])),'same_time_heading_difference_deg':float(np.rad2deg(dyaw)),'camera_exposure_offset_from_lidar_ms':(rawcam['timestamp_us']-row['timestamp_us'])/1000,'real_valid_modeled_actors':int(measured_valid[j].sum())})
            exposure_records[f]={'frame_idx':f,'images':[{'camera_id':0,'timestamp_us':record['timestamp_us'],'file':f'{f:03d}_0.png'}]}
    original_run={'status':'complete','kind':'real_undistorted_photographs','frames':[exposure_records[f] for f in frames]};(original_dir/'run.json').write_text(json.dumps(original_run,indent=2)+'\n')
    runs={};snapshots={};asset_versions_unchanged=False
    try:
        for name,calibration,camera_mode,actor_mode in variants:
            directory=a.output/name;directory.mkdir();run={'status':'started','renderer':'Omniscenegraph_3DGS','checkpoint':str(a.checkpoint.resolve()),'checkpoint_sha256':checkpoint_sha,'source_sha256':digest(a.source),'calibration':calibration,'camera_motion':camera_mode,'actor_motion':actor_mode,'fixed_exposure_frame':20,'frames':[]}
            with imageio.get_writer(str(directory/'camera_0.mp4'),fps=10,macro_block_size=2,ffmpeg_params=['-threads','2','-crf','18']) as writer:
                try:
                    for j,f in enumerate(frames):
                        realC,generatedC=camera_pairs[calibration,f];worldC=realC if camera_mode=='real' else generatedC;C=torch.tensor(S@worldC,dtype=torch.float32,device='cuda')
                        with torch.no_grad():
                            if actor_mode=='nexus':states.set_actors(rigid,f,f)
                            elif actor_mode=='fitted':
                                for key,value in originals.items():getattr(rigid,key)[f].copy_(value[f])
                            else:
                                rigid.instances_fv[f].copy_(torch.tensor(measured_valid[j],device='cuda'))
                                for m,T in gt_world[f].items():
                                    local=S@T;rigid.instances_trans[f,m].copy_(torch.tensor(local[:3,3],dtype=torch.float32,device='cuda'));rigid.instances_quats[f,m].copy_(torch.tensor(matrix_quaternion(local[:3,:3]),dtype=torch.float32,device='cuda'))
                            image,camera=camera_inputs(C,Ktensor,f,H,W,rigid.num_frames,cams);image['img_idx'].fill_(20*cams)
                            result=trainer(image,camera,novel_view=True);outputs=result[0] if isinstance(result,tuple) else result
                            rgb=(outputs['rgb'].clamp(0,1).cpu().numpy()*255).round().astype(np.uint8);imageio.imwrite(directory/f'{f:03d}_0.png',rgb);writer.append_data(rgb)
                            actualC=np.linalg.inv(S)@C.cpu().numpy();actors=[]
                            for m in range(M):
                                T=np.eye(4);T[:3,:3]=quaternion_matrix(rigid.instances_quats[f,m].detach().cpu().numpy());T[:3,3]=rigid.instances_trans[f,m].detach().cpu().numpy();T=np.linalg.inv(S)@T
                                actors.append({'model_index':m,'track_token':info[str(ids[m])]['id'],'valid':bool(rigid.instances_fv[f,m]),'pose_world':T.tolist(),'size_lwh':rigid.instances_size[m].detach().cpu().tolist()})
                            run['frames'].append({'frame_idx':f,'timestamp_us':states.records[f]['timestamp_us'],'file':f'{f:03d}_0.png','camera_world':actualC.tolist(),'intrinsics':K.tolist(),'actors':actors})
                            if f in [20,55,70,85,99]:
                                snapshots[name,f]={'camera_world':actualC,'actors':actors}
                                if f in [70,99]:imageio.imwrite(directory/f'{f:03d}_background.png',(outputs['Background_rgb'].clamp(0,1).cpu().numpy()*255).round().astype(np.uint8))
                        if j%10==0:print(f'TRAJECTORY_CONTROL {name} {j+1}/{len(frames)} frame={f}',flush=True)
                    run['status']='complete'
                finally:(directory/'run.json').write_text(json.dumps(run,indent=2)+'\n')
            runs[name]=run
        for calibration in ['factory','native']:
            if calibration=='factory':pairs=[('factory_rr','factory_nr','actors'),('factory_rn','factory_nn','actors'),('factory_rr','factory_rn','camera_world'),('factory_nr','factory_nn','camera_world')]
            else:pairs=[]
            for left,right,key in pairs:
                for a0,b0 in zip(runs[left]['frames'],runs[right]['frames']):assert a0[key]==b0[key],(left,right,key,a0['frame_idx'])
        asset_versions_unchanged=all(p._version==version for p,version in immutable);assert asset_versions_unchanged
    finally:
        with torch.no_grad():
            for key,value in originals.items():getattr(rigid,key).copy_(value)
        for key,value in originals.items():assert torch.equal(getattr(rigid,key),value)
    assert digest(a.checkpoint)==checkpoint_sha
    for row in motion:
        row['measured_camera_travel_m']=float(np.linalg.norm(np.array(row['measured_ego_world'])[:3,3]-np.array(motion[0]['measured_ego_world'])[:3,3]));row['nexus_camera_travel_m']=float(np.linalg.norm(np.array(row['nexus_ego_planar_world'])[:2,3]-np.array(motion[0]['nexus_ego_planar_world'])[:2,3]))
        R=np.array(motion[0]['measured_ego_world'])[:3,:3].T@np.array(row['measured_ego_world'])[:3,:3]
        row['measured_camera_rotation_change_deg']=float(np.rad2deg(np.arccos(np.clip((np.trace(R)-1)/2,-1,1))))
    audit={'status':'passed','frames_per_condition':len(frames),'conditions':[x[0] for x in variants],'checkpoint_sha256':checkpoint_sha,'checkpoint_file_unchanged':True,'static_dynamic_appearance_camera_parameters_unchanged':asset_versions_unchanged,'temporary_actor_states_restored_exactly':True,'same_actor_states_across_camera_controls':True,'same_camera_states_across_actor_controls':True,'native_correction_policy':'same per-frame right-multiplied learned correction applied to both measured and generated camera; disabled in factory controls','ego_height_tilt_policy':'measured sensor ego height/roll/pitch retained in both branches; Nexus changes XY/yaw only','measured_actor_policy':'annotation trajectories interpolated at camera exposure times, no gap filling; identical fixed road/asset registration as the Nexus branch','fitted_actor_policy':'additional native_fitted control uses original checkpoint per-frame poses, with no Nexus motion','reference_image_policy':'real photos resized/undistorted with identical K; only measured-camera variants have paired image ground truth','motion':motion}
    (a.output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');print('TRAJECTORY_CONTROL_COMPLETE '+str(a.output.resolve()),flush=True)


if __name__=='__main__':main()
