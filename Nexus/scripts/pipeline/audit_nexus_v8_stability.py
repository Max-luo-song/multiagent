"""Check actual v8 renders, camera invariance and physical box intersections."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,required=True);a=p.parse_args();v=a.experiment;old=v.parent/'optimization_v7'
    import cv2,imageio.v2 as imageio,numpy as np
    from nexus_3dgs.bundle import load_bundle
    from nexus_3dgs.geometry import resample_poses
    from nexus_3dgs.trajectory_control import register_measured_actor
    from nexus_3dgs.clearance import box_separation,rear_axle_to_box
    cv2.setNumThreads(4)
    before=json.loads((old/'raw_nexus_assets_prediction/run.json').read_text());after=json.loads((v/'nexus_stable/run.json').read_text());heading=json.loads((v/'heading_only/run.json').read_text())
    assert all(x['status']=='passed' for x in [before,after,heading])
    camera_error=max(float(np.abs(np.array(x['camera_world'])-np.array(y['camera_world'])).max()) for x,y in zip(before['frames'],after['frames']));assert camera_error==0
    baseline=json.loads((old/'assets_motion_validation/validation.json').read_text());current=json.loads((v/'observed_validation/validation.json').read_text());assert [r['frame'] for r in current['rows']]==list(range(20,100))
    early=[];static_differences=[];same_pose_differences=[]
    for row in current['rows']:
        f=row['frame'];A=imageio.imread(old/'raw_nexus_assets_prediction'/f'{f:03d}_0.png');B=imageio.imread(v/'heading_only'/f'{f:03d}_0.png')
        if f<90:
            assert np.array_equal(A,B),f;early.append(f)
        A=imageio.imread(old/'assets_motion_validation/new'/f'{f:03d}_0.png');B=imageio.imread(v/'observed_validation/new'/f'{f:03d}_0.png');C=imageio.imread(v/'observed_validation/old_same_camera'/f'{f:03d}_0.png')
        static=(imageio.imread(old/'assets_motion_validation'/f'{f:03d}_static_mask.png')>0)&(imageio.imread(v/'observed_validation'/f'{f:03d}_static_mask.png')>0)
        static_differences.append(float(abs(A.astype(float)-B)[static].max()));same_pose_differences.append(float(abs(B.astype(float)-C).max()))
    assert max(static_differences)==0
    assert max(same_pose_differences)<=1
    original,manifest=load_bundle(old.parent/'real_vs_nexus/inputs/goal');refined,new_manifest=load_bundle(v/'stability_candidates/heading_and_bus_path');bus=next(i for i,t in enumerate(original['track_tokens']) if str(t)=='441830199c6752a5');black=next(i for i,t in enumerate(original['track_tokens']) if str(t)=='2a3e0f8d92fa5791');k=manifest['current_index'];P=original['predicted_poses_world'];Q=refined['predicted_poses_world']
    np.testing.assert_array_equal(Q[:k+1],P[:k+1]);np.testing.assert_array_equal(Q[-1],P[-1]);np.testing.assert_array_equal(Q[:,0],P[:,0]);others=np.arange(len(original['track_tokens']));others=others[(others!=bus)&(others!=black)];np.testing.assert_array_equal(Q[:,others],P[:,others]);np.testing.assert_array_equal(Q[:,black,:3,3],P[:,black,:3,3])
    # Compare the actual rendered physical body boxes, including assets not
    # forecast by Nexus, using the same road/mounting policy as the renderer.
    scene=next((old/'registered_data').iterdir());meta=json.loads((scene/'nexus_scene.json').read_text());targets=[next(c['timestamp_us'] for c in meta['frames'][f]['images'] if c['camera_id']==0) for f in range(20,100)]
    oldp,oldvalid=resample_poses(original['timestamps_us'],P,original['valid'],targets);newp,newvalid=resample_poses(refined['timestamps_us'],Q,refined['valid'],targets);np.testing.assert_array_equal(oldvalid,newvalid)
    source=json.loads((old.parent/'optimization_v6/source/coherent_rear_go/run.json').read_text());bodies=[x for x in source['frames'][0]['images'][0]['scene_metadata']['actors'] if x['model_index']>=0];bytoken={x['track_token']:x['model_index'] for x in bodies};M=len(bodies);sizes=np.array([next(x['size_lwh'][:2] for x in bodies if x['model_index']==i) for i in range(M)]+[[5.176,2.297]])
    registration=source['ground_registration'];ground=registration['ground_model'];original_meta=json.loads((old.parent/'real_vs_nexus/inputs/scene/nexus_scene.json').read_text());gs=np.linalg.inv(ground.get('world_from_scene',original_meta['frames'][0]['ego_to_world']));aa,bb,cc=ground['coefficients_z_ax_by_c'];plane=gs.T@np.array([-aa,-bb,1.,-cc]);mounts=registration['object_mounts'];flat=registration.get('flat_track_mounts',{})
    oldworld=np.array([r['actors_world'] for r in baseline['actual_camera_and_actor_poses']]);newworld=np.array([r['actors_world'] for r in current['actual_camera_and_actor_poses']]);valid_bodies=json.loads((scene/'instances/frame_instances.json').read_text());oldboxes=np.zeros((80,M+1,3));newboxes=np.zeros_like(oldboxes);visible=np.ones((80,M+1),bool)
    info=json.loads((scene/'instances/instances_info.json').read_text());dataset_by_token={x['id']:int(i) for i,x in info.items()}
    for j,f in enumerate(range(20,100)):
        for m in range(M):
            token=next(x['track_token'] for x in bodies if x['model_index']==m);visible[j,m]=dataset_by_token[token] in valid_bodies[str(f)]
        for states,pred,boxes in [(oldworld,oldp,oldboxes),(newworld,newp,newboxes)]:
            world=states[j].copy()
            for slot,token in enumerate(original['track_tokens']):
                if not slot or str(token) not in bytoken:continue
                m=bytoken[str(token)];visible[j,m]=oldvalid[j,slot]
                if not oldvalid[j,slot]:continue
                key=str(m);isflat=key in flat;world[m]=register_measured_actor(pred[j,slot],np.array(flat[key] if isflat else mounts[key]),plane,isflat)
            boxes[j,:M,:2]=world[:,:2,3];boxes[j,:M,2]=np.arctan2(world[:,1,0],world[:,0,0]);ego=pred[j,0];boxes[j,M]=rear_axle_to_box(np.array([*ego[:2,3],np.arctan2(ego[1,0],ego[0,0])]),1.461)
    newly_overlapping=[];bus_model=bytoken['441830199c6752a5'];bus_clearance=[]
    for i in range(M+1):
        for j in range(i+1,M+1):
            mask=visible[:,i]&visible[:,j];d0=box_separation(oldboxes[:,i],sizes[i],oldboxes[:,j],sizes[j]);d1=box_separation(newboxes[:,i],sizes[i],newboxes[:,j],sizes[j])
            for f in np.flatnonzero(mask&(d0>1e-5)&(d1<=0)):newly_overlapping.append({'frame':int(f+20),'bodies':[i,j],'before_m':float(d0[f]),'after_m':float(d1[f])})
            if i==bus_model and j==M:bus_clearance=[float(d0[mask].min()),float(d1[mask].min())]
    summary={}
    for label,rows in [('stationary',[r for r in current['rows'] if r['frame']<=62]),('moving',[r for r in current['rows'] if r['frame']>=63]),('all',current['rows'])]:summary[label]={name:float(np.mean([r['metrics'][name]['full_psnr'] for r in rows])) for name in ['old_native','new']}
    report={'status':'passed' if not newly_overlapping else 'requires_review','all_80_nexus_cameras_bit_exact':True,'maximum_camera_matrix_difference':camera_error,'black_heading_only_early_frames_bit_exact':len(early),'background_pixels_outside_common_vehicle_masks_max_difference_8bit':max(static_differences),'same_physical_pose_render_max_difference_8bit':max(same_pose_differences),'given_history_goal_ego_other_actor_positions_preserved':True,'new_physical_box_overlap_events':newly_overlapping,'bus_ego_minimum_sat_separation_before_after_m':bus_clearance,'observed_view_full_psnr':summary,'scope':'single scene, front camera, 80 frames; physical boxes are a collision regression check, not a complete safety guarantee'}
    (v/'stability_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
