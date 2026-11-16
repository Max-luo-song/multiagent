"""Export candidate environment states; downstream policy exclusively owns ego.

This is a local integration contract, not a runnable WorldEngine asset or a
claim that a policy has been evaluated. Existing predictions remain unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from nexus_3dgs.bundle import load_bundle
from nexus_3dgs.scene_projection import pair_separations
from nexus_3dgs import nuplan


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def validate_environment(arrays):
    tokens=arrays['track_tokens'].astype(str)
    if len(set(tokens))!=len(tokens):raise ValueError('Duplicate actor token')
    if np.any(arrays['nexus_slots']==0) or any(t in ('ego','__ego__') for t in tokens):
        raise ValueError('Ego may not appear in environment-controlled traffic')
    poses=arrays['poses_world'];valid=arrays['valid']
    if poses.shape!=(*valid.shape,4,4):raise ValueError('Pose/validity dimensions disagree')
    if len(arrays['timestamps_us'])!=len(poses) or np.any(np.diff(arrays['timestamps_us'])<=0):raise ValueError('Invalid timestamps')
    if not np.isfinite(poses[valid]).all():raise ValueError('Non-finite valid pose')
    if not np.all(arrays['sizes_lwh']>0):raise ValueError('Nonpositive physical dimensions')
    rotation=poses[valid][:,:3,:3]
    if not np.allclose(rotation.transpose(0,2,1)@rotation,np.eye(3),atol=2e-5):raise ValueError('Invalid rotation')
    if not np.allclose(np.linalg.det(rotation),1.,atol=2e-5):raise ValueError('Improper rotation')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    run=json.loads((a.experiment/'raw_pair/run.json').read_text());assert run['status']=='complete'
    frozen=json.loads((a.experiment/'frozen_inputs.json').read_text())['inputs']
    scene=Path(frozen['registered_scene']['path']).parent;meta=json.loads((scene/'nexus_scene.json').read_text())
    plan=json.loads(Path(frozen['source_plan']['path']).read_text());scenario=next(s for s in plan['scenarios'] if s['token']==run['scenario_token'])
    hashes={key:sha(Path(frozen[key]['path'])) for key in ('visual_checkpoint','visual_config','registered_scene','asset_registration')}
    assert all(hashes[key]==frozen[key]['sha256'] for key in hashes)
    summaries={}
    for name in ('bus_stop','bus_follow_logged'):
        out=a.output/name;out.mkdir();(out/'diagnostics').mkdir()
        bundle,manifest=load_bundle(a.experiment/'raw_pair'/name);ci=manifest['current_index'];anchor=int(bundle['timestamps_us'][ci])
        assert sha(a.experiment/'raw_pair'/run['variants'][name]['prediction'])==run['variants'][name]['sha256']
        render_path=a.experiment/'renders'/name/'run.json';render=json.loads(render_path.read_text())
        assert render['bundle_sha256']==manifest['bundle_sha256'] and not render['later_offline_nexus_trajectory_edits']
        frames=render['frames'];first=frames[0]['scene_metadata']['actors'];actors=[x for x in first if x['model_index']>=0]
        ids=[x['model_index'] for x in actors];tokens=[x['track_token'] for x in actors]
        slots=np.array([x['nexus_slot'] if x['nexus_slot'] is not None else -1 for x in actors],dtype=np.int64)
        poses=np.broadcast_to(np.eye(4),(len(frames),len(actors),4,4)).copy();valid=np.zeros(poses.shape[:2],dtype=bool)
        for t,frame in enumerate(frames):
            byid={x['model_index']:x for x in frame['scene_metadata']['actors']}
            for m,index in enumerate(ids):
                actor=byid[index];assert actor['track_token']==tokens[m]
                poses[t,m]=actor['body_world'];valid[t,m]=actor['valid']
        arrays={'timestamps_us':np.array([f['timestamp_us'] for f in frames],dtype=np.int64),'poses_world':poses,
                'valid':valid,'sizes_lwh':np.array([x['size_lwh'] for x in actors]),'track_tokens':np.array(tokens),
                'model_indices':np.array(ids,dtype=np.int64),'nexus_slots':slots}
        validate_environment(arrays);np.savez_compressed(out/'environment_render_states.npz',**arrays)
        # Preserve the actual native samples separately; these are annotation
        # centres, while render states are mounted asset body centres.
        native={key:bundle[key].copy() for key in ('timestamps_us',)}
        non_ego=bundle['nexus_slots']!=0
        native.update(track_tokens=bundle['track_tokens'][non_ego],nexus_slots=bundle['nexus_slots'][non_ego],
                      predicted_poses_world=bundle['predicted_poses_world'][:,non_ego],valid=bundle['valid'][:,non_ego])
        np.savez_compressed(out/'native_generated_traffic.npz',**native)
        history=bundle['reference_poses_world'][:ci+1,0]
        np.savez_compressed(out/'ego_history.npz',timestamps_us=bundle['timestamps_us'][:ci+1],poses_world=history)
        np.savez_compressed(out/'diagnostics/nexus_ego_future.npz',timestamps_us=bundle['timestamps_us'][ci+1:],
                            poses_world=bundle['predicted_poses_world'][ci+1:,0],valid=bundle['valid'][ci+1:,0])
        velocity=(history[-1,:2,3]-history[-2,:2,3])/((int(bundle['timestamps_us'][ci])-int(bundle['timestamps_us'][ci-1]))/1e6)
        write(out/'ego_initial_state.json',{'timestamp_us':anchor,'pose_world':history[-1].tolist(),'position_reference':'rear_axle',
              'rear_axle_to_center_m':1.461,'size_lwh_m':[5.176,2.297,1.777],
              'velocity_world_xy_mps':velocity.tolist(),'velocity_method':'backward difference of logged history only',
              'future_controller':'external_policy','ego_future_not_supplied_to_policy':True})
        camera_records={}
        for c in meta['identity']['cameras']:
            intr=np.loadtxt(scene/'intrinsics'/f'{c}.txt');extr=np.loadtxt(scene/'extrinsics'/f'{c}.txt')
            camera_records[nuplan.CAMERAS[c]]={'camera_id':c,'intrinsic_raw_1920x1080':[[float(intr[0]),0.,float(intr[2])],[0.,float(intr[1]),float(intr[3])],[0.,0.,1.]],
                'distortion':intr[4:].tolist(),'camera_to_ego':extr.tolist(),'render_size_wh':[960,540]}
        write(out/'sensor_calibration.json',{'source':str(scene),'available_reconstruction_cameras':camera_records,
              'validated_final_camera_ids':[0],'camera_axes':'OpenCV right/down/forward','new_policy_pose_requires_new_render':True})
        light_rows=[]
        with nuplan.connect(run['database']) as db:
            for f in frames:
                lidar_token=meta['frames'][f['frame']]['lidar_token']
                for row in db.execute('SELECT lane_connector_id,status FROM traffic_light_status WHERE lidar_pc_token=?',(bytes.fromhex(lidar_token),)):
                    light_rows.append({'timestamp_us':f['timestamp_us'],'lidar_token':lidar_token,'lane_connector_id':row['lane_connector_id'],'status':row['status']})
        write(out/'traffic_light_reference.json',{'status':'exported_logged_states','records':light_rows,
              'policy':'recorded LiDAR sample matched to render frame; no invented light values','map_lane_linking_validated':False})
        write(out/'map_reference.json',{'dataset':'nuplan','database':run['database'],'map_name':scenario['map_name'],
              'private_map_root':str(ROOT.parent/'data/nuplan/maps'),'metric_polygons_exported':False,'worldengine_map_registration_validated':False,
              'upstream_feature_file':frozen['source_raw_features']['path'],'upstream_feature_sha256':frozen['source_raw_features']['sha256']})
        yaw=np.arctan2(poses[:,:,1,0],poses[:,:,0,0]);boxes=np.concatenate((poses[:,:,:2,3],yaw[...,None]),axis=-1)
        sep,pairs=pair_separations(boxes,valid,arrays['sizes_lwh'][:,:2]);events=np.argwhere((sep<=0)&pairs)
        overlap_events=[{'timestamp_us':int(arrays['timestamps_us'][t]),'track_tokens':[tokens[i],tokens[j]],
                         'nexus_slots':[int(slots[i]),int(slots[j])],'signed_sat_separation_m':float(sep[t,i,j])} for t,i,j in events]
        native_bus=list(native['nexus_slots']).index(run['conditioned_slot'])
        request={'risk_goal':'Evaluate whether an external ego policy safely reacts to the leading bus stop/go condition',
                 'scenario_token':run['scenario_token'],'condition':name,'conditioned_track_token':str(native['track_tokens'][native_bus]),
                 'condition_source':'full bus future supplied as a condition, not free model output','horizon_seconds':8.,
                 'ego_owner':'downstream policy','other_generated_traffic':'saved open-loop Nexus samples; not yet reactive to a new ego policy'}
        write(out/'scene_request.json',request)
        blockers=[
            'WorldEngine renderer has no registered Omniscenegraph/v9 checkpoint adapter; preserve SH, sky and appearance basis when adding one.',
            'Configured VADv2 uses 8 cameras; only 4 participated in reconstruction and only camera 0 has final video validation.',
            'Metric map polygons, lane links, route/navigation and map-to-world integration are not yet validated.',
            'Native Nexus has 0.5s samples; render exposures are near 0.1s and irregular. Select and validate the policy clock/history window.',
            'The inspected Agent runner permits 40 or 60 frames; current demonstration has 80 render frames.',
            'Generated NPC paths are open-loop and do not recondition on a new policy ego history.',
        ]
        if len(events):blockers.append('Non-ego traffic already contains vehicle-box overlaps; scene validity must be resolved before policy-failure attribution.')
        files=[p for p in out.rglob('*') if p.is_file()]
        case={'schema_version':1,'artifact_type':'candidate_agent_environment_contract','status':'exported_for_integration_audit',
              'variant':name,'coordinate_frame':'nuplan global world, metres, seconds; rotations in SE(3)',
              'pose_conventions':{'environment_render_states':'mounted asset body centres','native_generated_traffic':'Nexus annotation-centre states',
                                  'ego_history':'logged rear-axle poses; no future control commands'},
              'ego_controller':'external_policy','policy_input_files':['environment_render_states.npz','ego_history.npz','ego_initial_state.json','sensor_calibration.json','map_reference.json','traffic_light_reference.json','scene_request.json'],
              'diagnostic_only_files':['diagnostics/nexus_ego_future.npz'],'source_bundle_sha256':manifest['bundle_sha256'],
              'visual_asset':{'checkpoint':frozen['visual_checkpoint'],'config':frozen['visual_config'],'registration':frozen['asset_registration'],
                              'scene':str(scene),'worldengine_native_asset_ready':False},
              'worldengine_ready':False,'closed_loop_executed':False,'policy_training_data_ready':False,
              'scene_environment_overlap_pair_frames':len(events),'scene_validity':'requires_review' if len(events) else 'geometry_samples_clear_only',
              'blocked_interfaces':blockers,'source_render_metadata':{'path':str(render_path),'sha256':sha(render_path)},
              'files_sha256':{str(f.relative_to(out)):sha(f) for f in files}}
        write(out/'case_manifest.json',case)
        audit={'status':'passed_export_checks','ego_excluded_from_environment':True,'ego_history_ends_at_current':True,
               'native_non_ego_samples_preserved':True,'render_body_states_copied_exactly':True,'source_prediction_unchanged':True,
               'actor_count':len(actors),'native_generated_non_ego_count':int(non_ego.sum()),'render_frames':len(frames),
               'nominal_render_interval_s':float(np.median(np.diff(arrays['timestamps_us'])))/1e6,
               'minimum_interval_s':float(np.diff(arrays['timestamps_us']).min())/1e6,'maximum_interval_s':float(np.diff(arrays['timestamps_us']).max())/1e6,
               'non_ego_overlap_events':overlap_events,'worldengine_ready':False,'closed_loop_executed':False}
        write(out/'export_audit.json',audit);summaries[name]={key:audit[key] for key in ('actor_count','native_generated_non_ego_count','render_frames','worldengine_ready')}
        summaries[name]['non_ego_overlap_pair_frames']=len(events)
        print('AGENT_CANDIDATE_EXPORTED',name,json.dumps(summaries[name]),flush=True)
    common=np.load(a.output/'bus_stop/ego_history.npz');other=np.load(a.output/'bus_follow_logged/ego_history.npz')
    for key in common.files:np.testing.assert_array_equal(common[key],other[key])
    write(a.output/'handoff_summary.json',{'status':'candidate_exports_complete','ego_owned_by_external_policy':True,
          'cases':summaries,'worldengine_integration_completed':False,'closed_loop_policy_evaluation_completed':False})


if __name__=='__main__':main()
