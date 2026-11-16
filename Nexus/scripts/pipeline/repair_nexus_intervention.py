"""Apply an explicitly labeled physical clearance filter to saved Nexus output.

Only ego timing along its generated path changes. Conditioned bus and all other
actors remain unchanged. This is a downstream filter, not a learned response.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from scipy.optimize import minimize
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO/'third_party/nuplan-devkit'))
from nexus_3dgs.bundle import load_bundle, save_bundle
from nexus_3dgs.geometry import resample_poses, pose
from nexus_3dgs.runtime import TrajectoryController
from nexus_3dgs.grounding import GroundedRegistration
from nexus_3dgs.clearance import box_separation, rear_axle_to_box


def xyyaw(values):
    return np.concatenate((values[..., :2, 3], np.arctan2(values[..., 1, 0], values[..., 0, 0])[..., None]), axis=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--ground-model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--clearance', type=float, default=.8)
    parser.add_argument('--limit-to-observed-travel', action='store_true',
        help='Conservative rendering envelope: cap ego progress at logged camera-platform travel in this reconstructed interval')
    args = parser.parse_args()
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    ego = get_pacifica_parameters()
    ego_size = np.array([ego.length, ego.width])
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)['models']['RigidNodes']
    rigid = SimpleNamespace(instance_ids=state['dataset_instance_ids'], instances_trans=state['instances_trans'],
                            instances_quats=state['instances_quats'], instances_fv=state['instances_fv'])
    rigid.num_frames, rigid.num_instances = rigid.instances_fv.shape
    scene = Path(json.loads(args.ground_model.read_text())['scene'])
    args.output.mkdir(parents=True, exist_ok=False)
    experiment = json.loads((args.experiment/'run.json').read_text())
    report = {'status':'started','source_experiment':str(args.experiment.resolve()),
              'conditioned_slot':experiment['conditioned_slot'],'method':'ego path timing optimization with physical OBB separation',
              'learned_nexus_weights_changed':False,'is_unmodified_nexus_output':False,
              'ego_physical_size_lw':ego_size.tolist(),'rear_axle_to_center_m':ego.rear_axle_to_center,
              'clearance_m':args.clearance,'optimization_dt_s':.05,'verification_dt_s':.01,
              'limit_to_observed_camera_platform_travel':args.limit_to_observed_travel,
              'scope':'sampled geometry only; no closed-loop or continuous-time safety guarantee','variants':{}}
    def save(): (args.output/'run.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        for name in ('bus_stop','bus_follow_logged'):
            bundle, metadata = load_bundle(args.experiment/name)
            current=metadata['current_index']; source_time=bundle['timestamps_us']
            start=int(source_time[current]); duration=(int(source_time[-1])-start)/1e6
            knots=(source_time[current:]-start)/1e6
            ctrl=TrajectoryController(args.experiment/name,scene,rigid)
            ground=GroundedRegistration(ctrl,args.ground_model,stabilize_all=True)
            source=xyyaw(bundle['predicted_poses_world'][current:,0])
            source[:,2]=np.unwrap(source[:,2])
            station=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(source[:,:2],axis=0),axis=-1))]
            observed_xy=bundle['reference_poses_world'][current:,0,:2,3]
            observed_travel=float(np.linalg.norm(np.diff(observed_xy,axis=0),axis=-1).sum())
            bounds=[(0,float(min(s,observed_travel) if args.limit_to_observed_travel else s)) for s in station[1:]]
            keep=np.r_[True,np.diff(station)>1e-8]
            if keep.sum()<2:raise ValueError('No generated ego path to time-scale')
            route_s,route=station[keep],source[keep]
            def route_pose(progress):
                return np.stack([np.interp(progress,route_s,route[:,i]) for i in range(3)],axis=-1)
            def obstacles(times):
                timestamps=start+np.rint(times*1e6).astype(np.int64)
                generated,gv=resample_poses(source_time,bundle['predicted_poses_world'],bundle['valid'],timestamps)
                result=np.zeros((len(times),rigid.num_instances,3)); valid=np.zeros(result.shape[:2],dtype=bool)
                for model_index in range(rigid.num_instances):
                    agent=next((a for a,m in ctrl.bindings.items() if m==model_index),None)
                    if agent is not None:
                        values,visible=generated[:,agent],gv[:,agent]
                    else:
                        refs=ground.reference_poses[model_index]
                        fi=np.array(sorted(refs))
                        ts=np.array([ctrl.scene['frames'][int(i)]['timestamp_us'] for i in fi])
                        values,v=resample_poses(ts,np.array([refs[int(i)] for i in fi])[:,None],np.ones((len(fi),1),bool),timestamps)
                        values,visible=values[:,0],v[:,0]
                    for j in np.flatnonzero(visible):
                        world=ground.world_from_scene @ ground.object_scene_pose(model_index,values[j])
                        result[j,model_index]=xyyaw(world)
                    valid[:,model_index]=visible
                return result,valid
            times=np.linspace(0,duration,int(round(duration/.05))+1)
            others,other_valid=obstacles(times)
            sizes=state['instances_size'].numpy()[:,:2]
            initial_velocity=float(np.linalg.norm(bundle['predicted_poses_world'][current,0,:2,3]-bundle['predicted_poses_world'][current-1,0,:2,3]) / metadata['sample_interval_s'])
            dt=metadata['sample_interval_s']
            def full(x):return np.r_[0.,x]
            def kinematics(x):
                velocity=np.diff(full(x))/dt
                acceleration=np.diff(np.r_[initial_velocity,velocity])/dt
                return velocity,acceleration
            def margins(x,times,others,valid):
                progress=np.interp(times,knots,full(x))
                centers=rear_axle_to_box(route_pose(progress),ego.rear_axle_to_center)
                sep=box_separation(centers[:,None,:],ego_size,others,sizes)
                return sep[valid]-args.clearance
            def constraints(x):
                velocity,accel=kinematics(x)
                return np.r_[velocity, 8.-velocity, accel+3., 2.-accel,
                             margins(x,times,others,other_valid)]
            def objective(x):
                _,accel=kinematics(x)
                return np.mean((x-station[1:])**2)+.05*np.mean(accel**2)
            initial=np.zeros(len(knots)-1)
            if constraints(initial).min() < -1e-6:
                raise ValueError('Even a stationary ego is infeasible for this path; a different route/planner is required')
            solution=minimize(objective,initial,method='SLSQP',bounds=bounds,
                              constraints=[{'type':'ineq','fun':constraints}],options={'maxiter':300,'ftol':1e-9})
            if not solution.success or constraints(solution.x).min() < -1e-5:
                raise ValueError('No feasible path timing: '+solution.message)
            dense=np.linspace(0,duration,int(round(duration/.01))+1)
            dense_others,dense_valid=obstacles(dense)
            pair_separation=box_separation(dense_others[:,:,None,:],sizes[:,None,:],
                                          dense_others[:,None,:,:],sizes[None,:,:])
            pairs=dense_valid[:,:,None] & dense_valid[:,None,:] & np.triu(np.ones((len(sizes),len(sizes)),bool),1)
            before=margins(station[1:],dense,dense_others,dense_valid)+args.clearance
            after=margins(solution.x,dense,dense_others,dense_valid)+args.clearance
            if after.min()<args.clearance-.01:
                raise ValueError('Independent 100 Hz verification failed')
            arrays={k:np.array(v,copy=True) for k,v in bundle.items()}
            new_route=route_pose(full(solution.x))
            for j,xyh in enumerate(new_route):
                previous=arrays['predicted_poses_world'][current+j,0].copy()
                replacement=pose(xyh[0],xyh[1],z=previous[2,3],yaw=xyh[2])
                # Keep the exported ego tilt; only XY and yaw follow the path.
                old_yaw=np.arctan2(previous[1,0],previous[0,0])
                replacement[:3,:3]=pose(0,0,yaw=xyh[2]-old_yaw)[:3,:3] @ previous[:3,:3]
                arrays['predicted_poses_world'][current+j,0]=replacement
            arrays['predicted_size_lw'][current:,0]=ego_size
            # Verify the interpolation the renderer will actually consume,
            # rather than only the optimizer's continuous route parameter.
            saved_ego,saved_valid=resample_poses(source_time,arrays['predicted_poses_world'][:,:1],
                arrays['valid'][:,:1],start+np.rint(dense*1e6).astype(np.int64))
            actual_centers=rear_axle_to_box(xyyaw(saved_ego[:,0]),ego.rear_axle_to_center)
            actual_separation=box_separation(actual_centers[:,None,:],ego_size,dense_others,sizes)
            after=actual_separation[dense_valid & saved_valid[:,0,None]]
            if after.min()<args.clearance-.01:
                raise ValueError('Saved trajectory interpolation failed physical clearance verification')
            np.testing.assert_array_equal(arrays['predicted_poses_world'][:current+1],bundle['predicted_poses_world'][:current+1])
            np.testing.assert_array_equal(arrays['predicted_poses_world'][:,1:],bundle['predicted_poses_world'][:,1:])
            velocity,accel=kinematics(solution.x)
            item={'status':'filtered','solver':solution.message,'source_bundle_sha256':metadata['bundle_sha256'],
                  'original_ego_progress_m':float(station[-1]),'filtered_ego_progress_m':float(solution.x[-1]),
                  'max_velocity_mps':float(velocity.max()),'acceleration_range_mps2':[float(accel.min()),float(accel.max())],
                  'before_physical_min_separation_m':float(before.min()),'after_physical_min_separation_m':float(after.min()),
                  'before_overlapping_box_samples_100hz':int((before<=0).sum()),'after_overlapping_box_samples_100hz':int((after<=0).sum()),
                  'ego_clearance_scope':'ego against all reconstructed actors; fixed physical dimensions and rear-axle offset',
                  'observed_camera_platform_travel_m':observed_travel,
                  'ego_travel_cap_m':observed_travel if args.limit_to_observed_travel else None,
                  'other_actor_overlapping_pair_samples_100hz':int(((pair_separation<=0)&pairs).sum()),
                  'history_and_all_other_actor_poses_unchanged':True}
            hit=((pair_separation<=0)&pairs)
            pair_details=[]
            for m,n in zip(*np.nonzero(hit.any(axis=0))):
                pair_details.append({'model_indices':[int(m),int(n)],
                    'track_tokens':[str(next(v['id'] for k,v in json.loads((scene/'instances/instances_info.json').read_text()).items() if int(k)==int(rigid.instance_ids[i]))) for i in (m,n)],
                    'nexus_agents':[next((int(a) for a,j in ctrl.bindings.items() if j==i),None) for i in (m,n)],
                    'overlap_samples':int(hit[:,m,n].sum()),
                    'first_time_s':float(dense[np.flatnonzero(hit[:,m,n])[0]]),
                    'initial_separation_m':float(pair_separation[0,m,n]) if dense_valid[0,m] and dense_valid[0,n] else None,
                    'minimum_separation_m':float(pair_separation[:,m,n][dense_valid[:,m]&dense_valid[:,n]].min())})
            item['other_actor_overlap_details']=pair_details
            save_bundle(args.output/name,arrays,{**metadata,'source_bundle_sha256':metadata['bundle_sha256'],
                        'trajectory_filter':item,'output_kind':'Nexus output plus downstream physical timing filter'})
            report['variants'][name]=item;save();print(name+' '+json.dumps(item),flush=True)
        report['status']='complete';save()
    except Exception as exc:
        report.update(status='failed',error=str(exc));save();raise


if __name__=='__main__':main()
