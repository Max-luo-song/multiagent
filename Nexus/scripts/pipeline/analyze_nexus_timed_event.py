"""Audit all paired seeds without converting conditional sensitivity into causality."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from nexus_3dgs.bundle import load_bundle
from nexus_3dgs.clearance import rear_axle_to_box
from nexus_3dgs.geometry import resample_poses
from nexus_3dgs.scene_projection import pair_separations
from nexus_3dgs import nuplan


def geometry(arr,meta,sizes,slot):
    ci=meta['current_index'];start=int(arr['timestamps_us'][ci])
    # More densely sample interpolated raw trajectories; not a continuous proof.
    times=np.arange(start+100000,int(arr['timestamps_us'][-1])+1,100000,dtype=np.int64)
    poses,valid=resample_poses(arr['timestamps_us'],arr['predicted_poses_world'],arr['valid'],times)
    xy=poses[...,:2,3];yaw=np.arctan2(poses[...,1,0],poses[...,0,0])
    boxes=np.concatenate([xy,yaw[...,None]],axis=-1)
    boxes[:,0]=rear_axle_to_box(boxes[:,0],1.461)
    sep,pairs=pair_separations(boxes,valid,sizes)
    hits=(sep<=0)&pairs
    ego_hits=hits[:,0].any(axis=-1)|hits[:,:,0].any(axis=-1)
    native=arr['predicted_poses_world'][ci:,:,:2,3]
    velocity=np.diff(native,axis=0)/.5
    acceleration=np.diff(velocity,axis=0)/.5
    vv=arr['valid'][ci:-1]&arr['valid'][ci+1:]
    av=vv[:-1]&vv[1:]
    speeds=np.linalg.norm(velocity,axis=-1);accels=np.linalg.norm(acceleration,axis=-1)
    close=np.linalg.norm(xy[:,1:]-xy[:,:1],axis=-1)
    nearby=[int(arr['nexus_slots'][i+1]) for i in range(close.shape[1])
            if i+1!=slot and ((close[:,i]<10)&valid[:,i+1]).any()]
    return {'scope':'80 future 0.1s interpolated samples, fixed logged dimensions; not exact mounted assets or continuous-time guarantee',
            'ego_overlapping_samples':int(ego_hits.sum()),'all_overlapping_pair_samples':int(hits.sum()),
            'ego_event_actor_overlapping_samples':int((sep[:,0,slot]<=0).sum()),
            'first_ego_overlap_s':float((times[ego_hits][0]-start)/1e6) if ego_hits.any() else None,
            'minimum_ego_event_actor_signed_gap_m':float(sep[:,0,slot].min()),
            'max_ego_speed_mps':float(speeds[:,0].max()),'max_ego_acceleration_mps2':float(accels[:,0].max()),
            'max_any_valid_speed_mps':float(speeds[vv].max()),'max_any_valid_acceleration_mps2':float(accels[av].max()),
            'ego_future_path_length_m':float(speeds[:,0].sum()*.5),
            'other_slots_with_center_within_10m_of_ego':nearby,
            'geometry_accepted':bool(not hits.any()),
            'overlap_samples':np.argwhere(hits).tolist()}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    run=json.loads((a.root/'raw/run.json').read_text());assert run['status']=='complete'
    first,meta=load_bundle(a.root/'raw'/run['pairs'][str(run['request']['seeds'][0])]['normal']['path'])
    slot=list(first['nexus_slots']).index(run['actor_slot'])
    with nuplan.connect(meta['database']) as db:
        anchor=nuplan.anchor_frame(db,meta['scenario_token']);objs={o['track_token']:o for o in nuplan.boxes(db,anchor['lidar_token'])}
        sizes=np.array([[5.176,2.297]]+[[objs[str(t)]['length'],objs[str(t)]['width']] for t in first['track_tokens'][1:]])
    result={'status':'audit_complete','technical_checks_passed':True,'seeds':{},'complete_interaction_chain_validated':False,
            'thresholds':{'response_position_difference_m':.5,'nearby_center_distance_m':10.,'ego_lateral_response_m':.5,'ego_speed_reduction_mps':1.},
            'notes':['All seeds are reported; render seed declared before sampling.',
                     'Responses are paired conditional differences, not proof of sequential causal mediation.',
                     'Joint generation can anticipate the event before its start.',
                     'Logged future existence masks are retained; no other actor future state values are supplied.',
                     'Raw tracks are not repaired or post-filtered; generated video may show failures.']}
    import torch
    for seed,pair in run['pairs'].items():
        bundles={n:load_bundle(a.root/'raw'/pair[n]['path']) for n in ('normal','event')}
        for key in ('timestamps_us','track_tokens','nexus_slots','valid','reference_poses_world'):
            np.testing.assert_array_equal(bundles['normal'][0][key],bundles['event'][0][key])
        hashes={pair[n]['initial_noise_sha256'] for n in bundles};assert len(hashes)==1
        checks={}
        for name in bundles:
            payload=torch.load(a.root/'raw'/pair[name]['path']/'prediction.pt',map_location='cpu',weights_only=False)
            pred=payload['sampled_tensor'];mask=payload['task_mask'].bool();inputs=payload['conditioned_scene_tensor']
            expected=torch.zeros_like(mask);expected[:,:,:5]=True
            condition=json.loads((a.root/'raw'/f'{name}_condition.json').read_text())
            expected[:,run['actor_slot'],condition['event_indices']]=True
            assert torch.equal(mask,expected) and not mask[:,0,5:].any()
            assert torch.equal(inputs[:,:,:5],payload['scene_tensor_gt'][:,:,:5])
            assert torch.equal(pred[mask],inputs[mask]) and (inputs[~mask]==0).all()
            checks[name]={'correct_local_mask':True,'exact_condition_preservation':True,'ego_and_other_futures_free':True,'hidden_future_values_zero':True}
        normal,nm=bundles['normal'];event,em=bundles['event'];ci=em['current_index']
        times=(event['timestamps_us'][ci+1:]-event['timestamps_us'][ci])/1e6
        delta=event['predicted_poses_world'][ci+1:,:,:2,3]-normal['predicted_poses_world'][ci+1:,:,:2,3]
        yaw=np.arctan2(event['reference_poses_world'][ci,0,1,0],event['reference_poses_world'][ci,0,0,0])
        lateral=delta[:,0]@np.array([-np.sin(yaw),np.cos(yaw)])
        geo={n:geometry(arr,manifest,sizes,slot) for n,(arr,manifest) in bundles.items()}
        after=times>=run['request']['start_s'];before=times<run['request']['start_s']
        third=[r for r in pair['responses'] if r['slot'] not in (0,run['actor_slot']) and r['max_position_difference_m']>.5]
        near=geo['event']['other_slots_with_center_within_10m_of_ego']
        ego_diff=np.linalg.norm(delta[:,0],axis=-1)
        speeds={n:np.linalg.norm(np.diff(ar['predicted_poses_world'][ci:,0,:2,3],axis=0),axis=-1)/.5
                for n,(ar,_) in bundles.items()}
        reduction=speeds['normal']-speeds['event']
        result['seeds'][seed]={'input_checks':checks,'geometry':geo,'responses':pair['responses'],
            'max_ego_position_difference_m':float(ego_diff.max()),
            'max_pre_event_ego_difference_m':float(ego_diff[before].max()) if before.any() else 0.,
            'max_after_event_ego_lateral_difference_m':float(np.abs(lateral[after]).max()),
            'ego_lateral_response_candidate':bool(np.abs(lateral[after]).max()>.5),
            'max_after_event_speed_reduction_vs_normal_mps':float(reduction[after].max()),
            'ego_braking_response_candidate':bool(reduction[after].max()>1.),
            'other_responding_slots':[r['slot'] for r in third],
            'nearby_other_responding_slots':[r['slot'] for r in third if r['slot'] in near],
            'event_constraints_executed':True,
            'complete_interaction_chain_validated':False,
            'acceptance':'failed_geometry' if not geo['event']['geometry_accepted'] else 'chain_not_verified'}
    renders=a.root/'renders'
    if renders.exists():
        result['render_geometry']={}
        for name in ('normal','event'):
            path=renders/name/'run.json'
            if not path.exists():continue
            render=json.loads(path.read_text());assert render['status']=='passed'
            expected=run['pairs'][str(run['request']['render_seed'])][name]
            assert render['source_prediction_sha256']==expected['prediction_sha256']
            hits=[f for f in render['frames'] if f['scene_metadata']['overlapping_actor_pairs']]
            ego=[f for f in hits if any(0 in p for p in f['scene_metadata']['overlapping_actor_pairs'])]
            target=[f for f in render['frames'] if f['scene_metadata']['ego_bus_sat_separation_m']<=0]
            result['render_geometry'][name]={'frames':len(render['frames']),'ego_overlapping_frames':len(ego),
                'ego_bus_overlapping_frames':len(target),'overlapping_pair_frame_samples':sum(len(f['scene_metadata']['overlapping_actor_pairs']) for f in hits),
                'minimum_ego_bus_gap_m':min(f['scene_metadata']['ego_bus_sat_separation_m'] for f in render['frames']),
                'checkpoint_unchanged':render['appearance_and_checkpoint_unchanged'],'raw_prediction_unmodified':not render['later_offline_nexus_trajectory_edits']}
    (a.root/'analysis.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({s:{k:v for k,v in d.items() if k not in ('geometry','responses','input_checks')} for s,d in result['seeds'].items()},indent=2))


if __name__=='__main__':main()
