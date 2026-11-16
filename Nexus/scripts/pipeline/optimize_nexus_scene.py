"""Project free Nexus actors against all reconstructed traffic, without hiding actors."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO))
from nexus_3dgs.bundle import load_bundle,save_bundle
from nexus_3dgs.kinematics import repair_isolated_half_turns,repair_low_speed_tail_slips
from nexus_3dgs.scene_traffic import SceneTraffic
from nexus_3dgs.scene_projection import pair_separations,project_forward_offsets


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-experiment',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--ground-model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repair-tail-kinematics',action='store_true',help='Also project low-speed terminal turn/lateral outliers before collision projection')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    source=json.loads((a.source_experiment/'run.json').read_text())
    root={**source,'status':'started','source_filter_experiment':str(a.source_experiment.resolve()),
          'method':'offline full-scene collision projection of free Nexus actors; longitudinal offsets',
          'other_actor_paths_modified':True,'is_unmodified_nexus_output':False,'variants':{}}
    def save(): (a.output/'run.json').write_text(json.dumps(root,indent=2)+'\n')
    save()
    try:
        for name in ('bus_stop','bus_follow_logged'):
            arrays,meta=load_bundle(a.source_experiment/name);current=meta['current_index']
            times_us=arrays['timestamps_us'];start=int(times_us[current]);knots=(times_us[current:]-start)/1e6
            conditioned=int(np.flatnonzero(arrays['nexus_slots']==source['conditioned_slot'])[0])
            fixed=np.zeros_like(arrays['valid'],bool);fixed[:current+1]=True;fixed[:,[0,conditioned]]=True
            initial={k:np.array(v,copy=True) for k,v in arrays.items()}
            initial['predicted_poses_world'],heading_changes=repair_isolated_half_turns(initial['predicted_poses_world'],initial['valid'],fixed)
            tail_changes=[]
            if a.repair_tail_kinematics:
                initial['predicted_poses_world'],tail_changes=repair_low_speed_tail_slips(
                    initial['predicted_poses_world'],initial['valid'],fixed,times_us,current)
                for change in tail_changes:
                    change['nexus_slot']=int(arrays['nexus_slots'][change['agent_index']])
                    change['track_token']=str(arrays['track_tokens'][change['agent_index']])
            working=a.output/'working'/name
            save_bundle(working,initial,meta)
            traffic=SceneTraffic(a.checkpoint,working,a.ground_model)
            dense=np.linspace(0,knots[-1],int(round(knots[-1]/.05))+1)
            boxes,valid=traffic.sample(start+np.rint(dense*1e6).astype(np.int64))
            sep,pairs=pair_separations(boxes,valid,traffic.sizes)
            hit=(sep<=0)&pairs
            movable=set()
            for i,j in zip(*np.nonzero(hit.any(axis=0))):
                eligible=[k for k in (i,j) if traffic.agents.get(k) not in (None,0,conditioned)]
                if not eligible:raise ValueError(f'Overlapping scene pair {i},{j} has no free Nexus actor')
                movable.update(eligible)
            movable=sorted(movable)
            changes=[]
            if movable:
                agents=[traffic.agents[i] for i in movable]
                headings=[np.arctan2(initial['predicted_poses_world'][current,j,1,0],initial['predicted_poses_world'][current,j,0,0]) for j in agents]
                directions=np.stack((np.cos(headings),np.sin(headings)),axis=-1)
                old,_=traffic.sample(np.array([times_us[current-1],times_us[current]]))
                previous=(old[1,movable,:2]-old[0,movable,:2])/(meta['sample_interval_s'])
                fixed_knots=np.zeros((len(movable),len(knots)),bool);fixed_knots[:,0]=True
                offsets,solver=project_forward_offsets(dense,knots,boxes,valid,traffic.sizes,movable,directions,fixed_knots,previous_velocity=previous)
                for index,agent in enumerate(agents):
                    initial['predicted_poses_world'][current:,agent,:2,3]+=offsets[index,:,None]*directions[index]
                    changes.append({'agent_index':int(agent),'nexus_slot':int(arrays['nexus_slots'][agent]),
                                    'track_token':str(arrays['track_tokens'][agent]),'forward_offsets_m':offsets[index].tolist()})
            else:solver={'message':'No remaining scene overlap; no positional projection applied'}
            np.testing.assert_array_equal(initial['valid'],arrays['valid'])
            np.testing.assert_array_equal(initial['predicted_poses_world'][:current+1],arrays['predicted_poses_world'][:current+1])
            np.testing.assert_array_equal(initial['predicted_poses_world'][:,[0,conditioned]],arrays['predicted_poses_world'][:,[0,conditioned]])
            preserved={k:source['variants'][name][k] for k in ('filtered_ego_progress_m','ego_travel_cap_m','observed_camera_platform_travel_m')
                       if k in source['variants'][name]}
            item={**preserved,'source_ego_filter_audit':source['variants'][name],
                  'source_supported_bundle_sha256':meta['bundle_sha256'],
                  'heading_repairs':heading_changes,'free_actor_position_repairs':changes,'solver':solver,
                  'kinematic_tail_repairs':tail_changes,
                  'conditions_history_ego_and_visibility_unchanged':True,
                  'all_logged_context_trajectories_unchanged':True,
                  'method':'offline scene projection; no Nexus weight updates or actor deletion'}
            save_bundle(a.output/name,initial,{**meta,'scene_projection':item})
            final=SceneTraffic(a.checkpoint,a.output/name,a.ground_model)
            verify=np.linspace(0,knots[-1],int(round(knots[-1]/.01))+1)
            checked=start+np.rint(verify*1e6).astype(np.int64)
            before,bv=traffic.sample(checked);after,av=final.sample(checked)
            np.testing.assert_array_equal(bv,av)
            before_sep,before_pairs=pair_separations(before,bv,traffic.sizes)
            after_sep,after_pairs=pair_separations(after,av,traffic.sizes)
            overlap=int(((after_sep<=0)&after_pairs).sum())
            if overlap:raise ValueError(f'Saved interpolated scene still has {overlap} overlapping pair samples')
            item.update(before_all_actor_overlap_samples_100hz=int(((before_sep<=0)&before_pairs).sum()),
                        after_all_actor_overlap_samples_100hz=overlap,
                        after_overlapping_box_samples_100hz=0,
                        other_actor_overlapping_pair_samples_100hz=0,other_actor_overlap_details=[],
                        after_all_actor_min_separation_m=float(after_sep[after_pairs].min()),
                        after_physical_min_separation_m=float(after_sep[:,0,1:][av[:,0,None]&av[:,1:]].min()))
            # Persist the independently checked result in both manifest and report.
            save_bundle(a.output/name,initial,{**meta,'scene_projection':item})
            root['variants'][name]=item;save();print(name+' '+json.dumps(item),flush=True)
        root['status']='complete';save()
    except Exception as exc:root.update(status='failed',error=str(exc));save();raise


if __name__=='__main__':main()
