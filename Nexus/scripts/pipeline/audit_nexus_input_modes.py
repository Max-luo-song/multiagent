"""Independently inspect serialized inputs, masks, model provenance and outputs."""
import csv
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def main():
    import numpy as np
    import torch
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT/'artifacts/nexus_3dgs/input_modes_20261026')
    root=parser.parse_args().root
    run=json.loads((root/'run.json').read_text());assert run['status']=='complete'
    protocol=run['protocol'];actor=protocol['actor_slot']
    runner=root/'source_snapshots/compare_nexus_input_modes.py'
    if not runner.exists():runner=ROOT/'scripts/pipeline/compare_nexus_input_modes.py'
    assert hashlib.sha256(runner.read_bytes()).hexdigest()==protocol['runner_sha256']
    assert hashlib.sha256((ROOT/'nexus_3dgs/input_mode_conditions.py').read_bytes()).hexdigest()==protocol['compiler_sha256']
    audit={'status':'passed','samples':{},'matched_history_controls':(root/'conditions/history_full_target_event.npz').exists(),'same_noise_by_seed':True,
           'model_parameters_unchanged':run['parameters_unchanged'],'complete_interaction_chain_validated':False}
    rows=[];grouped={}
    for key,item in run['samples'].items():
        path=root/key/'prediction.pt'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['prediction_sha256']
        p=torch.load(path,map_location='cpu',weights_only=False)
        from nexus_3dgs.bundle import decode
        values=p['conditioned_scene_tensor'];mask=p['task_mask'].bool();pred=p['sampled_tensor']
        mode=item['mode'];expected=torch.zeros_like(mask)
        if mode in ('bp_history','history_full_target','future_full_target'):expected[:,:,:5]=True
        expected[:,actor,:5]=True
        if mode!='bp_history':expected[:,actor,5:]=True
        if mode=='official_sg_fixed_ego':expected[:,0]=True
        assert torch.equal(mask,expected) and (values[~mask]==0).all()
        assert torch.equal(pred[mask],values[mask])
        assert bool(mask[:,0,5:].any())==(mode=='official_sg_fixed_ego')
        assert torch.equal(values[:,[i for i in range(values.shape[1]) if i!=actor],:5],p['scene_tensor_gt'][:,[i for i in range(values.shape[1]) if i!=actor],:5]) if mode in ('bp_history','history_full_target','future_full_target') else True
        assert item['noise_sha256']==run['noise_sha256_per_seed'][str(item['seed'])]
        if item['unknown_valid_history_tokens']:
            assert item['first_noise_time_unknown_history']==(0. if mode=='pure_sg_bad_schedule' else 1.)
        audit['samples'][key]={'payload_hash':True,'mask_correct':True,'conditions_exact':True,'unknown_inputs_zero':True,
                              'ego_future_free':mode!='official_sg_fixed_ego','schedule_matches_declared_control':True}
        q=item['quality'].copy();paired=run['pairs'].get(f'{mode}/seed_{item["seed"]:03d}',{})
        state=decode(pred[0].numpy());ref=decode(p['scene_tensor_gt'][0].numpy())
        previous_velocity=(state[0,4,:2]-state[0,3,:2])/.5
        next_velocity=(state[0,5,:2]-state[0,4,:2])/.5
        q.update(ego_velocity_channel_current_mps=float(np.linalg.norm(ref[0,4,4:6])),
                 ego_position_derived_recorded_history_speed_mps=float(np.linalg.norm(ref[0,4,:2]-ref[0,3,:2])/.5),
                 ego_first_future_interval_speed_mps=float(np.linalg.norm(next_velocity)),
                 ego_prediction_boundary_acceleration_mps2=float(np.linalg.norm(next_velocity-previous_velocity)/.5))
        heading=ref[0,4,2:4]/np.linalg.norm(ref[0,4,2:4])
        q.update(ego_first_future_longitudinal_speed_mps=float(next_velocity@heading),
                 ego_final_longitudinal_displacement_m=float((state[0,-1,:2]-state[0,4,:2])@heading))
        audit['samples'][key]['boundary_kinematics']=q
        row={'mode':mode,'seed':item['seed'],'variant':item['variant'],**q,
             **{k:paired.get(k) for k in ('ego_max_difference_m','ego_max_lateral_difference_m','third_max_difference_m')}}
        rows.append(row);grouped.setdefault(mode,[]).append(item)
    for variant in ('normal','event'):
        if (root/'conditions'/f'history_full_target_{variant}.npz').exists():
            with np.load(root/'conditions'/f'bp_history_{variant}.npz') as bp, np.load(root/'conditions'/f'history_full_target_{variant}.npz') as full:
                np.testing.assert_array_equal(bp['values'][:,:5],full['values'][:,:5])
    summaries={}
    for mode,items in grouped.items():
        events=[i for i in items if i['variant']=='event']
        pairs=[p for key,p in run['pairs'].items() if key.startswith(mode+'/')]
        summaries[mode]={'event_seeds':len(events),'event_exportable_bundles':sum(i['bundle_exported'] for i in events),
            'median_unknown_step_p95_m':float(np.median([i['quality']['unknown_adjacent_step_p95_m'] for i in events])),
            'max_unknown_step_m':max(i['quality']['unknown_adjacent_step_max_m'] for i in events),
            'max_ego_current_drift_m':max(i['quality']['ego_current_drift_from_reference_m'] for i in events),
            'event_ego_target_future_overlaps':[i['quality']['ego_target_future_overlap_samples'] for i in events],
            'event_ego_third_future_overlaps':[i['quality']['ego_third_future_overlap_samples'] for i in events],
            'event_generated_future_pair_overlaps':[i['quality']['future_overlaps_involving_generated_states'] for i in events],
            'ego_response_m':[p['ego_max_difference_m'] for p in pairs],
            'ego_lateral_response_m':[p['ego_max_lateral_difference_m'] for p in pairs],
            'third_response_m':[p['third_max_difference_m'] for p in pairs]}
        summaries[mode]['event_boundary_acceleration_mps2']=[audit['samples'][i['path']]['boundary_kinematics']['ego_prediction_boundary_acceleration_mps2'] for i in events]
    audit['mode_summaries']=summaries
    (root/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    header=sorted(set(k for r in rows for k in r))
    with (root/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,header);w.writeheader();w.writerows(rows)
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
