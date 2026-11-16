"""Replay saved predictions on CPU with explicit collision counting rules.

Native pair-count metrics are reconstructed as a consistency check. Independent
vehicle-frame and per-trajectory overlap counts are then compared with logged
ground truth. This is not a continuous-time physical crash detector.
"""

import argparse
import io
import hashlib
import json
from pathlib import Path
import statistics
import time

from run_nuplan_metrics import json_write
from run_repro_demo import file_sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run',type=Path,required=True)
    parser.add_argument('--recipe-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    reference,recipe,output=args.reference_run.resolve(),args.recipe_run.resolve(),args.output.resolve()
    ref=json.loads((reference/'run.json').read_text())
    control=json.loads((recipe/'run.json').read_text())
    assert ref['status']==control['status']=='complete'
    assert ref['plan_sha256']==control['plan_sha256']
    assert ref['scenarios']==control['scenarios'] and ref['seeds']==control['seeds']==[0,1,2]
    assert len(ref['records'])==384 and len(control['records'])==768
    samples,seeds=ref['scenarios'],ref['seeds']
    from collision_audit import FEATURE_SOURCE
    fingerprint=dict(reference_manifest_sha256=file_sha256(reference/'run.json'),
                     recipe_manifest_sha256=file_sha256(recipe/'run.json'),
                     runner_sha256=file_sha256(Path(__file__)),
                     engine_sha256=file_sha256(Path(__file__).with_name('collision_audit.py')),
                     feature_source_sha256=file_sha256(FEATURE_SOURCE))
    assert output not in (reference,recipe)
    output.mkdir(parents=True,exist_ok=True)
    (output/'scenes').mkdir(exist_ok=True)
    state_path=output/'run.json'
    if state_path.exists():
        state=json.loads(state_path.read_text())
        assert all(state[k]==v for k,v in fingerprint.items()), 'Audit source or input changed'
        if state['status']=='complete':
            print('SAFETY_AUDIT_ALREADY_COMPLETE\nAUDIT_DIR='+str(output),flush=True)
            return
    else:
        state=dict(fingerprint,status='started',scope='common64_discrete_rectangle_overlap_audit',
                   reference_run=str(reference),recipe_run=str(recipe),records={},training_updates=0,
                   gpu_inference=False,agent_slots=64,future_interval_seconds=.5,
                   notes=[
                       'All-agent geometry includes ego; non-ego-only geometry ignores all ego interactions.',
                       'Boundary contact counts as overlap. No temporal interpolation or collision severity is evaluated.',
                       'Vehicle-frame fraction and ever-overlapping-vehicle fraction have different denominators.',
                       'Generated-frame fraction omits each provided terminal goal from the numerator and denominator; '
                       'provided states remain obstacles for other agents.',
                       'Centers, headings and dimensions follow the saved scene-tensor representation. '
                       'Ground-truth overlaps are annotation geometry, not independently verified physical accidents.',
                       'This audit does not declare exact alignment with every aggregation detail in the paper.'])
    state['status']='running';json_write(state_path,state)
    print('AUDIT_DIR='+str(output)+'\nCPU_ONLY: 1152 saved prediction files, 128 ground-truth scenes',flush=True)
    try:
        import numpy as np
        import torch
        from collision_audit import scene_scaling,decode_scene,collision_counts,add_counts,finalize_counts
        torch.set_num_threads(2)
        scaling=scene_scaling()
        aggregate={}
        started=time.perf_counter()
        def load_prediction(directory,record):
            blob=(directory/record['prediction_file']).read_bytes()
            assert hashlib.sha256(blob).hexdigest()==record['sha256'], record['prediction_file']
            return torch.load(io.BytesIO(blob),map_location='cpu',weights_only=False)
        def decoded(payload):
            assert payload['valid_mask'].shape[0]==1
            if payload['valid_mask'].shape[1]>64:
                assert not payload['valid_mask'][:,64:,5:].bool().any(), 'Reference truncation removes valid future agents'
            valid=payload['valid_mask'][0,:64].bool().numpy().copy()
            gt=decode_scene(payload['scene_tensor_gt'][0,:64].numpy(),scaling)
            return gt,valid
        for s in samples:
            key=str(s['index']);cache=output/'scenes'/f'{s["index"]:03d}.json'
            if key in state['records']:
                assert file_sha256(cache)==state['records'][key]['sha256']
                bundle=json.loads(cache.read_text())
            else:
                bundle=dict(index=s['index'],token=s['token'],city=s['city'],records=[])
                gt_reference=None;valid_reference=None
                for seed in seeds:
                    baseline=load_prediction(reference,ref['records'][f'{s["index"]}:{seed}'])
                    gt,valid=decoded(baseline)
                    if gt_reference is None:
                        gt_reference,valid_reference=gt,valid
                        bundle['records'].append(dict(seed='reference',branch='recorded_gt',task='gt',
                            counts=collision_counts(gt,valid)))
                    else:
                        assert np.array_equal(valid,valid_reference)
                        assert np.array_equal(gt[:,:,[0,1,2,3,6,7]],gt_reference[:,:,[0,1,2,3,6,7]])
                    branch_payloads=[('saved_test_recipe_common64',baseline)]
                    for branch in control['branches']:
                        payload=load_prediction(recipe,control['records'][f'{branch}:{s["index"]}:{seed}'])
                        other_gt,other_valid=decoded(payload)
                        assert np.array_equal(valid,other_valid)
                        assert np.allclose(other_gt[:,:,[0,1,2,3,6,7]],gt_reference[:,:,[0,1,2,3,6,7]],atol=1e-5,rtol=1e-6)
                        branch_payloads.append((branch,payload))
                    for branch,payload in branch_payloads:
                        for task,prediction_key in (('bp','behavior_pred'),('goal','intent_conditioned_behavior_pred')):
                            states=decode_scene(payload[prediction_key][0,:64].numpy(),scaling)
                            bundle['records'].append(dict(seed=str(seed),branch=branch,task=task,
                                counts=collision_counts(states,valid,goal_conditioned=task=='goal')))
                assert len(bundle['records'])==19
                json_write(cache,bundle)
                state['records'][key]=dict(file=str(cache.relative_to(output)),sha256=file_sha256(cache),token=s['token'])
                json_write(state_path,state)
            for record in bundle['records']:
                for city in ('all',s['city']):
                    dest=aggregate.setdefault(record['seed'],{}).setdefault(city,{}).setdefault(record['branch'],{}).setdefault(record['task'],{})
                    add_counts(dest,record['counts'])
            if (s['index']+1)%8==0:
                print(f'SAFETY_AUDIT_PROGRESS scenes={s["index"]+1}/128 elapsed_seconds={time.perf_counter()-started:.1f}',flush=True)
        result={seed:{city:{branch:{task:finalize_counts(counts) for task,counts in tasks.items()}
                for branch,tasks in branches.items()} for city,branches in cities.items()} for seed,cities in aggregate.items()}
        json_write(output/'metrics_by_seed_city.json',result)
        expected_reference=json.loads((reference/'official_metrics_by_seed.json').read_text())
        expected_control=json.loads((recipe/'official_metrics_by_recipe_seed.json').read_text())
        comparisons=[]
        for seed in seeds:
            for branch,tasks in result[str(seed)]['all'].items():
                expected=expected_reference[str(seed)] if branch=='saved_test_recipe_common64' else expected_control[branch][str(seed)]
                for task,values in tasks.items():
                    name='collision_rate_bp' if task=='bp' else 'collision_rate_gp'
                    actual=values['all_agents']['upstream_integer_horizon_pair_ratio']
                    target=expected[name]['collision@avg']
                    record=dict(seed=seed,branch=branch,task=task,computed=actual,expected=target,absolute_error=abs(actual-target))
                    comparisons.append(record)
        json_write(output/'native_replay_check.json',comparisons)
        assert max(v['absolute_error'] for v in comparisons)<2e-7, 'Native collision replay differs; inspect native_replay_check.json'
        summary=dict(scope=state['scope'],seeds=seeds,scenarios=128,by_branch={},
                     native_pair_metrics_replayed=True,native_replay_comparisons=len(comparisons),
                     max_native_replay_error=max(v['absolute_error'] for v in comparisons),
                     definitions=state['notes'])
        for branch in ['recorded_gt','saved_test_recipe_common64']+control['branches']:
            summary['by_branch'][branch]={}
            tasks=['gt'] if branch=='recorded_gt' else ['bp','goal']
            selected=['reference'] if branch=='recorded_gt' else [str(seed) for seed in seeds]
            for task in tasks:
                summary['by_branch'][branch][task]={}
                for scope in ('all_agents','non_ego_only'):
                    dest={}
                    for metric in ('vehicle_frame_collision_fraction','vehicle_any_future_collision_fraction',
                                   'generated_vehicle_frame_collision_fraction','upstream_integer_horizon_pair_ratio'):
                        vals=[result[seed]['all'][branch][task][scope][metric]*100 for seed in selected]
                        dest[metric+'_percent']=dict(mean=statistics.mean(vals),seed_sample_sd=statistics.stdev(vals) if len(vals)>1 else None,by_seed=vals)
                    dest['missed_pair_events_10m_by_seed']=[result[seed]['all'][branch][task][scope]['pair_events_missed_by_10m'] for seed in selected]
                    summary['by_branch'][branch][task][scope]=dest
        json_write(output/'summary.json',summary)
        state.update(status='complete',evaluated_scenarios=128,evaluated_saved_prediction_files=1152,
                     evaluated_model_task_outputs=2304,ground_truth_scenes=128,native_pair_replay_verified=True)
        state.pop('error',None);json_write(state_path,state)
        for branch,tasks in summary['by_branch'].items():
            for task,scopes in tasks.items():
                counters=scopes['all_agents']
                print(branch+' '+task+' | vehicle_frame_overlap_percent='+str(counters['vehicle_frame_collision_fraction_percent']['mean'])+
                      ' | vehicle_any_overlap_percent='+str(counters['vehicle_any_future_collision_fraction_percent']['mean']),flush=True)
        print('NUPLAN_SAFETY_AUDIT_OK\nAUDIT_DIR='+str(output),flush=True)
    except Exception as exc:
        state.update(status='failed_resumable',error=repr(exc));json_write(state_path,state)
        raise


if __name__=='__main__':
    main()
