"""Verify independent predictions and replay position, overlap and road metrics.

Uses the previously validated CPU geometry engines. Common-agent diagnostics
use slots 0..63; native replay retains every slot of each published input.
Writes numeric experiment artifacts only.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import statistics
import time

from run_nuplan_metrics import json_write
from run_nuplan_multicity import audit_displacement
from run_repro_demo import CHECKPOINT_SHA256, file_sha256


def stats(values):
    return dict(mean=statistics.mean(values), by_seed=values,
                seed_sample_sd=statistics.stdev(values) if len(values)>1 else None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run,output=args.run_dir.resolve(),args.output.resolve()
    manifest=json.loads((run/'run.json').read_text())
    assert manifest['status']=='complete' and manifest['evaluated_scenarios']==256
    assert len(manifest['records'])==1536 and manifest['task_outputs']==3072
    assert manifest['model_parameter_versions_unchanged'] and manifest['training_updates']==0
    assert manifest['checkpoint_sha256']==CHECKPOINT_SHA256
    plan=json.loads((run/'frozen_plan.json').read_text())
    assert file_sha256(run/'frozen_plan.json')==manifest['plan_sha256']
    assert not ({s['trip'] for s in manifest['scenarios']}&set(plan['excluded_previous_trips']))
    assert len({s['trip'] for s in manifest['scenarios']})==32
    assert manifest['seeds']==[0,1,2]
    assert manifest['recipes']==['released_test_recipe','causal_training_recipe']
    assert output!=run
    fingerprint=dict(source_manifest_sha256=file_sha256(run/'run.json'),
        runner_sha256=file_sha256(Path(__file__)),
        engine_sha256={p:file_sha256(Path(__file__).with_name(p)) for p in
            ('collision_audit.py','road_kinematics_audit.py','run_nuplan_multicity.py')},
        upstream_source_sha256={p:file_sha256(Path(p)) for p in (
            'nuplan_extent/planning/training/preprocessing/features/scene_tensor.py',
            'nuplan_extent/planning/training/modeling/aggregated_metrics/traj_metric.py',
            'nuplan_extent/planning/training/modeling/aggregated_metrics/collision_rate.py')},
        native_metrics_sha256=file_sha256(run/'official_metrics_by_recipe_seed.json'),
        displacement_sha256=file_sha256(run/'common64_displacement_by_recipe_city_seed.json'))
    output.mkdir(parents=True,exist_ok=True);(output/'scenes').mkdir(exist_ok=True)
    state_path=output/'run.json'
    if state_path.exists():
        state=json.loads(state_path.read_text())
        assert all(state[k]==v for k,v in fingerprint.items()),'Audit source or input changed'
        if state['status']=='complete':
            print('INDEPENDENT_AUDIT_ALREADY_COMPLETE\nAUDIT_DIR='+str(output),flush=True);return
    else:
        state=dict(fingerprint,status='started',records={},gpu_inference=False,training_updates=0,
            common_agent_slots=64,definitions=[
                'Position errors use common first 64 slots, exclude ego/history/invalid points, '
                'and omit every provided last valid goal in the goal-conditioned task.',
                'Common overlap counts use 64 slots and all 16 discrete future frames at 0.5 s. '
                'All-agent geometry includes ego; non-ego-only ignores all ego interactions. '
                'Contact counts as overlap; recorded overlaps are annotation geometry, not verified crashes.',
                'Native metric replay retains all 129 or 64 input slots, each recipe map, '
                'and the source formulas and averaging. Native pair rates are not vehicle collision fractions.',
                'Common road geometry uses the released-input LANE/LANE_CONNECTOR map for both recipes. '
                'Eligibility requires recorded history within 2.75 m of a valid centerline segment '
                'and at least one valid future frame. Segments never bridge masked vertices.',
                'Nearest centerline deviation is not assigned-route or drivable-polygon compliance.',
                'Physical acceleration uses backward 0.5 s position differences projected onto current '
                'heading; jerk differentiates those components. Only complete valid stencils contribute. '
                'Units are m/s^2 and m/s^3. Provided goals remain in road and comfort diagnostics.',
                'Seed standard deviation describes three model draws, not population uncertainty. '
                'These 256 new scenes are an independent subset, not the complete paper benchmark.'])
    state['status']='running';json_write(state_path,state)
    print('CPU_ONLY: verify 1536 files; replay native metrics and audit common-agent geometry',flush=True)
    try:
        import numpy as np
        import torch
        from collision_audit import (scene_scaling,decode_scene,collision_counts,add_counts,finalize_counts,
                                     FEATURE_SOURCE)
        from road_kinematics_audit import (RoadGeometry,native_road_counts,common_road_counts,
            physical_kinematics,load_native_kinematics,native_kinematic_counts,
            finalize_road_kinematics,TRAJECTORY_SOURCE)
        torch.set_num_threads(2)
        state['feature_source_sha256']=file_sha256(FEATURE_SOURCE)
        state['native_trajectory_source_sha256']=file_sha256(TRAJECTORY_SOURCE)
        scaling=scene_scaling();functions=load_native_kinematics()
        means,std=(torch.from_numpy(a) for a in scaling)
        def decode_tensor(values):
            return values*2*std+means
        aggregate={};started=time.perf_counter()
        for sample in manifest['scenarios']:
            index=sample['index'];key=str(index);cache=output/'scenes'/f'{index:03d}.json'
            if key in state['records']:
                assert file_sha256(cache)==state['records'][key]['sha256']
                bundle=json.loads(cache.read_text())
            else:
                bundle=dict(index=index,token=sample['token'],city=sample['city'],trip=sample['trip'],records=[])
                common=None;eligible=None;gt_reference=None;valid_reference=None;native_maps={};stored_maps={}
                for seed in manifest['seeds']:
                    for recipe in manifest['recipes']:
                        record=manifest['records'][f'{recipe}:{index}:{seed}']
                        blob=(run/record['prediction_file']).read_bytes()
                        assert hashlib.sha256(blob).hexdigest()==record['sha256'],record['prediction_file']
                        payload=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=False)
                        assert payload['valid_mask'].shape[0]==1
                        valid=payload['valid_mask'][0].bool().numpy().copy()
                        gt=decode_scene(payload['scene_tensor_gt'][0].numpy(),scaling)
                        graph=payload['road_graph'][0].numpy();graph_valid=payload['road_graph_validity'][0].numpy()
                        if recipe not in native_maps:
                            native_maps[recipe]=RoadGeometry(graph,graph_valid,scaling)
                            stored_maps[recipe]=(graph.copy(),graph_valid.copy())
                        else:
                            assert np.array_equal(graph,stored_maps[recipe][0])
                            assert np.array_equal(graph_valid,stored_maps[recipe][1])
                        if common is None:
                            assert recipe=='released_test_recipe'
                            common=RoadGeometry(graph,graph_valid,scaling,centerlines_only=True)
                            assert len(common.points)>0,'Empty common centerline map'
                            gt_reference,valid_reference=gt[:64].copy(),valid[:64].copy()
                            distances=common.distances(gt_reference,valid_reference,segments=True)
                            eligible=(((distances[:,:5]<=2.75)&valid_reference[:,:5]).any(axis=-1)
                                      &valid_reference[:,5:].any(axis=-1))
                        assert np.array_equal(valid[:64],valid_reference)
                        assert np.allclose(gt[:64,:,:4],gt_reference[:,:,:4],atol=1e-5,rtol=1e-6)
                        assert np.allclose(gt[:64,:,6:8],gt_reference[:,:,6:8],atol=1e-5,rtol=1e-6)
                        matched={k:(v[:,:64].clone() if k in ('behavior_pred','intent_conditioned_behavior_pred',
                            'scene_tensor_gt','valid_mask','task_mask') else v) for k,v in payload.items()}
                        displacement=audit_displacement(matched,decode_tensor)
                        def analyze(states,goal_conditioned=False):
                            shared_states,shared_valid=states[:64],valid[:64]
                            overlap=collision_counts(shared_states,shared_valid,goal_conditioned)
                            native_overlap=(overlap if len(valid)==64 or not valid[64:,5:].any()
                                            else collision_counts(states,valid,goal_conditioned))
                            return dict(collision=overlap,native_collision=native_overlap,
                                road_kinematics=dict(
                                    native_road=native_road_counts(native_maps[recipe].distances(
                                        states,valid,native_float32=True),valid),
                                    common_points=common_road_counts(common.distances(shared_states,shared_valid),shared_valid,eligible),
                                    common_segments=common_road_counts(common.distances(shared_states,shared_valid,segments=True),shared_valid,eligible),
                                    physical_kinematics=physical_kinematics(shared_states,shared_valid),
                                    native_kinematics=native_kinematic_counts(states,valid,functions)))
                        if seed==0 and recipe=='released_test_recipe':
                            bundle['records'].append(dict(seed='reference',recipe='recorded_gt',task='gt',counts=analyze(gt)))
                        for task,prediction in (('bp','behavior_pred'),('goal','intent_conditioned_behavior_pred')):
                            states=decode_scene(payload[prediction][0].numpy(),scaling)
                            assert np.isfinite(states[valid]).all()
                            counts=analyze(states,task=='goal');counts['displacement']=displacement[task]
                            bundle['records'].append(dict(seed=str(seed),recipe=recipe,task=task,counts=counts))
                        bundle.setdefault('omitted_future_points',{})[f'{recipe}:{seed}']=int(valid[64:,5:].sum())
                assert len(bundle['records'])==13
                json_write(cache,bundle)
                state['records'][key]=dict(file=str(cache.relative_to(output)),sha256=file_sha256(cache),token=sample['token'])
                json_write(state_path,state)
            for entry in bundle['records']:
                for city in ('all',sample['city']):
                    target=aggregate.setdefault(entry['seed'],{}).setdefault(city,{}).setdefault(entry['recipe'],{}).setdefault(entry['task'],{})
                    add_counts(target,entry['counts'])
            if (index+1)%8==0:
                print(f'INDEPENDENT_AUDIT_PROGRESS scenes={index+1}/256 elapsed_seconds={time.perf_counter()-started:.1f}',flush=True)
        result={}
        for seed,cities in aggregate.items():
            for city,recipes in cities.items():
                for recipe,tasks in recipes.items():
                    for task,counts in tasks.items():
                        finalized=dict(collision=finalize_counts(counts['collision']),
                            native_collision=finalize_counts(counts['native_collision']),
                            road_kinematics=finalize_road_kinematics(counts['road_kinematics']))
                        if 'displacement' in counts:
                            finalized['displacement']=dict(counts['displacement'])
                            d=finalized['displacement'];d['mean_error_m']=d['error_sum_m']/d['valid_points']
                        result.setdefault(seed,{}).setdefault(city,{}).setdefault(recipe,{})[task]=finalized
        json_write(output/'metrics_by_seed_city.json',result)
        native=json.loads((run/'official_metrics_by_recipe_seed.json').read_text())
        displacement=json.loads((run/'common64_displacement_by_recipe_city_seed.json').read_text())
        checks=[];position_checks=[]
        for seed in ('0','1','2'):
            for recipe in manifest['recipes']:
                for task in ('bp','goal'):
                    values=result[seed]['all'][recipe][task]
                    expected=native[recipe][seed]['traj_metric_bp' if task=='bp' else 'traj_metric_gp']
                    actual={'offroad_rate':values['road_kinematics']['native_road']['scene_mean_ratio']}
                    actual.update({k:v['mean'] for k,v in values['road_kinematics']['native_kinematics'].items()})
                    actual['collision@avg']=values['native_collision']['all_agents']['upstream_integer_horizon_pair_ratio']
                    expected=dict(expected,**native[recipe][seed]['collision_rate_bp' if task=='bp' else 'collision_rate_gp'])
                    for metric,value in actual.items():
                        tolerance=(2e-7 if metric=='collision@avg' else 1e-6 if metric=='offroad_rate'
                                   else 5e-5+abs(expected[metric])*3e-5)
                        checks.append(dict(seed=seed,recipe=recipe,task=task,metric=metric,actual=value,
                            expected=expected[metric],absolute_error=abs(value-expected[metric]),
                            tolerance=tolerance,passed=abs(value-expected[metric])<=tolerance))
                    for city in displacement[seed]:
                        d=result[seed][city][recipe][task]['displacement'];e=displacement[seed][city][recipe][task]
                        position_checks.append(dict(seed=seed,recipe=recipe,task=task,city=city,
                            passed=d['valid_points']==e['valid_points'] and abs(d['mean_error_m']-e['mean_error_m'])<1e-9))
        json_write(output/'native_replay_check.json',checks)
        json_write(output/'position_replay_check.json',position_checks)
        assert len(checks)==96 and all(c['passed'] for c in checks),'Native metric replay mismatch'
        assert len(position_checks)==60 and all(c['passed'] for c in position_checks),'Position replay mismatch'
        summary=dict(status='complete',scenarios=256,new_trips=32,seeds=manifest['seeds'],
            prediction_files_verified=1536,model_task_outputs=3072,ground_truth_scenes=256,
            native_values_replayed=96,position_values_replayed=60,definitions=state['definitions'],by_recipe={})
        for recipe in ['recorded_gt']+manifest['recipes']:
            summary['by_recipe'][recipe]={}
            for task in (['gt'] if recipe=='recorded_gt' else ['bp','goal']):
                selected=['reference'] if recipe=='recorded_gt' else ['0','1','2']
                entries=[result[seed]['all'][recipe][task] for seed in selected]
                dest={}
                if task!='gt':
                    dest['displacement_m']=stats([v['displacement']['mean_error_m'] for v in entries])
                for scope in ('all_agents','non_ego_only'):
                    dest[scope]={metric+'_percent':stats([v['collision'][scope][metric]*100 for v in entries])
                        for metric in ('vehicle_frame_collision_fraction','vehicle_any_future_collision_fraction',
                                       'generated_vehicle_frame_collision_fraction')}
                    dest[scope]['physical_kinematics']={metric:stats([v['road_kinematics']['physical_kinematics'][scope][metric]['mean_abs']
                        for v in entries]) for metric in entries[0]['road_kinematics']['physical_kinematics'][scope]}
                for mode in ('common_points','common_segments'):
                    dest[mode]={metric+'_percent':stats([v['road_kinematics'][mode][metric]*100 for v in entries])
                        for metric in ('frame_deviation_fraction','vehicle_deviation_fraction')}
                summary['by_recipe'][recipe][task]=dest
        summary['causal_vs_released_error_reduction_percent']={task:100*(1-
            summary['by_recipe']['causal_training_recipe'][task]['displacement_m']['mean']/
            summary['by_recipe']['released_test_recipe'][task]['displacement_m']['mean']) for task in ('bp','goal')}
        json_write(output/'summary.json',summary)
        state.update(status='complete',evaluated_scenarios=256,verified_prediction_files=1536,
                     native_values_replayed=96,position_values_replayed=60)
        state.pop('error',None);json_write(state_path,state)
        for recipe,tasks in summary['by_recipe'].items():
            for task,values in tasks.items():
                print(recipe+' '+task+' displacement_m='+str(values.get('displacement_m',{}).get('mean'))+
                      ' overlap_percent='+str(values['all_agents']['vehicle_frame_collision_fraction_percent']['mean']),flush=True)
        print('NUPLAN_INDEPENDENT_AUDIT_OK\nAUDIT_DIR='+str(output),flush=True)
    except Exception as exc:
        state.update(status='failed_resumable',error=repr(exc));json_write(state_path,state);raise


if __name__=='__main__':
    main()
