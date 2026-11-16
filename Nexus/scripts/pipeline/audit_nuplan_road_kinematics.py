"""Replay native metrics and audit shared centerlines and physical derivatives."""

import argparse
import hashlib
import io
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
    ref=json.loads((reference/'run.json').read_text());control=json.loads((recipe/'run.json').read_text())
    assert ref['status']==control['status']=='complete'
    assert ref['plan_sha256']==control['plan_sha256'] and ref['scenarios']==control['scenarios']
    assert ref['seeds']==control['seeds']==[0,1,2]
    assert len(ref['records'])==384 and len(control['records'])==768
    from collision_audit import FEATURE_SOURCE
    from road_kinematics_audit import TRAJECTORY_SOURCE
    fingerprint=dict(reference_manifest_sha256=file_sha256(reference/'run.json'),recipe_manifest_sha256=file_sha256(recipe/'run.json'),
        runner_sha256=file_sha256(Path(__file__)),engine_sha256=file_sha256(Path(__file__).with_name('road_kinematics_audit.py')),
        decode_engine_sha256=file_sha256(Path(__file__).with_name('collision_audit.py')),
        feature_source_sha256=file_sha256(FEATURE_SOURCE),native_metric_source_sha256=file_sha256(TRAJECTORY_SOURCE))
    assert output not in (reference,recipe)
    output.mkdir(parents=True,exist_ok=True);(output/'scenes').mkdir(exist_ok=True)
    state_path=output/'run.json'
    if state_path.exists():
        state=json.loads(state_path.read_text())
        assert all(state[k]==v for k,v in fingerprint.items()),'Input or source changed'
        if state['status']=='complete':
            print('ROAD_KINEMATICS_ALREADY_COMPLETE\nAUDIT_DIR='+str(output),flush=True);return
    else:
        state=dict(fingerprint,status='started',records={},gpu_inference=False,training_updates=0,
            scope='common_centerline_distance_and_heading_projected_derivative_audit',
            reference_run=str(reference),recipe_run=str(recipe),common_agent_slots=64,
            definitions=[
                'Native metrics retain each recipe input map, original formulas and original averaging.',
                'Common road geometry uses only LANE/LANE_CONNECTOR from the saved reference map for every recipe.',
                'Point versus segment distances use the SAME ground-truth-derived eligible vehicles: at least one '
                'valid observed frame within 2.75 m of a centerline segment, and at least one valid future frame.',
                'Segment interpolation joins adjacent valid vertices only, never across masked gaps.',
                'These nearest-centerline distances are not assigned-route compliance or drivable-area polygon checks.',
                'Physical acceleration uses backward position differences at 0.5 s, projected onto current heading. '
                'Physical jerk is the backward time difference of those projected components.',
                'Only future frames with a completely valid finite-difference stencil contribute. All agents and '
                'non-ego-only results are separate; provided goals remain in comfort/road diagnostics.',
                'Physical components have m/s^2 and m/s^3 units; they are not silently substituted for upstream angular metrics.',
                'This diagnostic does not certify exact reproduction of the complete paper protocol.'])
    state['status']='running';json_write(state_path,state)
    print('AUDIT_DIR='+str(output)+'\nCPU_ONLY: native replay, shared centerlines, physical acceleration and jerk',flush=True)
    try:
        import numpy as np
        import torch
        from collision_audit import scene_scaling,decode_scene,add_counts
        from road_kinematics_audit import (RoadGeometry,native_road_counts,common_road_counts,physical_kinematics,
            load_native_kinematics,native_kinematic_counts,finalize_road_kinematics)
        torch.set_num_threads(2);scaling=scene_scaling();functions=load_native_kinematics()
        totals={};started=time.perf_counter()
        def load(directory,record):
            blob=(directory/record['prediction_file']).read_bytes()
            assert hashlib.sha256(blob).hexdigest()==record['sha256']
            return torch.load(io.BytesIO(blob),map_location='cpu',weights_only=False)
        def arrays(payload):
            assert payload['valid_mask'].shape[0]==1
            return (decode_scene(payload['scene_tensor_gt'][0].numpy(),scaling),
                    payload['valid_mask'][0].bool().numpy().copy())
        for sample in ref['scenarios']:
            index=sample['index'];key=str(index);cache=output/'scenes'/f'{index:03d}.json'
            if key in state['records']:
                assert file_sha256(cache)==state['records'][key]['sha256'];bundle=json.loads(cache.read_text())
            else:
                bundle=dict(index=index,token=sample['token'],city=sample['city'],records=[])
                gt_reference=None;valid_reference=None;common=None;eligible=None
                native_geometry={};stored_maps={}
                def analyze(states,valid,native):
                    native_distances=native.distances(states,valid,native_float32=True)
                    common_states,common_valid=states[:64],valid[:64]
                    point_distances=common.distances(common_states,common_valid)
                    segment_distances=common.distances(common_states,common_valid,segments=True)
                    return dict(native_road=native_road_counts(native_distances,valid),
                        common_points=common_road_counts(point_distances,common_valid,eligible),
                        common_segments=common_road_counts(segment_distances,common_valid,eligible),
                        physical_kinematics=physical_kinematics(common_states,common_valid),
                        native_kinematics=native_kinematic_counts(states,valid,functions))
                for seed in ref['seeds']:
                    base=load(reference,ref['records'][f'{index}:{seed}'])
                    branch_payloads=[('saved_test_recipe_common64',base)]+[(b,load(recipe,control['records'][f'{b}:{index}:{seed}'])) for b in control['branches']]
                    for branch,payload in branch_payloads:
                        gt,valid=arrays(payload)
                        graph=payload['road_graph'][0].numpy();graph_valid=payload['road_graph_validity'][0].numpy()
                        if branch not in native_geometry:
                            native_geometry[branch]=RoadGeometry(graph,graph_valid,scaling)
                            stored_maps[branch]=(graph.copy(),graph_valid.copy())
                        else:
                            assert np.array_equal(graph,stored_maps[branch][0]) and np.array_equal(graph_valid,stored_maps[branch][1])
                        if gt_reference is None:
                            gt_reference,valid_reference=gt[:64],valid[:64]
                            common=RoadGeometry(graph,graph_valid,scaling,centerlines_only=True)
                            assert len(common.points)>0,'Reference centerline map empty'
                            gt_segments=common.distances(gt_reference,valid_reference,segments=True)
                            eligible=(((gt_segments[:,:5]<=2.75)&valid_reference[:,:5]).any(axis=-1)
                                      &valid_reference[:,5:].any(axis=-1))
                            bundle['records'].append(dict(seed='reference',branch='recorded_gt',task='gt',
                                counts=analyze(gt,valid,native_geometry[branch])))
                        assert np.array_equal(valid[:64],valid_reference)
                        assert np.allclose(gt[:64,:,:4],gt_reference[:,:,:4],atol=1e-5,rtol=1e-6)
                        if valid.shape[0]>64:
                            assert not valid[64:,5:].any(),'Common-agent truncation removes future agents'
                        for task,prediction in (('bp','behavior_pred'),('goal','intent_conditioned_behavior_pred')):
                            states=decode_scene(payload[prediction][0].numpy(),scaling)
                            bundle['records'].append(dict(seed=str(seed),branch=branch,task=task,
                                counts=analyze(states,valid,native_geometry[branch])))
                assert len(bundle['records'])==19
                json_write(cache,bundle);state['records'][key]=dict(file=str(cache.relative_to(output)),sha256=file_sha256(cache),token=sample['token'])
                json_write(state_path,state)
            for record in bundle['records']:
                for city in ('all',sample['city']):
                    dest=totals.setdefault(record['seed'],{}).setdefault(city,{}).setdefault(record['branch'],{}).setdefault(record['task'],{})
                    add_counts(dest,record['counts'])
            if (index+1)%8==0:
                print(f'ROAD_KINEMATICS_PROGRESS scenes={index+1}/128 elapsed_seconds={time.perf_counter()-started:.1f}',flush=True)
        result={seed:{city:{branch:{task:finalize_road_kinematics(counts) for task,counts in tasks.items()}
            for branch,tasks in branches.items()} for city,branches in cities.items()} for seed,cities in totals.items()}
        json_write(output/'metrics_by_seed_city.json',result)
        original=json.loads((reference/'official_metrics_by_seed.json').read_text())
        controlled=json.loads((recipe/'official_metrics_by_recipe_seed.json').read_text())
        checks=[]
        for seed in ('0','1','2'):
            for branch,tasks in result[seed]['all'].items():
                expected=original[seed] if branch=='saved_test_recipe_common64' else controlled[branch][seed]
                for task,counts in tasks.items():
                    target=expected['traj_metric_bp' if task=='bp' else 'traj_metric_gp']
                    actual={'offroad_rate':counts['native_road']['scene_mean_ratio']}
                    actual.update({k:v['mean'] for k,v in counts['native_kinematics'].items()})
                    for metric,value in actual.items():
                        difference=abs(value-target[metric]);tolerance=5e-5+abs(target[metric])*3e-5 if metric!='offroad_rate' else 1e-6
                        checks.append(dict(seed=seed,branch=branch,task=task,metric=metric,actual=value,
                            expected=target[metric],absolute_error=difference,tolerance=tolerance,passed=difference<=tolerance))
        json_write(output/'native_replay_check.json',checks)
        assert len(checks)==126 and all(c['passed'] for c in checks),'Native replay mismatch; inspect native_replay_check.json'
        summary=dict(scope=state['scope'],scenarios=128,seeds=ref['seeds'],native_values_replayed=126,definitions=state['definitions'],by_branch={})
        def stats(values):
            return dict(mean=statistics.mean(values),seed_sample_sd=statistics.stdev(values) if len(values)>1 else None,by_seed=values)
        for branch in ['recorded_gt','saved_test_recipe_common64']+control['branches']:
            summary['by_branch'][branch]={}
            for task in (['gt'] if branch=='recorded_gt' else ['bp','goal']):
                selected=['reference'] if branch=='recorded_gt' else ['0','1','2']
                dest={}
                for mode in ('common_points','common_segments'):
                    dest[mode]={metric:stats([result[seed]['all'][branch][task][mode][metric]*100 for seed in selected])
                        for metric in ('frame_deviation_fraction','vehicle_deviation_fraction')}
                dest['physical_kinematics']={scope:{metric:stats([result[seed]['all'][branch][task]['physical_kinematics'][scope][metric]['mean_abs']
                    for seed in selected]) for metric in result[selected[0]]['all'][branch][task]['physical_kinematics'][scope]}
                    for scope in ('all_agents','non_ego_only')}
                summary['by_branch'][branch][task]=dest
        json_write(output/'summary.json',summary)
        state.update(status='complete',evaluated_scenarios=128,evaluated_saved_prediction_files=1152,ground_truth_scenes=128,
            native_values_replayed=126,model_updates=0);state.pop('error',None);json_write(state_path,state)
        for branch,tasks in summary['by_branch'].items():
            for task,metrics in tasks.items():
                print(branch+' '+task+' | centerline_segment_departure_percent='+str(metrics['common_segments']['vehicle_deviation_fraction']['mean']),flush=True)
        print('NUPLAN_ROAD_KINEMATICS_AUDIT_OK\nAUDIT_DIR='+str(output),flush=True)
    except Exception as exc:
        state.update(status='failed_resumable',error=repr(exc));json_write(state_path,state);raise


if __name__=='__main__':
    main()
