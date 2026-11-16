"""Save compact numeric results of a completed recipe control; no report/ZIP."""

import argparse
import json
from pathlib import Path
import statistics

from run_nuplan_metrics import json_write
from run_repro_demo import file_sha256


def stats(values):
    return {'mean':statistics.mean(values), 'seed_sample_sd':statistics.stdev(values), 'by_seed':values}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir',type=Path)
    args=parser.parse_args()
    run=args.run_dir.resolve()
    manifest=json.loads((run/'run.json').read_text())
    comparison=json.loads((run/'common_agent_comparison.json').read_text())
    official=json.loads((run/'official_metrics_by_recipe_seed.json').read_text())
    assert manifest['status']=='complete' and manifest['evaluated_scene_seed_recipes']==768
    assert manifest['model_parameter_versions_unchanged'] and manifest['training_updates']==0
    seeds=[str(s) for s in manifest['seeds']]
    reference='saved_test_recipe_common64'
    branches=[reference]+manifest['branches']
    result={'status':'complete', 'source_manifest_sha256':file_sha256(run/'run.json'),
            'scenarios':128, 'seeds':manifest['seeds'], 'new_prediction_files':768,
            'displacement_unit':'m', 'aggregation':'valid future half-second points; exclude ego, '
                 'history and invalid points; goal task additionally excludes each last valid goal point',
            'common_displacement':{}, 'native_official_collision_count_ratio_x100':{},
            'notes':['Seed standard deviation is not a population confidence interval.',
                     'Published-versus-causal changes only the current history-boundary velocity estimator.',
                     'Reference-versus-training jointly changes velocity, maps and agent capacity; do not '
                     'attribute its entire difference to velocity alone.',
                     'Collision code counts overlapping pairs per valid vehicle, not distinct collided vehicles.',
                     'Published checkpoint actual training preprocessing remains unconfirmed.']}
    for city in comparison[seeds[0]]:
        result['common_displacement'][city]={}
        for branch in branches:
            result['common_displacement'][city][branch]={}
            for task in ('bp','goal'):
                denominators={comparison[s][city][b][task]['valid_points'] for s in seeds for b in branches}
                assert len(denominators)==1
                summary=stats([comparison[s][city][branch][task]['mean_error_m'] for s in seeds])
                summary['valid_points_per_seed']=denominators.pop()
                result['common_displacement'][city][branch][task]=summary
    for branch in manifest['branches']:
        result['native_official_collision_count_ratio_x100'][branch]={task:stats([
            official[branch][s]['collision_rate_'+task]['collision@avg']*100 for s in seeds])
            for task in ('bp','gp')}
    all_values=result['common_displacement']['all']
    result['causal_vs_reference_error_reduction_percent']={task:100*(1-
        all_values['causal_history_recipe'][task]['mean']/all_values[reference][task]['mean']) for task in ('bp','goal')}
    result['causal_minus_published_error_m']={task:
        all_values['causal_history_recipe'][task]['mean']-all_values['published_training_recipe'][task]['mean'] for task in ('bp','goal')}
    json_write(run/'summary.json',result)
    for branch, values in all_values.items():
        print(branch+' | '+' | '.join(f'{task}={v["mean"]:.4f} +/- {v["seed_sample_sd"]:.4f} m' for task,v in values.items()))
    print('CAUSAL_VS_REFERENCE_REDUCTION_PERCENT='+str(result['causal_vs_reference_error_reduction_percent']))
    print('SUMMARY='+str(run/'summary.json'))


if __name__=='__main__':
    main()
