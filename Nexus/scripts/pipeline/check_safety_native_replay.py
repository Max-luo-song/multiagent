"""Compare every reconstructed integer-horizon collision value with saved metrics."""

import argparse
import json
from pathlib import Path

from run_nuplan_metrics import json_write
from run_repro_demo import file_sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-run',type=Path,required=True)
    parser.add_argument('--reference-run',type=Path,required=True)
    parser.add_argument('--recipe-run',type=Path,required=True)
    args=parser.parse_args()
    state=json.loads((args.audit_run/'run.json').read_text())
    assert state['status']=='complete' and state['native_pair_replay_verified']
    assert state['reference_manifest_sha256']==file_sha256(args.reference_run/'run.json')
    assert state['recipe_manifest_sha256']==file_sha256(args.recipe_run/'run.json')
    actual=json.loads((args.audit_run/'metrics_by_seed_city.json').read_text())
    baseline=json.loads((args.reference_run/'official_metrics_by_seed.json').read_text())
    control=json.loads((args.recipe_run/'official_metrics_by_recipe_seed.json').read_text())
    comparisons=[]
    for seed in ('0','1','2'):
        for branch,tasks in actual[seed]['all'].items():
            target=baseline[seed] if branch=='saved_test_recipe_common64' else control[branch][seed]
            for task,counts in tasks.items():
                counts=counts['all_agents']
                expected=target['collision_rate_bp' if task=='bp' else 'collision_rate_gp']
                for second in range(1,9):
                    index=second*2-1
                    denom=counts['valid_by_future_step'][index]
                    assert denom>0
                    value=counts['pair10_by_future_step'][index]/denom
                    goal=expected[f'collision@{second:.6f}s']
                    comparisons.append(dict(seed=seed,branch=branch,task=task,horizon_seconds=second,
                        reconstructed=value,source=goal,absolute_error=abs(value-goal)))
    assert len(comparisons)==144
    largest=max(c['absolute_error'] for c in comparisons)
    assert largest<2e-7, sorted(comparisons,key=lambda c:c['absolute_error'],reverse=True)[:5]
    json_write(args.audit_run/'native_horizon_replay_check.json',dict(status='passed',comparisons=144,
               maximum_absolute_error=largest,records=comparisons))
    print(f'NATIVE_HORIZON_REPLAY_CHECK_OK: 144/144 values, max_abs_error={largest:.3g}')


if __name__=='__main__':
    main()
