"""Freeze a 256-scene independent four-city holdout using metadata only."""

import argparse
from collections import defaultdict,Counter
from datetime import datetime,timezone
import json
from pathlib import Path

from nuplan_plan_queries import candidates,validate_plan_metadata,QUERY_SOURCE
from prepare_nuplan_multicity import rank
from run_nuplan_metrics import json_write
from run_repro_demo import file_sha256
from training_recipe_common import SOURCE_FILES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run',type=Path,required=True)
    parser.add_argument('--inventory',type=Path,default=Path('artifacts/nuplan_evaluation/validation_inventory.json'))
    parser.add_argument('--output',type=Path,default=Path('artifacts/nuplan_evaluation/independent256_plan_v1.json'))
    parser.add_argument('--selection-seed',type=int,default=20261022)
    args=parser.parse_args()
    reference=args.reference_run.resolve()
    baseline=json.loads((reference/'run.json').read_text())
    old_plan=json.loads((reference/'frozen_plan.json').read_text())
    assert baseline['status']=='complete' and baseline['evaluated_scenarios']==128
    excluded={s['trip'] for s in baseline['scenarios']}|set(old_plan['excluded_development_trips'])
    if args.output.exists():
        existing=json.loads(args.output.read_text())
        assert existing['reference_manifest_sha256']==file_sha256(reference/'run.json')
        assert existing['selection_seed']==args.selection_seed and len(existing['scenarios'])==256
        assert validate_plan_metadata(existing)==256
        print('EXISTING_HOLDOUT_PLAN='+str(args.output.resolve()),flush=True)
        return
    inventory=json.loads(args.inventory.read_text())
    assert inventory['expected_databases']==len(inventory['records'])==1381
    groups=defaultdict(lambda:defaultdict(list))
    for log in inventory['records']:
        assert 'error' not in log
        if log['trip'] not in excluded:
            groups[log['city']][log['trip']].append(log)
    assert len(groups)==4
    selected=[];rejected=[]
    for city in sorted(groups):
        used=0
        for trip in sorted(groups[city],key=lambda t:rank(args.selection_seed,city+':'+t)):
            picks=[]
            for log in sorted(groups[city][trip],key=lambda x:rank(args.selection_seed,x['log_name'])):
                pool=candidates(log['database'])
                for sample in sorted(pool,key=lambda x:rank(args.selection_seed,x['token'])):
                    if all(abs(sample['timestamp_us']-other['timestamp_us'])>=15_000_000 for other in picks):
                        picks.append(dict(sample,city=city,trip=trip,log_name=log['log_name'],database=log['database']))
                    if len(picks)==8:
                        break
                if len(picks)==8:
                    break
            if len(picks)<8:
                rejected.append(dict(city=city,trip=trip,eligible_spaced_anchors=len(picks)))
                continue
            selected.extend(sorted(picks,key=lambda s:(s['log_name'],s['timestamp_us'],s['token'])))
            used+=1
            print(f'HOLDOUT_PLAN_PROGRESS city={city} trips={used}/8 scenes={len(selected)}/256',flush=True)
            if used==8:
                break
        assert used==8, f'Only {used} independent eligible trips found in {city}'
    assert len(selected)==len({s['token'] for s in selected})==256
    assert len({s['trip'] for s in selected})==32
    assert set(Counter(s['city'] for s in selected).values())=={64}
    assert not (excluded&{s['trip'] for s in selected})
    assert not ({s['token'] for s in baseline['scenarios']}&{s['token'] for s in selected})
    for index,sample in enumerate(selected):
        sample['index']=index
    metric_reference=Path(old_plan['reference_configuration']).with_name('run.json')
    plan=dict(version='independent_four_city_256_v1',frozen_at=datetime.now(timezone.utc).isoformat(),
        scope='independent_balanced_validation_subset_not_full_paper_benchmark',
        reference_run=str(reference),reference_manifest_sha256=file_sha256(reference/'run.json'),
        inventory_sha256=file_sha256(args.inventory),selection_seed=args.selection_seed,
        excluded_previous_trips=sorted(excluded),rejected_trips=rejected,
        selection_rule='Metadata hash order; 8 new trips per city; 8 anchors per trip across log segments; >=15s spacing.',
        primary_recipe='released_test_recipe',secondary_recipe='causal_training_recipe',
        recipes=['released_test_recipe','causal_training_recipe'],seeds=[0,1,2],batch_size=4,
        sampling={'steps':32,'schedule':'full_sequence'},constraints={'mode':['map','sma','clip'],'gamma':[1.,1.,1.]},
        recipe_note='The released test recipe remains primary. The causal training recipe is a frozen diagnostic '
                    'that jointly changes map visibility, velocity preprocessing and a 64-slot agent cap; '
                    'it is not selected as a paper-equivalent best configuration.',
        configuration=str(reference/'config.yaml'),configuration_sha256=file_sha256(reference/'config.yaml'),
        metric_reference_manifest=str(metric_reference),metric_reference_sha256=file_sha256(metric_reference),
        preprocessing_source_sha256={p:file_sha256(Path(p)) for p in SOURCE_FILES},
        causal_helper_sha256=file_sha256(Path(__file__).with_name('prepare_nuplan_training_recipe.py')),
        scenario_query_source=str(QUERY_SOURCE),scenario_query_sha256=file_sha256(QUERY_SOURCE),scenarios=selected)
    assert validate_plan_metadata(plan)==256
    args.output.parent.mkdir(parents=True,exist_ok=True)
    json_write(args.output,plan)
    print('HOLDOUT_PLAN_OK: 256 unique scenes, 4 cities, 32 new trips; no prior trip reused',flush=True)
    print('PLAN='+str(args.output.resolve()),flush=True)


if __name__=='__main__':
    main()
