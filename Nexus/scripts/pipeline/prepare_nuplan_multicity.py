"""Freeze a metadata-only four-city holdout before looking at model outputs."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from run_nuplan_metrics import json_write
from run_repro_demo import file_sha256
from nuplan_plan_queries import candidates, validate_plan_metadata, QUERY_SOURCE


def rank(seed, text):
    return hashlib.sha256(f'{seed}:{text}'.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, default=Path('artifacts/nuplan_evaluation/validation_inventory.json'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/nuplan_evaluation/multicity128_plan_v2.json'))
    parser.add_argument('--selection-seed', type=int, default=20261021)
    args = parser.parse_args()
    if args.output.exists():
        plan = json.loads(args.output.read_text())
        assert len(plan['scenarios']) == 128
        assert plan['version'] == 'four_city_holdout_v2'
        assert validate_plan_metadata(plan) == 128
        print('EXISTING_FROZEN_PLAN=' + str(args.output.resolve()))
        return
    inventory = json.loads(args.inventory.read_text())
    assert len(inventory['records']) == inventory['expected_databases'] == 1381
    assert not any('error' in r for r in inventory['records'])
    pilot = Path('artifacts/nuplan_metrics/val_pilot_20261020_140122_172964/run.json')
    development = json.loads(pilot.read_text())
    excluded = {s['log_name'].rsplit('_', 2)[0] for s in development['selected_scenarios']}
    groups = defaultdict(lambda: defaultdict(list))
    for r in inventory['records']:
        if r['trip'] not in excluded:
            groups[r['city']][r['trip']].append(r)
    assert len(groups) == 4
    selected, rejected = [], []
    for city in sorted(groups):
        trips_used = 0
        for trip in sorted(groups[city], key=lambda t: rank(args.selection_seed, city+':'+t)):
            chosen = None
            for log in sorted(groups[city][trip], key=lambda r: rank(args.selection_seed, r['log_name'])):
                pool = candidates(log['database'])
                picks = []
                for s in sorted(pool, key=lambda s: rank(args.selection_seed, s['token'])):
                    if all(abs(s['timestamp_us']-p['timestamp_us']) >= 15_000_000 for p in picks):
                        picks.append(s)
                    if len(picks) == 4:
                        break
                if len(picks) == 4:
                    chosen = [dict(s, city=city, trip=trip, log_name=log['log_name'], database=log['database']) for s in sorted(picks, key=lambda s: s['timestamp_us'])]
                    break
                rejected.append({'log_name': log['log_name'], 'reason': 'fewer_than_four_spaced_eligible_anchors'})
            if chosen:
                selected.extend(chosen)
                trips_used += 1
                print(f'PLAN_PROGRESS city={city} trips={trips_used}/8 scenes={len(selected)}', flush=True)
            if trips_used == 8:
                break
        if trips_used != 8:
            raise RuntimeError(f'Insufficient eligible independent trips for {city}: {trips_used}')
    assert len(selected) == len({s['token'] for s in selected}) == 128
    assert len({s['trip'] for s in selected}) == 32
    assert not (excluded & {s['trip'] for s in selected})
    for i, s in enumerate(selected):
        s['index'] = i
    plan = {
        'version': 'four_city_holdout_v2', 'frozen_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'balanced_multicity_validation_subset_not_full_paper_benchmark',
        'selection_seed': args.selection_seed, 'inventory_sha256': file_sha256(args.inventory),
        'excluded_development_trips': sorted(excluded), 'rejected_logs': rejected,
        'selection_rule': 'hash_ordered_trips_then_logs_then_tokens; 8 trips/city; 4 anchors/trip; >=15s separation',
        'eligibility_rule': 'official valid_scenes row_num >=3 and row_num < count-1; tagged; >=2s past and >=8s future',
        'scenario_query_source': str(QUERY_SOURCE),
        'scenario_query_sha256': file_sha256(QUERY_SOURCE),
        'primary_preprocessing': 'official_reset_v', 'constraints': {'mode': ['map','sma','clip'], 'gamma': [1,1,1]},
        'sampling': {'steps': 32, 'schedule': 'full_sequence'},
        'model_configuration_source': 'nuplan_extent/planning/script/config/common/model/nexus.yaml',
        'model_configuration_sha256': file_sha256(Path('nuplan_extent/planning/script/config/common/model/nexus.yaml')),
        'reference_configuration': str(pilot.with_name('config.yaml').resolve()),
        'scenarios': selected,
    }
    assert validate_plan_metadata(plan) == 128
    json_write(args.output, plan)
    print('FROZEN_PLAN=' + str(args.output.resolve()), flush=True)
    print('SCENARIOS=128 CITIES=4 TRIPS=32', flush=True)


if __name__ == '__main__':
    main()
