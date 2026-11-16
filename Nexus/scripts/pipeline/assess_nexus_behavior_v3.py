"""Apply declared safety/behavior gates before expanding a development probe."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--geometry',type=Path,required=True)
    p.add_argument('--behavior',type=Path,required=True)
    p.add_argument('--reference-geometry',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    def records(folder,name):
        data=json.loads((folder/name).read_text());assert data['status']=='complete';return data['records']
    geometry=[r for r in records(a.geometry,'audit.json') if r['run']==a.run.name]
    behavior={(r['seed'],r['variant']):r for r in records(a.behavior,'behavior.json') if r['run']==a.run.name}
    reference={(r['seed'],r['variant']):r for r in records(a.reference_geometry,'audit.json') if r['run']=='order_candidate_v1'}
    run=json.loads((a.run/'run.json').read_text());assert run['status']=='complete'
    source={(r['seed'],r['variant']):r for r in run['records']}
    checks=[];failures=[]
    for row in geometry:
        key=(row['seed'],row['variant']);e=behavior[key]['metrics']['ego'];m=row['metrics'];old=reference[key]['metrics']
        tests={'dense_collisions_no_worse':m['dense']['all_vehicle_pair_samples_overlapping']<=old['dense']['all_vehicle_pair_samples_overlapping'],
            'road_center_no_worse':m['dense']['initially_onroad_center_off_samples']<=old['dense']['initially_onroad_center_off_samples'],
            'road_full_box_no_worse':m['dense']['drivable_full_box_off_samples']<=old['dense']['drivable_full_box_off_samples'],
            'acceleration_no_worse':m['kinematics']['all_agent_acceleration_over_5_samples']<=old['kinematics']['all_agent_acceleration_over_5_samples'],
            'unrequested_reverse_le_0p2m':e['reverse_distance_body_axis_m']<=.2,
            'ego_lateral_rms_le_0p25mps':e['xy_lateral_velocity_rms_mps']<=.25,
            'ego_velocity_residual_rms_le_0p5mps':e['xy_velocity_head_rms_mps']<=.5,
            'moving_heading_error_le_15deg':e['moving_heading_error_max_deg'] is None or e['moving_heading_error_max_deg']<=15,
            'ego_yaw_rate_le_0p6rps':source[key]['metrics']['ego_peak_yaw_rate_rps']<=.6001}
        checks.append({'seed':key[0],'variant':key[1],'checks':tests})
        failures.extend({'seed':key[0],'variant':key[1],'criterion':name} for name,passed in tests.items() if not passed)
    assert len(checks)==len(run['records'])>0
    all_seeds={key[0] for key in source}=={0,1,2} and len(source)==6
    status='rejected' if failures else ('three_seed_candidate_passed' if all_seeds else 'probe_ready_for_validation')
    decision={'status':status,'review_run':str(a.run.resolve()),'promoted_to_default':False,
        'checks':checks,'failures':failures,'validated_three_fixed_seeds':all_seeds,
        'criteria_scope':'engineering gates, not a safety certificate; no required left turn or progress maximization',
        'next_step':'retain V2 reference; no expanded sampling' if failures else 'inspect actual v9 bodies and complete missing seed validation',
        'limitations':['one development scene','recorded background futures and existence masks','no route legality or full bicycle guarantee','no closed-loop evaluation']}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():raise FileExistsError(a.output)
    a.output.write_text(json.dumps(decision,indent=2,ensure_ascii=False)+'\n')
    print(status,json.dumps(failures,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
