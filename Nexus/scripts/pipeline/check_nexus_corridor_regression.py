"""Probe the real C-lane regression; diagnostic only, not a delivered forecast."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts/pipeline')]


def main():
    import numpy as np
    import torch
    from nuplan_extent.planning.training.modeling.models.diffusion.omega_cruise_corridor import CorridorCruiseConfig,CorridorCruiseRefiner
    from nexus_3dgs.trajectory_acceptance import evaluate
    out=ROOT/'artifacts/nexus_kinematic_20261027';old=ROOT/'artifacts/nexus_multicar_upgrade_check_20261027'
    folder=out/'candidate_v2_cruise/kinematic/seed_000/normal'
    payload=torch.load(folder/'prediction.pt',map_location='cpu',weights_only=False)
    torch.set_num_threads(2)
    config=CorridorCruiseConfig(start_call=1,control_iterations=300,final_control_iterations=300,
                              collision_weight=3000,road_weight=1000,following_weight=0,motion_qp=False,
                              max_acceleration_mps2=4.5,acceleration_weight=100,jerk_weight=2,clearance_m=.3)
    refiner=CorridorCruiseRefiner(config,payload['conditioned_scene_tensor'].cuda(),payload['valid_mask'].cuda(),
                                payload['task_mask'].cuda(),route_path=out/'routes.npz')
    assert refiner.lane_risk_relaxation==[1.,1.]
    event=torch.load(out/'candidate_v2_cruise/kinematic/seed_000/event/prediction.pt',map_location='cpu',weights_only=False)
    conditional=CorridorCruiseRefiner(config,event['conditioned_scene_tensor'].cuda(),event['valid_mask'].cuda(),
                                    event['task_mask'].cuda(),route_path=out/'routes.npz')
    assert conditional.lane_risk_relaxation==[.1,1.]
    refiner.set_drivable_sdf(old/'inputs/drivable_sdf.npz');refiner.set_background_replay(old/'inputs/background_replay.npz')
    refiner.previous_controls=torch.from_numpy(np.load(folder/'controls.npy')).cuda()
    anchor=payload['sampled_tensor'].cuda()
    result=refiner(anchor,torch.ones_like(anchor),torch.ones_like(anchor),{})
    evaluated=dict(payload);evaluated['sampled_tensor']=result.cpu()
    gate=evaluate(evaluated,reference_root=old,lane_root=out/'lanes',variant='normal')
    print('CORRIDOR_PROBE',json.dumps(gate,ensure_ascii=False),flush=True)
    assert gate['primary_three_accepted'],gate['original_lane']
    speed=gate['motion']['E']['xy_interval_speeds_mps']
    assert speed[-1]>=10 and sum(speed)*.5>=80,speed
    report={'status':'passed','scope':'offline optimizer regression probe only; not a new Nexus forecast or postprocessed deliverable',
            'normal_relaxation':[1,1],'event_relaxation':[.1,1],
            'real_normal_C_outside_lane_before':13,'after':gate['original_lane']['C']['box_outside_original_lane_samples'],
            'acceptance':gate}
    (out/'corridor_regression_check.json').write_text(json.dumps(report,indent=2)+'\n')
    print('CORRIDOR_REGRESSION_CHECK_PASSED',flush=True)


if __name__=='__main__':main()
