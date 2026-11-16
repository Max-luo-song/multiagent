"""Verify the same free warm start follows curved real lanes without braking."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    import torch
    from nuplan_extent.planning.training.modeling.models.diffusion.omega_cruise_initialized import CorridorCruiseConfig,InitializedCruiseRefiner
    from nexus_3dgs.trajectory_acceptance import evaluate
    torch.set_num_threads(2)
    out=ROOT/'artifacts/nexus_kinematic_20261027';old=ROOT/'artifacts/nexus_multicar_upgrade_check_20261027'
    controls=[];records=[]
    for variant in ['normal','event']:
        p=torch.load(out/'candidate_v2_cruise/kinematic/seed_000'/variant/'prediction.pt',map_location='cpu',weights_only=False)
        config=CorridorCruiseConfig(following_weight=0,motion_qp=False)
        refiner=InitializedCruiseRefiner(config,p['conditioned_scene_tensor'],p['valid_mask'],p['task_mask'],route_path=out/'routes.npz')
        u=refiner.lane_following_controls();controls.append(u)
        proposal,_,_=refiner.rollout(u,p['sampled_tensor'][:,refiner.indices])
        expected=p['conditioned_scene_tensor'][:,refiner.indices]
        assert torch.equal(proposal[refiner.keep],expected[refiner.keep])
        payload=dict(p);sample=p['sampled_tensor'].clone();sample[:,refiner.indices]=proposal;payload['sampled_tensor']=sample
        gate=evaluate(payload,reference_root=old,lane_root=out/'lanes',variant=variant)
        if variant=='normal':
            for role in ['E','C']:
                assert gate['original_lane'][role]['box_outside_original_lane_samples']==0,gate['original_lane']
            assert min(gate['motion']['E']['xy_interval_speeds_mps'])>12
        records.append({'variant':variant,'original_lane':gate['original_lane'],'ego_speed':gate['motion']['E']['xy_interval_speeds_mps']})
    assert torch.equal(controls[0],controls[1])
    result={'status':'passed','scope':'initial control proposal only, not a forecast or postprocessed deliverable',
            'same_normal_event_initial_controls':True,'given_inputs_exact':True,'checks':records}
    (out/'cruise_initialization_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
