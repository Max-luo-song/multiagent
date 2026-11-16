"""Frozen-case Nexus sampling with optional control-space guidance and a gate."""
import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT/'artifacts/nexus_multicar_upgrade_check_20261027'
OUT = ROOT/'artifacts/nexus_kinematic_20261027'
sys.path[:0] = [str(ROOT), str(ROOT/'scripts/pipeline'), str(ROOT/'third_party/nuplan-devkit'), str(ROOT/'third_party/alf')]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',default='candidate_v4_cruise_initialized')
    parser.add_argument('--seeds',type=int,nargs='+',default=[0]);args=parser.parse_args()
    frozen=json.loads((OLD/'protocol.json').read_text())
    protected=frozen['source_sha256'].copy()
    new_files=[ROOT/'nuplan_extent/planning/training/modeling/models/diffusion/omega_cruise_initialized.py',
               ROOT/'nuplan_extent/planning/training/modeling/models/diffusion/omega_cruise_corridor.py',
               ROOT/'nuplan_extent/planning/training/modeling/models/diffusion/omega_cruise.py',
               ROOT/'nuplan_extent/planning/training/modeling/models/diffusion/omega_kinematic.py',
               ROOT/'nexus_3dgs/trajectory_acceptance.py', Path(__file__)]
    protected.update({str(p):digest(p) for p in new_files})
    frozen_inputs={str(p):digest(p) for p in [OUT/'routes.npz',OUT/'lane_corridors_sdf.npz',
        OLD/'inputs/drivable_sdf.npz',OLD/'inputs/background_replay.npz',
        OLD/'map_event_input_candidate_v1/A_normal.npy',OLD/'map_event_input_candidate_v1/A_event.npy']}
    def check_source():
        for name,h in protected.items():assert digest(name)==h,name
        for name,h in frozen_inputs.items():assert digest(name)==h,name
    check_source()
    stage=OUT/args.stage;stage.mkdir(exist_ok=False)
    os.environ.update(MPLBACKEND='Agg',WANDB_MODE='disabled',NUPLAN_DATA_STORE='local',PYTHONDONTWRITEBYTECODE='1')
    for key in ('GT_REPLACE','EXTRAPOLATE','NO_PHYPE'):
        if os.environ.get(key):raise ValueError('Unset sampler override '+key)
    import numpy as np
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from nuplan_extent.planning.training.preprocessing.features.scene_tensor import SceneTensor
    from nuplan_extent.planning.training.data_augmentation.reset_v import ResetVAugmentor
    from nuplan_extent.planning.training.modeling.models.diffusion.omega_cruise_initialized import CorridorCruiseConfig as KinematicConfig, InitializedCruiseRefiner as KinematicRefiner
    from nexus_3dgs.bundle import FEATURE_MEANS,FEATURE_STD,export_arrays,save_bundle,decode
    from nexus_3dgs.trajectory_acceptance import evaluate
    from run_repro_demo import CHECKPOINT_SHA256,model_state_dict
    from run_nexus_omega_v2 import segments_from_map
    torch.set_num_threads(2)
    record=frozen['scenario'];source=ROOT/'artifacts/nuplan_evaluation/independent256_v1'
    rawpath=source/'raw_features'/f'{record["index"]:03d}_{record["token"]}.pt'
    raw=torch.load(rawpath,map_location='cpu',weights_only=False)
    scene=SceneTensor(**{k:v.numpy().copy() for k,v in raw.items()})
    scene=ResetVAugmentor().augment({'scene_tensor':scene},{})[0]['scene_tensor']
    reference=scene.tensor[...,:8].copy()
    compiled={}
    for variant in ('normal','event'):
        with np.load(OLD/'normal_route_factorial_v1'/f'conditions_{variant}.npz') as data:
            values=data['repaired'].copy();keep=data['keep'].copy();valid=data['valid'].copy()
        supplied=np.load(OLD/'map_event_input_candidate_v1'/f'A_{variant}.npy')
        values[2,5:]=((supplied[5:]-FEATURE_MEANS)/(2*FEATURE_STD)).astype(values.dtype)
        assert keep[:,:5].all() and keep[2].all()
        assert not keep[0,5:].any() and not keep[12,5:].any() and (values[~keep]==0).all()
        compiled[variant]=(values,keep,valid)
        np.savez_compressed(stage/f'conditions_{variant}.npz',values=values,keep=keep,valid=valid)
    np.testing.assert_array_equal(compiled['normal'][0][:,:5],compiled['event'][0][:,:5])
    config=KinematicConfig(collision_weight=3000.,road_weight=1000.,following_weight=0.,
                           motion_qp=False,max_acceleration_mps2=4.5,acceleration_weight=100.,
                           jerk_weight=2.,clearance_m=.3,control_iterations=65,
                           final_control_iterations=180)
    checkpoint=ROOT.parent/'checkpoints/nuplan.ckpt';assert digest(checkpoint)==CHECKPOINT_SHA256
    plan={'source_protocol':str(OLD/'protocol.json'),'scenario':record,'seeds':args.seeds,
          'method':'kinematic','config':vars(config),'source_sha256':protected,'input_sha256':frozen_inputs,
          'checkpoint_sha256':CHECKPOINT_SHA256,'raw_features_sha256':digest(rawpath),
          'route_sha256':digest(OUT/'routes.npz'),'ego_C_future_free':True,
          'given_A_full_future':True,'common_A_forward_shift_m':2.,
          'conditioned_history_s':2.,'forecast_s':8.,'event_window_s':[.5,2.5],
          'event_lateral_shift_m':3.,'same_noise_for_normal_event':True,
          'extra_information':'recorded 97-actor background and future existence mask; SDK map routes',
          'algorithm':'same nonholonomic rollout plus observation-derived soft cruise and terminal speed recovery',
          'speed_target_source':'current observed speed only, no logged future speeds',
          'lane_body_prior':'whole observed-size box; relaxed when given front actor enters this lane',
          'lane_sdf_sha256':digest(OUT/'lane_corridors_sdf.npz'),
          'baseline_stage':'candidate_v1','control_initialization':'same map-only lane-following warm start for normal/event; future controls remain free',
          'model_training_updates':0,'offline_output_repairs':False,
          'trust_region':'soft physical anchor preference; not exact Gaussian KL',
          'handoff_policy':'export approved bundle only after full independent acceptance; rejected samples remain diagnostic'}
    plan['speed_behavior_gate']={
        'normal_E_final_speed_min_fraction_current':.85,
        'normal_E_any_interval_speed_min_fraction_current':.85,
        'normal_E_progress_min_fraction_current_speed_times_horizon':.9,
        'event_E_early_speed_reduction_vs_normal_min_mps':1.,
        'event_E_final_speed_min_fraction_observed_following_speed':.8,
        'scope':'Frozen scene033 diagnostic, not a universal cruise legality rule; physical checks remain required'}
    write(stage/'protocol.json',plan)
    for p in new_files:shutil.copy2(p,stage/p.name)
    print('KINEMATIC_LOADING_MODEL',flush=True)
    cfg=OmegaConf.load(source/'config.yaml');model=instantiate(cfg.model)
    model.load_state_dict(model_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)),strict=True)
    model=model.cuda().eval().requires_grad_(False);versions=[p._version for p in model.parameters()]
    model.diffuser.constrain_mode=[];model.diffuser.constraint_gamma=[]
    model.diffuser._generate_scheduling_matrix=functools.partial(model.diffuser._generate_scheduling_matrix,zero_init=False)
    segments=torch.as_tensor(segments_from_map(scene.road_graph,scene.road_graph_validity),device='cuda')
    run={'status':'running','records':[],'pairs':[],'protocol':plan};write(stage/'run.json',run)
    for seed in args.seeds:
        noise_hash=None;pair_records=[]
        for variant,(values,keep,valid) in compiled.items():
            check_source();random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            inputs=torch.from_numpy(values).unsqueeze(0).cuda();mask=torch.from_numpy(keep).unsqueeze(0).cuda()
            validity=torch.from_numpy(valid.astype(np.float32)).unsqueeze(0).cuda()
            features=SceneTensor(tensor=inputs,validity=validity,
                                 road_graph=torch.from_numpy(scene.road_graph.copy()).unsqueeze(0).cuda(),
                                 road_graph_validity=torch.from_numpy(scene.road_graph_validity.copy()).unsqueeze(0).cuda())
            noise=torch.randn_like(inputs);h=hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest()
            if noise_hash is None:noise_hash=h
            else:assert h==noise_hash
            refiner=KinematicRefiner(config,inputs,validity,mask,segments,route_path=OUT/'routes.npz')
            refiner.set_drivable_sdf(OLD/'inputs/drivable_sdf.npz')
            refiner.set_background_replay(OLD/'inputs/background_replay.npz')
            model.diffuser.anchor_refiner=refiner
            print('CRUISE_SAMPLE_START',seed,variant,'controlled',len(refiner.controlled_slots),flush=True)
            start=time.monotonic()
            with torch.no_grad():
                pred=model.forward_inference({'scene_tensor':features,'task_mask':mask},noise=noise,return_intermidates=False)['sampled_tensor'].cpu()
            torch.cuda.synchronize();elapsed=time.monotonic()-start
            torch.testing.assert_close(pred[mask.cpu()],inputs.cpu()[mask.cpu()],rtol=0,atol=0)
            folder=stage/'kinematic'/f'seed_{seed:03d}'/variant;folder.mkdir(parents=True)
            payload={'sampled_tensor':pred,'scene_tensor_gt':torch.from_numpy(reference).unsqueeze(0),
                     'valid_mask':validity.cpu(),'task_mask':mask.cpu(),'conditioned_scene_tensor':inputs.cpu()}
            torch.save(payload,folder/'prediction.pt');write(folder/'guidance.json',refiner.records)
            np.save(folder/'controls.npy',refiner.previous_controls.cpu().numpy())
            evaluation=evaluate(payload,reference_root=OLD,lane_root=OUT/'lanes',variant=variant)
            observed=decode(values)
            observed_speed=float(np.linalg.norm((observed[0,4,:2]-observed[0,3,:2])/.5))
            speeds=evaluation['motion']['E']['xy_interval_speeds_mps']
            progress=float(sum(speeds)*.5)
            speed_check={'scope':'predeclared scene033 behavior benchmark; not a general speed rule',
                         'observed_reference_speed_mps':observed_speed,
                         'final_interval_speed_mps':speeds[-1],
                         'forecast_progress_m':progress}
            if variant=='normal':
                speed_check.update(final_speed_threshold_mps=.85*observed_speed,
                                   minimum_interval_speed_mps=float(min(speeds)),
                                   minimum_interval_speed_threshold_mps=.85*observed_speed,
                                   progress_threshold_m=.9*observed_speed*8.)
                speed_check['accepted']=(speeds[-1]>=speed_check['final_speed_threshold_mps']
                                         and min(speeds)>=speed_check['minimum_interval_speed_threshold_mps']
                                         and progress>=speed_check['progress_threshold_m'])
            else:
                leader_speed=float(np.linalg.norm(observed[2,-1,4:6]))
                normal_min=pair_records[0]['acceptance']['motion']['E']['minimum_first2p5s_xy_speed_mps']
                event_min=evaluation['motion']['E']['minimum_first2p5s_xy_speed_mps']
                speed_check.update(final_speed_threshold_mps=.8*min(observed_speed,leader_speed),
                                   early_additional_slowdown_mps=normal_min-event_min,
                                   early_additional_slowdown_threshold_mps=1.)
                speed_check['accepted']=(speeds[-1]>=speed_check['final_speed_threshold_mps']
                                         and speed_check['early_additional_slowdown_mps']>=1.)
            evaluation['speed_behavior']=speed_check
            evaluation['approved_for_downstream'] &= speed_check['accepted']
            write(folder/'acceptance.json',evaluation)
            row={'method':'kinematic','seed':seed,'variant':variant,'seconds':elapsed,
                 'path':str(folder.relative_to(stage)),'noise_sha256':h,
                 'prediction_sha256':digest(folder/'prediction.pt'),'acceptance':evaluation,
                 'controlled_slots':refiner.controlled_slots,'lane_risk_relaxation':refiner.lane_risk_relaxation,'bundle_exported':False}
            if evaluation['approved_for_downstream']:
                arrays,meta=export_arrays(payload,{'scenario_token':record['token'],'database':record['database'],
                                                 'method':'kinematic','variant':variant,'seed':seed,
                                                 'prediction_sha256':row['prediction_sha256'],
                                                 'in_sampler_optimization':True,'offline_trajectory_postprocessing':False,
                                                 'independent_acceptance_passed':True})
                save_bundle(folder/'approved_bundle',arrays,meta);row['bundle_exported']=True
            run['records'].append(row);pair_records.append(row);write(stage/'run.json',run)
            print('CRUISE_SAMPLE_COMPLETE',seed,variant,'seconds',round(elapsed,1),
                  'primary',evaluation['primary_three_accepted'],'full',evaluation['full_scene_accepted'],
                  'EA',evaluation['roles']['EA']['overlap_samples'],'EC',evaluation['roles']['EC']['overlap_samples'],
                  'lane',evaluation['original_lane'],'motion',evaluation['motion']['E'],flush=True)
        run['pairs'].append({'method':'kinematic','seed':seed,
                             'primary_pair_accepted':all(r['acceptance']['primary_three_accepted'] for r in pair_records),
                             'speed_behavior_pair_accepted':all(r['acceptance']['speed_behavior']['accepted'] for r in pair_records),
                             'primary_and_speed_pair_accepted':all(r['acceptance']['primary_three_accepted'] and r['acceptance']['speed_behavior']['accepted'] for r in pair_records),
                             'full_scene_pair_accepted':all(r['acceptance']['approved_for_downstream'] for r in pair_records)})
        write(stage/'run.json',run)
    check_source();assert versions==[p._version for p in model.parameters()]
    assert digest(checkpoint)==CHECKPOINT_SHA256
    run.update(status='complete',model_parameters_unchanged=True,existing_upgrade_source_unchanged=True)
    write(stage/'run.json',run);print('CRUISE_STAGE_COMPLETE',flush=True)


if __name__=='__main__':
    main()
