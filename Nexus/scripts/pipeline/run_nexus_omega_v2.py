"""Second-stage structural and inherited-constraint ablations for Nexus.

The original project provides audited event compilation/export; experiment
code and outputs live in this worktree. No weights or old artifacts change.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BRIDGE = Path('/inspire/hdd3/project/intelligent-driving-agent/public/workspace/hzy/nexus_repro/Nexus')
CREATED_OUTPUT = None


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def restore_history_velocity(original,raw_tensor,valid):
    """Use observed history only, preserving all historical pose/size bits."""
    import numpy as np
    scale=np.array([104,104,1,1,4,4,5,1.6])
    mean=np.array([0,0,0,0,0,0,4.5,2])
    history=original[:,:5]*scale+mean
    observed=raw_tensor[:,:5,:8]*scale+mean
    velocity=np.empty_like(history[...,4:6])
    velocity[:,0]=observed[:,0,4:6]
    eligible=valid[:,:4]&valid[:,1:5]
    velocity[:,1:]=np.where(eligible[...,None],np.diff(history[...,:2],axis=1)/.5,observed[:,1:,4:6])
    output=original.copy();output[:,:5,4:6]=(velocity/4).astype(original.dtype)
    return output


def segments_from_map(graph, valid):
    import numpy as np
    # Schema: normalized XY, one-hot LANE / LANE_CONNECTOR / STOP_LINE /
    # CROSSWALK, speed. Keep segments within lane/connector polylines only.
    xy = graph[..., :2]*104.
    point_valid = valid.any(-1) if valid.ndim == graph.ndim else valid
    allowed = point_valid.astype(bool) & (graph[..., 2:4].sum(-1) > .5)
    active = allowed[:, :-1] & allowed[:, 1:]
    seg = np.stack([xy[:, :-1], xy[:, 1:]], -2)[active]
    return seg[np.linalg.norm(seg[:, 1]-seg[:, 0], axis=-1) > 1e-4].astype('float32')


def audit(prediction, history, valid, keep, actor_slot, segments):
    """Independent NumPy SAT using observed dimensions and rear-axle offset."""
    import numpy as np
    from nexus_3dgs.bundle import decode
    states, ref = decode(prediction), decode(history)
    ii, jj = np.triu_indices(len(states), 1)
    size = ref[:, 4, 6:8].copy()
    centers = states[..., :2].copy()
    yaw = np.arctan2(states[..., 3], states[..., 2])
    h = np.stack([np.cos(yaw), np.sin(yaw)], -1)
    centers[0] += 1.461*h[0]
    basis = np.stack([h, np.stack([-h[..., 1], h[..., 0]], -1)], -2)
    axes = np.concatenate([basis[ii], basis[jj]], -2)
    ri = (np.abs(np.sum(basis[ii, :, None]*axes[..., None, :], -1))*size[ii, None, None]/2).sum(-1)
    rj = (np.abs(np.sum(basis[jj, :, None]*axes[..., None, :], -1))*size[jj, None, None]/2).sum(-1)
    sep = (np.abs(((centers[jj]-centers[ii])[..., None, :]*axes).sum(-1))-ri-rj).max(-1)
    pair_valid = valid[ii] & valid[jj]
    pair_valid[:, :5] = False
    overlap = (sep < 0) & pair_valid
    bus_pair = np.flatnonzero((ii == 0) & (jj == actor_slot))[0]
    ego_rows = (ii == 0)
    v = np.diff(states[..., :2], axis=1)/.5
    vv = valid[:, :-1] & valid[:, 1:]
    acc = np.diff(v, axis=1)/.5
    av = vv[:, :-1] & vv[:, 1:]
    jerk = np.diff(acc, axis=1)/.5
    jv = av[:, :-1] & av[:, 1:]
    heading_rate = np.abs(np.arctan2(np.sin(np.diff(yaw)), np.cos(np.diff(yaw))))/.5
    size_error = np.abs(states[..., 6:8]-size[:, None]).max(-1)
    result = {'physical_box_pair_frames_overlapping':int(overlap.sum()),
        'ego_any_overlap_frames':int(overlap[ego_rows].any(0).sum()),
        'ego_bus_overlap_frames':int(overlap[bus_pair].sum()),
        'minimum_ego_bus_sat_separation_m':float(sep[bus_pair, pair_valid[bus_pair]].min()),
        'ego_peak_speed_mps':float(np.linalg.norm(v[0, 4:][vv[0, 4:]], axis=-1).max()),
        'ego_peak_acceleration_mps2':float(np.linalg.norm(acc[0, 3:][av[0, 3:]], axis=-1).max()),
        'ego_peak_jerk_mps3':float(np.linalg.norm(jerk[0, 2:][jv[0, 2:]], axis=-1).max()),
        'ego_peak_yaw_rate_rps':float(heading_rate[0, 4:][vv[0, 4:]].max()),
        'max_future_size_error_m':float(size_error[:, 5:][valid[:, 5:]].max()),
        'all_agent_acceleration_over_5mps2_samples':int(((np.linalg.norm(acc[:, 3:], axis=-1)>5)&av[:, 3:]).sum()),
        'collision_scope':'fixed observed dimensions; oriented boxes; ego rear axle +1.461m; 16 discrete samples; predicted vehicles only',
        'future_presence':'logged validity; no all-obstacle or continuous-time safety claim'}
    if len(segments):
        def distances(points):
            a, d = segments[:, 0], segments[:, 1]-segments[:, 0]
            q = points[..., None, :]-a
            u = np.clip((q*d).sum(-1)/np.maximum((d*d).sum(-1), 1e-8), 0, 1)
            return np.linalg.norm(q-u[..., None]*d, axis=-1).min(-1)
        distance = distances(states[..., :2])
        eligible = distances(ref[:, 4, :2]) < 2.5
        road_valid = valid[:, 5:] & eligible[:, None]
        result['lane_segment_proxy_over_2p5m_frames'] = int(((distance[:, 5:]>2.5)&road_valid).sum())
        result['lane_segment_proxy_valid_frames'] = int(road_valid.sum())
        result['ego_max_lane_segment_distance_m'] = float(distance[0, 5:][valid[0, 5:]].max())
    return result


def main():
    global CREATED_OUTPUT
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bridge-root', type=Path, default=DEFAULT_BRIDGE)
    p.add_argument('--request', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--methods', nargs='+', default=['baseline','anchor'])
    p.add_argument('--iterations', type=int, default=40)
    p.add_argument('--kl-budget', type=float, default=20.)
    p.add_argument('--collision-weight', type=float, default=30.)
    p.add_argument('--solver', choices=['adam_physical','adam','lbfgs'], default='adam_physical')
    p.add_argument('--reference-run', type=Path, help='Require exact agreement with a saved candidate')
    p.add_argument('--native-constraints',choices=['released','none','map_clip'],default='released')
    p.add_argument('--profile',choices=['legacy','structural'],default='legacy')
    p.add_argument('--acceleration-weight',type=float,default=2.)
    p.add_argument('--clearance',type=float,default=.15)
    p.add_argument('--dense-substeps',type=int,default=5)
    p.add_argument('--background-replay',type=Path)
    p.add_argument('--history-velocity',choices=['released','backward_xy'],default='released')
    p.add_argument('--drivable-sdf',type=Path)
    p.add_argument('--kl-weight',type=float,default=.02)
    p.add_argument('--road-weight',type=float,default=2.)
    p.add_argument('--acceleration-limit',type=float,default=5.)
    p.add_argument('--lock-predicted-heading',action='store_true')
    p.add_argument('--following-weight',type=float,default=0.)
    p.add_argument('--motion-qp',action='store_true')
    p.add_argument('--ego-motion-weight',type=float,default=0.)
    p.add_argument('--ego-heading-only',action='store_true')
    p.add_argument('--ego-curvature-weight',type=float,default=0.)
    args = p.parse_args()
    if min(args.ego_motion_weight,args.ego_curvature_weight)<0 or ((args.ego_motion_weight or args.ego_curvature_weight) and args.profile!='structural'):
        raise ValueError('Ego motion preference needs structural profile and nonnegative weight')
    if args.ego_heading_only and (args.profile!='structural' or args.lock_predicted_heading):
        raise ValueError('Ego heading optimization needs structural profile and unlocked heading')
    bridge = args.bridge_root.resolve()
    sys.path[:0] = [str(ROOT), str(bridge/'scripts/pipeline'), str(bridge),
                   str(bridge/'third_party/nuplan-devkit'), str(bridge/'third_party/alf')]
    os.environ.update(MPLBACKEND='Agg', WANDB_MODE='disabled', NUPLAN_DATA_STORE='local',
                      NUMBA_CACHE_DIR=str(ROOT/'artifacts/nexus_omega/numba_cache'))
    for key in ('GT_REPLACE','EXTRAPOLATE','NO_PHYPE'):
        if os.environ.get(key):
            raise ValueError('Unset sampler override '+key)
    if len(set(args.seeds)) != len(args.seeds) or any(s<0 for s in args.seeds):
        raise ValueError('Distinct nonnegative seeds required')
    if not set(args.methods) <= {'baseline','anchor','schedule','omega'} or len(set(args.methods))!=len(args.methods):
        raise ValueError('Unknown or repeated method')
    args.output.mkdir(parents=True, exist_ok=False)
    CREATED_OUTPUT = args.output
    import numpy as np
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from nuplan_extent.planning.training.preprocessing.features.scene_tensor import SceneTensor
    from nuplan_extent.planning.training.data_augmentation.reset_v import ResetVAugmentor
    from nuplan_extent.planning.training.modeling.models.diffusion.omega_guidance import AnchorRefiner, GuidanceConfig
    from nexus_3dgs.bundle import decode, export_arrays, save_bundle
    from nexus_3dgs.timed_event import compile_event
    from run_repro_demo import CHECKPOINT_SHA256, model_state_dict
    torch.set_num_threads(2)
    request_path = args.request or bridge/'artifacts/nexus_3dgs/timed_event_20261026/event_request.json'
    request = json.loads(request_path.read_text())
    if request['render_seed'] not in args.seeds:
        raise ValueError('The predeclared render seed must be included')
    request['seeds'] = args.seeds
    write(args.output/'request.json', request)
    source = bridge/request['source_run']
    plan = json.loads((source/'frozen_plan.json').read_text())
    record = next(s for s in plan['scenarios'] if s['token']==request['scenario_token'])
    raw_path = source/'raw_features'/f'{record["index"]:03d}_{record["token"]}.pt'
    checkpoint = bridge.parent/'checkpoints/nuplan.ckpt'
    assert digest(checkpoint) == CHECKPOINT_SHA256
    raw = torch.load(raw_path, map_location='cpu', weights_only=False)
    scene = SceneTensor(**{k:v.numpy().copy() for k,v in raw.items()})
    scene = ResetVAugmentor().augment({'scene_tensor':scene}, {})[0]['scene_tensor']
    original = scene.tensor[..., :8].copy()
    valid = scene.validity.astype(bool).copy()
    valid[~valid[:, :5].any(-1)] = False
    if args.history_velocity=='backward_xy':
        # Correct only observed historical velocity channels. Map, slots,
        # history poses and specified future actor conditions are unchanged.
        original=restore_history_velocity(original,raw['tensor'].numpy(),valid)
        np.testing.assert_array_equal(original[:,:5,:4],scene.tensor[:,:5,:4])
    gt_cpu = torch.from_numpy(original).unsqueeze(0)
    # Nexus attention forms an outer product by matrix multiplication, so
    # model validity must retain the original floating-point representation.
    valid_cpu = torch.from_numpy(valid.astype(scene.validity.dtype)).unsqueeze(0)
    # Resolve slot using the previously audited identity metadata. No DB
    # rebuild, sensor conversion or future values enter condition compilation.
    old_bundle = bridge/'artifacts/nexus_3dgs/real_vs_nexus/inputs/goal/manifest.json'
    meta = json.loads(old_bundle.read_text())
    if meta['scenario_token'] != request['scenario_token']:
        raise ValueError('This pilot uses the audited identity manifest for a89c1931e7b25976 only')
    slot = next(b['nexus_slot'] for b in meta['agent_bindings'] if b['track_token']==request['actor_track_token'])
    compiled = {name:compile_event(original, valid, slot, request, name) for name in ('normal','event')}
    for name, (_, _, cond_audit) in compiled.items():
        write(args.output/f'{name}_condition.json', cond_audit)
    config = GuidanceConfig(iterations=args.iterations, kl_budget=args.kl_budget,
                            collision_weight=args.collision_weight, solver=args.solver)
    config.acceleration_weight=args.acceleration_weight
    config.clearance_m=args.clearance
    config.kl_weight=args.kl_weight
    config.road_weight=args.road_weight
    config.max_acceleration_mps2=args.acceleration_limit
    refiner_type=AnchorRefiner
    if args.profile=='structural':
        from nuplan_extent.planning.training.modeling.models.diffusion.omega_structural import StructuralConfig, StructuralRefiner
        config=StructuralConfig(**asdict(config),dense_substeps=args.dense_substeps,
                                optimize_heading=not args.lock_predicted_heading,following_weight=args.following_weight,
                                ego_heading_only=args.ego_heading_only,
                                motion_qp=args.motion_qp)
        refiner_type=StructuralRefiner
        if args.ego_motion_weight or args.ego_curvature_weight:
            from nuplan_extent.planning.training.modeling.models.diffusion.omega_behavior import BehaviorConfig,BehaviorRefiner
            config=BehaviorConfig(**asdict(config),ego_motion_weight=args.ego_motion_weight,
                                  ego_curvature_weight=args.ego_curvature_weight)
            refiner_type=BehaviorRefiner
    segments = segments_from_map(scene.road_graph, scene.road_graph_validity)
    source_files=[Path(__file__),ROOT/'nuplan_extent/planning/training/modeling/models/diffusion/omega_guidance.py',
                  ROOT/'nuplan_extent/planning/training/modeling/models/diffusion/scenediffusion.py',
                  ROOT/'nexus_3dgs/timed_event.py']
    if args.profile=='structural':source_files.append(ROOT/'nuplan_extent/planning/training/modeling/models/diffusion/omega_structural.py')
    if args.motion_qp:source_files.append(ROOT/'nuplan_extent/planning/training/modeling/models/diffusion/omega_motion_qp.py')
    if args.ego_motion_weight or args.ego_curvature_weight:source_files.append(ROOT/'nuplan_extent/planning/training/modeling/models/diffusion/omega_behavior.py')
    for file in source_files:
        target=args.output/'sources'/file.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,target)
    write(args.output/'protocol.json', {'paper':'https://arxiv.org/html/2512.07661v4',
        'implementation':'independent OMEGA-inspired; not official OMEGA reproduction',
        'config':asdict(config), 'seeds':args.seeds, 'methods':args.methods,
        'profile':args.profile,'native_constraints':args.native_constraints,
        'history_velocity':args.history_velocity,
        'background_replay':str(args.background_replay) if args.background_replay else None,
        'drivable_sdf':str(args.drivable_sdf) if args.drivable_sdf else None,
        'scenario':record, 'conditions':'same history + local bus event; ego future free',
        'future_presence':'logged mask', 'raw_features_sha256':digest(raw_path),
        'checkpoint_sha256':CHECKPOINT_SHA256,'lane_segments':len(segments),
        'source_sha256':{str(f.relative_to(ROOT)):digest(f) for f in source_files},
        'planned_render_seed':request['render_seed'], 'training_updates':0, 'offline_trajectory_postprocessing':False,
        'differences_from_paper':['finite-difference dynamics instead of auxiliary bicycle controls',
            'oriented physical boxes instead of two circles', 'local lane-segment proxy instead of route DFS/drivable polygons',
            'no adversarial SE-IBR; prescribed actor event retained', 'hyperparameters chosen for engineering pilot, not paper defaults']})
    print('OMEGA loading existing official model', flush=True)
    cfg = OmegaConf.load(source/'config.yaml')
    model = instantiate(cfg.model)
    model.load_state_dict(model_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)), strict=True)
    model = model.cuda().eval().requires_grad_(False)
    versions = [p._version for p in model.parameters()]
    if args.native_constraints=='none':
        model.diffuser.constrain_mode=[]
        model.diffuser.constraint_gamma=[]
    elif args.native_constraints=='map_clip':
        model.diffuser.constrain_mode=['map','clip']
        model.diffuser.constraint_gamma=[1.,1.]
    assert str(ROOT) in sys.modules[model.diffuser.__class__.__module__].__file__
    print(f'OMEGA model loaded; lane_segments={len(segments)}', flush=True)
    report = {'status':'running','records':[],'pairs':[]}
    write(args.output/'run.json', report)
    original_mode = model.diffuser.scheduling_matrix
    for method in args.methods:
        for seed in args.seeds:
            predictions, noise_hashes = {}, []
            for name, (tensor, keep, _) in compiled.items():
                random.seed(seed); np.random.seed(seed)
                torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                inputs = torch.from_numpy(tensor).unsqueeze(0).cuda()
                mask = torch.from_numpy(keep).unsqueeze(0).cuda()
                validity = valid_cpu.cuda()
                features = SceneTensor(tensor=inputs, validity=validity,
                    road_graph=torch.from_numpy(scene.road_graph.copy()).unsqueeze(0).cuda(),
                    road_graph_validity=torch.from_numpy(scene.road_graph_validity.copy()).unsqueeze(0).cuda())
                noise = torch.randn_like(inputs)
                noise_hash = hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest()
                noise_hashes.append(noise_hash)
                refiner = None
                if method in ('anchor','omega'):
                    refiner = refiner_type(config, inputs, validity, mask, torch.as_tensor(segments,device='cuda'))
                    if args.background_replay:
                        if not hasattr(refiner,'set_background_replay'):
                            raise ValueError('Background replay requires structural profile')
                        refiner.set_background_replay(args.background_replay)
                    if args.drivable_sdf:
                        if not hasattr(refiner,'set_drivable_sdf'):raise ValueError('SDF requires structural profile')
                        refiner.set_drivable_sdf(args.drivable_sdf)
                model.diffuser.anchor_refiner = refiner
                model.diffuser.scheduling_matrix = 'omega_two_phase' if method in ('schedule','omega') else original_mode
                model.eval()
                start = time.monotonic()
                with torch.no_grad():
                    prediction = model.forward_inference({'scene_tensor':features,'task_mask':mask},
                        noise=noise, return_intermidates=False)['sampled_tensor'].detach().cpu()
                torch.cuda.synchronize()
                elapsed = time.monotonic()-start
                torch.testing.assert_close(prediction[mask.cpu()], inputs.cpu()[mask.cpu()], rtol=0, atol=0)
                assert torch.isfinite(prediction[valid_cpu.bool()]).all()
                assert not mask[0, 0, 5:].any() and (inputs[~mask]==0).all()
                out = args.output/method/f'seed_{seed:03d}'/name
                out.mkdir(parents=True)
                payload = {'sampled_tensor':prediction, 'scene_tensor_gt':gt_cpu, 'valid_mask':valid_cpu,
                           'task_mask':mask.cpu(), 'conditioned_scene_tensor':inputs.cpu()}
                path = out/'prediction.pt'
                torch.save(payload, path)
                arrays, metadata = export_arrays(payload, {'database':record['database'],
                    'scenario_token':record['token'],'prediction':str(path.resolve()),'prediction_sha256':digest(path),
                    'seed':seed,'omega_method':method,'condition_source':'history + explicit local event',
                    'in_sampler_optimization':refiner is not None,'offline_trajectory_postprocessing':False})
                save_bundle(out, arrays, metadata)
                metrics = audit(prediction[0].numpy(), original, valid, keep, slot, segments)
                row = {'method':method,'seed':seed,'variant':name,'seconds':elapsed,
                    'path':str(out.relative_to(args.output)),'noise_sha256':noise_hash,
                    'conditions_exact':True, 'ego_future_free':True,'metrics':metrics}
                if refiner:
                    write(out/'guidance.json', refiner.records)
                    row['max_step_kl'] = max((r['kl_max'] for r in refiner.records),default=0.)
                if args.reference_run:
                    reference = args.reference_run/method/f'seed_{seed:03d}'/name/'prediction.pt'
                    old = torch.load(reference,map_location='cpu',weights_only=False)['sampled_tensor']
                    row['reference_max_abs_difference'] = float((old-prediction).abs().max())
                    torch.testing.assert_close(old,prediction,rtol=0,atol=0)
                if method=='baseline':
                    previous = bridge/f'artifacts/nexus_3dgs/timed_event_20261026/raw/seed_{seed:03d}/{name}/prediction.pt'
                    if previous.exists():
                        old = torch.load(previous,map_location='cpu',weights_only=False)['sampled_tensor']
                        row['saved_baseline_max_abs_difference'] = float((old-prediction).abs().max())
                report['records'].append(row)
                predictions[name] = decode(prediction[0].numpy())
                write(args.output/'run.json', report)
                print(f'OMEGA {method} seed={seed} {name} seconds={elapsed:.1f} metrics={metrics}',flush=True)
            assert noise_hashes[0] == noise_hashes[1]
            difference = np.linalg.norm(predictions['event'][..., :2]-predictions['normal'][..., :2], axis=-1)
            report['pairs'].append({'method':method,'seed':seed,'same_initial_noise':True,
                'ego_max_position_response_m':float(difference[0, 5:][valid[0, 5:]].max())})
    assert versions == [p._version for p in model.parameters()]
    report.update(status='complete',model_parameters_unchanged=True)
    write(args.output/'run.json', report)
    print('OMEGA_COMPLETE',flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        if CREATED_OUTPUT is not None:
            path=CREATED_OUTPUT/'run.json'
            report=json.loads(path.read_text()) if path.exists() else {'records':[]}
            report.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=repr(exc))
            write(path,report)
        raise
