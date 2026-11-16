"""Generate paired local-time actor conditions with ego/NPC futures free."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts/pipeline')]
from run_repro_demo import CHECKPOINT_SHA256, file_sha256, model_state_dict


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    request = json.loads(a.request.read_text())
    a.output.mkdir(parents=True, exist_ok=False)
    write(a.output/'request.json', request)
    os.environ.update(MPLBACKEND='Agg', WANDB_MODE='disabled', NUPLAN_DATA_STORE='local')
    # These optional sampler switches must not silently change the experiment.
    for key in ('GT_REPLACE','EXTRAPOLATE'):
        if os.environ.get(key):raise ValueError(f'Unset sampler override {key}')
    import numpy as np
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from nuplan_extent.planning.training.preprocessing.features.scene_tensor import SceneTensor
    from nuplan_extent.planning.training.data_augmentation.reset_v import ResetVAugmentor
    from nexus_3dgs.bundle import decode, export_arrays, save_bundle
    from nexus_3dgs.timed_event import compile_event
    from diagnose_nexus_interaction import audit_variant
    torch.set_num_threads(2)
    source = ROOT/request['source_run']
    plan = json.loads((source/'frozen_plan.json').read_text())
    record = next(s for s in plan['scenarios'] if s['token']==request['scenario_token'])
    raw_path = source/'raw_features'/f'{record["index"]:03d}_{record["token"]}.pt'
    checkpoint = ROOT.parent/'checkpoints/nuplan.ckpt'
    if file_sha256(checkpoint) != CHECKPOINT_SHA256:raise ValueError('Checkpoint hash changed')
    raw = torch.load(raw_path,map_location='cpu',weights_only=False)
    scene = SceneTensor(**{k:v.numpy().copy() for k,v in raw.items()})
    scene = ResetVAugmentor().augment({'scene_tensor':scene},{})[0]['scene_tensor']
    original = scene.tensor[...,:8].copy()
    validity = scene.validity.copy()
    validity[~validity[:,:5].any(axis=-1)] = 0
    gt = torch.from_numpy(original).unsqueeze(0)
    valid = torch.from_numpy(validity).unsqueeze(0)
    # Recover identities from reference annotations using the existing audited bridge.
    _, reference = export_arrays({'sampled_tensor':gt,'scene_tensor_gt':gt,'valid_mask':valid},
        {'database':record['database'],'scenario_token':record['token']})
    binding = next(b for b in reference['agent_bindings'] if b['track_token']==request['actor_track_token'])
    slot = binding['nexus_slot']
    compiled = {}
    for name in ('normal','event'):
        tensor,keep,audit = compile_event(original,validity,slot,request,name)
        compiled[name] = (tensor,keep,audit)
        write(a.output/f'{name}_condition.json',audit)
    np.testing.assert_array_equal(compiled['normal'][1],compiled['event'][1])
    seeds = request['seeds']
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s<0 for s in seeds):
        raise ValueError('Expected distinct nonnegative integer seeds')
    if request['render_seed'] not in seeds:raise ValueError('render_seed must be predeclared in seeds')
    report = {'status':'running','request':request,'database':record['database'],'scenario_token':record['token'],
              'actor_slot':slot,'checkpoint_sha256':CHECKPOINT_SHA256,'raw_feature_sha256':file_sha256(raw_path),
              'config_sha256':file_sha256(source/'config.yaml'),'runner_sha256':file_sha256(Path(__file__)),
              'compiler_sha256':file_sha256(ROOT/'nexus_3dgs/timed_event.py'),
              'training_updates':0,'ego_future_conditioned':False,'future_presence_mask':'logged; shared in both variants',
              'trajectory_postprocessing':False,'pairs':{}}
    write(a.output/'run.json',report)
    print('TIMED_EVENT loading model',flush=True)
    cfg = OmegaConf.load(source/'config.yaml')
    model = instantiate(cfg.model)
    model.load_state_dict(model_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)),strict=True)
    model = model.cuda().eval().requires_grad_(False)
    versions = [x._version for x in model.parameters()]
    report['sampler']={'scheduling_matrix':model.diffuser.scheduling_matrix,'constraints':model.diffuser.constrain_mode,
                       'fill_scene_tensor':model.diffuser.fill_scene_tensor}
    for seed in seeds:
        predictions = {}
        pair = {}
        for name,(tensor,keep,audit) in compiled.items():
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            inputs = torch.from_numpy(tensor).unsqueeze(0).cuda()
            mask = torch.from_numpy(keep).unsqueeze(0).cuda()
            features = SceneTensor(tensor=inputs,validity=valid.cuda(),road_graph=torch.from_numpy(scene.road_graph.copy()).unsqueeze(0).cuda(),
                                   road_graph_validity=torch.from_numpy(scene.road_graph_validity.copy()).unsqueeze(0).cuda())
            noise = torch.randn_like(inputs)
            model.eval()
            with torch.inference_mode():
                pred = model.forward_inference({'scene_tensor':features,'task_mask':mask},noise=noise,return_intermidates=False)['sampled_tensor'].cpu()
            torch.testing.assert_close(pred[mask.cpu()],inputs.cpu()[mask.cpu()],rtol=0,atol=0)
            if not torch.isfinite(pred[valid.bool()]).all():raise ValueError('Nonfinite valid output')
            if mask[0,0,5:].any() or (inputs[~mask]!=0).any():raise ValueError('Future input leakage')
            payload = {'sampled_tensor':pred,'scene_tensor_gt':gt,'valid_mask':valid,
                       'task_mask':mask.cpu(),'conditioned_scene_tensor':inputs.cpu()}
            out = a.output/f'seed_{seed:03d}'/name
            out.mkdir(parents=True)
            path = out/'prediction.pt';torch.save(payload,path)
            context = {'database':record['database'],'scenario_token':record['token'],'seed':seed,
                       'prediction':str(path.resolve()),'prediction_sha256':file_sha256(path),'event_variant':name,
                       'event_request':request,'condition_source':audit['condition_source']}
            arrays,metadata = export_arrays(payload,context)
            save_bundle(out,arrays,metadata)
            predictions[name] = decode(pred[0].numpy())
            pair[name] = {'path':str(out.relative_to(a.output)), 'prediction_sha256':file_sha256(path),
                          'initial_noise_sha256':hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest(),
                          'conditions_preserved_exactly':True,'ego_future_free':True,'unconditioned_values_zero':True,
                          'native_predicted_size_geometry':audit_variant(predictions[name],validity.astype(bool),slot)}
            print(f'TIMED_EVENT seed={seed} {name} exported',flush=True)
        assert pair['normal']['initial_noise_sha256']==pair['event']['initial_noise_sha256']
        rows = []
        for actor in np.flatnonzero(validity[:,5:].any(axis=1)):
            active = validity[actor,5:].astype(bool)
            delta = np.linalg.norm(predictions['event'][actor,5:,:2]-predictions['normal'][actor,5:,:2],axis=-1)
            rows.append({'slot':int(actor),'directly_conditioned':bool(actor==slot),'max_position_difference_m':float(delta[active].max()),
                         'mean_position_difference_m':float(delta[active].mean())})
        pair['responses'] = rows
        report['pairs'][str(seed)] = pair
        write(a.output/'run.json',report)
    if versions != [x._version for x in model.parameters()]:raise ValueError('Model weights changed')
    report.update(status='complete',model_parameters_unchanged=True)
    write(a.output/'run.json',report)
    print('TIMED_EVENT_COMPLETE',flush=True)


if __name__=='__main__':main()
