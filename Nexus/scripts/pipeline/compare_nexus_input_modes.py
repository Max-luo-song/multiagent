"""Run declared BP/SG condition variants, retaining broken-schedule controls."""
import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts/pipeline')]
from run_repro_demo import CHECKPOINT_SHA256,file_sha256,model_state_dict


def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def quality(states,valid,keep,reference,actor,third):
    import numpy as np
    from nexus_3dgs.clearance import rear_axle_to_box
    from nexus_3dgs.scene_projection import pair_separations
    yaw=np.arctan2(states[...,3],states[...,2])
    boxes=np.concatenate([states[...,:2],yaw[...,None]],axis=-1).transpose(1,0,2)
    boxes[:,0]=rear_axle_to_box(boxes[:,0],1.461)
    physical=reference[:,4,6:8].copy();physical[0]=[5.176,2.297]
    # Inactive padded actors get harmless physical sizes and a false valid mask.
    physical[~valid[:,4]]=np.maximum(physical[~valid[:,4]],.1)
    sep,pairs=pair_separations(boxes,valid.T,physical)
    hits=(sep<=0)&pairs
    fixed=keep.all(axis=-1).T
    generated_pair=~(fixed[:,:,None]&fixed[:,None,:])
    generated_hits=hits&generated_pair
    future=valid[:,5:]
    unknown_future=future&~keep[:,5:].all(axis=-1)
    adjacent=valid[:,:-1]&valid[:,1:]
    generated_adjacent=adjacent&~(keep[:,:-1].all(axis=-1)&keep[:,1:].all(axis=-1))
    displacement=np.linalg.norm(np.diff(states[...,:2],axis=1),axis=-1)
    steps=displacement[generated_adjacent]
    speeds=displacement/.5
    ego_vel=np.diff(states[0,4:,:2],axis=0)/.5
    ego_acc=np.diff(ego_vel,axis=0)/.5
    headings=np.linalg.norm(states[...,2:4],axis=-1)
    sizes=states[...,6:8]
    return {'scope':'native 0.5s samples; fixed physical reference dimensions; not continuous or rendered asset geometry',
        'finite_valid_states':bool(np.isfinite(states[valid]).all()),
        'unconditioned_future_samples':int(unknown_future.sum()),
        'unconditioned_future_nonpositive_size_samples':int(((sizes<=0).any(axis=-1)[:,5:]&unknown_future).sum()),
        'unconditioned_future_undefined_heading_samples':int(((headings<1e-6)[:,5:]&unknown_future).sum()),
        'unknown_adjacent_step_p95_m':float(np.percentile(steps,95)) if steps.size else None,
        'unknown_adjacent_step_max_m':float(steps.max()) if steps.size else None,
        'ego_current_drift_from_reference_m':float(np.linalg.norm(states[0,4,:2]-reference[0,4,:2])),
        'third_current_drift_from_reference_m':float(np.linalg.norm(states[third,4,:2]-reference[third,4,:2])),
        'ego_target_current_signed_gap_m':float(sep[4,0,actor]),
        'ego_target_history_overlaps':int((sep[:5,0,actor]<=0).sum()),
        'all_history_pair_overlaps':int(hits[:5].sum()),
        'ego_target_future_overlap_samples':int((sep[5:,0,actor]<=0).sum()),
        'ego_third_future_overlap_samples':int((sep[5:,0,third]<=0).sum()),
        'all_future_pair_overlaps':int(hits[5:].sum()),
        'future_overlaps_involving_generated_states':int(generated_hits[5:].sum()),
        'ego_max_future_speed_mps':float(np.linalg.norm(ego_vel,axis=-1).max()),
        'ego_max_future_acceleration_mps2':float(np.linalg.norm(ego_acc,axis=-1).max()),
        'ego_future_path_m':float(np.linalg.norm(ego_vel,axis=-1).sum()*.5),
        'target_final_xy':states[actor,-1,:2].tolist(),
        'max_generated_future_speed_mps':float(speeds[:,4:][valid[:,4:-1]&valid[:,5:]&~keep[:,5:].all(axis=-1)].max()) if unknown_future.any() else None}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',type=Path,default=ROOT/'artifacts/nexus_3dgs/multicar_event_033_20261026')
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/nexus_3dgs/input_modes_20261026')
    p.add_argument('--velocity-preprocessing',choices=('official_reset_v','raw_velocity'),default='official_reset_v')
    p.add_argument('--modes',nargs='+',choices=('bp_history','history_full_target','future_full_target','pure_sg_free_ego','official_sg_fixed_ego','pure_sg_bad_schedule'))
    a=p.parse_args();a.case=a.case.resolve();a.output=a.output.resolve();a.output.mkdir(exist_ok=False)
    os.environ.update(MPLBACKEND='Agg',WANDB_MODE='disabled',NUPLAN_DATA_STORE='local')
    if any(os.environ.get(k) for k in ('GT_REPLACE','EXTRAPOLATE')):raise ValueError('Unset sampler overrides')
    import numpy as np
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from nuplan_extent.planning.training.preprocessing.features.scene_tensor import SceneTensor
    from nuplan_extent.planning.training.data_augmentation.reset_v import ResetVAugmentor
    from nexus_3dgs.bundle import decode,export_arrays,save_bundle
    from nexus_3dgs.input_mode_conditions import compile_mode,MODES
    torch.set_num_threads(2)
    request=json.loads((a.case/'event_request.json').read_text())
    case=json.loads((a.case/'raw/run.json').read_text());actor=case['actor_slot']
    source=ROOT/request['source_run'];plan=json.loads((source/'frozen_plan.json').read_text())
    record=next(r for r in plan['scenarios'] if r['token']==request['scenario_token'])
    raw_path=source/'raw_features'/f'{record["index"]:03d}_{record["token"]}.pt'
    raw=torch.load(raw_path,map_location='cpu',weights_only=False)
    scene=SceneTensor(**{k:v.numpy().copy() for k,v in raw.items()})
    if a.velocity_preprocessing=='official_reset_v':
        scene=ResetVAugmentor().augment({'scene_tensor':scene},{})[0]['scene_tensor']
    original=scene.tensor[...,:8].copy();valid=scene.validity.copy().astype(bool)
    valid[~valid[:,:5].any(axis=-1)]=False
    reference=decode(original)
    # The case's raw run contains the same reference/identity binding.
    from nexus_3dgs.bundle import load_bundle
    arrays,_=load_bundle(a.case/'reference')
    third=int(arrays['nexus_slots'][list(arrays['track_tokens']).index(request['third_actor_track_token_for_audit'])])
    checkpoint=ROOT.parent/'checkpoints/nuplan.ckpt'
    if file_sha256(checkpoint)!=CHECKPOINT_SHA256:raise ValueError('Checkpoint changed')
    jobs=[]
    for mode in MODES:
        if a.modes and mode not in a.modes:continue
        seeds=[0] if mode in ('official_sg_fixed_ego','pure_sg_bad_schedule') else [0,1,2]
        for seed in seeds:
            for variant in (('event',) if mode=='pure_sg_bad_schedule' else ('normal','event')):
                jobs.append({'mode':mode,'seed':seed,'variant':variant})
    protocol={'status':'declared_before_sampling','case':str(a.case),'scenario_token':request['scenario_token'],
        'actor_slot':actor,'third_slot':third,'jobs':jobs,'shared_target_longitudinal_offset_m':2.,'lateral_displacement_m':3.,
        'history_event_window_s':[-2.,0.],'future_event_window_s':[.5,2.5],
        'future_presence_mask':'logged; identical across modes','checkpoint_sha256':CHECKPOINT_SHA256,
        'raw_feature_sha256':file_sha256(raw_path),'runner_sha256':file_sha256(Path(__file__)),
        'compiler_sha256':file_sha256(ROOT/'nexus_3dgs/input_mode_conditions.py'),
        'plan_sha256':file_sha256(ROOT/'docs/nexus_input_modes_comparison_20261026.md'),
        'velocity_preprocessing':a.velocity_preprocessing,
        'scope':'single-scene functional comparison; not a benchmark; no 3DGS assets for this clip'}
    write(a.output/'protocol.json',protocol)
    gt=torch.from_numpy(original).unsqueeze(0)
    # The temporal attention computes a matrix product of this mask, requiring
    # the same floating validity representation as the original feature builder.
    valid_tensor=torch.from_numpy(valid.astype(np.float32)).unsqueeze(0)
    compiled={}
    for job in jobs:
        key=(job['mode'],job['variant'])
        if key not in compiled:
            compiled[key]=compile_mode(original,valid,actor,*key)
            tensor,keep,condition=compiled[key]
            folder=a.output/'conditions';folder.mkdir(exist_ok=True)
            np.savez_compressed(folder/f'{key[0]}_{key[1]}.npz',values=tensor,keep=keep,valid=valid)
            write(folder/f'{key[0]}_{key[1]}.json',condition)
    for variant in ('normal','event'):
        if ('bp_history',variant) in compiled and ('history_full_target',variant) in compiled:
            x=compiled[('bp_history',variant)];y=compiled[('history_full_target',variant)]
            np.testing.assert_array_equal(x[0][:,:5],y[0][:,:5])
        targets=[compiled[(mode,variant)][0][actor] for mode in ('future_full_target','pure_sg_free_ego','official_sg_fixed_ego') if (mode,variant) in compiled]
        for target in targets[1:]:np.testing.assert_array_equal(target,targets[0])
    print('INPUT_MODES loading model; jobs',len(jobs),flush=True)
    cfg=OmegaConf.load(source/'config.yaml');model=instantiate(cfg.model)
    model.load_state_dict(model_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)),strict=True)
    model=model.cuda().eval().requires_grad_(False);versions=[v._version for v in model.parameters()]
    base_schedule=model.diffuser._generate_scheduling_matrix
    report={'status':'running','protocol':protocol,'samples':{},'pairs':{},'no_training':True,
            'sampler':{'steps':model.diffuser.steps,'constraints':model.diffuser.constrain_mode,'denoise_history_runtime_override':True}}
    predictions={};noise_hashes={}
    for index,job in enumerate(jobs):
        mode,seed,variant=job['mode'],job['seed'],job['variant'];key=f'{mode}/seed_{seed:03d}/{variant}'
        values,keep,condition=compiled[(mode,variant)]
        model.diffuser._generate_scheduling_matrix=base_schedule if mode=='pure_sg_bad_schedule' else functools.partial(base_schedule,zero_init=False)
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        inputs=torch.from_numpy(values).unsqueeze(0).cuda();mask=torch.from_numpy(keep).unsqueeze(0).cuda()
        features=SceneTensor(tensor=inputs,validity=valid_tensor.cuda(),road_graph=torch.from_numpy(scene.road_graph.copy()).unsqueeze(0).cuda(),
                             road_graph_validity=torch.from_numpy(scene.road_graph_validity.copy()).unsqueeze(0).cuda())
        noise=torch.randn_like(inputs);noise_sha=hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest()
        if seed in noise_hashes:assert noise_hashes[seed]==noise_sha
        noise_hashes[seed]=noise_sha
        schedule=model.diffuser._generate_scheduling_matrix(inputs)
        effective,_=model.diffuser._filling_scene_tensor(inputs,noise.clone(),mask,schedule)
        unknown_history=valid[:,:5]&~keep[:,:5].all(axis=-1)
        effective_unknown=effective[0,0,:,:5][unknown_history]
        if mode!='pure_sg_bad_schedule' and effective_unknown.size:assert (effective_unknown==1).all()
        assert (values[~keep]==0).all()
        model.eval()
        with torch.inference_mode():
            pred=model.forward_inference({'scene_tensor':features,'task_mask':mask},noise=noise,return_intermidates=False)['sampled_tensor'].cpu()
        torch.testing.assert_close(pred[mask.cpu()],inputs.cpu()[mask.cpu()],rtol=0,atol=0)
        payload={'sampled_tensor':pred,'scene_tensor_gt':gt,'valid_mask':valid_tensor,'task_mask':mask.cpu(),'conditioned_scene_tensor':inputs.cpu()}
        out=a.output/key;out.mkdir(parents=True);torch.save(payload,out/'prediction.pt')
        states=decode(pred[0].numpy());predictions[key]=states
        item={'mode':mode,'seed':seed,'variant':variant,'path':key,'prediction_sha256':file_sha256(out/'prediction.pt'),
            'noise_sha256':noise_sha,'condition_exact':True,'unknown_values_zero':True,'condition':condition,
            'unknown_valid_history_tokens':int(unknown_history.sum()),
            'first_noise_time_unknown_history':float(effective_unknown[0]) if effective_unknown.size else None,
            'quality':quality(states,valid,keep,reference,actor,third)}
        try:
            ar,me=export_arrays(payload,{'database':record['database'],'scenario_token':record['token'],'mode':mode,'variant':variant,
                'seed':seed,'prediction':str(out/'prediction.pt'),'prediction_sha256':item['prediction_sha256'],
                'input_mode_comparison':True,'shared_target_longitudinal_input_offset_m':2.})
            save_bundle(out,ar,me);item['bundle_exported']=True
        except ValueError as exc:item.update(bundle_exported=False,bundle_error=str(exc))
        report['samples'][key]=item
        write(a.output/'run.json',report)
        print(f'INPUT_MODE {index+1}/{len(jobs)} {key}',flush=True)
    for mode in MODES:
        for seed in (0,1,2):
            nk=f'{mode}/seed_{seed:03d}/normal';ek=f'{mode}/seed_{seed:03d}/event'
            if nk not in predictions or ek not in predictions:continue
            normal,event=predictions[nk],predictions[ek]
            d=event[:,5:,:2]-normal[:,5:,:2];dist=np.linalg.norm(d,axis=-1)
            speed={name:np.linalg.norm(np.diff(values[0,4:,:2],axis=0),axis=-1)/.5 for name,values in [('normal',normal),('event',event)]}
            pair={'ego_max_difference_m':float(dist[0].max()),'ego_max_lateral_difference_m':float(np.abs(d[0,:,1]).max()),
                  'ego_max_speed_reduction_mps':float((speed['normal']-speed['event']).max()),
                  'third_max_difference_m':float(dist[third,valid[third,5:]].max()),
                  'ego_future_is_condition':mode=='official_sg_fixed_ego',
                  'same_input_histories_normal_vs_event':False if mode in ('bp_history','history_full_target') else True,
                  'causal_chain_proven':False}
            report['pairs'][f'{mode}/seed_{seed:03d}']=pair
    assert versions==[v._version for v in model.parameters()]
    report.update(status='complete',parameters_unchanged=True,noise_sha256_per_seed=noise_hashes)
    write(a.output/'run.json',report)
    print('INPUT_MODES_COMPLETE',flush=True)


if __name__=='__main__':main()
