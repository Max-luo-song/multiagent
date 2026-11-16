"""Run frozen V4 control-space guidance on independent selected scenes."""
import functools,hashlib,json,os,random,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/nexus_multiscene_20261027'
sys.path[:0]=[str(ROOT),str(ROOT/'scripts/pipeline'),str(ROOT/'third_party/nuplan-devkit'),str(ROOT/'third_party/alf')]
os.environ.update(MPLBACKEND='Agg',WANDB_MODE='disabled',NUPLAN_DATA_STORE='local',PYTHONDONTWRITEBYTECODE='1')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def main():
    import numpy as np,torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from nuplan_extent.planning.training.preprocessing.features.scene_tensor import SceneTensor
    from nuplan_extent.planning.training.modeling.models.diffusion.omega_cruise_initialized import CorridorCruiseConfig,InitializedCruiseRefiner
    from nexus_3dgs.bundle import decode
    from nexus_3dgs.multiscene_acceptance import evaluate
    from run_repro_demo import CHECKPOINT_SHA256,model_state_dict
    from run_nexus_omega_v2 import segments_from_map
    torch.set_num_threads(2)
    for key in ('GT_REPLACE','EXTRAPOLATE','NO_PHYPE'):
        if os.environ.get(key):raise ValueError('Unset '+key)
    protocol=json.loads((OUT/'protocol.json').read_text());assert protocol['status']=='prepared'
    checkpoint=ROOT.parent/'checkpoints/nuplan.ckpt';assert digest(checkpoint)==CHECKPOINT_SHA256==protocol['checkpoint_sha256']
    protected=protocol['source_sha256'].copy()
    for p in [Path(__file__),ROOT/'nexus_3dgs/multiscene_acceptance.py',ROOT/'scripts/pipeline/prepare_nexus_multiscene_validation.py',ROOT/'scripts/pipeline/nexus_multiscene_routes.py']:protected[str(p)]=digest(p)
    inputs={str(p):digest(p) for p in OUT.rglob('*') if p.is_file() and p.suffix in ('.npz','.pt','.wkb')}
    def check():
        for p,h in protected.items():assert digest(p)==h,p
        for p,h in inputs.items():assert digest(p)==h,p
    check();stage=OUT/'frozen_v4_seed0';stage.mkdir(exist_ok=False)
    run={'status':'running','protocol':protocol,'source_sha256':protected,'input_sha256':inputs,'records':[],'skipped':[],'pairs':[]};write(stage/'run.json',run)
    cfg=OmegaConf.load(ROOT/'artifacts/nuplan_evaluation/independent256_v1/config.yaml');model=instantiate(cfg.model)
    model.load_state_dict(model_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)),strict=True)
    model=model.cuda().eval().requires_grad_(False);versions=[p._version for p in model.parameters()]
    model.diffuser.constrain_mode=[];model.diffuser.constraint_gamma=[]
    model.diffuser._generate_scheduling_matrix=functools.partial(model.diffuser._generate_scheduling_matrix,zero_init=False)
    for info in protocol['cases']:
        idx=info['scenario']['index'];case=OUT/f'case_{idx:03d}';roles=info['roles'];A=roles['A'];C=roles['C'];pair=[]
        if not info['given_A_input_usable']:
            run['skipped'].append({'index':idx,'reason':'prescribed A input failed predeclared physical checks','precheck':info['input_precheck']});write(stage/'run.json',run);continue
        scene=torch.load(case/'scene.pt',weights_only=False);noisehash=None;warm=None;predictions={}
        for variant in ('normal','event'):
            check();random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
            with np.load(case/f'conditions_{variant}.npz') as data:values=data['values'];keep=data['keep'];valid=data['valid']
            assert not keep[[0,C],5:].any() and keep[A].all() and (values[~keep]==0).all()
            inputs_gpu=torch.from_numpy(values).unsqueeze(0).cuda();mask=torch.from_numpy(keep).unsqueeze(0).cuda();validity=torch.from_numpy(valid.astype(np.float32)).unsqueeze(0).cuda()
            features=SceneTensor(tensor=inputs_gpu,validity=validity,road_graph=torch.from_numpy(scene['road_graph']).unsqueeze(0).cuda(),road_graph_validity=torch.from_numpy(scene['road_graph_validity']).unsqueeze(0).cuda())
            segments=torch.as_tensor(segments_from_map(scene['road_graph'],scene['road_graph_validity']),device='cuda')
            noise=torch.randn_like(inputs_gpu);nh=hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest()
            if noisehash is None:noisehash=nh
            else:assert nh==noisehash
            refiner=InitializedCruiseRefiner(CorridorCruiseConfig(**protocol['config']),inputs_gpu,validity,mask,segments,route_path=case/'map/routes.npz')
            refiner.set_drivable_sdf(case/'reference/inputs/drivable_sdf.npz')
            if info['background_count']:refiner.set_background_replay(case/'reference/inputs/background_replay.npz')
            initial=refiner.lane_following_controls().detach().cpu()
            if warm is None:warm=initial
            else:assert torch.equal(warm,initial)
            model.diffuser.anchor_refiner=refiner;print('SAMPLE_START',idx,variant,'actors',int(valid[:,4].sum()),flush=True);start=time.monotonic()
            with torch.no_grad():pred=model.forward_inference({'scene_tensor':features,'task_mask':mask},noise=noise,return_intermidates=False)['sampled_tensor'].cpu()
            torch.cuda.synchronize();elapsed=time.monotonic()-start
            torch.testing.assert_close(pred[mask.cpu()],inputs_gpu.cpu()[mask.cpu()],rtol=0,atol=0)
            assert torch.isfinite(pred[validity.cpu().bool()]).all()
            payload={'sampled_tensor':pred,'scene_tensor_gt':torch.from_numpy(scene['reference']).unsqueeze(0),'valid_mask':validity.cpu(),'task_mask':mask.cpu(),'conditioned_scene_tensor':inputs_gpu.cpu()}
            folder=stage/f'case_{idx:03d}'/variant;folder.mkdir(parents=True)
            torch.save(payload,folder/'prediction.pt');write(folder/'guidance.json',refiner.records);np.save(folder/'controls.npy',refiner.previous_controls.cpu().numpy())
            evaluation=evaluate(payload,reference_root=case/'reference',lane_root=case/'map/lanes',variant=variant,actor_slot=A,third_slot=C)
            write(folder/'acceptance.json',evaluation)
            row={'index':idx,'variant':variant,'seed':0,'seconds':elapsed,'path':str(folder.relative_to(stage)),'noise_sha256':nh,'prediction_sha256':digest(folder/'prediction.pt'),'acceptance':evaluation,'approved_bundle_exported':False}
            run['records'].append(row);pair.append(row);predictions[variant]=decode(pred[0].numpy());write(stage/'run.json',run)
            print('SAMPLE_COMPLETE',idx,variant,round(elapsed,1),'primary',evaluation['primary_three_accepted'],'full',evaluation['full_scene_accepted'],'overlaps',evaluation['role_any_overlap_samples'],'lane',evaluation['original_lane'],'Espeed',evaluation['motion']['E']['xy_interval_speeds_mps'],flush=True)
        ns=predictions['normal'];es=predictions['event'];n=pair[0]['acceptance']['motion']['E']['xy_interval_speeds_mps'];e=pair[1]['acceptance']['motion']['E']['xy_interval_speeds_mps'];v0=info['observed_speeds_mps']['E']
        from audit_nexus_omega_v2 import sample_positions
        from nexus_3dgs.scene_projection import pair_separations
        mixed=ns[[0,A]].copy();mixed[1]=es[A]
        xy,yaw,seen=sample_positions(mixed,valid[[0,A]],np.arange(1,81)*.1,np);boxes=np.concatenate([xy.transpose(1,0,2),yaw.T[...,None]],-1);boxes[:,0,:2]+=1.461*np.c_[np.cos(boxes[:,0,2]),np.sin(boxes[:,0,2])]
        sep,vp=pair_separations(boxes,seen.T,decode(scene['reference'])[[0,A],4,6:8])
        stats={'index':idx,'primary_pair_accepted':all(r['acceptance']['primary_three_accepted'] for r in pair),'full_scene_pair_accepted':all(r['acceptance']['full_scene_accepted'] for r in pair),'same_noise_and_warm_start':True,'normal_min_E_speed_mps':min(n),'normal_final_E_speed_mps':n[-1],'normal_progress_m':sum(n)*.5,'event_early_min_E_speed_mps':min(e[:5]),'event_final_E_speed_mps':e[-1],'early_extra_slowdown_mps':min(n[:5])-min(e[:5]),'normal_E_min_speed_fraction_observed':min(n)/v0,'normal_progress_fraction_observed_cruise':sum(n)*.5/(v0*8),'E_max_position_change_m':float(np.linalg.norm(ns[0,5:,:2]-es[0,5:,:2],axis=-1).max()),'C_max_position_change_m':float(np.linalg.norm(ns[C,5:,:2]-es[C,5:,:2],axis=-1).max()),'normal_E_plus_event_A_overlap_samples':int(((sep[:,0,1]<0)&vp[:,0,1]).sum()),'normal_E_plus_event_A_min_gap_m':float(sep[:,0,1].min()),'strict_causal_chain_proven':False}
        run['pairs'].append(stats);write(stage/'run.json',run);print('PAIR_COMPLETE',stats,flush=True)
    check();assert versions==[p._version for p in model.parameters()];assert digest(checkpoint)==CHECKPOINT_SHA256
    run.update(status='complete',weights_unchanged=True,existing_source_unchanged=True);write(stage/'run.json',run);print('MULTISCENE_COMPLETE',flush=True)
if __name__=='__main__':main()
