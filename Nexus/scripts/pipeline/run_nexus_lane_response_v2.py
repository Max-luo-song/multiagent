"""Map lane-choice pilot and matched B-trajectory transfer interventions."""
import functools,hashlib,json,os,random,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/nexus_lane_response_v2_20261027'
sys.path[:0]=[str(ROOT),str(ROOT/'scripts/pipeline'),str(ROOT/'third_party/nuplan-devkit'),str(ROOT/'third_party/alf')]
os.environ.update(MPLBACKEND='Agg',WANDB_MODE='disabled',NUPLAN_DATA_STORE='local',PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(OUT/'mpl_cache'))
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def main():
 import numpy as np,torch
 from hydra.utils import instantiate
 from omegaconf import OmegaConf
 from nuplan_extent.planning.training.preprocessing.features.scene_tensor import SceneTensor
 from nuplan_extent.planning.training.modeling.models.diffusion.omega_lane_choice import LaneChoiceConfig,LaneChoiceRefiner
 from nexus_3dgs.chain_acceptance import evaluate
 from nexus_3dgs.chain_metrics import summarize
 from run_repro_demo import CHECKPOINT_SHA256,model_state_dict
 from run_nexus_omega_v2 import segments_from_map
 torch.set_num_threads(2)
 for key in ('GT_REPLACE','EXTRAPOLATE','NO_PHYPE'):
  if os.environ.get(key):raise ValueError('Unset '+key)
 protocol=json.loads((OUT/'protocol.json').read_text());assert protocol['status']=='prepared'
 protocol['chain_thresholds']={'main_speed_difference_mps':.75,'transfer_speed_difference_mps':.5,'cross_path_difference_m':.5,'position_difference_m':1.,'normal_C_with_event_B_must_overlap_for_strong_support':True,'scope':'pilot visibility diagnostics; not universal driving rules'}
 protocol['variants']=['normal','event','transfer_control','transfer_B_response'];protocol['model_training_updates']=0;protocol['main_B_C_futures_free']=True;protocol['transfer_B_future']='main model-generated normal/event, diagnostic only';protocol['C_future_always_free']=True
 protected=protocol['source_sha256'].copy()
 for p in [Path(__file__),ROOT/'nexus_3dgs/chain_acceptance.py',ROOT/'nexus_3dgs/chain_metrics.py',ROOT/'scripts/pipeline/prepare_nexus_chain_cases.py',ROOT/'scripts/pipeline/compile_nexus_chain_brake.py',ROOT/'scripts/pipeline/nexus_chain_routes.py',ROOT/'nuplan_extent/planning/training/modeling/models/diffusion/omega_lane_choice.py',ROOT/'scripts/pipeline/prepare_nexus_lane_response.py',ROOT/'scripts/pipeline/prepare_nexus_lane_response_v2.py',ROOT/'scripts/pipeline/select_nexus_lane_response.py']:
  protected[str(p)]=digest(p)
 inputs={}
 for info in protocol['cases']:
  case=OUT/f"case_{info['scenario']['index']:03d}"
  for p in case.rglob('*'):
   if p.is_file() and p.suffix in ('.pt','.npz','.wkb'):inputs[str(p)]=digest(p)
 def check():
  for p,h in protected.items():assert digest(p)==h,p
  for p,h in inputs.items():assert digest(p)==h,p
 check();checkpoint=ROOT.parent/'checkpoints/nuplan.ckpt';assert digest(checkpoint)==CHECKPOINT_SHA256
 stage=OUT/'pilot_seed0';stage.mkdir(exist_ok=False)
 run={'status':'running','protocol':protocol,'source_sha256':protected,'input_sha256':inputs,'records':[],'cases':[],'skipped':[]};write(stage/'run.json',run)
 config=OmegaConf.load(ROOT/'artifacts/nuplan_evaluation/independent256_v1/config.yaml');model=instantiate(config.model)
 model.load_state_dict(model_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)),strict=True);model=model.cuda().eval().requires_grad_(False);versions=[p._version for p in model.parameters()]
 model.diffuser.constrain_mode=[];model.diffuser.constraint_gamma=[];model.diffuser._generate_scheduling_matrix=functools.partial(model.diffuser._generate_scheduling_matrix,zero_init=False)
 for info in protocol['cases']:
  idx=info['scenario']['index'];case=OUT/f'case_{idx:03d}';roles=info['roles'];A,B,C=[roles[r] for r in ('A','B','C')]
  if not info['given_A_input_usable']:
   run['skipped'].append({'index':idx,'reason':'A input physical/background precheck failed','precheck':info['input_precheck'],'background':info.get('given_A_background_conflicts',[])});write(stage/'run.json',run);continue
  assert B!=0,'B-transfer diagnostics currently require free true ego; select an ordinary B actor'
  scene=torch.load(case/'scene.pt',map_location='cpu',weights_only=False);payloads={};acceptance={};noises={};warms={};mask_groups={}
  for variant in protocol['variants']:
   check();random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
   fixed=variant.startswith('transfer_');source='event' if variant=='event' else 'normal'
   with np.load(case/f'conditions_{source}.npz') as z:values=z['values'].copy();keep=z['keep'].copy();valid=z['valid']
   if fixed:
    donor='normal' if variant=='transfer_control' else 'event';values[B,5:]=payloads[donor]['sampled_tensor'][0,B,5:].numpy();keep[B,5:]=True
    assert np.array_equal(values[A],np.load(case/'conditions_normal.npz')['values'][A])
   assert keep[A].all() and not keep[C,5:].any() and (values[~keep]==0).all()
   if not fixed:assert not keep[B,5:].any()
   group='diagnostic' if fixed else 'main'
   if group in mask_groups:np.testing.assert_array_equal(mask_groups[group],keep)
   else:mask_groups[group]=keep.copy()
   inp=torch.from_numpy(values).unsqueeze(0).cuda();mask=torch.from_numpy(keep).unsqueeze(0).cuda();validity=torch.from_numpy(valid.astype(np.float32)).unsqueeze(0).cuda()
   features=SceneTensor(tensor=inp,validity=validity,road_graph=torch.from_numpy(scene['road_graph']).unsqueeze(0).cuda(),road_graph_validity=torch.from_numpy(scene['road_graph_validity']).unsqueeze(0).cuda())
   segments=torch.as_tensor(segments_from_map(scene['road_graph'],scene['road_graph_validity']),device='cuda');noise=torch.randn_like(inp);noisehash=hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest()
   if group in noises:assert noises[group]==noisehash
   else:noises[group]=noisehash
   refiner=LaneChoiceRefiner(LaneChoiceConfig(**protocol['config']),inp,validity,mask,segments,route_path=case/'choice_map/routes.npz',choice_path=case/'map/lane_choices.npz');refiner.set_drivable_sdf(case/'reference/inputs/drivable_sdf.npz')
   if info['background_count']:refiner.set_background_replay(case/'reference/inputs/background_replay.npz')
   seeds=refiner.maneuver_seeds();warm=torch.stack([u.detach().cpu() for _,u in seeds]);assert [label for label,_ in seeds]==protocol['lane_choice']['candidate_seeds']
   if group in warms:assert torch.equal(warms[group],warm)
   else:warms[group]=warm
   model.diffuser.anchor_refiner=refiner;print('CHAIN_SAMPLE_START',idx,variant,'actors',int(valid[:,4].sum()),flush=True);start=time.monotonic()
   with torch.no_grad():pred=model.forward_inference({'scene_tensor':features,'task_mask':mask},noise=noise,return_intermidates=False)['sampled_tensor'].cpu()
   torch.cuda.synchronize();seconds=time.monotonic()-start;torch.testing.assert_close(pred[mask.cpu()],inp.cpu()[mask.cpu()],rtol=0,atol=0);assert torch.isfinite(pred[validity.cpu().bool()]).all()
   folder=stage/f'case_{idx:03d}'/variant;folder.mkdir(parents=True)
   payload={'sampled_tensor':pred,'scene_tensor_gt':torch.from_numpy(scene['reference']).unsqueeze(0),'valid_mask':validity.cpu(),'task_mask':mask.cpu(),'conditioned_scene_tensor':inp.cpu()};torch.save(payload,folder/'prediction.pt');write(folder/'guidance.json',refiner.records);write(folder/'lane_choice.json',refiner.choice_records);np.save(folder/'controls.npy',refiner.previous_controls.cpu().numpy())
   metrics=evaluate(payload,reference_root=case/'reference',lane_root=case/'map/lanes',variant='normal' if variant in ('normal','transfer_control') else variant,actor_slot=A,reactor_slot=B,third_slot=C,fixed_reactor=fixed);write(folder/'acceptance.json',metrics)
   row={'index':idx,'variant':variant,'seed':0,'path':str(folder.relative_to(stage)),'seconds':seconds,'noise_sha256':noisehash,'prediction_sha256':digest(folder/'prediction.pt'),'acceptance':metrics,'B_future_free':not fixed,'C_future_free':True,'B_donor':donor if fixed else None,'diagnostic_only':fixed,'post_output_trajectory_edits':False};run['records'].append(row);payloads[variant]=payload;acceptance[variant]=metrics;write(stage/'run.json',run)
   print('LANE_CHOICE',variant,refiner.choice_records,flush=True);print('CHAIN_SAMPLE_COMPLETE',idx,variant,round(seconds,1),'primary',metrics['primary_three_accepted'],'full',metrics['full_scene_accepted'],'role_overlap',metrics['role_any_overlap_samples'],'Bmin',min(metrics['motion']['B']['xy_interval_speeds_mps']),'Cmin',min(metrics['motion']['C']['xy_interval_speeds_mps']),flush=True)
  result=summarize(payloads,acceptance,roles);result.update(index=idx,roles=roles);run['cases'].append(result);write(stage/'run.json',run);print('CHAIN_CASE_COMPLETE',idx,'supported',result['chain_supported_in_this_generator'],'Bchange',result['B_response']['maximum_speed_difference_mps'],'Cchange',result['C_response']['maximum_speed_difference_mps'],'Ctransfer',result['C_response_to_B_only_transfer']['maximum_speed_difference_mps'],flush=True)
 check();assert versions==[p._version for p in model.parameters()];assert digest(checkpoint)==CHECKPOINT_SHA256;run.update(status='complete',weights_unchanged=True,existing_upgrade_source_unchanged=True);write(stage/'run.json',run);print('CHAIN_COMPLETE',flush=True)
if __name__=='__main__':main()
