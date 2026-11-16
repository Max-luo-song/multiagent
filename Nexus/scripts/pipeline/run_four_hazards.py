"""Run matched normal/event Nexus samples; never repair sampled outputs."""
import functools,hashlib,json,os,random,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/nexus_four_hazards_20261110'
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
 from run_repro_demo import CHECKPOINT_SHA256,model_state_dict
 from run_nexus_omega_v2 import segments_from_map
 from audit_four_hazards import evaluate,compare
 torch.set_num_threads(2);protocol=json.loads((OUT/'protocol.json').read_text());assert protocol['status']=='prepared'
 checkpoint=ROOT.parent/'checkpoints/nuplan.ckpt';assert digest(checkpoint)==CHECKPOINT_SHA256==protocol['checkpoint_sha256']
 protected={str(p):digest(p) for p in list((ROOT/'nuplan_extent/planning/training/modeling/models/diffusion').glob('omega*.py'))+[Path(__file__),ROOT/'scripts/pipeline/audit_four_hazards.py',ROOT/'scripts/pipeline/prepare_four_hazards.py',ROOT/'scripts/pipeline/nexus_multiscene_routes.py']}
 inputs={str(p):digest(p) for c in protocol['cases'] for p in (OUT/c['case']).rglob('*') if p.is_file() and p.suffix in ('.npz','.pt','.wkb')}
 def check():
  for p,h in {**protected,**inputs}.items():assert digest(p)==h,p
 stage=OUT/'frozen_v4_seed0';stage.mkdir(exist_ok=False)
 run={'status':'running','protocol':protocol,'source_sha256':protected,'input_sha256':inputs,'records':[],'pairs':[]};write(stage/'run.json',run)
 cfg=OmegaConf.load(ROOT/'artifacts/nuplan_evaluation/independent256_v1/config.yaml');model=instantiate(cfg.model)
 model.load_state_dict(model_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)),strict=True);model=model.cuda().eval().requires_grad_(False);versions=[p._version for p in model.parameters()]
 model.diffuser.constrain_mode=[];model.diffuser.constraint_gamma=[];model.diffuser._generate_scheduling_matrix=functools.partial(model.diffuser._generate_scheduling_matrix,zero_init=False)
 for info in protocol['cases']:
  case=OUT/info['case'];scene=torch.load(case/'scene.pt',weights_only=False);warm=None;nh=None;predictions={}
  for variant in ('normal','event'):
   check();random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
   with np.load(case/f'conditions_{variant}.npz') as d:values=d['values'];keep=d['keep'];valid=d['valid']
   assert not keep[0,5:].any() and all(keep[s].all() for s in info['fixed_slots']) and (values[~keep]==0).all()
   x=torch.from_numpy(values).unsqueeze(0).cuda();mask=torch.from_numpy(keep).unsqueeze(0).cuda();v=torch.from_numpy(valid.astype(np.float32)).unsqueeze(0).cuda()
   features=SceneTensor(tensor=x,validity=v,road_graph=torch.from_numpy(scene['road_graph']).unsqueeze(0).cuda(),road_graph_validity=torch.from_numpy(scene['road_graph_validity']).unsqueeze(0).cuda())
   segments=torch.as_tensor(segments_from_map(scene['road_graph'],scene['road_graph_validity']),device='cuda');noise=torch.randn_like(x);noisehash=digest_bytes(noise.cpu().numpy().tobytes())
   if nh is None:nh=noisehash
   else:assert nh==noisehash
   refiner=InitializedCruiseRefiner(CorridorCruiseConfig(**protocol['config']),x,v,mask,segments,route_path=case/'map/routes.npz');refiner.set_drivable_sdf(case/'reference/inputs/drivable_sdf.npz')
   initial=refiner.lane_following_controls().detach().cpu()
   if warm is None:warm=initial
   else:assert torch.equal(warm,initial)
   model.diffuser.anchor_refiner=refiner;start=time.monotonic();print('SAMPLE_START',info['kind'],variant,'actors',len(info['active_slots']),flush=True)
   with torch.no_grad():pred=model.forward_inference({'scene_tensor':features,'task_mask':mask},noise=noise,return_intermidates=False)['sampled_tensor'].cpu()
   torch.cuda.synchronize();elapsed=time.monotonic()-start;torch.testing.assert_close(pred[mask.cpu()],x.cpu()[mask.cpu()],rtol=0,atol=0);assert torch.isfinite(pred[v.cpu().bool()]).all()
   payload={'sampled_tensor':pred,'scene_tensor_gt':torch.from_numpy(scene['reference']).unsqueeze(0),'valid_mask':v.cpu(),'task_mask':mask.cpu(),'conditioned_scene_tensor':x.cpu()}
   folder=stage/info['kind']/variant;folder.mkdir(parents=True);torch.save(payload,folder/'prediction.pt');write(folder/'guidance.json',refiner.records);np.save(folder/'controls.npy',refiner.previous_controls.cpu().numpy())
   audit=evaluate(payload,case,info,variant);write(folder/'acceptance.json',audit);predictions[variant]=payload
   row={'kind':info['kind'],'index':info['scenario']['index'],'variant':variant,'seconds':elapsed,'path':str(folder.relative_to(stage)),'noise_sha256':noisehash,'prediction_sha256':digest(folder/'prediction.pt'),'acceptance':audit};run['records'].append(row);write(stage/'run.json',run)
   print('SAMPLE_COMPLETE',info['kind'],variant,round(elapsed,1),'accepted',audit['accepted'],'pairs',audit['pairs'],'offroad',audit['body_offroad_samples'],'E_speed',audit['E_motion']['xy_interval_speeds_mps'],flush=True)
  pair=compare(predictions['normal'],predictions['event'],info);pair['kind']=info['kind'];run['pairs'].append(pair);write(stage/'run.json',run);print('PAIR_COMPLETE',pair,flush=True)
 check();assert versions==[p._version for p in model.parameters()] and digest(checkpoint)==CHECKPOINT_SHA256;run.update(status='complete',weights_unchanged=True,protected_source_unchanged=True);write(stage/'run.json',run);print('FOUR_HAZARDS_COMPLETE',flush=True)
def digest_bytes(b):return hashlib.sha256(b).hexdigest()
if __name__=='__main__':main()
