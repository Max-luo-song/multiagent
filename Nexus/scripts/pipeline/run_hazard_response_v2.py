"""Causal event replanning and explicit lane-change intent for four hazards."""
import copy,functools,hashlib,json,os,random,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/nexus_hazard_response_v2_20261110'
sys.path[:0]=[str(ROOT),str(ROOT/'scripts/pipeline'),str(ROOT/'third_party/nuplan-devkit'),str(ROOT/'third_party/alf')]
os.environ.update(MPLBACKEND='Agg',WANDB_MODE='disabled',NUPLAN_DATA_STORE='local',PYTHONDONTWRITEBYTECODE='1')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 import numpy as np,torch
 from hydra.utils import instantiate
 from omegaconf import OmegaConf
 from shapely import wkb
 from nuplan_extent.planning.training.preprocessing.features.scene_tensor import SceneTensor
 from nuplan_extent.planning.training.modeling.models.diffusion.omega_cruise_initialized import CorridorCruiseConfig
 from hazard_response_refiner import ResponseIntentRefiner
 from nexus_3dgs.bundle import decode,FEATURE_MEANS,FEATURE_STD
 from nexus_3dgs.geometry import pose
 from run_repro_demo import CHECKPOINT_SHA256,model_state_dict
 from run_nexus_omega_v2 import segments_from_map
 from prepare_nexus_multiscene_validation import raster
 from audit_four_hazards import evaluate,compare
 torch.set_num_threads(2);protocol=json.loads((OUT/'protocol.json').read_text());assert protocol['status']=='prepared'
 checkpoint=ROOT.parent/'checkpoints/nuplan.ckpt';assert digest(checkpoint)==CHECKPOINT_SHA256==protocol['checkpoint_sha256']
 stage=OUT/'response_seed0';stage.mkdir(exist_ok=True)
 protected={str(p):digest(p) for p in list((ROOT/'nuplan_extent/planning/training/modeling/models/diffusion').glob('omega*.py'))+[Path(__file__),ROOT/'scripts/pipeline/hazard_response_refiner.py',ROOT/'scripts/pipeline/prepare_hazard_response_v2.py',ROOT/'scripts/pipeline/audit_four_hazards.py']}
 frozen={str(p):digest(p) for c in protocol['cases'] for p in (OUT/c['case']).rglob('*') if p.is_file() and p.suffix in ['.npz','.pt','.wkb']}
 previous=json.loads((stage/'run.json').read_text()) if (stage/'run.json').exists() else None
 run={'status':'running','protocol':protocol,'source_sha256':protected,'input_sha256':frozen,'records':previous['records'] if previous else [],'pairs':previous['pairs'] if previous else [],'method':'normal prefix is rolled out before event detection; event continuation is a new Nexus call with transformed observed history; no manual E pose repair or waypoint conditioning','resume_note':'completed predictions and immutable raw sampled stages are reused after correcting a prefix-assertion indexing error; original source and run saved in source_versions' if previous else None};write(stage/'run.json',run)
 cfg=OmegaConf.load(ROOT/'artifacts/nuplan_evaluation/independent256_v1/config.yaml');model=instantiate(cfg.model);model.load_state_dict(model_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)),strict=True);model=model.cuda().eval().requires_grad_(False);versions=[p._version for p in model.parameters()]
 model.diffuser.constrain_mode=[];model.diffuser.constraint_gamma=[];model.diffuser._generate_scheduling_matrix=functools.partial(model.diffuser._generate_scheduling_matrix,zero_init=False)
 def encode(s):return ((s-FEATURE_MEANS)/(2*FEATURE_STD)).astype('float32')
 def check():
  for p,h in {**protected,**frozen}.items():assert digest(p)==h,p
 def variant_case(case,folder,variant,info):
  if folder.exists():return folder
  shutil.copytree(case,folder)
  if info['response_intent']=='adjacent_lane_change':
   if variant=='event':shutil.copy2(folder/'map/routes_event.npz',folder/'map/routes.npz')
   shutil.copy2(folder/f'map/lane_corridors_{variant}.npz',folder/'map/lane_corridors_sdf.npz')
  return folder
 def reanchor(case,folder,info,normal):
  if folder.exists():
   transform=json.loads((folder/'replan.json').read_text());T=np.array(transform['old_from_new']);return T[:2,3],T[:2,:2],round(info['causal_replan_trigger_s']/.5),transform
  shutil.copytree(case,folder);trigger=info['causal_replan_trigger_s'];n=round(trigger/.5);prefix=decode(normal['sampled_tensor'][0].numpy());origin=prefix[0,4+n,:2];angle=np.arctan2(prefix[0,4+n,3],prefix[0,4+n,2]);R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]);T=pose(*origin,yaw=angle);W=np.array(info['world_from_nexus'])@T
  with np.load(case/'conditions_event.npz') as z:original=z['values'];keep=z['keep'];valid=z['valid']
  source=decode(original);tt=np.arange(-2,8.01,.5);query=tt+trigger;physical=np.zeros_like(source)
  for actor in info['active_slots']:
   current=prefix[actor] if actor==0 else source[actor]
   for k in range(8):physical[actor,:,k]=np.interp(query,tt,current[:,k])
   yaw=np.interp(query,tt,np.unwrap(np.arctan2(current[:,3],current[:,2])));physical[actor,:,2:4]=np.c_[np.cos(yaw),np.sin(yaw)]
   after=query>8
   if after.any():physical[actor,after,:2]=current[-1,:2]+(query[after]-8)[:,None]*current[-1,4:6]
  physical[0,:5]=prefix[0,n:n+5]
  physical[info['active_slots'],:,:2]=(physical[info['active_slots'],:,:2]-origin)@R;physical[info['active_slots'],:,2:4]=physical[info['active_slots'],:,2:4]@R;physical[info['active_slots'],:,4:6]=physical[info['active_slots'],:,4:6]@R
  values=np.where(keep,encode(physical),0);values[~valid.any(-1)]=0;assert not keep[0,5:].any() and (values[~keep]==0).all();np.savez_compressed(folder/'conditions_event.npz',values=values,keep=keep,valid=valid)
  scene=torch.load(case/'scene.pt',weights_only=False);graph=scene['road_graph'].copy();graph[...,:2]=((graph[...,:2]*104-origin)@R)/104;scene['road_graph']=graph;scene['reference']=encode(physical);torch.save(scene,folder/'scene.pt')
  with np.load(case/'map/routes.npz') as z:routes={k:((z[k]-origin)@R).astype('float32') for k in z.files}
  np.savez_compressed(folder/'map/routes.npz',**routes)
  road=wkb.loads((folder/'reference/inputs/drivable.wkb').read_bytes());raster(road,W,folder/'reference/inputs/drivable_sdf.npz',[-180.,-180.],.25,(1441,1441));lane=wkb.loads((folder/'map/lanes/0.wkb').read_bytes());grid=raster(lane,W,None,[-100.,-100.],.2,(1001,1501));np.savez_compressed(folder/'map/lane_corridors_sdf.npz',sdf=grid[None],slots=np.array([0]),origin=np.array([-100.,-100.]),resolution=.2)
  write(folder/'reference/protocol.json',{'scenario':info['scenario'],'world_from_nexus':W.tolist(),'source_sha256':{}})
  transform={'old_from_new':T.tolist(),'world_from_new':W.tolist(),'trigger_s':trigger,'normal_prefix_prediction_sha256':digest(stage/info['kind']/'normal/prediction.pt'),'history_times_global_s':query[:5].tolist(),'E_future_conditioned':False,'event_information_introduced_only_here':True};write(folder/'replan.json',transform)
  return origin,R,n,transform
 for info in sorted(protocol['cases'],key=lambda c:['brake','reveal','rear','oncoming'].index(c['kind'])):
  kind=info['kind'];case=OUT/info['case'];results={};normal=None
  for variant in ['normal','event']:
   check();folder=stage/kind/variant;folder.mkdir(parents=True,exist_ok=True);is_replan=variant=='event' and info['response_intent']=='stop_without_reacceleration'
   done=next((r for r in run['records'] if r['kind']==kind and r['variant']==variant),None)
   if done is not None:
    assert digest(folder/'prediction.pt')==done['prediction_sha256'];results[variant]=torch.load(folder/'prediction.pt',weights_only=False)
    if variant=='normal':normal=results[variant]
    print('RESPONSE_SAMPLE_REUSED',kind,variant,flush=True);continue
   if is_replan:origin,R,n,transform=reanchor(case,folder/'sampling_inputs',info,normal);samplecase=folder/'sampling_inputs'
   else:samplecase=variant_case(case,folder/'sampling_inputs',variant,info)
   scene=torch.load(samplecase/'scene.pt',weights_only=False)
   with np.load(samplecase/f'conditions_{variant}.npz') as d:values=d['values'];keep=d['keep'];valid=d['valid']
   parameters=copy.deepcopy(protocol['config']);parameters.update(anchor_xy_weight=.01,anchor_speed_weight=.01,anchor_longitudinal_weight=.005,collision_weight=10000.,control_iterations=80,final_control_iterations=240)
   intent='stop' if is_replan else 'lane_change' if variant=='event' and info['response_intent']=='adjacent_lane_change' else 'cruise'
   if intent=='lane_change':parameters.update(lane_preference_weight=200.,lateral_acceleration_limit_mps2=3.,entering_actor_yield_weight=0.)
   random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0);x=torch.from_numpy(values).unsqueeze(0).cuda();mask=torch.from_numpy(keep).unsqueeze(0).cuda();v=torch.from_numpy(valid.astype('float32')).unsqueeze(0).cuda();noise=torch.randn_like(x)
   features=SceneTensor(tensor=x,validity=v,road_graph=torch.from_numpy(scene['road_graph']).unsqueeze(0).cuda(),road_graph_validity=torch.from_numpy(scene['road_graph_validity']).unsqueeze(0).cuda());segments=torch.as_tensor(segments_from_map(scene['road_graph'],scene['road_graph_validity']),device='cuda')
   refiner=ResponseIntentRefiner(CorridorCruiseConfig(**parameters),x,v,mask,segments,route_path=samplecase/'map/routes.npz',response_intent=intent,stop_slot=info['roles'].get(info.get('stop_role')),stop_deadline_s=info.get('stop_deadline_global_s',6.5)-info.get('causal_replan_trigger_s',0),target_sdf_path=samplecase/'map/lane_corridors_target.npz' if intent=='lane_change' else None);refiner.set_drivable_sdf(samplecase/'reference/inputs/drivable_sdf.npz');model.diffuser.anchor_refiner=refiner
   print('RESPONSE_SAMPLE_START',kind,variant,intent,flush=True);start=time.monotonic()
   raw_reused=(folder/'raw_stage_prediction.pt').exists()
   if raw_reused:sample=torch.load(folder/'raw_stage_prediction.pt',weights_only=False)['sampled_tensor']
   else:
    with torch.no_grad():sample=model.forward_inference({'scene_tensor':features,'task_mask':mask},noise=noise,return_intermidates=False)['sampled_tensor'].cpu()
   torch.cuda.synchronize();seconds=time.monotonic()-start;assert torch.isfinite(sample[v.cpu().bool()]).all();torch.testing.assert_close(sample[mask.cpu()],x.cpu()[mask.cpu()],rtol=0,atol=0)
   raw_payload={'sampled_tensor':sample,'scene_tensor_gt':torch.from_numpy(scene['reference']).unsqueeze(0),'valid_mask':v.cpu(),'task_mask':mask.cpu(),'conditioned_scene_tensor':x.cpu()}
   if not raw_reused:torch.save(raw_payload,folder/'raw_stage_prediction.pt');write(folder/'guidance.json',refiner.records);np.save(folder/'controls.npy',refiner.previous_controls.detach().cpu().numpy())
   if is_replan:
    # A temporal rollout composition: execute the previously sampled prefix,
    # then execute the fresh plan from the same exact current pose.
    continuation=decode(sample[0].numpy());continuation[...,:2]=continuation[...,:2]@R.T+origin;continuation[...,2:4]=continuation[...,2:4]@R.T;continuation[...,4:6]=continuation[...,4:6]@R.T
    pred=normal['sampled_tensor'][0].numpy().copy();pred[0,5+n:]=encode(continuation[0,5:21-n]);np.testing.assert_array_equal(pred[0,:5+n],normal['sampled_tensor'][0,0,:5+n].numpy())
    with np.load(case/'conditions_event.npz') as z:combined_values=z['values'];combined_keep=z['keep'];combined_valid=z['valid']
    for actor in info['fixed_slots']:pred[actor]=combined_values[actor]
    finalscene=torch.load(case/'scene.pt',weights_only=False);payload={'sampled_tensor':torch.from_numpy(pred).unsqueeze(0),'scene_tensor_gt':torch.from_numpy(finalscene['reference']).unsqueeze(0),'valid_mask':torch.from_numpy(combined_valid.astype('float32')).unsqueeze(0),'task_mask':torch.from_numpy(combined_keep).unsqueeze(0),'conditioned_scene_tensor':torch.from_numpy(combined_values).unsqueeze(0),'staged_rollout':transform}
   else:payload=raw_payload
   torch.save(payload,folder/'prediction.pt');auditinfo=copy.deepcopy(info)
   if kind=='reveal' and variant=='normal':auditinfo['roles'].pop('S');auditinfo['active_slots'].remove(info['roles']['S']);auditinfo['kind']='clear_road_control'
   audit=evaluate(payload,case,auditinfo,variant);write(folder/'acceptance.json',audit);results[variant]=payload
   if variant=='normal':normal=payload
   row={'kind':kind,'variant':variant,'index':info['scenario']['index'],'path':str(folder.relative_to(stage)),'seconds':seconds,'intent':intent,'sampling_parameters':parameters,'raw_stage_reused':raw_reused,'noise_sha256':hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest(),'prediction_sha256':digest(folder/'prediction.pt'),'acceptance':audit};run['records'].append(row);write(stage/'run.json',run);print('RESPONSE_SAMPLE_COMPLETE',kind,variant,round(seconds,1),'accepted',audit['accepted'],'pairs',audit['pairs'],'E_speed',audit['E_motion']['xy_interval_speeds_mps'],flush=True)
  if not any(p['kind']==kind for p in run['pairs']):
   pair=compare(results['normal'],results['event'],info);pair.update(kind=kind,same_noise_and_warm_start=False,comparison_scope='same original ego history; stopping event shares executed normal prefix and then replans; lane-change event has explicit adjacent-lane intent');run['pairs'].append(pair);write(stage/'run.json',run)
 check();assert versions==[p._version for p in model.parameters()] and digest(checkpoint)==CHECKPOINT_SHA256;run.update(status='complete',weights_unchanged=True,protected_source_unchanged=True);write(stage/'run.json',run);print('RESPONSE_GENERATION_COMPLETE',flush=True)
if __name__=='__main__':main()
