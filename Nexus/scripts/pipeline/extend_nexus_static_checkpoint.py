"""Adapt the preserved checkpoint to additional real reconstruction observations.

Original shared observations retain their learned camera, exposure, and traffic
parameters exactly. New observations supervise static geometry only; they do not
extend or replace Nexus predictions.
"""
import argparse,copy,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'Omniscenegraph')]

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('checkpoint','extended-config','anchors','output'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--source-scene',type=Path,help='Available original scene metadata when the old config data path was removed')
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 import torch,yaml
 from nexus_3dgs.geometry import quaternion_matrix
 state=torch.load(a.checkpoint,map_location='cpu',weights_only=False);cfg=yaml.safe_load((a.checkpoint.parent/'config.yaml').read_text());extended=yaml.safe_load(a.extended_config.read_text());oldscene=Path(cfg['data']['data_root'])/str(cfg['data']['scene_idx']);newscene=Path(extended['data']['data_root'])/str(extended['data']['scene_idx'])
 if a.source_scene:oldscene=a.source_scene
 oldmeta,newmeta=[json.loads((s/'nexus_scene.json').read_text()) for s in (oldscene,newscene)];oldinfo,newinfo=[json.loads((s/'instances/instances_info.json').read_text()) for s in (oldscene,newscene)]
 assert oldmeta['frames'][0]['ego_to_world']==newmeta['frames'][0]['ego_to_world'];assert oldmeta['identity']['scenario_token']==newmeta['identity']['scenario_token'];oldcams=cfg['data']['pixel_source']['cameras'];newcams=extended['data']['pixel_source']['cameras'];n0,n1=len(oldmeta['frames']),len(newmeta['frames']);assert n1>n0
 for frame0,frame1 in zip(oldmeta['frames'],newmeta['frames']):
  assert frame0['ego_to_world']==frame1['ego_to_world']
  records={i['camera_id']:i for i in frame1['images']}
  for image in frame0['images']:assert image['timestamp_us']==records[image['camera_id']]['timestamp_us'] and image['ego_to_world']==records[image['camera_id']]['ego_to_world']
 for model in state['models'].values():
  for key,value in list(model.items()):
   if key.endswith('weight') and torch.is_tensor(value) and value.ndim==2 and len(value)==n0*len(oldcams):
    rows=value.reshape(n0,len(oldcams),-1);result=torch.empty((n1,len(newcams),value.shape[1]),dtype=value.dtype)
    for cpos,c in enumerate(newcams):
     if c in oldcams:
      j=oldcams.index(c);result[:n0,cpos]=rows[:,j];result[n0:,cpos]=rows[-1,j]
      assert torch.equal(result[:n0,cpos],rows[:,j])
     else:result[:,cpos]=rows.mean((0,1))
    model[key]=result.reshape(n1*len(newcams),-1)
 w=state['models']['RigidNodes'];S=np.linalg.inv(oldmeta['frames'][0]['ego_to_world']);newbytoken={v['id']:(int(k),v) for k,v in newinfo.items()};oldids=w['dataset_instance_ids'].clone();M=len(oldids);trans=torch.zeros((n1,M,3));quats=torch.zeros((n1,M,4));quats[:,:,0]=1;fv=torch.zeros((n1,M),dtype=torch.bool)
 trans[:n0]=w['instances_trans'];quats[:n0]=w['instances_quats'];fv[:n0]=w['instances_fv'];mapped=[]
 for m,oldid in enumerate(oldids.tolist()):
  token=oldinfo[str(oldid)]['id'];newid,record=newbytoken[token];mapped.append(newid);ann0=oldinfo[str(oldid)]['frame_annotations'];ann1=record['frame_annotations'];mounts=[]
  for f,T in zip(ann0['frame_idx'],ann0['obj_to_world']):
   if f<n0 and bool(fv[f,m]):
    B=np.eye(4);B[:3,:3]=quaternion_matrix(quats[f,m].numpy());B[:3,3]=trans[f,m].numpy();mounts.append(np.linalg.inv(S@T)@B)
  mount=np.eye(4)
  if mounts:
   mount[:3,:3]=Rotation.from_matrix(np.array(mounts)[:,:3,:3]).mean().as_matrix();mount[:3,3]=np.mean(np.array(mounts)[:,:3,3],axis=0)
  for f,T in zip(ann1['frame_idx'],ann1['obj_to_world']):
   if f>=n0:
    B=S@T@mount;trans[f,m]=torch.tensor(B[:3,3]);quats[f,m]=torch.tensor(Rotation.from_matrix(B[:3,:3]).as_quat()[[3,0,1,2]]);fv[f,m]=True
 for key,val in [('instances_trans',trans),('instances_quats',quats),('instances_fv',fv)]:assert torch.equal(w[key],val[:n0]);w[key]=val
 w['dataset_instance_ids']=torch.tensor(mapped,dtype=oldids.dtype)
 # Original Gaussian order is unchanged, so precomputed real-LiDAR anchors
 # for the original prefix remain valid; synthetic road additions are omitted.
 N=len(state['models']['Background']['_means'])
 with np.load(a.anchors) as source:
  values={k:source[k][:N] for k in source.files};assert len(values['valid'])==N
 np.savez_compressed(a.output/'anchors.npz',**values)
 cfg['data']=extended['data'];cfg['trainer']['optim']['num_iters']=18000
 cfg['trainer']['losses']['depth']['w']=.1
 cfg['trainer']['losses']['scene_reconstruction']={'exclude_vehicle_pixels':True,'sky_alpha_w':.02,'sky_rgb_w':.05,'occupied_alpha_w':.01}
 cfg['data']['pixel_source']['load_scene_semantics']=True
 for ctrl in [cfg['trainer']['gaussian_ctrl_general_cfg'],cfg['model']['Background'].setdefault('ctrl',{}),cfg['model']['RigidNodes']['ctrl']]:ctrl.update(refine_interval=100000,reset_alpha_interval=100000,stop_split_at=0)
 opts=cfg['trainer']['gaussian_optim_general_cfg'];opts['xyz'].update(lr=.00005,lr_final=.00005);opts['sh_dc']['lr']=.0005;opts['sh_rest']['lr']=.00001;opts['opacity']['lr']=.01;opts['scaling']['lr']=.001;opts['rotation']['lr']=.0002;cfg['model']['Sky']['optim']['all']['lr']=.002
 cfg['logging'].update(vis_freq=0,saveckpt_freq=1000)
 torch.save(state,a.output/'checkpoint_final.pth');(a.output/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
 report={'status':'passed','old_frames':n0,'extended_real_frames':n1,'old_cameras':oldcams,'extended_cameras':newcams,'original_shared_camera_and_appearance_rows_exact':True,'original_traffic_poses_visibility_geometry_exact':True,'source_gaussians_unchanged':True,'old_to_new_dataset_ids':list(zip(oldids.tolist(),mapped)),'supported_static_anchors':int(values['valid'].sum()),'scope':'offline reconstruction supervision only; prediction bundle and generated cameras are not modified'}
 (a.output/'extension.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
