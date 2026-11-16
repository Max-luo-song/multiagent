"""Isolate measured native panel and original vehicle contributions at actual poses."""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'Omniscenegraph')]

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ('checkpoint','render','indices','output'):p.add_argument('--'+key,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 import numpy as np,torch,yaml
 from PIL import Image
 from gsplat import rasterization
 from models.gaussians.basics import quat_to_rotmat,quat_mult,spherical_harmonics
 from models.gaussians.support import confine_gaussian_support
 from nexus_3dgs.runtime import TrajectoryController
 from nexus_3dgs.grounding import GroundedRegistration
 w=torch.load(a.checkpoint,map_location='cpu',weights_only=False)['models']['RigidNodes'];cfg=yaml.safe_load((a.checkpoint.parent/'config.yaml').read_text());run=json.loads((a.render/'run.json').read_text());g=run['ground_registration'];model=6
 rigid=SimpleNamespace(instance_ids=w['dataset_instance_ids'],instances_trans=w['instances_trans'],instances_quats=w['instances_quats'],instances_fv=w['instances_fv']);rigid.num_frames,rigid.num_instances=rigid.instances_fv.shape
 controller=TrajectoryController(run['bundle'],Path(g['ground_model']['scene']),rigid);ground=a.output/'ground.json';ground.write_text(json.dumps(g['ground_model']));reg=GroundedRegistration(controller,ground,stabilize_all=g['stabilize_all_rigid_instances']);reg.object_mounts={int(k):np.array(v) for k,v in g['object_mounts'].items()};reg.flat_mounts={int(k):np.array(v) for k,v in g.get('flat_track_mounts',{}).items()};controller.grounding=reg
 ids=torch.nonzero(w['points_ids'].reshape(-1)==model)[:,0];local=w['_means'][ids].cuda();ql=torch.nn.functional.normalize(w['_quats'][ids].cuda(),dim=-1);scale=w['_scales'][ids].exp().cuda().clamp(max=cfg['model']['RigidNodes']['ctrl'].get('physical_max_std_m',1e9));scale,valid=confine_gaussian_support(local,scale,quat_to_rotmat(ql),w['instances_size'][model].cuda().expand(len(ids),-1),sigma=3.)
 native=np.zeros(len(w['_means']),bool);native[np.load(a.indices)]=True;native=torch.tensor(native[ids],device='cuda');opacity=w['_opacities'][ids,0].sigmoid().cuda()*valid;sh=torch.cat((w['_features_dc'][ids,None],w['_features_rest'][ids]),1).cuda();rows=[]
 for f in (20,70,99):
  record=next(r for r in run['frames'] if r['frame_idx']==f);im=record['images'][0];C=torch.tensor(controller.scene_from_world@np.array(im['scene_metadata']['camera_world']),dtype=torch.float32,device='cuda');K=torch.tensor(im['scene_metadata']['intrinsics'],dtype=torch.float32,device='cuda')
  with controller.edit_frame(f,im['timestamp_us']):
   q=torch.nn.functional.normalize(rigid.instances_quats[f,model].cuda(),dim=-1);R=quat_to_rotmat(q[None])[0];mu=local@R.T+rigid.instances_trans[f,model].cuda();quats=quat_mult(q.expand(len(ids),-1),ql)
  colors=(spherical_harmonics(3,torch.nn.functional.normalize(mu-C[:3,3],dim=-1),sh)+.5).clamp_min(0)
  maps={}
  for name,mask in [('native',native),('original',~native),('both',torch.ones_like(native))]:
   with torch.no_grad():rgb,alpha,_=rasterization(means=mu,quats=quats,scales=scale,opacities=opacity*mask,colors=colors,viewmats=torch.linalg.inv(C)[None],Ks=K[None],width=960,height=540,near_plane=.1,far_plane=1e10,packed=False,render_mode='RGB')
   rgb=rgb[0].cpu().numpy();alpha=alpha[0,:,:,0].cpu().numpy();maps[name]=(rgb,alpha);Image.fromarray((rgb.clip(0,1)*255).astype(np.uint8)).save(a.output/f'{f:03d}_{name}.png');Image.fromarray((alpha*255).astype(np.uint8)).save(a.output/f'{f:03d}_{name}_alpha.png')
  core=maps['native'][1]>.98;delta=np.abs(maps['native'][0]-maps['both'][0])[core]
  rows.append({'frame':f,'native_opaque_pixels':int(core.sum()),'native_vs_composite_core_rgb_mean':float(delta.mean()) if core.any() else None,'native_vs_composite_core_rgb_p90':float(np.quantile(delta,.9)) if core.any() else None})
 (a.output/'occlusion.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows),flush=True)
if __name__=='__main__':main()
