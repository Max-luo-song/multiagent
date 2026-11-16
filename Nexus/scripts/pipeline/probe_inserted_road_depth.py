"""Compare LiDAR-grounded tire contacts with native reconstructed-road depth."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/nexus_3dgs/asset_insert_20261028'
sys.path[:0]=[str(ROOT),str(ROOT/'Omniscenegraph')]
def main():
 import numpy as np,torch
 from nexus_3dgs.replay import load_trainer,camera_inputs
 run=json.loads((OUT/'probe_8000_native_zoom/run.json').read_text());p=json.loads((OUT/'insertion_protocol.json').read_text());meta=json.loads((Path(p['background_scene'])/'nexus_scene.json').read_text());S=np.linalg.inv(meta['frames'][0]['ego_to_world']);W=np.linalg.inv(S);ckpt=Path(run['checkpoint']);aabb=json.loads((ckpt.parent/'scene_geometry.json').read_text())['scene_aabb'];trainer,cfg=load_trainer(ckpt,scene_aabb=aabb);rigid=trainer.models['RigidNodes'];original=rigid.get_gaussians;rigid.get_gaussians=lambda cam:None;trainer.render_each_class=False;samples=[]
 try:
  for row in run['frames']['event']:
   if row['frame'] not in [0,40,80]:continue
   C=np.array(row['camera_scene']);K=np.array(row['intrinsics']);V=np.linalg.inv(C);Kinv=np.linalg.inv(K)
   with torch.no_grad():
    image,cam=camera_inputs(torch.tensor(C,dtype=torch.float32,device='cuda'),torch.tensor(K,dtype=torch.float32,device='cuda'),40,540,960,rigid.num_frames,len(cfg.data.pixel_source.cameras));result=trainer(image,cam,novel_view=True);result=result[0] if isinstance(result,tuple) else result;depth=result['depth'].cpu().numpy()[...,0];alpha=result['opacity'].cpu().numpy()[...,0]
   for actor in row['actors']:
    T=np.array(actor['body_world']);size=np.array(actor['size_lwh'])
    for x in [-.3,0,.3]:
     for y in [-.3,0,.3]:
      contact=T@np.r_[x*size[0],y*size[1],-size[2]/2,1.];camera=V@S@contact;pixel=K@camera[:3];uv=pixel[:2]/pixel[2];u,v=np.round(uv).astype(int)
      if not(2<=u<958 and 2<=v<538) or alpha[v,u]<.9:continue
      d=float(np.median(depth[v-1:v+2,u-1:u+2]));scene=C[:3,3]+C[:3,:3]@(Kinv@np.r_[u,v,1.])*d;world=W@np.r_[scene,1.];samples.append({'time_s':row['time_s'],'role':actor['role'],'contact_xyz_world':contact[:3].tolist(),'native_road_xyz_world':world[:3].tolist(),'road_minus_contact_z_m':float(world[2]-contact[2])})
 finally:rigid.get_gaussians=original
 dz=np.array([s['road_minus_contact_z_m'] for s in samples]);result={'status':'complete','samples':samples,'median_road_minus_contact_z_m':float(np.median(dz)),'q10_q50_q90_m':np.quantile(dz,[.1,.5,.9]).tolist(),'scope':'native expected depth is a visual-surface diagnostic, not independent terrain truth'}
 (OUT/'native_road_depth_audit.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k!='samples'},flush=True)
if __name__=='__main__':main()
