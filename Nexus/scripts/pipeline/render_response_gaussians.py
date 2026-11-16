"""Native Gaussian depth compositing for the four frozen response events."""
import argparse,json,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/nexus_response_gaussians_20261110';sys.path[:0]=[str(ROOT),str(ROOT/'Omniscenegraph')]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--probe',action='store_true');args=parser.parse_args()
 import numpy as np,torch,imageio.v2 as imageio
 from scipy.interpolate import PchipInterpolator
 from nexus_3dgs.replay import load_trainer,camera_inputs,road_rotation
 from nexus_3dgs.inserted_vehicle import GaussianVehicle
 from nexus_3dgs.geometry import matrix_quaternion
 torch.set_num_threads(4);p=json.loads((OUT/'protocol.json').read_text());bg=p['background'];scene=Path(bg['background_scene']);meta=json.loads((scene/'nexus_scene.json').read_text());S=np.linalg.inv(meta['frames'][0]['ego_to_world']);egos=np.array([f['ego_to_world'] for f in meta['frames']]);checkpoint=Path(p['checkpoint']);checkpoint_hash=sha(checkpoint);geom=json.loads((checkpoint.parent/'scene_geometry.json').read_text());trainer,cfg=load_trainer(checkpoint,scene_aabb=geom['scene_aabb']);trainer.render_each_class=False;rigid=trainer.models['RigidNodes'];original=rigid.get_gaussians;versions=[(x,x._version) for model in trainer.models.values() for x in model.parameters()]
 plane=np.array(bg['road_ground']['world_plane']);profile=p['ground_profile'];knots=np.array(profile['station_and_height_offset_m']);curve=PchipInterpolator(knots[:,0],knots[:,1]);deriv=curve.derivative();forward=np.array(profile['forward_world']);po=np.array(profile['origin_world']);state={};vehicles={}
 def pose(xy,yaw,height=0):
  st=float((xy-po)@forward);cl=float(np.clip(st,knots[0,0],knots[-1,0]));z=-(plane[0]*xy[0]+plane[1]*xy[1]+plane[3])/plane[2]+float(curve(cl));grad=-plane[:2]/plane[2]+float(deriv(cl))*forward if cl==st else -plane[:2]/plane[2];up=np.r_[-grad,1.];up/=np.linalg.norm(up);T=np.eye(4);T[:3,:3]=road_rotation(yaw,up);T[:3,3]=[*xy,z];T[:3,3]+=T[:3,2]*height/2;return T
 def get(cam):
  entries=[]
  for role,T in state['poses'].items():
   if role=='E' and state['view']=='driver':continue
   v=vehicles[role];entries.append(v.gaussians(T,cam.camtoworlds[:3,3]));shadow=T.copy();shadow[:3,3]-=T[:3,2]*(v.size[2]/2-.035);entries.append({'_means':torch.tensor(shadow[:3,3],dtype=torch.float32,device='cuda')[None],'_quats':torch.tensor(matrix_quaternion(shadow[:3,:3]),dtype=torch.float32,device='cuda')[None],'_scales':torch.tensor([[v.size[0]*.34,v.size[1]*.36,.008]],dtype=torch.float32,device='cuda'),'_opacities':torch.tensor([[.28]],device='cuda'),'_rgbs':torch.tensor([[.04,.04,.04]],device='cuda')})
  return {k:torch.cat([e[k] for e in entries]) for k in entries[0]}
 rigid.get_gaussians=get;K=torch.tensor([[600.,0,480.],[0,600.,270.],[0,0,1]],device='cuda');records=[];start=time.monotonic()
 try:
  for case in p['cases']:
   kind=case['kind'];z=np.load(OUT/f'{kind}.npz');roles=case['roles'];vehicles={r:GaussianVehicle(ROOT.parents[1]/'3dgs_pro'/('3.ply' if r=='E' or kind=='reveal' and r=='A' else '1.ply'),z['sizes'][i]) for i,r in enumerate(roles)};folder=OUT/('probe' if args.probe else 'render')/kind;folder.mkdir(parents=True,exist_ok=False);writers={} if args.probe else {view:imageio.get_writer(folder/f'{view}.mp4',fps=10,macro_block_size=2,ffmpeg_params=['-threads','2','-crf','18']) for view in ['observer','driver']};rows=[]
   try:
    for k in ([0,20,40,60,80] if args.probe else range(81)):
     world={r:pose(z['xy'][i,k],z['yaw'][i,k],vehicles[r].size[2]) for i,r in enumerate(roles)};state['poses']={r:S@T for r,T in world.items()}
     E=world['E'];h=E[:3,0];up=E[:3,2]
     for view in ['observer','driver']:
      state['view']=view
      if view=='driver':
       direction=-h if kind=='rear' else h;eye=E[:3,3]-up*vehicles['E'].size[2]/2+up*1.45+h*(0 if kind=='rear' else 1.5)
      else:
       # Follow behind the rearmost vehicle so rear approach is visible too.
       index=int(np.argmin(z['station'][:,k]));T=world[roles[index]];direction=np.array(bg['background_camera_anchor_ego'])[:3,0].copy();direction/=np.linalg.norm(direction);left=np.cross(up,direction);center=T[:3,3]-left*z['lateral'][index,k];eye=center-direction*12+up*2.5;direction=direction-up*.06;direction/=np.linalg.norm(direction)
      right=np.cross(direction,up);right/=np.linalg.norm(right);down=np.cross(direction,right);C=np.eye(4);C[:3,:3]=np.column_stack([right,down,direction]);C[:3,3]=eye;nearest=int(np.argmin(np.linalg.norm(egos[:,:2,3]-eye[:2],axis=-1)))
      with torch.no_grad():
       image,cam=camera_inputs(torch.tensor(S@C,dtype=torch.float32,device='cuda'),K,nearest,540,960,rigid.num_frames,len(cfg.data.pixel_source.cameras));image['img_idx']=torch.tensor([bg['background_camera_anchor_frame']*len(cfg.data.pixel_source.cameras)],device='cuda');result=trainer(image,cam,novel_view=True);result=result[0] if isinstance(result,tuple) else result;assert torch.isfinite(result['rgb']).all();rgb=(result['rgb'].clamp(0,1).cpu().numpy()*255).round().astype('uint8')
      if writers:writers[view].append_data(rgb)
      if k in [0,20,40,60,80]:imageio.imwrite(folder/f'{view}_{k:03d}.jpg',rgb)
      rows.append({'k':k,'view':view,'camera_world':C.tolist()})
     if k%20==0:print('HAZARD_GAUSSIAN',kind,k,flush=True)
   finally:
    for w in writers.values():w.close()
   records.append({'kind':kind,'folder':str(folder),'assets':{r:v.info for r,v in vehicles.items()},'frames':rows});print('HAZARD_GAUSSIAN_CASE_COMPLETE',kind,flush=True)
 finally:rigid.get_gaussians=original
 assert all(x._version==v for x,v in versions) and sha(checkpoint)==checkpoint_hash
 for case in p['cases']:assert sha(case['prediction'])==case['prediction_sha256']
 (OUT/('probe_run.json' if args.probe else 'render_run.json')).write_text(json.dumps({'status':'complete','native_gaussian_depth_compositing':True,'background_weights_unchanged':True,'predictions_unchanged':True,'checkpoint_sha256':checkpoint_hash,'cases':records,'seconds':time.monotonic()-start,'scope':p['scope']},ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
