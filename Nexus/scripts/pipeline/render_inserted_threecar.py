"""Native 3DGS synthetic replay: scene121 background + supplied cars + Nexus motion."""
import argparse,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/nexus_3dgs/asset_insert_20261028'
sys.path[:0]=[str(ROOT),str(ROOT/'Omniscenegraph')]
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--probe',action='store_true');parser.add_argument('--camera-lift',type=float,default=0.);parser.add_argument('--camera-side',type=float,default=0.);parser.add_argument('--zoom-end',type=float,default=1.);parser.add_argument('--ground-profile',type=Path);args=parser.parse_args();args.output.mkdir(exist_ok=False,parents=True)
 import numpy as np,torch,imageio.v2 as imageio
 from PIL import Image,ImageDraw,ImageFont
 from shapely import wkb
 from nexus_3dgs.replay import load_trainer,camera_inputs,road_rotation
 from nexus_3dgs.inserted_vehicle import GaussianVehicle
 from nexus_3dgs.geometry import matrix_quaternion
 torch.set_num_threads(6);p=json.loads((OUT/'insertion_protocol.json').read_text());z=np.load(OUT/'placed_trajectories.npz');tt=z['times_s'];plane=np.array(p['road_ground']['world_plane']);normal=plane[:3];scene=Path(p['background_scene']);meta=json.loads((scene/'nexus_scene.json').read_text());S=np.linalg.inv(meta['frames'][0]['ego_to_world']);E0=np.array(p['background_camera_anchor_ego']);base_frame=p['background_camera_anchor_frame']
 geom=json.loads((args.checkpoint.parent/'scene_geometry.json').read_text());before=sha(args.checkpoint);trainer,cfg=load_trainer(args.checkpoint,scene_aabb=geom['scene_aabb']);rigid=trainer.models['RigidNodes'];original_get=rigid.get_gaussians;versions=[(x,x._version) for m in trainer.models.values() for x in m.parameters()]
 trainer.render_each_class=False
 vehicles={r:GaussianVehicle(a['source'],a['target_original_lw']) for r,a in p['asset_assignment'].items()}
 for r,v in vehicles.items():assert np.allclose(v.size,p['asset_assignment'][r]['render_lwh'])
 def height(xy):return -(plane[0]*xy[0]+plane[1]*xy[1]+plane[3])/plane[2]
 profile=None;curve=None
 if args.ground_profile:
  from scipy.interpolate import PchipInterpolator
  profile=json.loads(args.ground_profile.read_text());knots=np.array(profile['station_and_height_offset_m']);curve=PchipInterpolator(knots[:,0],knots[:,1]);derivative=curve.derivative()
 def road_pose(xy,yaw,vehicle=False):
  up=normal;z=height(xy)
  if vehicle and curve is not None:
   direction=np.array(profile['forward_world']);station=float((xy-profile['origin_world'])@direction);st=float(np.clip(station,knots[0,0],knots[-1,0]));z+=float(curve(st));gradient=-normal[:2]/normal[2]+(float(derivative(st)) if station==st else 0.)*direction;up=np.r_[-gradient,1.];up/=np.linalg.norm(up)
  T=np.eye(4);T[:3,:3]=road_rotation(yaw,up);T[:3,3]=[xy[0],xy[1],z];return T
 intr=np.loadtxt(scene/'intrinsics/0.txt');K=np.array([[intr[0]/2,0,intr[2]/2],[0,intr[1]/2,intr[3]/2],[0,0,1.]])
 # Keep the real calibrated pinhole and constant exposure. The observation
 # camera follows the slow B progress, identically in normal/event videos.
 camera_record=next(x for x in meta['frames'][base_frame]['images'] if x['camera_id']==0);extr=np.loadtxt(scene/'extrinsics/0.txt');C0=np.array(camera_record['ego_to_world'])@extr;yaw0=np.arctan2(E0[1,0],E0[0,0]);mount=np.linalg.inv(road_pose(E0[:2,3],yaw0))@C0
 source_b=z['event_xy_world'][1];h0=E0[:2,0]/np.linalg.norm(E0[:2,0]);progress=np.maximum.accumulate(np.maximum((source_b-source_b[0])@h0,0))*.85
 original_egos=np.array([x['ego_to_world'] for x in meta['frames']]);cameras=[];frame_ids=[]
 for d in progress:
  xy=E0[:2,3]+h0*d;T=road_pose(xy,yaw0);C=T@mount;C[:3,3]+=normal*args.camera_lift+T[:3,1]*args.camera_side;cameras.append(S@C);frame_ids.append(int(np.argmin(np.linalg.norm(original_egos[:,:2,3]-xy,axis=-1))))
 cameras=np.array(cameras);Kcuda=torch.tensor(K,dtype=torch.float32,device='cuda')
 state={'poses':None};pose_rows=[]
 def inserted_get(cam):
  entries=[]
  for role,pose in state['poses'].items():entries.append(vehicles[role].gaussians(pose,cam.camtoworlds[:3,3]))
  # A native thin Gaussian contact-shadow proxy, not an image overlay.
  for role,pose in state['poses'].items():
   size=vehicles[role].size;shadow=pose.copy();shadow[:3,3]-=pose[:3,2]*(size[2]/2-.035)
   entries.append({'_means':torch.tensor(shadow[:3,3],dtype=torch.float32,device='cuda')[None],'_quats':torch.tensor(matrix_quaternion(shadow[:3,:3]),dtype=torch.float32,device='cuda')[None],'_scales':torch.tensor([[size[0]*.34,size[1]*.36,.008]],dtype=torch.float32,device='cuda'),'_opacities':torch.tensor([[.28]],device='cuda'),'_rgbs':torch.tensor([[.04,.04,.04]],device='cuda')})
  return {k:torch.cat([e[k] for e in entries],0) for k in entries[0]}
 rigid.get_gaussians=inserted_get
 indices=[0,20,40,60,80] if args.probe else list(range(81));videos={};poses_meta={};start=time.monotonic()
 try:
  for variant in ['normal','event']:
   folder=args.output/variant;folder.mkdir();poses_meta[variant]=[];writer=None if args.probe else imageio.get_writer(folder/'camera_0.mp4',fps=10,macro_block_size=2,ffmpeg_params=['-threads','2','-crf','18'])
   try:
    for k in indices:
     poses={};row=[]
     for i,role in enumerate(['A','B','C']):
      xy=z[f'{variant}_xy_world'][i,k];yaw=z[f'{variant}_yaw_world'][i,k];T=road_pose(xy,yaw,vehicle=True);T[2,3]+=vehicles[role].size[2]/(2*T[2,2]);poses[role]=S@T;row.append({'role':role,'center_xy_world':xy.tolist(),'yaw_world':float(yaw),'body_world':T.tolist(),'size_lwh':vehicles[role].size.tolist()})
     state['poses']=poses
     with torch.no_grad():
      u=min(float(tt[k])/5.,1.);zoom=1.+(args.zoom_end-1.)*(3*u*u-2*u*u*u);intrinsic=K.copy();intrinsic[0,0]*=zoom;intrinsic[1,1]*=zoom
      image,cam=camera_inputs(torch.tensor(cameras[k],dtype=torch.float32,device='cuda'),torch.tensor(intrinsic,dtype=torch.float32,device='cuda'),frame_ids[k],540,960,rigid.num_frames,len(cfg.data.pixel_source.cameras));image['img_idx']=torch.tensor([base_frame*len(cfg.data.pixel_source.cameras)],device='cuda',dtype=torch.long);result=trainer(image,cam,novel_view=True);result=result[0] if isinstance(result,tuple) else result
      assert torch.isfinite(result['rgb']).all();rgb=(result['rgb'].clamp(0,1).cpu().numpy()*255).round().astype('uint8')
     if writer is not None:writer.append_data(rgb)
     if args.probe or k in [0,20,40,60,80]:imageio.imwrite(folder/f'{k:03d}.png',rgb)
     poses_meta[variant].append({'frame':k,'time_s':float(tt[k]),'camera_scene':cameras[k].tolist(),'intrinsics':intrinsic.tolist(),'zoom':zoom,'actors':row})
     if k%20==0:print('INSERT_RENDER',variant,k,flush=True)
   finally:
    if writer is not None:writer.close()
   if writer is not None:
    with imageio.get_reader(folder/'camera_0.mp4') as reader:count=sum(1 for _ in reader);fps=reader.get_meta_data()['fps']
    assert count==81 and fps==10;videos[variant]={'frames':count,'fps':fps,'sha256':sha(folder/'camera_0.mp4')}
 finally:rigid.get_gaussians=original_get
 assert all(x._version==version for x,version in versions)
 assert sha(args.checkpoint)==before
 for a in p['asset_assignment'].values():assert sha(a['source'])==a['source_sha256']
 for path,h in p['source_prediction_sha256'].items():assert sha(path)==h
 report={'status':'complete','probe_only':args.probe,'checkpoint':str(args.checkpoint),'checkpoint_sha256':before,'background_and_original_models_unchanged':True,'asset_sources_unchanged':True,'Nexus_predictions_unchanged':True,'protocol':str(OUT/'insertion_protocol.json'),'asset_info':{r:v.info for r,v in vehicles.items()},'frames':poses_meta,'videos':videos,'seconds':time.monotonic()-start,'camera_policy':'calibrated pinhole with declared observation offset, shared smooth following of B-event progress; constant exposure; not original Nexus ego camera','camera_lift_m':args.camera_lift,'camera_side_offset_m':args.camera_side,'native_gaussian_depth_compositing':True,'contact_shadow':'one thin native Gaussian proxy per car; no physical illumination simulation','scope':'synthetic three-car traffic only, scene121 background with scene059 Nexus+guidance motion; not an original059 reconstruction or full52-actor replay'}
 report.update(native_focal_zoom_end=args.zoom_end,vehicle_ground_profile=profile,renderer_sha256=sha(Path(__file__)),asset_loader_sha256=sha(ROOT/'nexus_3dgs/inserted_vehicle.py'))
 write(args.output/'run.json',report)
 fontpath=ROOT/'artifacts/nexus_3dgs/cache/fonts/NotoSansSC-Medium.ttf';font=ImageFont.truetype(str(fontpath),20)
 if args.probe:
  board=Image.new('RGB',(1920,5*580),'white');draw=ImageDraw.Draw(board)
  for j,k in enumerate(indices):
   for c,v in enumerate(['normal','event']):board.paste(Image.open(args.output/v/f'{k:03d}.png'),(c*960,j*580+35));draw.text((c*960+12,j*580+6),f'{v} {tt[k]:.1f}s',font=font,fill='black')
  board.save(args.output/'probe_contact.jpg')
 print('INSERT_RENDER_COMPLETE',args.output,flush=True)
if __name__=='__main__':main()
