"""Fit one shared SE2 relocation of saved three-car trajectories to a real road."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/nexus_3dgs/asset_insert_20261028'
sys.path[:0]=[str(ROOT),str(ROOT/'scripts/pipeline'),str(ROOT/'third_party/nuplan-devkit')]
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 import numpy as np,torch
 from shapely.geometry import Polygon,Point,LineString
 from shapely.ops import unary_union
 from shapely import vectorized
 from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
 from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
 from nexus_3dgs.bundle import decode
 from nexus_3dgs.scene_projection import pair_separations
 from audit_nexus_omega_v2 import sample_positions
 choice=json.loads((OUT/'chosen_background.json').read_text());row=choice['scenario'];roles={'A':39,'B':41,'C':44};rolelist=list(roles)
 scene=next((OUT/'processed').glob('*'));meta=json.loads((scene/'nexus_scene.json').read_text());S=np.linalg.inv(meta['frames'][0]['ego_to_world']);W=np.linalg.inv(S)
 api=get_maps_api(str(ROOT.parent/'data/nuplan/maps'),'nuplan-maps-v1.0',row['map_name']);bl=api.get_map_object(choice['B_lane'],L.LANE);al=api.get_map_object(choice['AC_lane'],L.LANE);road=bl.polygon.union(al.polygon)
 (OUT/'road_corridor.wkb').write_bytes(road.wkb);(OUT/'B_lane.wkb').write_bytes(bl.polygon.wkb);(OUT/'AC_lane.wkb').write_bytes(al.polygon.wkb)
 E=np.array(choice['camera_ego_to_world']);base_station=bl.baseline_path.linestring.project(Point(E[:2,3]));line=bl.baseline_path.linestring
 raw={};source_hash={};times=np.arange(81)*.1
 for variant in ['normal','event']:
  file=ROOT/'artifacts/nexus_lane_response_v2_20261027/pilot_seed0/case_059'/variant/'prediction.pt';payload=torch.load(file,map_location='cpu',weights_only=False);state=decode(payload['sampled_tensor'][0].numpy())[[roles[r] for r in rolelist]];valid=payload['valid_mask'][0].bool().numpy()[[roles[r] for r in rolelist]];assert valid.all();xy,yaw,_=sample_positions(state,valid,times,np);raw[variant]=(state,xy,yaw);source_hash[str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
 sizes=raw['normal'][0][:,4,6:8];assets={};render_sizes=[]
 for i,role in enumerate(rolelist):
  file=Path('/inspire/hdd3/project/intelligent-driving-agent/public/workspace/hzy/3dgs_pro')/('3.ply' if role=='B' else '1.ply')
  # These two inspected PLYs contain only little-endian float vertex fields.
  # Read their bounds without adding packages to the Nexus inference env.
  with file.open('rb') as f:
   fields=[];count=None
   while True:
    line_header=f.readline().decode().strip()
    if line_header.startswith('format '):assert line_header=='format binary_little_endian 1.0'
    if line_header.startswith('element vertex '):count=int(line_header.split()[-1])
    if line_header.startswith('property '):
     _,kind,name=line_header.split();assert kind=='float';fields.append((name,'<f4'))
    if line_header=='end_header':break
   v=np.fromfile(f,dtype=np.dtype(fields),count=count)
  xyz=np.stack([v[k] for k in ('x','y','z')],-1);extent=np.ptp(xyz,axis=0);factor=float(min(sizes[i]/extent[:2])*.97);size=extent*factor;render_sizes.append(size[:2]);assets[role]={'source':str(file),'source_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'target_original_lw':sizes[i].tolist(),'uniform_scale':factor,'render_lwh':size.tolist()}
 render_sizes=np.array(render_sizes);origin=raw['normal'][0][1,4,:2];h=raw['normal'][0][1,4,2:4];source_yaw=float(np.arctan2(h[1],h[0]));c_along=float((raw['normal'][0][2,4,:2]-origin)@h)
 def boxes(xy,yaw):return np.concatenate([xy.transpose(1,0,2),yaw.T[...,None]],-1)
 def body(q,a,size):
  R=np.array([[np.cos(a),np.sin(a)],[-np.sin(a),np.cos(a)]]);return Polygon(np.array([[1,1],[1,-1],[-1,-1],[-1,1]])*size/2@R+q)
 def audit(xy,yaw,variant):
  sep,pair=pair_separations(boxes(xy,yaw),np.ones((81,3),bool),render_sizes);off=0;normal_wrong=0;event_end_wrong=0;normal_role_wrong={r:0 for r in rolelist};min_road=1e9
  for k,t in enumerate(times):
   for i,r in enumerate(rolelist):
    p=body(xy[i,k],yaw[i,k],render_sizes[i]);off+=not road.buffer(.02).covers(p)
    if variant=='normal':
     bad=not (al if r=='A' else bl).polygon.buffer(.02).covers(p);normal_wrong+=bad;normal_role_wrong[r]+=int(bad)
    if variant=='event' and t>=7:event_end_wrong+=not (al if r=='C' else bl).polygon.buffer(.02).covers(p)
  return {'overlap_samples':int(((sep<0)&pair).sum()),'minimum_pair_gap_m':float(sep[pair].min()),'body_outside_two_lane_corridor_samples':int(off),'normal_wrong_lane_samples':int(normal_wrong),'normal_role_wrong':normal_role_wrong,'event_last_second_wrong_lane_samples':int(event_end_wrong)}
 # Retarget arc progress and lane-relative offsets, removing the original
 # road bend rather than forcing a curved trajectory into a straight lane.
 with np.load(ROOT/'artifacts/nexus_lane_response_v2_20261027/case_059/map/routes.npz') as data:source_line=LineString(data['route_41']);source_left_line=LineString(data['route_39'])
 def frame_at(curve,station):
  p0=np.array(curve.interpolate(station).coords[0]);a=np.array(curve.interpolate(max(station-.1,0)).coords[0]);b=np.array(curve.interpolate(min(station+.1,curve.length)).coords[0]);direction=(b-a)/max(np.linalg.norm(b-a),1e-8);return p0,direction
 source_station_B=source_line.project(Point(origin));source_station_C=source_line.project(Point(raw['normal'][0][2,4,:2]));c_along=source_station_C-source_station_B
 source_A=raw['normal'][0][0,4,:2];sa=source_line.project(Point(source_A));qa,ha=frame_at(source_line,sa);source_gap=float((source_A-qa)@np.array([-ha[1],ha[0]]));assert source_gap>2
 profiles={}
 for variant,(state,xy,yaw) in raw.items():
  station=np.zeros((3,81));lateral=np.zeros_like(station);heading_relative=np.zeros_like(station)
  for i in range(3):
   for k in range(81):
    st=source_line.project(Point(xy[i,k]));q,h=frame_at(source_line,st);station[i,k]=st-source_station_B;lateral[i,k]=(xy[i,k]-q)@np.array([-h[1],h[0]]);heading_relative[i,k]=yaw[i,k]-np.arctan2(h[1],h[0])
  profiles[variant]=(station,lateral,heading_relative)
 candidates=[];best=None
 for distance in [16.,20.,12.,24.]:
  target_station=base_station+distance-c_along;target,heading=frame_at(line,target_station);left_point=np.array(al.baseline_path.linestring.interpolate(al.baseline_path.linestring.project(Point(target))).coords[0]);target_gap=float((left_point-target)@np.array([-heading[1],heading[0]]));factor=target_gap/source_gap
  for adjustment in [0.,-.1,.1,-.2,.2,-.3,.3,-.4,.4]:
   metrics={};adapted={}
   for variant,(station,lateral,relative) in profiles.items():
    xy=np.zeros((3,81,2));yaw=np.zeros((3,81))
    for i in range(3):
     for k in range(81):
      q,h=frame_at(line,target_station+station[i,k]);qs,hs=frame_at(source_line,source_station_B+station[i,k]);ql=np.array(source_left_line.interpolate(source_left_line.project(Point(qs))).coords[0]);source_spacing=float((ql-qs)@np.array([-hs[1],hs[0]]));qt=np.array(al.baseline_path.linestring.interpolate(al.baseline_path.linestring.project(Point(q))).coords[0]);target_spacing=float((qt-q)@np.array([-h[1],h[0]]));local_factor=target_spacing/max(source_spacing,1.);xy[i,k]=q+(lateral[i,k]*local_factor+adjustment)*np.array([-h[1],h[0]]);yaw[i,k]=np.arctan2(h[1],h[0])+relative[i,k]
    metrics[variant]=audit(xy,yaw,variant);adapted[variant]=(xy,yaw)
   violations=sum(m['overlap_samples']+m['body_outside_two_lane_corridor_samples']+m['normal_wrong_lane_samples']+m['event_last_second_wrong_lane_samples'] for m in metrics.values());score=violations*1e6+abs(adjustment)*10+abs(distance-16)*.01
   rec={'C_initial_ahead_camera_m':distance,'lateral_adjustment_m':adjustment,'lateral_lane_spacing_factor':factor,'target_B_initial_station':target_station,'source_B_current_station':source_station_B,'violation_count':violations,'metrics':metrics};candidates.append(rec)
   if best is None or score<best[0]:best=(score,rec,adapted)
  if best[1]['violation_count']==0:break
 score,chosen,adapted=best
 write(OUT/'placement_search.json',{'method':'map Frenet transfer','previous_rigid_failure':str(OUT/'placement_search_rigid_failed.json'),'candidates':candidates,'chosen':chosen})
 
 if chosen['violation_count']:
  np.savez_compressed(OUT/'placement_failed_points.npz',normal_xy_world=adapted['normal'][0],normal_yaw_world=adapted['normal'][1],event_xy_world=adapted['event'][0],event_yaw_world=adapted['event'][1],render_sizes_lw=render_sizes)
 assert chosen['violation_count']==0,chosen
 for variant,(xy,yaw) in adapted.items():
  velocity=np.diff(xy[:,::5],axis=1)/.5;acceleration=np.diff(velocity,axis=1)/.5;chosen['metrics'][variant]['max_native_acceleration_mps2']=float(np.linalg.norm(acceleration,axis=-1).max());assert np.linalg.norm(acceleration,axis=-1).max()<=5
  chosen['metrics'][variant]['speeds_mps']={role:np.linalg.norm(velocity[i],axis=-1).tolist() for i,role in enumerate(rolelist)}
 np.savez_compressed(OUT/'placed_trajectories.npz',times_s=times,normal_xy_world=adapted['normal'][0],normal_yaw_world=adapted['normal'][1],event_xy_world=adapted['event'][0],event_yaw_world=adapted['event'][1],render_sizes_lw=render_sizes)
 # Fit a road plane from real points inside these SDK lanes, excluding all
 # annotated objects. The fit uses the full 120m needed by the new trajectories.
 info=json.loads((scene/'instances/instances_info.json').read_text());objects={}
 for obj in info.values():
  ann=obj['frame_annotations']
  for frame,p,size in zip(ann['frame_idx'],ann['obj_to_world'],ann['box_size']):objects.setdefault(int(frame),[]).append((np.array(p),np.array(size)))
 collected=[]
 for frame in range(0,101,8):
  points=np.fromfile(scene/'lidar'/f'{frame:03d}.bin',np.float32).reshape(-1,6)[::4,:3];pose=np.array(meta['frames'][frame]['ego_to_world']);world=points@pose[:3,:3].T+pose[:3,3];local=world@S[:3,:3].T+S[:3,3];mask=vectorized.contains(road.buffer(-.15),world[:,0],world[:,1])&(local[:,2]>-4)&(local[:,2]<5);world=world[mask]
  for pose,size in objects.get(frame,[]):
   q=(world-pose[:3,3])@pose[:3,:3];world=world[~np.all(np.abs(q)<size/2+.2,axis=-1)]
  collected.append(world@S[:3,:3].T+S[:3,3])
 points=np.concatenate(collected);cells=np.floor(points[:,:2]);_,group=np.unique(cells,axis=0,return_inverse=True);order=np.argsort(group);samples=[]
 for ids in np.split(order,np.flatnonzero(np.diff(group[order]))+1):
  if len(ids)<4:continue
  q=np.median(points[ids],axis=0);q[2]=np.quantile(points[ids,2],.25);samples.append(q)
 samples=np.array(samples);A=np.c_[samples[:,:2],np.ones(len(samples))];rng=np.random.default_rng(0);inliers=np.zeros(len(samples),bool)
 for _ in range(500):
  ids=rng.choice(len(samples),3,replace=False)
  if abs(np.linalg.det(A[ids]))<1e-6:continue
  coef=np.linalg.solve(A[ids],samples[ids,2]);hit=np.abs(A@coef-samples[:,2])<.08
  if np.linalg.norm(coef[:2])<.15 and hit.sum()>inliers.sum():inliers=hit
 assert inliers.sum()>50 and inliers.mean()>.5
 for _ in range(3):coef=np.linalg.lstsq(A[inliers],samples[inliers,2],rcond=None)[0];inliers=np.abs(A@coef-samples[:,2])<.08
 plane=S.T@np.r_[-coef[:2],1.,-coef[2]];plane/=np.linalg.norm(plane[:3])
 ground={'status':'validated','coefficients_scene':coef.tolist(),'world_plane':plane.tolist(),'inlier_rmse_m':float(np.sqrt(np.mean((A[inliers]@coef-samples[inliers,2])**2))),'inlier_fraction':float(inliers.mean()),'cells':len(samples),'supported_bounds_scene':[samples[inliers,:2].min(0).tolist(),samples[inliers,:2].max(0).tolist()],'source':'real LiDAR inside selected SDK lanes; annotated objects excluded'}
 write(OUT/'insertion_ground.json',ground)
 protocol={'status':'prepared','background_scenario':row,'background_scene':str(scene),'background_camera_anchor_ego':E.tolist(),'background_camera_anchor_frame':40,'B_lane':bl.id,'AC_lane':al.id,'roles':roles,'asset_assignment':assets,'source_prediction_sha256':source_hash,'source_trajectory_scene':59,'source_and_derived_scope':'synthetic three-car asset insertion on scene121; only A/B/C from saved Nexus plus behavior guidance','other_original_background_traffic':'not included in the synthetic traffic layer; background geometry remains from reconstruction','mapping_method':'map Frenet station/lateral transfer','placement':chosen,'road_ground':ground,'trajectory_policy':'shared map Frenet transfer for every role/variant; original time and longitudinal arc progress retained; lateral offsets scaled by spatially varying source/target paired-lane spacing; original road bend removed; XY/yaw are derived for new background and independently rechecked; no new Nexus inference','model_training_updates':'background reconstruction only; Nexus unchanged','photorealistic_quality_pending':True}
 write(OUT/'insertion_protocol.json',protocol);print('INSERTION_PREPARED',chosen,'ground',ground['inlier_rmse_m'],flush=True)
if __name__=='__main__':main()
