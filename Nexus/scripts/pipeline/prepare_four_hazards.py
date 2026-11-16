"""Compile four minimal edited traffic scenes on real nuPlan roads."""
import hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/nexus_four_hazards_20261110'
SOURCE=ROOT/'artifacts/nuplan_evaluation/independent256_v1'
sys.path[:0]=[str(ROOT),str(ROOT/'scripts/pipeline'),str(ROOT/'third_party/nuplan-devkit'),str(ROOT/'third_party/alf')]
os.environ.update(MPLBACKEND='Agg',NUPLAN_DATA_STORE='local',PYTHONDONTWRITEBYTECODE='1')
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 import numpy as np,torch
 from shapely.geometry import Point,LineString,Polygon,box
 from shapely.ops import unary_union
 from shapely import wkb
 from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
 from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
 from nuplan.common.actor_state.state_representation import Point2D
 from nexus_3dgs.bundle import decode,FEATURE_MEANS,FEATURE_STD
 from nexus_3dgs.geometry import pose
 from nexus_3dgs import nuplan
 from run_nexus_omega_v2 import restore_history_velocity
 from nexus_multiscene_routes import prepare_routes
 from prepare_nexus_multiscene_validation import raster
 torch.set_num_threads(2)
 scan=json.loads((OUT/'history_candidate_scan.json').read_text())['candidates']
 base=json.loads((ROOT/'artifacts/nexus_multiscene_20261027/protocol.json').read_text())
 protocol={'status':'preparing','history_s':2,'forecast_s':8,'dt_s':.5,'render_dt_s':.1,
   'seed':0,'config':base['config'],'checkpoint_sha256':base['checkpoint_sha256'],
   'model':'Nexus + frozen V4 InitializedCruiseRefiner; no weight training',
   'population_scope':'minimal edited role scenes: only E/A, plus inserted stationary S for reveal; other original traffic omitted',
   'E_future':'all zero values and false condition mask, generated freely',
   'information_scope':'full given A future, supplied stationary S including history, recorded existence; not online closed-loop or visibility-limited perception',
   'selection':'history geometry and input feasibility only; no generated response used',
   'offline_output_repairs':False,'training_updates':0,'cases':[],'attempts':[]}
 if (OUT/'protocol.json').exists():
  previous=json.loads((OUT/'protocol.json').read_text());protocol['cases']=previous['cases'];protocol['attempts']=previous['attempts']
 (OUT/'prepared').mkdir(exist_ok=True);write(OUT/'protocol.json',protocol)
 maps={};times=np.arange(17)*.5;dense=np.arange(81)*.1
 names={'brake':'前车无障碍急刹','reveal':'前车变道暴露静止车','rear':'后车高速接近','oncoming':'对向车辆越过中心线'}
 def tangent(line,station):
  p=np.array(line.interpolate(max(0,station-.1)).coords[0]);q=np.array(line.interpolate(min(line.length,station+.1)).coords[0]);d=q-p;return d/max(np.linalg.norm(d),1e-8)
 def lane(api,xy,head):
  choices=[]
  for layer in [L.LANE,L.LANE_CONNECTOR]:
   for obj in api.get_proximal_map_objects(Point2D(*xy),8,[layer])[layer]:
    line=obj.baseline_path.linestring;h=tangent(line,line.project(Point(xy)));dot=float(h@head)
    if dot<.85:continue
    choices.append((line.distance(Point(xy))+2*(1-dot)+(0 if obj.polygon.covers(Point(xy)) else 10),obj))
  if not choices:raise ValueError('no heading-compatible lane')
  return min(choices,key=lambda x:(x[0],str(x[1].id)))[1]
 def body(q,h,size,ego=False):
  h=np.asarray(h);h=h/max(np.linalg.norm(h),1e-8);center=q+(1.461*h if ego else 0);side=np.array([-h[1],h[0]])
  return Polygon(center+np.array([[1,1],[1,-1],[-1,-1],[-1,1]])*size/2@np.stack([h,side]))
 def profile(v,start,decel,minimum=0):
  if decel==0:return v*times,np.full(17,v)
  tau=np.maximum(times-start,0);duration=max((v-minimum)/decel,0);u=np.minimum(tau,duration)
  return v*np.minimum(times,start)+v*u-.5*decel*u*u+minimum*np.maximum(tau-duration,0),np.maximum(v-decel*tau,minimum)
 def smooth(t,start,end):
  u=np.clip((t-start)/(end-start),0,1);return 10*u**3-15*u**4+6*u**5
 for kind in ['brake','reveal','rear','oncoming']:
  if any(c['kind']==kind for c in protocol['cases']):continue
  success=False
  for attempt,candidate in enumerate(scan[kind]):
   row=candidate['scenario'];A=candidate['A'];case=OUT/'prepared'/f'{kind}_{row["index"]:03d}_{attempt:02d}_v3'
   case.mkdir();ref=case/'reference';(ref/'normal_route_factorial_v1').mkdir(parents=True);(ref/'inputs').mkdir()
   record={'kind':kind,'index':row['index'],'A':A,'case':str(case.relative_to(OUT))}
   try:
    rawfile=SOURCE/'raw_features'/f"{row['index']:03d}_{row['token']}.pt";raw=torch.load(rawfile,map_location='cpu',weights_only=False)
    original=raw['tensor'].numpy()[...,:8].copy();raw_valid=raw['validity'].numpy().astype(bool)
    valid=np.zeros_like(raw_valid);valid[0]=raw_valid[0];valid[A]=True;assert raw_valid[[0,A],:5].all() and valid[0].all()
    states=restore_history_velocity(original,original,raw_valid);physical=decode(states)
    with nuplan.connect(row['database']) as db:anchor=nuplan.anchor_frame(db,row['token'])
    e=physical[0,4];W=pose(anchor['x'],anchor['y'],yaw=nuplan.ego_yaw(anchor))@np.linalg.inv(pose(e[0],e[1],yaw=np.arctan2(e[3],e[2])));R=W[:2,:2];offset=W[:2,3]
    api=maps.setdefault(row['map_name'],get_maps_api(str(ROOT.parent/'data/nuplan/maps'),'nuplan-maps-v1.0',row['map_name']))
    el=lane(api,e[:2]@R.T+offset,e[2:4]@R.T);aa=physical[A,4];al=lane(api,aa[:2]@R.T+offset,aa[2:4]@R.T)
    def connected(a,b):
     frontier=[a];seen=set()
     for _ in range(8):
      next_nodes=[]
      for obj in frontier:
       key=(type(obj).__name__,str(obj.id))
       if key==(type(b).__name__,str(b.id)):return True
       if key in seen:continue
       seen.add(key);next_nodes.extend(obj.outgoing_edges)
      frontier=next_nodes
     return False
    if kind in ['brake','reveal','rear'] and not (connected(el,al) or connected(al,el)):raise ValueError('E/A not on connected directed lane segments')
    target=None;shift=0.
    if kind=='oncoming':
     distance=float(el.polygon.distance(al.polygon))
     if distance>.35:raise ValueError(f'opposing lanes separated by gap/median {distance:.2f}m')
     shift=float((e[:2]+1.461*e[2:4]-aa[:2])@np.array([-aa[3],aa[2]]))*.78
    if kind=='reveal':
     candidates=[];world=aa[:2]@R.T+offset
     for obj in api.get_proximal_map_objects(Point2D(*world),8,[L.LANE])[L.LANE]:
      if str(obj.id)==str(al.id):continue
      line=obj.baseline_path.linestring;st=line.project(Point(world));hh=tangent(line,st)
      if hh@(aa[2:4]@R.T)<.95 or al.polygon.distance(obj.polygon)>.35:continue
      q=np.array(line.interpolate(st).coords[0]);delta=(q-world)@R
      lateral=float(delta@np.array([-aa[3],aa[2]]))
      if 2.6<abs(lateral)<4.6:candidates.append((abs(abs(lateral)-3.4),obj,lateral))
     if not candidates:raise ValueError('no adjacent same-direction lane for reveal')
     _,target,shift=min(candidates,key=lambda x:(x[0],str(x[1].id)))
    keep=np.zeros_like(states,dtype=bool);keep[:,:5]=True;keep[A]=True
    values=np.where(keep,states,0);values[~valid.any(-1)]=0
    np.savez_compressed(ref/'normal_route_factorial_v1/conditions_normal.npz',repaired=values,keep=keep,valid=valid)
    write(ref/'protocol.json',{'scenario':row,'world_from_nexus':W.tolist(),'source_sha256':{}})
    prepare_routes(ref,case/'map',A)
    with np.load(case/'map/routes.npz') as d:routes={k:d[k] for k in d.files}
    route=LineString(routes[f'route_{A}']);station=route.project(Point(aa[:2]));correction=aa[:2]-np.array(route.interpolate(station).coords[0]);va=float(np.linalg.norm(aa[4:6]));ve=float(np.linalg.norm(e[4:6]))
    if route.length-station<va*8+5:raise ValueError('A route ends before complete horizon')
    def state_on_route(distance):
     given=physical[A].copy();given[5:,6:8]=physical[A,4,6:8];pos=[];heads=[]
     for x in distance:
      pos.append(np.array(route.interpolate(station+float(x)).coords[0])+correction);heads.append(tangent(route,station+float(x)))
     given[4:,:2]=pos;given[4:,2:4]=heads;return given
    params={};fixed=[A];S=None
    if kind=='brake':
     nd,_=profile(va,0,0);ed,_=profile(va,1.5,6.5);params={'brake_start_s':1.5,'A_deceleration_mps2':6.5,'no_forward_dynamic_obstacles_in_generated_population':True}
    elif kind=='reveal':
     nd,_=profile(va,.5,3.0);ed,_=profile(va,0,0);S=int(np.flatnonzero(~raw_valid[:,4])[-1]);fixed.append(S);valid[S]=True;keep[S]=True
     separation=max(va*3.4,va*.5+va*va/6+(aa[6]+4.5)/2+3)
     q=np.array(route.interpolate(station+separation).coords[0])+correction;h=tangent(route,station+separation)
     physical[S]=np.tile(np.r_[q,h,0.,0.,4.5,1.9],(21,1));states[S]=((physical[S]-FEATURE_MEANS)/(2*FEATURE_STD)).astype(states.dtype);original[S]=states[S]
     params={'lane_change_start_s':.5,'lane_change_end_s':2.5,'lateral_shift_m':shift,'static_S_slot':S,'static_S_source':'synthetic stationary vehicle inserted ahead using current A route, no E future','A_to_S_initial_along_m':separation,'normal_A':'brakes from0.5s at3m/s2 before S','occlusion_is_visual_not_model_observation_mask':True}
    elif kind=='rear':
     late_brake=max(2.,candidate['ttc_s']-.6)
     nd,_=profile(va,0,3.5,ve);ed,_=profile(va,late_brake,4.5,ve);params={'rear_A_initial_speed_mps':va,'E_observed_speed_mps':ve,'initial_rear_ttc_s':candidate['ttc_s'],'normal_A_brake_start_s':0.,'event_A_brake_start_s':late_brake,'event_A_braking_mps2':4.5}
    else:
     nd,_=profile(va,0,0);ed=nd.copy();params={'crossing_start_s':.5,'peak_intrusion_s':2.0,'return_start_s':2.8,'return_end_s':4.5,'lateral_shift_m':shift,'opposing_lane_boundary_gap_m':distance}
    generated={};checks={}
    frame=api._load_vector_map_layer('drivable_area');patch=box(offset[0]-250,offset[1]-250,offset[0]+250,offset[1]+250)
    region=unary_union([g.intersection(patch) for g in frame[frame.geometry.intersects(patch)].geometry]);region=region if region.is_valid else region.buffer(0)
    for variant,distance_along in [('normal',nd),('event',ed)]:
     given=state_on_route(distance_along)
     if variant=='event' and kind in ['reveal','oncoming']:
      w=smooth(times,.5,2.5) if kind=='reveal' else smooth(times,.5,2.)*(1-smooth(times,2.8,4.5))
      heads=given[4:,2:4].copy();given[4:,:2]+=shift*w[:,None]*np.c_[-heads[:,1],heads[:,0]]
     velocity=np.diff(given[4:,:2],axis=0)/.5;speed=np.linalg.norm(velocity,axis=-1);angle=np.arctan2(velocity[:,1],velocity[:,0]);moving=speed>.05
     heads=given[5:,2:4].copy();heads[moving]=np.c_[np.cos(angle[moving]),np.sin(angle[moving])];given[5:,2:4]=heads;given[5:,4:6]=velocity;given[:5]=physical[A,:5]
     v=np.where(keep,states,0);v[~valid.any(-1)]=0;v[A,5:]=((given[5:]-FEATURE_MEANS)/(2*FEATURE_STD)).astype(v.dtype)
     if S is not None:v[S]=states[S]
     assert not keep[0,5:].any() and (v[~keep]==0).all()
     np.testing.assert_array_equal(v[[0,A],:5,:4],states[[0,A],:5,:4])
     np.savez_compressed(case/f'conditions_{variant}.npz',values=v,keep=keep,valid=valid)
     interp=np.c_[np.interp(dense,np.arange(-2,8.01,.5),given[:,0]),np.interp(dense,np.arange(-2,8.01,.5),given[:,1])];angles=np.interp(dense,np.arange(-2,8.01,.5),np.unwrap(np.arctan2(given[:,3],given[:,2])))
     off=0;static_hits=0
     for q,ang in zip(interp,angles):
      b=body(q,[np.cos(ang),np.sin(ang)],aa[6:8]);bw=Polygon(np.array(b.exterior.coords)@R.T+offset);off+=not region.buffer(.02).covers(bw)
      if S is not None:static_hits+=b.intersection(body(physical[S,4,:2],physical[S,4,2:4],physical[S,4,6:8])).area>1e-5
     acc=np.linalg.norm(np.diff(given[:,:2],n=2,axis=0)/.25,axis=-1)[3:];checks[variant]={'A_offroad_samples':int(off),'A_static_overlap_samples':int(static_hits),'A_max_native_acceleration_mps2':float(acc.max())};generated[variant]=given
    if any(c['A_offroad_samples'] or c['A_static_overlap_samples'] or c['A_max_native_acceleration_mps2']>9 for c in checks.values()):raise ValueError('prescribed A failed physical precheck '+str(checks))
    if S is not None:
     sb=body(physical[S,4,:2],physical[S,4,2:4],physical[S,4,6:8]);sw=Polygon(np.array(sb.exterior.coords)@R.T+offset)
     if not region.buffer(.02).covers(sw):raise ValueError('inserted static car not fully on road')
     eye=e[:2]+(1.461+e[6]/2-.5)*e[2:4];ab=body(aa[:2],aa[2:4],aa[6:8]);sp=physical[S,4];front=sp[:2]-sp[2:4]*sp[6]/2;side=np.array([-sp[3],sp[2]])
     occluded=float(np.mean([LineString([eye,front+u*side*sp[7]/2]).intersects(ab) for u in np.linspace(-1,1,41)]));params['initial_static_front_face_rays_blocked_fraction']=occluded
     if occluded<.95:raise ValueError(f'static car not initially occluded enough {occluded:.2f}')
    eb=body(e[:2],e[2:4],e[6:8],ego=True)
    if not el.polygon.buffer(.02).covers(Polygon(np.array(eb.exterior.coords)@R.T+offset)):raise ValueError('ego initial body outside selected lane')
    (ref/'inputs/drivable.wkb').write_bytes(region.wkb);raster(region,W,ref/'inputs/drivable_sdf.npz',[-220.,-220.],.25,(1761,1761))
    egolane=wkb.loads((case/'map/lanes/0.wkb').read_bytes());grid=raster(egolane,W,None,[-60.,-80.],.15,(1068,1601))
    np.savez_compressed(case/'map/lane_corridors_sdf.npz',sdf=grid[None],slots=np.array([0]),origin=np.array([-60.,-80.]),resolution=.15)
    (case/'map/initial_E_lane.wkb').write_bytes(el.polygon.wkb);(case/'map/initial_A_lane.wkb').write_bytes(al.polygon.wkb)
    if target is not None:(case/'map/target_A_lane.wkb').write_bytes(target.polygon.wkb)
    torch.save({'reference':original,'road_graph':raw['road_graph'].numpy(),'road_graph_validity':raw['road_graph_validity'].numpy()},case/'scene.pt')
    info={'kind':kind,'title':names[kind],'case':str(case.relative_to(OUT)),'scenario':row,'roles':{'E':0,'A':A,**({'S':S} if S is not None else {})},'fixed_slots':fixed,'active_slots':np.flatnonzero(valid.any(-1)).tolist(),'raw_sha256':digest(rawfile),'parameters':params,'input_precheck':checks,'observed_speeds_mps':{'E':ve,'A':va},'world_from_nexus':W.tolist(),'lanes':{'E':str(el.id),'A':str(al.id),'target_A':str(target.id) if target else None},'selection_candidate':candidate,'given_A_input_usable':True}
    write(case/'preparation.json',info);protocol['cases'].append(info);record['status']='accepted_input';success=True
    print('HAZARD_PREPARED',kind,row['index'],'A',A,'params',params,'checks',checks,flush=True)
   except Exception as exc:
    record.update(status='rejected_input',reason=str(exc));write(case/'rejection.json',record);print('HAZARD_REJECT',kind,row['index'],str(exc),flush=True)
   protocol['attempts'].append(record);write(OUT/'protocol.json',protocol)
   if success:break
  if not success:print('HAZARD_NO_USABLE_INPUT',kind,flush=True)
 protocol['status']='prepared' if len(protocol['cases'])==4 else 'incomplete_inputs';write(OUT/'protocol.json',protocol)
 print('FOUR_HAZARDS_PREPARATION',protocol['status'],len(protocol['cases']),flush=True)

if __name__=='__main__':main()
