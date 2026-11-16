"""Response-focused revision; retain every previous experiment unchanged."""
import copy,hashlib,json,os,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'artifacts/nexus_four_hazards_20261110';OUT=ROOT/'artifacts/nexus_hazard_response_v2_20261110';SOURCE=ROOT/'artifacts/nuplan_evaluation/independent256_v1'
sys.path[:0]=[str(ROOT),str(ROOT/'scripts/pipeline'),str(ROOT/'third_party/nuplan-devkit'),str(ROOT/'third_party/alf')]
os.environ.update(NUPLAN_DATA_STORE='local',MPLBACKEND='Agg',PYTHONDONTWRITEBYTECODE='1')
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 import numpy as np,torch
 from shapely import wkb
 from shapely.geometry import Polygon,LineString,Point,box
 from shapely.ops import unary_union
 from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
 from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
 from nuplan.common.actor_state.state_representation import Point2D
 from nexus_3dgs.bundle import decode,FEATURE_MEANS,FEATURE_STD
 from run_nexus_omega_v2 import restore_history_velocity
 from prepare_nexus_multiscene_validation import raster
 torch.set_num_threads(2);OUT.mkdir(exist_ok=True);old=json.loads((BASE/'protocol.json').read_text());protocol=copy.deepcopy(old);protocol.update(status='preparing',cases=[],attempts=[],parent_protocol=str(BASE/'protocol.json'),model='Nexus + disclosed response intent and physical control guidance; weights unchanged',E_future='never conditioned; first two events replan from rolled-out normal prefix at detection; lane-change cases have soft route intent',information_scope='brake/reveal event information introduced at detection stage; rear/oncoming edited A known during sampling; minimal role scenes',revision='close following, monotone stop, E changes lane for rear and oncoming')
 if (OUT/'protocol.json').exists():protocol=json.loads((OUT/'protocol.json').read_text())
 (OUT/'prepared').mkdir(exist_ok=True)
 def body(q,h,size,ego=False):
  h=h/max(np.linalg.norm(h),1e-8);q=q+(1.461*h if ego else 0);return Polygon(np.array([[1,1],[1,-1],[-1,-1],[-1,1]])*size/2@np.array([[h[0],h[1]],[-h[1],h[0]]])+q)
 def precheck(case,info):
  W=np.array(info['world_from_nexus']);road=wkb.loads((case/'reference/inputs/drivable.wkb').read_bytes()).buffer(.02);checks={}
  for variant in ['normal','event']:
   with np.load(case/f'conditions_{variant}.npz') as d:s=decode(d['values']);valid=d['valid'];keep=d['keep']
   assert not keep[0,5:].any();off={};hits=0
   for slot in info['fixed_slots']:
    if not valid[slot,4]:continue
    dense=np.arange(81)*.1;xy=np.c_[np.interp(dense,np.arange(-2,8.01,.5),s[slot,:,0]),np.interp(dense,np.arange(-2,8.01,.5),s[slot,:,1])];yaw=np.interp(dense,np.arange(-2,8.01,.5),np.unwrap(np.arctan2(s[slot,:,3],s[slot,:,2])));off[slot]=0
    for q,a in zip(xy,yaw):
     b=body(q,np.array([np.cos(a),np.sin(a)]),s[slot,4,6:8]);poly=Polygon(np.array(b.exterior.coords)@W[:2,:2].T+W[:2,3]);off[slot]+=not road.covers(poly)
   checks[variant]={'given_actor_body_offroad_samples':off};assert not any(off.values()),checks
  info['input_precheck_v2']=checks
 for kind in ['brake','reveal']:
  if any(c['kind']==kind for c in protocol['cases']):continue
  info=copy.deepcopy(next(c for c in old['cases'] if c['kind']==kind));case=OUT/'prepared'/kind;shutil.copytree(BASE/info['case'],case);info['case']=str(case.relative_to(OUT));info['causal_replan_trigger_s']=1.5;info['stop_deadline_global_s']=6.5;info['response_intent']='stop_without_reacceleration';info['stop_role']='A' if kind=='brake' else 'S';A=info['roles']['A']
  if kind=='brake':
   with np.load(case/'conditions_normal.npz') as d:s=decode(d['values']);e=s[0,4];a=s[A,4]
   h=e[2:4]/np.linalg.norm(e[2:4]);oldgap=float((a[:2]-e[:2]-1.461*h)@h-(a[6]+e[6])/2);shift=oldgap-12.
   for variant in ['normal','event']:
    with np.load(case/f'conditions_{variant}.npz') as d:values=d['values'].copy();keep=d['keep'];valid=d['valid']
    given=decode(values[A]);given[:,:2]-=shift*h;values[A]=((given-FEATURE_MEANS)/(2*FEATURE_STD)).astype(values.dtype);np.savez_compressed(case/f'conditions_{variant}.npz',values=values,keep=keep,valid=valid)
   scene=torch.load(case/'scene.pt',weights_only=False);original=decode(scene['reference']);original[A,:,:2]-=shift*h;scene['reference']=((original-FEATURE_MEANS)/(2*FEATURE_STD)).astype(scene['reference'].dtype);torch.save(scene,case/'scene.pt')
   info['parameters'].update(initial_net_following_gap_m=12.,previous_initial_gap_m=oldgap,A_history_and_future_backward_shift_m=shift,brake_start_s=1.5);info['title']='近距离跟驰：前车急刹，自车制动停车'
  else:
   with np.load(case/'conditions_event.npz') as d:values=d['values'].copy();keep=d['keep'].copy();valid=d['valid'].copy()
   S=info['roles']['S'];values[S]=0;keep[S]=False;valid[S]=False;np.savez_compressed(case/'conditions_normal.npz',values=values,keep=keep,valid=valid)
   info['parameters'].update(normal_A='same lane change, road ahead clear; no S supplied',S_first_supplied_to_model_at_s=1.5);info['title']='前车变道暴露静止车：自车减速停车'
  precheck(case,info);write(case/'preparation.json',info);protocol['cases'].append(info);write(OUT/'protocol.json',protocol);print('RESPONSE_INPUT_PREPARED',kind,flush=True)
 # Road-layout scan may still be running; rerunning appends missing cases only.
 if not (OUT/'road_layouts.json').exists():write(OUT/'protocol.json',protocol);return
 layouts=json.loads((OUT/'road_layouts.json').read_text())['layouts'];maps={}
 def tangent(line,station):
  a=np.array(line.interpolate(max(0,station-.2)).coords[0]);b=np.array(line.interpolate(min(line.length,station+.2)).coords[0]);return (b-a)/max(np.linalg.norm(b-a),1e-8)
 def route(obj):
  first=obj;chain=[obj];seen={(type(obj).__name__,str(obj.id))};length=0.
  for _ in range(12):
   h=tangent(first.baseline_path.linestring,0);options=[]
   for edge in first.incoming_edges:
    key=(type(edge).__name__,str(edge.id))
    if key in seen:continue
    hh=tangent(edge.baseline_path.linestring,edge.baseline_path.linestring.length);options.append((1-float(hh@h),str(edge.id),edge))
   if not options or length>110:break
   _,_,first=min(options,key=lambda a:(a[0],a[1]));seen.add((type(first).__name__,str(first.id)));chain.insert(0,first);length+=first.baseline_path.linestring.length
  last=obj;length=0.
  for _ in range(20):
   h=tangent(last.baseline_path.linestring,last.baseline_path.linestring.length);options=[]
   for edge in last.outgoing_edges:
    key=(type(edge).__name__,str(edge.id))
    if key in seen:continue
    line=edge.baseline_path.linestring;hh=tangent(line,0);he=tangent(line,line.length);options.append((2-float(hh@h)-float(hh@he),str(edge.id),edge))
   if not options or length>220:break
   _,_,last=min(options,key=lambda a:(a[0],a[1]));seen.add((type(last).__name__,str(last.id)));chain.append(last);length+=last.baseline_path.linestring.length
  points=np.concatenate([np.array(x.baseline_path.linestring.coords) for x in chain]);points=points[np.r_[True,np.linalg.norm(np.diff(points,axis=0),axis=-1)>1e-5]]
  return LineString(points),unary_union([o.polygon for o in chain])
 for kind in ['rear','oncoming']:
  if any(c['kind']==kind for c in protocol['cases']):continue
  candidates=[r for r in layouts if kind!='oncoming' or r['opposite_direction']]
  candidates.sort(key=lambda r:(r['scenario']['index'] not in ([170,9,18] if kind=='rear' else [174]),abs(r['E_speed_mps']-8),r['scenario']['index']))
  for attempt,layout in enumerate(candidates):
   row=layout['scenario'];case=OUT/'prepared'/f'{kind}_{row["index"]:03d}_{attempt:02d}'
   if case.exists():continue
   case.mkdir();record={'kind':kind,'index':row['index'],'case':str(case.relative_to(OUT))}
   try:
    if row['map_name'] not in maps:maps[row['map_name']]=get_maps_api(str(ROOT.parent/'data/nuplan/maps'),'nuplan-maps-v1.0',row['map_name'])
    api=maps[row['map_name']];el=api.get_map_object(layout['E_lane'],L.LANE);W=np.array(layout['world_from_nexus']);R=W[:2,:2];offset=W[:2,3]
    rawfile=SOURCE/'raw_features'/f"{row['index']:03d}_{row['token']}.pt";raw=torch.load(rawfile,map_location='cpu',weights_only=False);original=raw['tensor'].numpy()[...,:8].copy();rv=raw['validity'].numpy().astype(bool);history=restore_history_velocity(original,original,rv);physical=decode(history);e=physical[0,4];ve=float(np.linalg.norm(e[4:6]));world=e[:2]@R.T+offset;eh=e[2:4]@R.T
    options=layout['same_direction']
    if kind=='oncoming':
     op=layout['opposite_direction'][0];options=[x for x in options if x['lateral_m']*op['lateral_m']<0]
     if not options:raise ValueError('escape lane is not away from opposing lane')
    target_rec=min(options,key=lambda x:abs(abs(x['lateral_m'])-3.5));target=api.get_map_object(target_rec['id'],L.LANE);eline,ecorr=route(el);tline,tcorr=route(target);station=eline.project(Point(world));q=np.array(eline.interpolate(station).coords[0]);lateral=float((world-q)@np.array([-eh[1],eh[0]]))
    stations=np.arange(0,eline.length,.5);own=np.array([np.array(eline.interpolate(x).coords[0])+lateral*np.array([-tangent(eline,x)[1],tangent(eline,x)[0]]) for x in stations]);ownline=LineString(own);station=ownline.project(Point(world))
    if ownline.length-station<max(130,ve*8+10):raise ValueError('ego forward lane corridor too short')
    target_world=np.array(tline.interpolate(tline.project(Point(world))).coords[0]);target_shift=float((target_world-world)@np.array([-eh[1],eh[0]]))
    maneuver=[]
    for q in own:
     s=ownline.project(Point(q))-station;u=np.clip((s-ve*.35)/(ve*3.6),0,1);w=10*u**3-15*u**4+6*u**5;z=np.array(tline.interpolate(tline.project(Point(q))).coords[0]);maneuver.append(q*(1-w)+z*w)
    maneuver=np.array(maneuver);A=1;valid=np.zeros_like(rv);valid[[0,A]]=True;keep=np.zeros_like(history,dtype=bool);keep[[0,A],:5]=True;keep[A]=True
    va=ve+7 if kind=='rear' else max(6.,min(9.,ve));parameters={'E_target_lane_id':str(target.id),'E_lane_change_shift_m':target_shift,'E_route_intent':'soft spatial lane change toward adjacent same-direction lane; no E future values conditioned','initial_rear_ttc_s':25/(va-ve)}
    if kind=='rear':
     aline=ownline;astation=station+1.461-30.;al=el;acorr=ecorr
     if astation<va*2+2:raise ValueError('insufficient incoming lane for synthetic rear car history')
     parameters.update(initial_net_gap_m=30-(e[6]+4.8)/2,rear_A_initial_speed_mps=va,E_observed_speed_mps=ve,event_A_brake_start_s=20.,event_A='continues current lane at high speed; no lane change',normal_A='brakes immediately to observed E speed')
    else:
     al=api.get_map_object(op['id'],L.LANE);aline,acorr=route(al);desired=world+eh*(ve+va)*4.2;astation=aline.project(Point(desired));ap=np.array(aline.interpolate(astation).coords[0]);ahead=float((ap-world)@eh)
     if not 35<ahead<100 or aline.length-astation<va*8+5 or astation<va*2+2:raise ValueError('insufficient opposing lane horizon')
     parameters.update(crossing_start_s=.5,peak_intrusion_s=2.,return_start_s=100.,return_end_s=101.,event_A='cross center line then remain in invaded lane; never returns during clip',opposing_lane_boundary_gap_m=float(el.polygon.distance(al.polygon)),historical_estimated_encounter_s=ahead/(ve+va))
    (case/'map/lanes').mkdir(parents=True);(case/'reference/inputs').mkdir(parents=True);(case/'reference/normal_route_factorial_v1').mkdir()
    orig_route=((own-offset)@R).astype('float32');event_route=((maneuver-offset)@R).astype('float32');aroute=((np.array(aline.coords)-offset)@R).astype('float32')
    np.savez_compressed(case/'map/routes.npz',**{'route_0':orig_route,'route_1':aroute});np.savez_compressed(case/'map/routes_event.npz',**{'route_0':event_route,'route_1':aroute})
    for name,geometry in [('lanes/0.wkb',ecorr),('lanes/1.wkb',acorr),('original_E_corridor.wkb',ecorr),('target_E_corridor.wkb',tcorr),('escape_corridor.wkb',unary_union([ecorr,tcorr])),('initial_E_lane.wkb',el.polygon),('initial_A_lane.wkb',al.polygon),('target_A_lane.wkb',target.polygon)]: (case/'map'/name).write_bytes(geometry.wkb)
    frame=api._load_vector_map_layer('drivable_area');patch=box(offset[0]-250,offset[1]-250,offset[0]+250,offset[1]+250);region=unary_union([g.intersection(patch) for g in frame[frame.geometry.intersects(patch)].geometry]);(case/'reference/inputs/drivable.wkb').write_bytes(region.wkb);raster(region,W,case/'reference/inputs/drivable_sdf.npz',[-220.,-220.],.25,(1761,1761))
    for variant,corridor in [('normal',ecorr),('event',unary_union([ecorr,tcorr]))]:
     grid=raster(corridor,W,None,[-100.,-100.],.2,(1001,1501));np.savez_compressed(case/f'map/lane_corridors_{variant}.npz',sdf=grid[None],slots=np.array([0]),origin=np.array([-100.,-100.]),resolution=.2)
    shutil.copy2(case/'map/lane_corridors_normal.npz',case/'map/lane_corridors_sdf.npz')
    for variant in ['normal','event']:
     tt=np.arange(-2,8.01,.5);distance=va*tt
     if kind=='rear' and variant=='normal':
      tau=np.maximum(tt,0);u=np.minimum(tau,(va-ve)/3.5);distance=np.where(tt<=0,va*tt,va*u-1.75*u*u+ve*np.maximum(tau-(va-ve)/3.5,0))
     given=np.zeros((21,8));given[:,6:8]=[4.8,1.9]
     for j,x in enumerate(distance):
      s=astation+x;given[j,:2]=(np.array(aline.interpolate(s).coords[0])-offset)@R;given[j,2:4]=tangent(aline,s)@R
     if kind=='oncoming' and variant=='event':
      for j,t in enumerate(tt):
       if t<=.5:continue
       q=given[j,:2]@R.T+offset;target_q=np.array(ownline.interpolate(ownline.project(Point(q))).coords[0]);u=np.clip((t-.5)/1.5,0,1);w=10*u**3-15*u**4+6*u**5;given[j,:2]=((1-w)*q+w*target_q-offset)@R
     velocity=np.diff(given[:,:2],axis=0)/.5;given[0,4:6]=va*given[0,2:4];given[1:,4:6]=velocity;moving=np.linalg.norm(velocity,axis=-1)>.05;given[1:,2:4][moving]=velocity[moving]/np.linalg.norm(velocity[moving],axis=-1,keepdims=True)
     values=np.zeros_like(history);values[0,:5]=history[0,:5];values[A]=((given-FEATURE_MEANS)/(2*FEATURE_STD)).astype(values.dtype);np.savez_compressed(case/f'conditions_{variant}.npz',values=values,keep=keep,valid=valid)
     if variant=='normal':original[A]=values[A]
    info={'kind':kind,'title':'高速后车逼近：自车变道让行' if kind=='rear' else '对向车持续侵入：自车变道避让','case':str(case.relative_to(OUT)),'scenario':row,'roles':{'E':0,'A':A},'active_slots':[0,A],'fixed_slots':[A],'parameters':parameters,'observed_speeds_mps':{'E':ve,'A':va},'world_from_nexus':W.tolist(),'lanes':{'E':str(el.id),'A':str(al.id),'target_E':str(target.id)},'response_intent':'adjacent_lane_change','synthetic_A':'map-aligned synthetic event vehicle; original other traffic omitted','given_A_input_usable':True}
    write(case/'reference/protocol.json',{'scenario':row,'world_from_nexus':W.tolist(),'source_sha256':{}});torch.save({'reference':original,'road_graph':raw['road_graph'].numpy(),'road_graph_validity':raw['road_graph_validity'].numpy()},case/'scene.pt');precheck(case,info);write(case/'preparation.json',info);protocol['cases'].append(info);record['status']='accepted';protocol['attempts'].append(record);write(OUT/'protocol.json',protocol);print('RESPONSE_INPUT_PREPARED',kind,row['index'],parameters,flush=True);break
   except Exception as exc:
    record.update(status='rejected',reason=str(exc));write(case/'rejection.json',record);protocol['attempts'].append(record);write(OUT/'protocol.json',protocol);print('RESPONSE_INPUT_REJECTED',kind,row['index'],str(exc),flush=True)
 protocol['status']='prepared' if len(protocol['cases'])==4 else 'incomplete';write(OUT/'protocol.json',protocol);print('RESPONSE_PREPARATION',protocol['status'],len(protocol['cases']),flush=True)
if __name__=='__main__':main()
