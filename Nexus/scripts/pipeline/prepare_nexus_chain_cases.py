"""Prepare close-following chain cases; all A/B/C histories are observed."""
import hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/nexus_chain_20261027'
SOURCE=ROOT/'artifacts/nuplan_evaluation/independent256_v1'
sys.path[:0]=[str(ROOT),str(ROOT/'scripts/pipeline'),str(ROOT/'third_party/nuplan-devkit'),str(ROOT/'third_party/alf')]
os.environ.update(MPLBACKEND='Agg',NUPLAN_DATA_STORE='local',PYTHONDONTWRITEBYTECODE='1')
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def raster(region,W,path,origin,resolution,shape):
    import numpy as np
    from shapely import affinity,vectorized
    from scipy.ndimage import distance_transform_edt
    R=W[:2,:2];t=W[:2,3];shift=-R.T@t
    local=affinity.affine_transform(region,[R[0,0],R[1,0],R[0,1],R[1,1],shift[0],shift[1]])
    xx,yy=np.meshgrid(np.arange(shape[1])*resolution+origin[0],np.arange(shape[0])*resolution+origin[1])
    inside=vectorized.contains(local,xx,yy)
    if path is None:
        from PIL import Image,ImageDraw
        mask=Image.new('L',(shape[1],shape[0]),0);draw=ImageDraw.Draw(mask)
        for poly in (local.geoms if hasattr(local,'geoms') else [local]):
            points=(np.array(poly.exterior.coords)-origin)/resolution
            draw.polygon([tuple(v) for v in points],fill=1)
            for ring in poly.interiors:
                points=(np.array(ring.coords)-origin)/resolution
                draw.polygon([tuple(v) for v in points],fill=0)
        inside=np.asarray(mask).astype(bool)
    sdf=np.where(inside,distance_transform_edt(inside)*resolution-resolution/2,-distance_transform_edt(~inside)*resolution+resolution/2).astype('float32')
    if path:
        sdf=(distance_transform_edt(inside)-distance_transform_edt(~inside))*resolution
    if path:np.savez_compressed(path,sdf=sdf,origin=np.array(origin,dtype='float32'),resolution=np.float32(resolution))
    return sdf

def main():
    import numpy as np,torch
    from shapely import wkb
    from shapely.geometry import box,Polygon
    from shapely.ops import unary_union
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
    from nuplan_extent.planning.training.preprocessing.features.scene_tensor import SceneTensor
    from nuplan_extent.planning.training.data_augmentation.reset_v import ResetVAugmentor
    from nexus_3dgs.bundle import export_arrays,save_bundle,decode,FEATURE_MEANS,FEATURE_STD
    from nexus_3dgs import nuplan
    from run_nexus_omega_v2 import restore_history_velocity
    from nexus_chain_routes import prepare_routes
    torch.set_num_threads(2)
    chosen=json.loads((OUT/'chosen.json').read_text())['cases']
    selected={c['scenario']['index']:(c['roles']['A'],c['roles']['B'],c['roles']['C']) for c in chosen}
    plan=json.loads((SOURCE/'frozen_plan.json').read_text())['scenarios']
    frozen=json.loads((ROOT/'artifacts/nexus_kinematic_20261027/candidate_v4_cruise_initialized/protocol.json').read_text())
    protocol={'status':'preparing','indices':list(selected),'roles':{str(k):{'B':v[1],'A':v[0],'C':v[2]} for k,v in selected.items()},
        'selection':'chosen.json: history-only A-adjacent-front, B, close same-lane following C; actual role slots; fixed before generation',
        'seeds':[0],'variants':['normal','event'],'history_s':2,'forecast_s':8,'event_window_s':[.5,3.],'event_lateral_shift_m':'observed A-to-B lateral distance',
        'A_forward_offset_m':0,'A_future':'map continuation from current pose/speed, event road-relative quintic shift toward ego',
        'ego_C_future_free':True,'B_C_future_free':True,'config':frozen['config'],'checkpoint_sha256':frozen['checkpoint_sha256'],
        'source_sha256':frozen['source_sha256'],'training_updates':0,'offline_output_repairs':False,
        'extra_information':'recorded future existence masks and background replay, same as original case',
        'speed_policy':'report relative normal/event speeds and recovery; scene033 cruise thresholds diagnostic only, not universal acceptance',
        'input_adaptations':'generic A/C slots; corridor raster expanded to [-100,180]x[-100,100], same .1m SDK polygon raster algorithm; zero A forward shift',
        'previous_input_preflight':str(ROOT/'artifacts/nexus_multiscene_preflight_v1_20261027'),
        'failures_retained':True,'strict_causal_chain_proven':False,'cases':[]}
    write(OUT/'protocol.json',protocol)
    for idx,(A,B,C) in selected.items():
        record=next(r for r in plan if r['index']==idx);case=OUT/f'case_{idx:03d}'
        if (case/'preparation.json').exists() and (case/'scene.pt').exists():
            protocol['cases'].append(json.loads((case/'preparation.json').read_text()));continue
        case.mkdir()
        ref=case/'reference';ref.mkdir();(ref/'inputs').mkdir();(ref/'normal_route_factorial_v1').mkdir()
        rawpath=SOURCE/'raw_features'/f'{idx:03d}_{record["token"]}.pt';raw=torch.load(rawpath,map_location='cpu',weights_only=False)
        scene=SceneTensor(**{k:v.numpy().copy() for k,v in raw.items()});scene=ResetVAugmentor().augment({'scene_tensor':scene},{})[0]['scene_tensor']
        original=scene.tensor[...,:8].copy();valid=scene.validity.astype(bool).copy();valid[~valid[:,4]]=False
        assert valid[[0,A,B,C]].all()
        states=restore_history_velocity(original,raw['tensor'].numpy(),valid)
        gt=torch.from_numpy(original).unsqueeze(0)
        ar,meta=export_arrays({'sampled_tensor':gt,'scene_tensor_gt':gt,'valid_mask':raw['validity'].unsqueeze(0)}, {'database':record['database'],'scenario_token':record['token']})
        save_bundle(ref/'identity_reference',ar,meta);W=np.array(meta['world_from_nexus']);R=W[:2,:2];t=W[:2,3]
        keep=np.zeros_like(states,dtype=bool);keep[:,:5]=True;keep[A]=True
        values=np.where(keep,states,0)
        np.savez_compressed(ref/'normal_route_factorial_v1/conditions_normal.npz',repaired=values,keep=keep,valid=valid)
        rp={'scenario':record,'world_from_nexus':W.tolist(),'source_sha256':protocol['source_sha256']}
        write(ref/'protocol.json',rp)
        prepare_routes(ref,case/'map',C,B)
        mapdir=case/'map'
        mp=json.loads((mapdir/'preparation.json').read_text());mp['history_source']='unshifted observed history; only historical velocity reconstructed';write(mapdir/'preparation.json',mp)
        if not all((mapdir/'lanes'/f'{i}.wkb').exists() for i in (A,B,C)):raise ValueError(f'Initial role lane missing {idx}')
        api=get_maps_api(str(ROOT.parent/'data/nuplan/maps'),'nuplan-maps-v1.0',record['map_name'])
        frame=api._load_vector_map_layer('drivable_area');patch=box(t[0]-250,t[1]-250,t[0]+250,t[1]+250)
        region=unary_union([g.intersection(patch) for g in frame[frame.geometry.intersects(patch)].geometry]);region=region if region.is_valid else region.buffer(0)
        (ref/'inputs/drivable.wkb').write_bytes(region.wkb)
        raster(region,W,ref/'inputs/drivable_sdf.npz',[-220.,-220.],.25,(1761,1761))
        # Same SDK corridor algorithm; wider common raster avoids curve clipping.
        grids=[raster(wkb.loads((mapdir/'lanes'/f'{i}.wkb').read_bytes()),W,None,[-100.,-100.],.1,(2001,2801)) for i in (B,C)]
        np.savez_compressed(mapdir/'lane_corridors_sdf.npz',sdf=np.stack(grids),slots=np.array([B,C]),origin=np.array([-100.,-100.]),resolution=.1)
        predicted={b['track_token'] for b in meta['agent_bindings'] if valid[b['nexus_slot']].any()}
        times=np.arange(81)*100000
        with nuplan.connect(record['database']) as db:
            anchor=nuplan.anchor_frame(db,record['token']);frames=nuplan.nearest_frames(db,anchor['lidar_timestamp']+times)
            annotations=[nuplan.boxes(db,f['lidar_token']) for f in frames]
        objects={o['track_token']:o for obs in annotations for o in obs if o['track_token'] not in predicted and o['category_name'] in ('vehicle','pedestrian','bicycle') and np.linalg.norm(np.array([o['x'],o['y']])-t)<=200}
        names=sorted(objects);lookup={k:i for i,k in enumerate(names)};M=len(names)
        centers=np.zeros((81,M,2),np.float32);head=np.zeros_like(centers);head[...,0]=1;bv=np.zeros((81,M),bool)
        sizes=np.array([[objects[k]['length'],objects[k]['width']] for k in names],np.float32).reshape(-1,2)
        wyaw=np.arctan2(W[1,0],W[0,0])
        for k,obs in enumerate(annotations):
            for o in obs:
                i=lookup.get(o['track_token'])
                if i is None:continue
                centers[k,i]=(np.array([o['x'],o['y']])-t)@R;ang=o['yaw']-wyaw;head[k,i]=[np.cos(ang),np.sin(ang)];bv[k,i]=True
        np.savez_compressed(ref/'inputs/background_replay.npz',times_s=np.arange(81,dtype=np.float32)*.1,centers=centers,heading=head,sizes=sizes,valid=bv)
        physical=decode(states);initial=physical[A,4];speed=np.linalg.norm(initial[4:6])
        with np.load(mapdir/'routes.npz') as data:points=data[f'route_{A}']
        dist=np.r_[0,np.linalg.norm(np.diff(points,axis=0),axis=-1).cumsum()]
        from shapely.geometry import LineString,Point
        start=LineString(points).project(Point(initial[:2]));stations=start+np.arange(17)*.5*speed
        normal=physical[A].copy();xy=np.stack([np.interp(stations,dist,points[:,j]) for j in (0,1)],-1);xy+=initial[:2]-xy[0];normal[4:,:2]=xy
        normal[5:,4:6]=np.diff(xy,axis=0)/.5;ang=np.arctan2(normal[5:,5],normal[5:,4]);normal[5:,2:4]=np.c_[np.cos(ang),np.sin(ang)]
        eh=physical[B,4,2:4];lateral=(initial[:2]-physical[B,4,:2])@np.array([-eh[1],eh[0]])
        event=normal.copy();u=np.clip((np.arange(17)*.5-.5)/2.5,0,1);shift=-lateral*(10*u**3-15*u**4+6*u**5)
        h=normal[4:,2:4];event[4:,:2]+=shift[:,None]*np.c_[-h[:,1],h[:,0]]
        event[5:,4:6]=np.diff(event[4:,:2],axis=0)/.5;ang=np.arctan2(event[5:,5],event[5:,4]);event[5:,2:4]=np.c_[np.cos(ang),np.sin(ang)]
        prechecks={}
        for variant,given in [('normal',normal),('event',event)]:
            v=values.copy();v[A,5:]=((given[5:]-FEATURE_MEANS)/(2*FEATURE_STD)).astype(v.dtype)
            np.savez_compressed(case/f'conditions_{variant}.npz',values=v,keep=keep,valid=valid)
            tt=np.arange(1,81)*.1;xy=np.c_[np.interp(tt,np.arange(-2,8.01,.5),given[:,0]),np.interp(tt,np.arange(-2,8.01,.5),given[:,1])];ang=np.interp(tt,np.arange(-2,8.01,.5),np.unwrap(np.arctan2(given[:,3],given[:,2])))
            off=0
            for q,a in zip(xy,ang):
                rot=np.array([[np.cos(a),np.sin(a)],[-np.sin(a),np.cos(a)]]);corn=np.array([[1,1],[1,-1],[-1,-1],[-1,1]])*initial[6:8]/2@rot+q
                off+=not region.buffer(.02).covers(Polygon(corn@R.T+t))
            acc=np.linalg.norm(np.diff(given[:,:2],n=2,axis=0)/.25,axis=-1)[3:]
            prechecks[variant]={'A_box_offroad_samples':int(off),'A_max_xy_acceleration_mps2':float(acc.max())}
        torch.save({'reference':original,'road_graph':scene.road_graph,'road_graph_validity':scene.road_graph_validity},case/'scene.pt')
        info={'scenario':record,'roles':{'E':0,'A':A,'B':B,'C':C},'raw_sha256':digest(rawpath),'background_count':M,'cut_in_side':'right' if lateral>0 else 'left','observed_speeds_mps':{r:float(np.linalg.norm(physical[i,4,4:6])) for r,i in [('E',0),('A',A),('B',B),('C',C)]},'input_precheck':prechecks,'A_forward_shift_m':0,'E_C_future_used_for_input':False,'given_A_input_usable':all(v['A_box_offroad_samples']==0 and v['A_max_xy_acceleration_mps2']<=5 for v in prechecks.values())}
        rp.update(background_tokens=names);write(ref/'protocol.json',rp);write(case/'preparation.json',info);protocol['cases'].append(info);write(OUT/'protocol.json',protocol)
        print('PREPARED',idx,info,flush=True)
    protocol['status']='prepared'
    protocol['causal_diagnostic']=json.loads((OUT/'chosen.json').read_text())['diagnostic_design'];write(OUT/'protocol.json',protocol)
if __name__=='__main__':main()
