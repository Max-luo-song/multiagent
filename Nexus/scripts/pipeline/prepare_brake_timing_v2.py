"""Move the sudden brake later so the video includes sustained normal following."""
import copy,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'artifacts/nexus_four_hazards_20261110';OUT=BASE/'brake_timing_v2'
sys.path[:0]=[str(ROOT),str(ROOT/'scripts/pipeline')]
def main():
 import numpy as np
 from shapely import wkb
 from shapely.geometry import Polygon,LineString,Point
 from nexus_3dgs.bundle import decode,FEATURE_MEANS,FEATURE_STD
 protocol=json.loads((BASE/'protocol.json').read_text());info=copy.deepcopy(next(c for c in protocol['cases'] if c['kind']=='brake'))
 OUT.mkdir(exist_ok=False);case=OUT/'case_brake';shutil.copytree(BASE/info['case'],case);info['previous_case']=str(BASE/info['case']);info['case']='case_brake';t=np.arange(17)*.5
 with np.load(case/'conditions_normal.npz') as d:values=d['values'].copy();keep=d['keep'];valid=d['valid']
 A=info['roles']['A'];given=decode(values[A]);aa=given[4].copy();va=info['observed_speeds_mps']['A'];start=5.;decel=6.5
 with np.load(case/'map/routes.npz') as d:route=LineString(d[f'route_{A}'])
 station=route.project(Point(aa[:2]));correction=aa[:2]-np.array(route.interpolate(station).coords[0]);tau=np.maximum(t-start,0);u=np.minimum(tau,va/decel);distance=va*np.minimum(t,start)+va*u-.5*decel*u*u
 for j,x in enumerate(distance[1:],5):
  s=station+x;given[j,:2]=np.array(route.interpolate(s).coords[0])+correction;a=np.array(route.interpolate(max(0,s-.1)).coords[0]);b=np.array(route.interpolate(min(route.length,s+.1)).coords[0]);given[j,2:4]=(b-a)/max(np.linalg.norm(b-a),1e-8)
 velocity=np.diff(given[4:,:2],axis=0)/.5;speed=np.linalg.norm(velocity,axis=-1);angle=np.arctan2(velocity[:,1],velocity[:,0]);moving=speed>.05;heads=given[5:,2:4].copy();heads[moving]=np.c_[np.cos(angle[moving]),np.sin(angle[moving])];given[5:,2:4]=heads;given[5:,4:6]=velocity
 values[A,5:]=((given[5:]-FEATURE_MEANS)/(2*FEATURE_STD)).astype(values.dtype)
 W=np.array(info['world_from_nexus']);road=wkb.loads((case/'reference/inputs/drivable.wkb').read_bytes()).buffer(.02);dense=np.arange(81)*.1;angles=np.interp(dense,np.arange(-2,8.01,.5),np.unwrap(np.arctan2(given[:,3],given[:,2])));xy=np.c_[np.interp(dense,np.arange(-2,8.01,.5),given[:,0]),np.interp(dense,np.arange(-2,8.01,.5),given[:,1])];off=0
 for q,a in zip(xy,angles):
  c,s=np.cos(a),np.sin(a);p=np.array([[1,1],[1,-1],[-1,-1],[-1,1]])*aa[6:8]/2@np.array([[c,s],[-s,c]])+q;off+=not road.covers(Polygon(p@W[:2,:2].T+W[:2,3]))
 acc=float(np.linalg.norm(np.diff(given[:,:2],n=2,axis=0)/.25,axis=-1)[3:].max());assert off==0 and acc<=9,(off,acc)
 np.savez_compressed(case/'conditions_event.npz',values=values,keep=keep,valid=valid);info['parameters'].update(brake_start_s=start,timing_revision_reason='show longer normal following and avoid emphasizing the v1 finite-horizon end acceleration; no E future prescribed')
 info['input_precheck']['event'].update(A_offroad_samples=off,A_max_native_acceleration_mps2=acc);(case/'preparation.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')
 protocol.update(status='prepared',cases=[info],parent_protocol=str(BASE/'protocol.json'),revision_reason='later brake event for a clearer normal-following segment; v1 preserved',attempts=[]);(OUT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n');print('BRAKE_V2_PREPARED',start,'offroad',off,'acc',acc,flush=True)
if __name__=='__main__':main()
