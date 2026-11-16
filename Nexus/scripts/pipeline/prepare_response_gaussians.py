"""Transfer frozen response trajectories to the available Gaussian background."""
import json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/nexus_response_gaussians_20261110';SRC=ROOT/'artifacts/nexus_hazard_response_v2_20261110';BG=ROOT/'artifacts/nexus_3dgs/asset_insert_20261028'
sys.path[:0]=[str(ROOT),str(ROOT/'scripts/pipeline'),str(ROOT/'third_party/nuplan-devkit')]
def main():
 import numpy as np,torch
 from shapely.geometry import LineString,Point
 from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
 from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
 from nexus_3dgs.bundle import decode
 from audit_nexus_omega_v2 import sample_positions
 OUT.mkdir(exist_ok=True);p=json.loads((BG/'insertion_protocol.json').read_text());api=get_maps_api(str(ROOT.parent/'data/nuplan/maps'),'nuplan-maps-v1.0','sg-one-north');lane=api.get_map_object(p['B_lane'],L.LANE);line=lane.baseline_path.linestring;E=np.array(p['background_camera_anchor_ego']);base=line.project(Point(E[:2,3]))+25;run=json.loads((SRC/'response_seed0/run.json').read_text());cases=[]
 def frame(curve,s):
  q=np.array(curve.interpolate(s).coords[0]);a=np.array(curve.interpolate(max(0,s-.1)).coords[0]);b=np.array(curve.interpolate(min(curve.length,s+.1)).coords[0]);h=(b-a)/np.linalg.norm(b-a);return q,h
 for info in run['protocol']['cases']:
  case=SRC/info['case'];file=SRC/'response_seed0'/info['kind']/'event/prediction.pt';payload=torch.load(file,weights_only=False);state=decode(payload['sampled_tensor'][0].numpy());active=info['active_slots'];s=state[active];times=np.arange(81)*.1;xy,yaw,seen=sample_positions(s,payload['valid_mask'][0].numpy().astype(bool)[active],times,np);xy[0]+=1.461*np.c_[np.cos(yaw[0]),np.sin(yaw[0])]
  with np.load(case/'map/routes.npz') as z:source=LineString(z['route_0'])
  origin=source.project(Point(state[0,4,:2]));placed=np.zeros_like(xy);heads=np.zeros_like(yaw);stations=np.zeros_like(yaw);laterals=np.zeros_like(yaw)
  for i in range(len(active)):
   for k in range(81):
    st=source.project(Point(xy[i,k]));q,h=frame(source,st);lat=float((xy[i,k]-q)@np.array([-h[1],h[0]]));station=base+st-origin
    assert 0<=station<=line.length,(info['kind'],station,line.length)
    target,hh=frame(line,station);placed[i,k]=target+lat*np.array([-hh[1],hh[0]]);heads[i,k]=np.arctan2(hh[1],hh[0])+np.arctan2(np.sin(yaw[i,k]-np.arctan2(h[1],h[0])),np.cos(yaw[i,k]-np.arctan2(h[1],h[0])));stations[i,k]=station;laterals[i,k]=lat
  np.savez_compressed(OUT/f"{info['kind']}.npz",times=times,xy=placed,yaw=heads,station=stations,lateral=laterals,sizes=s[:,4,6:8])
  cases.append({'kind':info['kind'],'title':info['title'],'roles':list(info['roles']),'slots':active,'prediction':str(file),'prediction_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'original_scenario':info['scenario']['index'],'scope':'synthetic road replay on scene121, not original scene reconstruction','oncoming_topology_limit':info['kind']=='oncoming'})
 out={'background':p,'checkpoint':json.loads((BG/'render_final/run.json').read_text())['checkpoint'],'ground_profile':json.loads((BG/'vehicle_ground_profile.json').read_text()),'base_station':base,'cases':cases,'mapping':'same longitudinal arc progress and metric lateral offsets, background road curvature substituted; timing unchanged; no new Nexus generation','scope':'counterfactual asset insertion; scene121 background has same-direction lanes, so oncoming clip represents wrong-way intrusion, not a true reconstructed two-way centerline'};(OUT/'protocol.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print('GAUSSIAN_PLACEMENT_READY',len(cases),base,line.length,flush=True)
if __name__=='__main__':main()
