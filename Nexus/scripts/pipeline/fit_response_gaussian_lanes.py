"""Fit shared background lane spacing; audit actual asset footprints first."""
import json,sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/nexus_response_gaussians_20261110'
sys.path[:0]=[str(ROOT),str(ROOT/'third_party/nuplan-devkit')]
def main():
 import numpy as np
 from shapely.geometry import Polygon
 from shapely.ops import unary_union
 from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
 from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
 from nexus_3dgs.scene_projection import pair_separations
 api=get_maps_api(str(ROOT.parent/'data/nuplan/maps'),'nuplan-maps-v1.0','sg-one-north');line=api.get_map_object('52395',L.LANE).baseline_path.linestring;road=unary_union([api.get_map_object(k,L.LANE).polygon for k in ['52395','52553','52012']]).buffer(.02);run=json.loads((OUT/'render_run.json').read_text());inputs={};best=None
 for c in run['cases']:
  with np.load(OUT/f"{c['kind']}.npz") as z:data={k:z[k] for k in z.files}
  inputs[c['kind']]=(data,np.array([v['render_extent_lwh'][:2] for v in c['assets'].values()]))
 for factor,offset in [(1.,-.4),(.95,-.3),(.9,-.2),(.9,-.4),(.85,-.3),(.85,-.5)]:
  results={};total=0
  for kind,(data,sizes) in inputs.items():
   xy=np.zeros_like(data['xy']);yaw=np.zeros_like(data['yaw'])
   for i in range(len(sizes)):
    for k in range(81):
     st=data['station'][i,k];q=np.array(line.interpolate(st).coords[0]);a=np.array(line.interpolate(max(0,st-.1)).coords[0]);b=np.array(line.interpolate(min(line.length,st+.1)).coords[0]);h=(b-a)/np.linalg.norm(b-a);ang=np.arctan2(h[1],h[0]);rel=data['yaw'][i,k]-ang;xy[i,k]=q+(data['lateral'][i,k]*factor+offset)*np.array([-h[1],h[0]]);yaw[i,k]=ang+np.arctan2(factor*np.sin(rel),np.cos(rel))
   boxes=np.concatenate([xy.transpose(1,0,2),yaw.T[...,None]],-1);sep,vp=pair_separations(boxes,np.ones(boxes.shape[:2],bool),sizes);hits=int(((sep<0)&vp).sum());off=0
   for t in range(81):
    for i in range(len(sizes)):
     a=yaw[i,t];rot=np.array([[np.cos(a),np.sin(a)],[-np.sin(a),np.cos(a)]]);poly=Polygon(np.array([[1,1],[1,-1],[-1,-1],[-1,1]])*sizes[i]/2@rot+xy[i,t]);off+=not road.covers(poly)
   results[kind]={'xy':xy,'yaw':yaw,'overlap_samples':hits,'offroad_samples':off};total+=hits+off
  print('FIT',factor,offset,total,flush=True)
  if total==0:best=(factor,offset,results);break
 assert best is not None,'No physically checked background placement'
 factor,offset,results=best
 for kind,(data,_) in inputs.items():
  shutil.copy2(OUT/f'{kind}.npz',OUT/f'{kind}.unfitted.npz');data['xy']=results[kind]['xy'];data['yaw']=results[kind]['yaw'];data['lateral']=data['lateral']*factor+offset;np.savez_compressed(OUT/f'{kind}.npz',**data)
 p=json.loads((OUT/'protocol.json').read_text());p['background_lane_fit']={'lateral_scale':factor,'lateral_offset_m':offset,'all_4_cases_no_render_asset_overlap_or_outside_3lane_background_at_81_samples':True};p['mapping']='longitudinal arc progress and time unchanged; shared background lateral spacing fit and offset; heading transformed consistently; original Nexus predictions untouched';(OUT/'protocol.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n')
 (OUT/'lane_fit.json').write_text(json.dumps({'factor':factor,'offset':offset,'cases':{k:{a:b for a,b in v.items() if a not in ['xy','yaw']} for k,v in results.items()}},indent=2)+'\n')
 (OUT/'render').rename(OUT/'render_unfitted');(OUT/'render_run.json').rename(OUT/'render_unfitted_run.json')
if __name__=='__main__':main()
