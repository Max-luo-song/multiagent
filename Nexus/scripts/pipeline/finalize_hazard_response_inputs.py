"""Complete optional target-lane grids after the road-layout preparation.

Existing grids are retained byte for byte, allowing validation after sampling.
"""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/nexus_hazard_response_v2_20261110'
sys.path[:0]=[str(ROOT),str(ROOT/'scripts/pipeline')]
def main():
 import numpy as np
 from shapely import wkb
 from prepare_nexus_multiscene_validation import raster
 p=json.loads((OUT/'protocol.json').read_text());changed=False
 for c in p['cases']:
  case=OUT/c['case']
  if c['kind']=='oncoming' and 'initial_rear_ttc_s' in c['parameters']:
   c['parameters'].pop('initial_rear_ttc_s');(case/'preparation.json').write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n');changed=True
  if c['response_intent']!='adjacent_lane_change':continue
  file=case/'map/lane_corridors_target.npz'
  if not file.exists():
   corridor=wkb.loads((case/'map/target_E_corridor.wkb').read_bytes());grid=raster(corridor,np.array(c['world_from_nexus']),None,[-100.,-100.],.2,(1001,1501));np.savez_compressed(file,sdf=grid[None],slots=np.array([0]),origin=np.array([-100.,-100.]),resolution=.2)
  with np.load(file) as z:assert z['sdf'].shape==(1,1001,1501) and z['slots'].tolist()==[0]
 if changed:(OUT/'protocol.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n')
 print('RESPONSE_INPUT_COMPLETION_OK',flush=True)
if __name__=='__main__':main()
