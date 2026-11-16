"""Rank A-cut-in / B / following-C triples using observed history only."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];SOURCE=ROOT/'artifacts/nuplan_evaluation/independent256_v1';OUT=ROOT/'artifacts/nexus_chain_20261027'
sys.path.insert(0,str(ROOT))
def main():
 import numpy as np,torch
 from nexus_3dgs.bundle import decode
 torch.set_num_threads(2);OUT.mkdir(exist_ok=False)
 plan=json.loads((SOURCE/'frozen_plan.json').read_text())['scenarios'];ranked=[]
 for row in plan:
  raw=torch.load(SOURCE/'raw_features'/f'{row["index"]:03d}_{row["token"]}.pt',map_location='cpu',weights_only=False)
  s=decode(raw['tensor'].numpy()[...,:8]);valid=raw['validity'].numpy().astype(bool)
  eligible=valid.all(-1)&(s[:,4,6]>3)&(s[:,4,7]>1.4)
  vel=(s[:,4,:2]-s[:,3,:2])/.5;speed=np.linalg.norm(vel,axis=-1);eligible &=speed>2
  for B in np.flatnonzero(eligible):
   h=s[B,4,2:4];h=h/max(np.linalg.norm(h),1e-8);side=np.array([-h[1],h[0]])
   xy=s[:,4,:2]-s[B,4,:2];x=xy@h;y=xy@side;aligned=(s[:,4,2:4]@h)>.94
   ahead=eligible&aligned&(x>5)&(x<24)&(abs(y)>2.4)&(abs(y)<4.5)
   behind=eligible&aligned&(x<-5)&(x>-28)&(abs(y)<1.15)&(speed>=speed[B]-.8)
   for C in np.flatnonzero(behind):
    # B is a rear axle only when it is the real ego slot0.
    gap=-x[C]-(s[B,4,6]+s[C,4,6])/2+(1.461 if B==0 else 0)-(1.461 if C==0 else 0)
    headway=gap/max(speed[C],.1)
    if gap<2 or headway>1.9:continue
    for A in np.flatnonzero(ahead):
     if C==A:continue
     # Do not prescribe the real ego as the event vehicle.
     if A==0:continue
     score=40-abs(x[A]-10)-abs(headway-1)*7-abs(speed[B]-speed[C])*2+min(speed[B],12)+(5 if B==0 else 0)
     ranked.append({'scenario':row,'roles':{'A':int(A),'B':int(B),'C':int(C)},'score':float(score),'A_relative_xy_m':[float(x[A]),float(y[A])],'C_relative_xy_m':[float(x[C]),float(y[C])],'BC_initial_body_gap_m':float(gap),'C_time_headway_s':float(headway),'speeds_mps':{r:float(speed[i]) for r,i in [('A',A),('B',B),('C',C)]}})
 ranked.sort(key=lambda x:(-x['score'],x['scenario']['index'],x['roles']['B'],x['roles']['A'],x['roles']['C']))
 result={'status':'screened','scanned_scenes':len(plan),'selection_uses_future_XY':False,'future_information_used':'full recorded actor existence only','rules':'parallel A adjacent-front; C same-lane behind B, similar/faster speed, 2..28m body gap, <=1.9s headway; all vehicle-shaped full-presence actors','ranked':ranked}
 (OUT/'selection.json').write_text(json.dumps(result,indent=2)+'\n')
 for r in ranked[:18]:print(r['scenario']['index'],r['scenario']['map_name'],r['roles'],'gap',round(r['BC_initial_body_gap_m'],2),'headway',round(r['C_time_headway_s'],2),'speeds',r['speeds_mps'],'Axy',r['A_relative_xy_m'],flush=True)
 print('TOTAL_TRIPLES',len(ranked),'SCENES',len({r['scenario']['index'] for r in ranked}),flush=True)
if __name__=='__main__':main()
