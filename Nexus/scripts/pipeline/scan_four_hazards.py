"""History-only geometric candidates for four requested edited hazards."""
import json, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/nexus_four_hazards_20261110'
SOURCE=ROOT/'artifacts/nuplan_evaluation/independent256_v1'
sys.path[:0]=[str(ROOT),str(ROOT/'scripts/pipeline'),str(ROOT/'third_party/nuplan-devkit'),str(ROOT/'third_party/alf')]
os.environ.update(MPLBACKEND='Agg',NUPLAN_DATA_STORE='local')

def main():
 import numpy as np, torch
 from nexus_3dgs.bundle import decode
 torch.set_num_threads(2)
 OUT.mkdir(exist_ok=True)
 plan=json.loads((SOURCE/'frozen_plan.json').read_text())['scenarios']
 candidates={x:[] for x in ['brake','reveal','rear','oncoming']}
 for row in plan:
  file=SOURCE/'raw_features'/f"{row['index']:03d}_{row['token']}.pt"
  d=torch.load(file,map_location='cpu',weights_only=False)
  s=decode(d['tensor'].numpy()[...,:8]);v=d['validity'].numpy().astype(bool)
  if not v[0].all():continue
  h=s[0,4,2:4];h=h/max(np.linalg.norm(h),1e-8);side=np.array([-h[1],h[0]])
  ve=np.linalg.norm((s[0,4,:2]-s[0,3,:2])/.5)
  if not 3<=ve<=20:continue
  center=s[0,4,:2]+1.461*h
  active=np.flatnonzero(v[:,:5].all(-1));front=[]
  for a in active:
   if a==0:continue
   q=s[a,4,:2]-center;along=float(q@h);lat=float(q@side);dot=float(s[a,4,2:4]@h)
   va=float(np.linalg.norm((s[a,4,:2]-s[a,3,:2])/.5));length=(s[0,4,6]+s[a,4,6])/2
   item={'scenario':row,'A':int(a),'E_speed':float(ve),'A_speed':va,'along_m':along,'lateral_m':lat,'heading_dot':dot,'net_gap_m':abs(along)-float(length)}
   if 10<along<45 and abs(lat)<.8 and dot>.98 and 6<va<20 and abs(va-ve)<2.5 and 7<along-length<26:
    others=[int(b) for b in active if b not in [0,a] and 2<((s[b,4,:2]-s[a,4,:2])@h)<50 and abs((s[b,4,:2]-s[a,4,:2])@side)<1.8]
    rec=dict(item,score=abs(float(along-length)-12)+2*abs(va-ve)+8*len(others),ahead_A_obstacle_slots=others)
    candidates['brake'].append(rec);front.append(item)
   if -80<along<-10 and abs(lat)<1.2 and dot>.94 and va>ve+3 and va<30 and 1.5<(abs(along)-length)/(va-ve)<6:
    candidates['rear'].append(dict(item,ttc_s=float((abs(along)-length)/(va-ve)),score=abs((abs(along)-length)/(va-ve)-2.5)+abs(lat)))
   if 25<along<120 and 2.5<abs(lat)<6 and dot<-.97 and va>4 and 2.3<along/(ve+va)<5.:
    candidates['oncoming'].append(dict(item,closing_time_s=along/(ve+va),score=abs(along/(ve+va)-3.2)+abs(abs(lat)-3.5)))
  for item in front:
   a=item['A'];stationary=[]
   for b in active:
    if b in [0,a]:continue
    vb=float(np.linalg.norm((s[b,4,:2]-s[b,3,:2])/.5));q=s[b,4,:2]-s[a,4,:2]
    if vb<.4 and 12<q@h<40 and abs((s[b,4,:2]-center)@side)<.9:
     stationary.append({'slot':int(b),'ahead_A_m':float(q@h)})
   candidates['reveal'].append(dict(item,score=abs(item['net_gap_m']-14)+abs(item['A_speed']-ve)-30*bool(stationary),stationary_candidates=stationary))
 for name,items in candidates.items():
  items.sort(key=lambda x:(x['score'],x['scenario']['index'],x['A']))
  print('CANDIDATES',name,len(items),json.dumps([{k:v for k,v in x.items() if k!='scenario'}|{'index':x['scenario']['index']} for x in items[:8]]),flush=True)
 (OUT/'history_candidate_scan.json').write_text(json.dumps({'selection_uses':'observed history geometry and history existence only; E exists throughout; edited A is retained throughout synthetic future; no actor future positions','candidates':candidates},ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
