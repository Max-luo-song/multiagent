"""Disclosed physical stopping-gap term; preserve free E future and model weights."""
import hashlib,json,shutil
from pathlib import Path
import run_four_hazards as runner
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'artifacts/nexus_four_hazards_20261110';OUT=BASE/'brake_stopping_v3'
def main():
 OUT.mkdir(exist_ok=False);parent=BASE/'brake_timing_v2';protocol=json.loads((parent/'protocol.json').read_text());shutil.copytree(parent/'case_brake',OUT/'case_brake')
 protocol.update(parent_protocol=str(parent/'protocol.json'),revision_reason='timing v2 leaves small stopping distance at forecast end; same inputs, add physical relative stopping gap during sampling',model='Nexus + frozen V4 refiner + disclosed relative stopping-gap adapter; no weight training',runtime_refiner_adapter={'path':str(Path(__file__)),'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'change':'same-direction given lead: clearance>=0.5+relu(vE²/(2*4.5)-vA²/(2*6.5))+0.3*vE, soft weight3000; native future samples; applies identically to normal/event; not a continuous-time guarantee'})
 (OUT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
 import torch
 import nuplan_extent.planning.training.modeling.models.diffusion.omega_cruise_initialized as module
 base=module.InitializedCruiseRefiner
 class StoppingGapRefiner(base):
  def control_objective(self,controls,anchor):
   total,state,parts=super().control_objective(controls,anchor);p=state*self.scale+self.mean;center=p[0,:,5:,:2].clone();h=p[0,:,5:,2:4];h=h/h.norm(dim=-1,keepdim=True).clamp_min(1e-8);center[0]+=1.461*h[0];cost=state.sum()*0
   v=p[0,:,5:,4:6].norm(dim=-1)
   for other in torch.nonzero(self.given_future).flatten().tolist():
    if not bool(self.initial_ahead[0,other]):continue
    delta=center[other]-center[0];along=(delta*h[0]).sum(-1);lateral=(delta*torch.stack([-h[0,:,1],h[0,:,0]],-1)).sum(-1).abs();width=(self.sizes[0,0,1]+self.sizes[0,other,1])/2+.3;gate=((width+.4-lateral)/.4).clamp(0,1)
    aligned=(h[0]*h[other]).sum(-1)>.7;extent=(self.sizes[0,0,0]+self.sizes[0,other,0])/2
    required=.5+torch.relu(v[0].square()/(2*self.config.braking_limit_mps2)-v[other].square()/(2*6.5))+.3*v[0]
    cost+=(torch.relu(required-(along-extent)).square()*gate*aligned*self.valid[0,0,5:]*self.valid[0,other,5:]).mean()
   parts['relative_stopping_gap']=cost;return total+3000*cost,state,parts
 module.InitializedCruiseRefiner=StoppingGapRefiner;runner.OUT=OUT;runner.main()
 runfile=OUT/'frozen_v4_seed0/run.json';run=json.loads(runfile.read_text());run['source_sha256'][str(Path(__file__))]=protocol['runtime_refiner_adapter']['sha256'];runfile.write_text(json.dumps(run,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
