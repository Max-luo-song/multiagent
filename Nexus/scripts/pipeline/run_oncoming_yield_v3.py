"""Add disclosed physical yielding for a known oncoming lane intrusion.

No learned weights or original refiner source files are modified. The existing
given-actor corridor activation decides when yielding applies; the adapter
adds initially-ahead, opposite-heading given actors to its eligibility mask.
"""
import hashlib,json,shutil
from pathlib import Path
import run_four_hazards as runner
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'artifacts/nexus_four_hazards_20261110';OUT=BASE/'oncoming_yield_v3'

def main():
 OUT.mkdir(exist_ok=False);parent=BASE/'oncoming_timing_v2';protocol=json.loads((parent/'protocol.json').read_text());shutil.copytree(parent/'case_oncoming',OUT/'case_oncoming')
 protocol.update(parent_protocol=str(parent/'protocol.json'),revision_reason='timing v2 produced7 overlap samples; same inputs, extend existing given-actor lane-yield eligibility to opposing actors initially ahead',model='Nexus + frozen V4 refiner + disclosed opposing-intrusion yield adapter; no weight training',runtime_refiner_adapter={'path':str(Path(__file__)),'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'change':'initial_ahead eligibility includes initially ahead opposite-heading GIVEN actors, with original time-varying lane intrusion activation; applies identically to normal/event'})
 (OUT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
 import torch
 import nuplan_extent.planning.training.modeling.models.diffusion.omega_cruise_initialized as module
 base=module.InitializedCruiseRefiner
 class OpposingIntrusionRefiner(base):
  def __init__(self,*args,**kwargs):
   super().__init__(*args,**kwargs)
   center=self.initial[:,:2].clone();center[0]+=1.461*self.initial_heading[0]
   delta=center[None]-center[:,None];along=(delta*self.initial_heading[:,None]).sum(-1);alignment=(self.initial_heading[:,None]*self.initial_heading[None]).sum(-1)
   opposing=(along>0)&(alignment<-.7)&self.given_future[None]
   self.initial_ahead|=opposing
 module.InitializedCruiseRefiner=OpposingIntrusionRefiner;runner.OUT=OUT;runner.main()
 runfile=OUT/'frozen_v4_seed0/run.json';run=json.loads(runfile.read_text());run['source_sha256'][str(Path(__file__))]=protocol['runtime_refiner_adapter']['sha256'];runfile.write_text(json.dumps(run,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
