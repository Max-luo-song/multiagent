"""Explicit response intents inside Nexus sampling, never E waypoint conditions."""
import numpy as np
import torch
from nuplan_extent.planning.training.modeling.models.diffusion.omega_cruise_initialized import InitializedCruiseRefiner

class ResponseIntentRefiner(InitializedCruiseRefiner):
 def __init__(self,*args,response_intent='cruise',stop_slot=None,stop_deadline_s=5.,target_sdf_path=None,**kwargs):
  self.response_intent=response_intent;super().__init__(*args,**kwargs)
  self.stop_deadline=max(1,min(16,round(stop_deadline_s/.5)))
  if response_intent=='stop':
   self.cruise_eligible[0]=False
   lead=int(torch.nonzero(self.indices==stop_slot).flatten()[0]);last=self.physical_history[0,lead,-1];head=self.initial_heading[0];extent=(self.sizes[0,0,0]+self.sizes[0,lead,0])/2
   self.stop_target=last[:2]-(extent+1.461+3.)*head
  if response_intent=='lane_change':
   with np.load(target_sdf_path) as z:self.target_sdf=self.scene.new_tensor(z['sdf'][0]);self.target_origin=self.scene.new_tensor(z['origin']);self.target_resolution=float(z['resolution'])
 def project_controls(self,controls):
  super().project_controls(controls)
  if self.response_intent=='stop':
   # Once a stationary blocking obstacle is detected, accelerating toward it
   # is excluded; speed and braking timing remain optimization variables.
   with torch.no_grad():controls[0,:,0].clamp_(max=0.)
 def control_objective(self,controls,base):
  total,state,parts=super().control_objective(controls,base);p=state*self.scale+self.mean
  if self.response_intent=='stop':
   speed=(p[0,0,5:,4:6]*p[0,0,5:,2:4]).sum(-1);deadline=self.stop_deadline-1
   terminal=(speed[deadline:]**2).mean();gap=((p[0,0,5+deadline,:2]-self.stop_target)*self.initial_heading[0]).sum()
   position=gap.square();parts.update(stop_terminal_speed=terminal,stop_terminal_position=position)
   return total+2000*terminal+80*position,state,parts
  if self.response_intent=='lane_change':
   xy=p[0,0,5:,:2];h=p[0,0,5:,2:4];center=xy+1.461*h;side=torch.stack([-h[:,1],h[:,0]],-1);size=self.sizes[0,0];signs=state.new_tensor([[1,1],[1,-1],[-1,-1],[-1,1]])
   corners=center[:,None]+h[:,None]*size[0]/2*signs[None,:,0:1]+side[:,None]*size[1]/2*signs[None,:,1:2];H,W=self.target_sdf.shape
   grid=(corners-self.target_origin)/(self.target_resolution*state.new_tensor([W-1,H-1]))*2-1
   distance=torch.nn.functional.grid_sample(self.target_sdf[None,None],grid.reshape(1,1,-1,2),mode='bilinear',padding_mode='border',align_corners=True).reshape(16,4).amin(-1)
   body=torch.relu(.12-distance[8:]).square().mean();parts['finish_in_escape_lane']=body
   return total+20000*body,state,parts
  return total,state,parts
