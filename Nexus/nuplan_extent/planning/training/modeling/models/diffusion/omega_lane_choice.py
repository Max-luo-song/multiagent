"""Opt-in map lane alternatives during Nexus denoising; no C future conditions.

Identical maneuver seeds and objective are used with/without an event. This is
an explicit added tactical search, not a claim of native learned lane choice.
"""
from dataclasses import dataclass
import numpy as np
import torch
from .omega_cruise_initialized import CorridorCruiseConfig,InitializedCruiseRefiner

@dataclass
class LaneChoiceConfig(CorridorCruiseConfig):
    choice_iterations: int = 100
    alternative_center_weight: float = 4.
    lane_change_cost: float = 25.
    maneuver_body_weight: float = 10000.

class LaneChoiceRefiner(InitializedCruiseRefiner):
    def __init__(self,*args,choice_path,**kwargs):
        super().__init__(*args,**kwargs)
        with np.load(choice_path) as z:
            self.choice_slot=int(z['slot']);self.target_points=z['target_route'].copy()
            self.departure_points=z['source_route'].copy()
            self.change_windows=z['change_windows'].copy()
        with np.load(str(choice_path).replace('lane_choices.npz','lane_modes_sdf.npz')) as z:
            self.mode_sdf=self.scene.new_tensor(z['sdf'])
            self.mode_origin=self.scene.new_tensor(z['origin']);self.mode_resolution=float(z['resolution'])
        self.mode='keep'
        self.choice_actor=int(torch.nonzero(self.indices==self.choice_slot).flatten().item())
        assert bool(self.controlled[self.choice_actor])
        for layer,actor in enumerate(self.lane_actors):
            if actor==self.choice_actor:self.lane_risk_relaxation[layer]=1.
        self.choice_routes=[]
        for p in (self.departure_points,self.target_points):
            pp=self.scene.new_tensor(p);self.choice_routes.append((pp[:-1],pp[1:]-pp[:-1]))
        # Remove the old single-lane route attraction/yielding for C only.
        # Its body still uses the source+adjacent SDK polygon union SDF;
        # collision, dynamic following, road and actuator costs remain active.
        self.routes.pop(self.choice_actor,None)
        self.choice_records=[]

    def common_objective(self,controls,base):
        total,state,parts=super().control_objective(controls,base)
        xy=(state*self.scale+self.mean)[0,self.choice_actor,5:,:2]
        distances=[]
        for start,delta in self.choice_routes:
            q=xy[:,None]-start
            u=((q*delta).sum(-1)/delta.square().sum(-1).clamp_min(1e-8)).clamp(0,1)
            distances.append((q-u[...,None]*delta).square().sum(-1).amin(-1))
        d=torch.stack(distances);mask=self.valid[0,self.choice_actor,5:]
        lane=(d.amin(0)*mask).sum()/mask.sum().clamp_min(1)
        # Smooth maneuver effort, zero at the departure centerline and bounded
        # near one at the adjacent centerline. No event flag enters this term.
        change=(d[0]/(d[0]+d[1]).clamp_min(.01))[-4:].mean()
        parts.update(alternative_lane_center=lane,lane_change_preference=change)
        return total+self.config.alternative_center_weight*lane+self.config.lane_change_cost*change,state,parts

    def control_objective(self,controls,base):
        total,state,parts=self.common_objective(controls,base)
        physical=state*self.scale+self.mean;actor=self.choice_actor
        xy=physical[0,actor,5:,:2];h=physical[0,actor,5:,2:4]
        if self.choice_slot==0:xy=xy+1.461*h
        side=torch.stack([-h[:,1],h[:,0]],-1);size=self.sizes[0,actor]
        signs=state.new_tensor([[1,1],[1,-1],[-1,-1],[-1,1]])
        corners=xy[:,None]+h[:,None]*size[0]/2*signs[None,:,0:1]+side[:,None]*size[1]/2*signs[None,:,1:2]
        H,W=self.mode_sdf.shape[-2:];grid=(corners-self.mode_origin)/(self.mode_resolution*state.new_tensor([W-1,H-1]))*2-1
        layer=0 if self.mode=='keep' else 1
        distance=torch.nn.functional.grid_sample(self.mode_sdf[layer][None,None],grid.reshape(1,1,-1,2),mode='bilinear',padding_mode='border',align_corners=True).reshape(16,4).amin(-1)
        # A candidate denotes a spatial maneuver, not a prescribed C path:
        # keep candidates stay in the source lane; change candidates finish in
        # the adjacent lane in the second half. Speed, timing and controls vary.
        mask=self.valid[0,actor,5:].clone()
        if self.mode!='keep':mask[:8]=False
        body=(torch.relu(.15-distance).square()*mask).sum()/mask.sum().clamp_min(1)
        parts['maneuver_body']=body
        return total+self.config.maneuver_body_weight*body,state,parts

    def maneuver_seeds(self):
        base=self.lane_following_controls()
        # C was removed from self.routes: reconstruct its keep-lane warm start
        # through exactly the same map/observation pure-pursuit integration.
        actor=self.choice_actor;self.routes[actor]=self.choice_routes[0]
        base=self.lane_following_controls();self.routes.pop(actor)
        seeds=[('keep',base)]
        initial=self.initial[actor].detach().cpu().numpy();speed=float(self.initial_speed[actor])
        source=self.departure_points;target=self.target_points
        def project(points,p):
            a=points[:-1];d=np.diff(points,axis=0);length=np.linalg.norm(d,axis=-1)
            u=np.clip(((p-a)*d).sum(-1)/np.maximum(length**2,1e-8),0,1)
            j=int(np.argmin(((p-a-u[:,None]*d)**2).sum(-1)))
            return float(np.r_[0,length.cumsum()][j]+u[j]*length[j])
        stations=np.r_[0,np.linalg.norm(np.diff(source,axis=0),axis=-1).cumsum()]
        origin=project(source,initial[:2])
        # Construct map-only lateral transition paths at the observed speed.
        # These are optimizer initializations, not waypoint conditions/costs.
        target_st=np.r_[0,np.linalg.norm(np.diff(target,axis=0),axis=-1).cumsum()]
        for begin,end in self.change_windows:
            tt=np.arange(0,8.01,.1);st=np.minimum(origin+speed*tt,stations[-1])
            points=np.c_[[np.interp(st,stations,source[:,k]) for k in (0,1)]].T
            closest=[]
            for p in points:
                s=project(target,p);closest.append([np.interp(s,target_st,target[:,k]) for k in (0,1)])
            u=np.clip((tt-begin)/(end-begin),0,1);w=10*u**3-15*u**4+6*u**5
            path=points*(1-w[:,None])+np.array(closest)*w[:,None]
            path_st=np.r_[0,np.linalg.norm(np.diff(path,axis=0),axis=-1).cumsum()]
            xy=initial[:2].copy();yaw=float(np.arctan2(initial[3],initial[2]));previous=0.;controls=base.clone()
            for t in range(16):
                st=project(path,xy);look=min(st+max(4.,speed*.8),path_st[-1])
                goal=np.array([np.interp(look,path_st,path[:,k]) for k in (0,1)])
                rel=goal-xy;lat=rel@np.array([-np.sin(yaw),np.cos(yaw)])
                k=2*lat/max(float(rel@rel),1.);limit=min(self.config.curvature_limit_per_m,self.config.lateral_acceleration_limit_mps2/max(speed**2,1.))
                k=float(np.clip(k,-limit,limit));step=self.config.curvature_slew_limit_per_ms*.5;k=float(np.clip(k,previous-step,previous+step));previous=k
                controls[actor,t,1]=k/.02
                for _ in range(5):
                    angle=speed*k*.1;xy+=speed*.1*np.array([np.cos(yaw+angle/2),np.sin(yaw+angle/2)]);yaw+=angle
            self.project_controls(controls);seeds.append((f'adjacent_{begin:g}_{end:g}',controls))
        return seeds

    def __call__(self,anchor,coefficient,variance,constrain):
        if self.previous_controls is None and self.calls+1>=self.config.start_call:
            base=torch.where(self.keep,self.scene,anchor[:,self.indices]).detach()
            with torch.inference_mode(False),torch.enable_grad():
                base=base.clone();best=None;records=[];seeds=self.maneuver_seeds()
                # First form a shared traffic response, then compare C modes
                # against identical provisional trajectories for every other
                # vehicle. This prevents unrelated actors' optimizer minima
                # from deciding whether C changes lanes.
                self.mode='keep';warm=seeds[0][1].clone().requires_grad_()
                opt=torch.optim.Adam([warm],lr=self.config.control_learning_rate)
                for _ in range(self.config.choice_iterations):
                    opt.zero_grad(set_to_none=True);loss,_,_=self.control_objective(warm,base);loss.backward();opt.step();self.project_controls(warm)
                shared=warm.detach().clone()
                for label,seed in seeds:
                    self.mode='keep' if label=='keep' else 'change'
                    initial=shared.clone()
                    if label!='keep':initial[self.choice_actor]=seed[self.choice_actor]
                    u=initial.requires_grad_();optimizer=torch.optim.Adam([u],lr=self.config.control_learning_rate)
                    best_cost=float('inf');best_u=u.detach().clone()
                    for _ in range(self.config.choice_iterations):
                        optimizer.zero_grad(set_to_none=True);loss,_,_=self.control_objective(u,base)
                        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite tactical objective')
                        if float(loss)<best_cost:best_cost=float(loss);best_u=u.detach().clone()
                        loss.backward();u.grad[:self.choice_actor]=0;u.grad[self.choice_actor+1:]=0
                        optimizer.step();self.project_controls(u)
                    loss,_,_=self.control_objective(u,base)
                    if float(loss)<best_cost:best_cost=float(loss);best_u=u.detach().clone()
                    common,_,parts=self.common_objective(best_u,base);score=float(common)
                    _,_,mode_parts=self.control_objective(best_u,base);mode_violation=float(mode_parts['maneuver_body'])
                    # Reject a branch that did not realize its spatial mode.
                    feasible=mode_violation<.0025
                    records.append({'candidate':label,'common_objective':score,'mode_body_cost':mode_violation,'realized_maneuver':feasible})
                    if feasible and (best is None or score<best[0]):best=(score,label,best_u)
                if best is None:
                    best=(float(self.common_objective(shared,base)[0]),'keep',shared)
                    records.append({'fallback':'no realized candidate; retain shared keep proposal; independent output gate still required'})
                self.mode='keep' if best[1]=='keep' else 'change'
                self.previous_controls=best[2];self.choice_records.append({'call':self.calls+1,'selected':best[1],'candidates':records,'scope':'same shared traffic and common costs; map maneuver priors only; no fixed C future'})
        return super().__call__(anchor,coefficient,variance,constrain)
