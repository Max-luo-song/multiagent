"""Feasible lane-following initialization, never future actor conditioning."""
import numpy as np
import torch

from .omega_cruise_corridor import CorridorCruiseConfig, CorridorCruiseRefiner


class InitializedCruiseRefiner(CorridorCruiseRefiner):
    def lane_following_controls(self):
        """Pure-pursuit warm start from observed pose/speed and spatial map.

        It only initializes free acceleration/curvature variables. Normal and
        event have identical warm starts when observed histories are identical.
        The optimizer can change every future control after initialization.
        """
        initial=self.initial.detach().cpu().numpy()
        initial_speed=self.initial_speed.detach().cpu().numpy()
        controlled=self.controlled.detach().cpu().numpy()
        horizon=self.scene.shape[2]-5
        controls=np.zeros((len(initial),horizon,2),np.float32)
        for actor,(start,delta) in self.routes.items():
            if not controlled[actor]:continue
            aa=start.detach().cpu().numpy();dd=delta.detach().cpu().numpy()
            lengths=np.linalg.norm(dd,axis=-1)
            stations=np.r_[0,np.cumsum(lengths)]
            points=np.concatenate([aa,aa[-1:]+dd[-1:]],axis=0)
            xy=initial[actor,:2].copy()
            yaw=float(np.arctan2(initial[actor,3],initial[actor,2]))
            speed=float(initial_speed[actor]);previous=0.
            for t in range(horizon):
                q=xy-aa;u=np.clip((q*dd).sum(-1)/np.maximum((dd*dd).sum(-1),1e-8),0,1)
                nearest=int(np.argmin(((q-u[:,None]*dd)**2).sum(-1)))
                station=stations[nearest]+u[nearest]*lengths[nearest]
                lookahead=max(4.,speed*.8)
                target_station=min(station+lookahead,stations[-1])
                target=np.array([np.interp(target_station,stations,points[:,axis]) for axis in (0,1)])
                relative=target-xy
                lateral=float(relative@np.array([-np.sin(yaw),np.cos(yaw)]))
                curvature=2*lateral/max(float(relative@relative),1.)
                limit=min(self.config.curvature_limit_per_m,
                          self.config.lateral_acceleration_limit_mps2/max(speed*speed,1.))
                curvature=float(np.clip(curvature,-limit,limit))
                step=self.config.curvature_slew_limit_per_ms*.5
                curvature=float(np.clip(curvature,previous-step,previous+step));previous=curvature
                controls[actor,t,1]=curvature/.02
                for _ in range(5):
                    angle=speed*curvature*.1
                    xy+=speed*.1*np.array([np.cos(yaw+angle/2),np.sin(yaw+angle/2)])
                    yaw+=angle
        result=self.scene.new_tensor(controls)
        self.project_controls(result)
        return result

    def __call__(self,anchor,coefficient,variance,constrain):
        if self.previous_controls is None and self.calls+1>=self.config.start_call:
            self.previous_controls=self.lane_following_controls()
        return super().__call__(anchor,coefficient,variance,constrain)
