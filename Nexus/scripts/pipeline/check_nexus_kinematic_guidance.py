"""Meaningful checks: fixed inputs, no sliding, lane retention and cut-in braking."""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    import torch
    from nuplan_extent.planning.training.modeling.models.diffusion.omega_kinematic import KinematicConfig, KinematicRefiner
    from nuplan_extent.planning.training.modeling.models.diffusion.omega_structural import sat_separation
    torch.set_num_threads(2)
    scale = np.array([104, 104, 1, 1, 4, 4, 5, 1.6], np.float32)
    mean = np.array([0, 0, 0, 0, 0, 0, 4.5, 2], np.float32)
    times = np.arange(-2, 8.01, .5)
    observations = np.zeros((3, 21, 8), np.float32)
    for i, (x, y, speed) in enumerate([(0, 0, 12), (8, 3.3, 10), (4, -3.8, 12)]):
        observations[i, :, :2] = np.c_[x+times*speed, np.full(21, y)]
        observations[i, :, 2] = 1
        observations[i, :, 4] = speed
        observations[i, :, 6:8] = [4.5, 1.8]
    keep = np.zeros_like(observations, bool)
    keep[:, :5] = True
    keep[1] = True
    valid = torch.ones(1, 3, 21)
    reports = []
    with tempfile.TemporaryDirectory(prefix='nexus_control_check_') as directory:
        route_path = Path(directory)/'routes.npz'
        points = np.linspace(-30, 150, 181)
        np.savez(route_path, route_0=np.c_[points, points*0], route_2=np.c_[points, points*0-3.8])
        for variant in ('normal', 'cutin'):
            supplied = observations.copy()
            if variant == 'cutin':
                u = np.clip((times[5:]-.5)/2, 0, 1)
                supplied[1, 5:, 1] -= 3.3*(10*u**3-15*u**4+6*u**5)
                velocity = np.diff(supplied[1, 4:, :2], axis=0)/.5
                angle = np.arctan2(velocity[:, 1], velocity[:, 0])
                supplied[1, 5:, 2:4] = np.c_[np.cos(angle), np.sin(angle)]
                supplied[1, 5:, 4:6] = velocity
            encoded = torch.from_numpy((supplied-mean)/scale)[None]
            mask = torch.from_numpy(keep)[None]
            conditions = torch.where(mask, encoded, 0)
            anchor = encoded.clone()
            # The normal neural proposal drifts despite no event. The free
            # controls should recover lane keeping without fixed waypoints.
            if variant == 'normal':
                anchor[0, 0, 5:, 1] += torch.linspace(0, 4, 16)/104
            config = KinematicConfig(start_call=1, control_iterations=220,
                                     final_control_iterations=220, following_weight=0,
                                     collision_weight=3000, road_weight=1000,
                                     acceleration_weight=100, jerk_weight=2,
                                     motion_qp=False)
            refiner = KinematicRefiner(config, conditions, valid, mask, route_path=route_path)
            result = refiner(anchor, torch.ones_like(anchor), torch.ones_like(anchor), {})
            assert torch.equal(result[mask], conditions[mask])
            physical = result*torch.tensor(scale)+torch.tensor(mean)
            h = physical[0, :, 5:, 2:4]
            v = physical[0, :, 5:, 4:6]
            assert ((v[..., 0]*h[..., 1]-v[..., 1]*h[..., 0]).abs()<1e-4).all()
            assert (v[..., 0]*h[..., 0]+v[..., 1]*h[..., 1] >= -1e-5).all()
            xy, head, seen = refiner.interpolate(physical, valid.bool())
            centers = xy.clone();centers[:, 0] += 1.461*head[:, 0]
            gap = sat_separation(centers[:, 0], head[:, 0], refiner.sizes[:, 0, None],
                                 centers[:, 1], head[:, 1], refiner.sizes[:, 1, None])
            assert float(gap.min()) >= 0, (variant, float(gap.min()))
            third_gap = sat_separation(centers[:, 0], head[:, 0], refiner.sizes[:, 0, None],
                                       centers[:, 2], head[:, 2], refiner.sizes[:, 2, None])
            assert float(third_gap.min()) >= 0, (variant, 'EC', float(third_gap.min()))
            lane_offset = float(physical[0, 0, 5:, 1].abs().max())
            if variant == 'normal':
                assert lane_offset < .6, (variant, lane_offset)
            ego_speed = v[0].norm(dim=-1)
            if variant == 'cutin':
                assert float(ego_speed[:5].min()) < 10.5, ego_speed.tolist()
            reports.append({'variant':variant, 'min_EA_gap_m':float(gap.min()),
                            'min_EC_gap_m':float(third_gap.min()),
                            'ego_min_first2p5s_speed_mps':float(ego_speed[:5].min()),
                            'max_ego_lane_offset_m':lane_offset,
                            'given_conditions_exact':True, 'future_E_C_free':True})
    print(json.dumps({'status':'passed','checks':reports}, indent=2))


if __name__ == '__main__':
    main()
