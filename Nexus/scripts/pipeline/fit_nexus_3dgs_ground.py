"""Estimate a local road plane from real LiDAR, excluding annotated objects."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    scene = json.loads((args.scene/'nexus_scene.json').read_text())
    if scene['status'] != 'prepared':
        raise ValueError('Incomplete input scene')
    origin = np.linalg.inv(np.asarray(scene['frames'][0]['ego_to_world']))
    info = json.loads((args.scene/'instances/instances_info.json').read_text())
    objects = {}
    for item in info.values():
        annotations = item['frame_annotations']
        for frame, p, size in zip(annotations['frame_idx'], annotations['obj_to_world'], annotations['box_size']):
            objects.setdefault(frame, []).append((origin @ np.asarray(p), np.asarray(size)))
    indices = np.linspace(0, len(scene['frames'])-1, 7).round().astype(int)
    collected = []
    for index in indices:
        raw = np.fromfile(args.scene/'lidar'/f'{index:03d}.bin', dtype=np.float32).reshape(-1, 6)[::4, :3]
        transform = origin @ np.asarray(scene['frames'][index]['ego_to_world'])
        points = raw @ transform[:3, :3].T+transform[:3, 3]
        keep = (np.isfinite(points).all(axis=1) & (points[:, 0] > -8) & (points[:, 0] < 55)
            & (np.abs(points[:, 1]) < 22) & (points[:, 2] > -2.) & (points[:, 2] < .6))
        points = points[keep]
        for p, size in objects.get(int(index), []):
            local = (points-p[:3, 3]) @ p[:3, :3]
            points = points[~np.all(np.abs(local) < size/2+.15, axis=1)]
        collected.append(points)
    points = np.concatenate(collected)
    # Equal weighting per metre cell prevents dense near-sensor returns from
    # dominating; a low quantile suppresses remaining above-road clutter.
    cells = np.floor(points[:, :2]).astype(int)
    _, inverse = np.unique(cells, axis=0, return_inverse=True)
    order = np.argsort(inverse)
    splits = np.flatnonzero(np.diff(inverse[order]))+1
    samples = []
    for group in np.split(order, splits):
        if len(group) >= 4:
            sample = np.median(points[group], axis=0)
            sample[2] = np.quantile(points[group, 2], .25)
            samples.append(sample)
    samples = np.asarray(samples)
    design = np.c_[samples[:, :2], np.ones(len(samples))]
    rng = np.random.default_rng(0)
    best = np.zeros(len(samples), dtype=bool)
    for _ in range(300):
        chosen = rng.choice(len(samples), 3, replace=False)
        if abs(np.linalg.det(design[chosen])) < 1e-6:
            continue
        coef = np.linalg.solve(design[chosen], samples[chosen, 2])
        if np.linalg.norm(coef[:2]) > .15:
            continue
        inliers = np.abs(design @ coef-samples[:, 2]) < .08
        if inliers.sum() > best.sum():
            best = inliers
    if best.sum() < 50 or best.mean() < .45:
        raise ValueError('Insufficient planar road support; no grounding model emitted')
    for _ in range(3):
        coef = np.linalg.lstsq(design[best], samples[best, 2], rcond=None)[0]
        best = np.abs(design @ coef-samples[:, 2]) < .08
    error = design[best] @ coef-samples[best, 2]
    report = {'status': 'validated', 'scene': str(args.scene.resolve()), 'identity': scene['identity'],
        'world_from_scene': scene['frames'][0]['ego_to_world'],
        'source': 'real merged LiDAR; annotated objects excluded; metre-cell low quantiles',
        'frame_indices': indices.tolist(), 'coefficients_z_ax_by_c': coef.tolist(),
        'candidate_cells': len(samples), 'inlier_cells': int(best.sum()),
        'inlier_fraction': float(best.mean()), 'inlier_rmse_m': float(np.sqrt(np.mean(error**2))),
        'supported_xy_bounds_scene': [samples[best, :2].min(axis=0).tolist(), samples[best, :2].max(axis=0).tolist()],
        'scope': 'local planar-road rendering assumption; not a universal terrain model'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
