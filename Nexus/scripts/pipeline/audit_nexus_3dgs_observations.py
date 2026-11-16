"""Count actual LiDAR/color observations and per-pixel occlusion conflicts.

Uses original read-only DB calibration and the prepared raw point clouds.
It does not infer that an existing Gaussian has a valid appearance merely
because it has an instance ID.
"""
import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from nexus_3dgs import nuplan


def visibility(points_world, camera_world, K, height, width, tolerance):
    w2c = np.linalg.inv(camera_world)
    camera_points = points_world @ w2c[:3, :3].T+w2c[:3, 3]
    depth = camera_points[:, 2]
    projected = camera_points @ K.T
    uv = projected[:, :2]/(depth[:, None]+1e-6)
    valid = (np.isfinite(uv).all(axis=1) & (depth > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < width)
        & (uv[:, 1] >= 0) & (uv[:, 1] < height))
    selected = np.flatnonzero(valid)
    pixels = uv[valid].astype(np.int64)
    ids = pixels[:, 1]*width+pixels[:, 0]
    near = np.full(height*width, np.inf)
    far = np.full(height*width, -np.inf)
    counts = np.bincount(ids, minlength=height*width)
    np.minimum.at(near, ids, depth[valid]); np.maximum.at(far, ids, depth[valid])
    visible = np.zeros(len(points_world), dtype=bool)
    visible[selected] = depth[valid] <= near[ids]+tolerance
    occupied = counts > 0
    return valid, visible, {'fov_points': len(selected), 'occupied_pixels': int(occupied.sum()),
        'duplicate_pixels': int((counts > 1).sum()),
        'depth_conflict_pixels': int(((far-near > tolerance) & occupied).sum()),
        'occluded_fov_points': int((valid & ~visible).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cameras', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--tolerance', type=float, default=.2)
    parser.add_argument('--bundle', type=Path, help='Identify actors actually generated in the future')
    args = parser.parse_args()
    scene = json.loads((args.scene/'nexus_scene.json').read_text())
    info = json.loads((args.scene/'instances/instances_info.json').read_text())
    per_frame = {}
    actors = {}
    for dataset_id, item in info.items():
        actors[dataset_id] = {'track_token': item['id'], 'class_name': item['class_name'], 'raw_lidar_points': 0,
            'fov_points': {str(c):0 for c in args.cameras}, 'visible_points': {str(c):0 for c in args.cameras},
            'visible_frames': {str(c):0 for c in args.cameras}, 'visible_union_points': 0,
            'visibility_bit_counts': {}}
        ann = item['frame_annotations']
        for index, p, size in zip(ann['frame_idx'], ann['obj_to_world'], ann['box_size']):
            per_frame.setdefault(index, []).append((dataset_id, np.asarray(p), np.asarray(size)))
    stats = {str(c): {'fov_points':0, 'occupied_pixels':0, 'duplicate_pixels':0,
        'depth_conflict_pixels':0, 'occluded_fov_points':0, 'available_images':0} for c in args.cameras}
    locator = nuplan.SensorLocator([
        Path('/inspire/hdd/global_public/public_datas/nuPlan/sensor_blobs_full'),
        Path('/inspire/hdd/global_public/public_datas/nuPlan/sensor_blobs_mini')])
    with nuplan.connect(scene['identity']['database']) as db:
        calibrations = nuplan.cameras(db, [nuplan.CAMERAS[c] for c in args.cameras])
        for frame in scene['frames'][::args.stride]:
            index = frame['frame_idx']
            raw = np.fromfile(args.scene/'lidar'/f'{index:03d}.bin', dtype=np.float32).reshape(-1, 6)[:, :3]
            ego = np.asarray(frame['ego_to_world'])
            points = raw @ ego[:3, :3].T+ego[:3, 3]
            views, union = {}, np.zeros(len(points), dtype=bool)
            visibility_bits = np.zeros(len(points), dtype=np.uint16)
            for calibration in calibrations.values():
                c = str(calibration['id'])
                image = nuplan.nearest_image(db, calibration['token'], frame['timestamp_us'])
                if not locator.resolve(image['filename_jpg']):
                    continue
                stats[c]['available_images'] += 1
                K = calibration['intrinsics'].copy(); K[:2] /= 2
                in_fov, seen, current = visibility(points, nuplan.ego_pose(image) @ calibration['extrinsics'],
                    K, calibration['height']//2, calibration['width']//2, args.tolerance)
                views[c] = in_fov, seen
                union |= seen
                visibility_bits[seen] |= (1 << calibration['id'])
                for key, value in current.items():
                    stats[c][key] += int(value)
            for dataset_id, p, size in per_frame.get(index, []):
                # Reject by a world AABB before the exact rotated-box test.
                extent = np.abs(p[:3, :3]) @ (size/2)
                candidates = np.flatnonzero(np.all(np.abs(points-p[:3, 3]) < extent, axis=1))
                local = (points[candidates]-p[:3, 3]) @ p[:3, :3]
                inside = candidates[np.all(np.abs(local) < size/2, axis=1)]
                actor = actors[dataset_id]
                actor['raw_lidar_points'] += len(inside)
                actor['visible_union_points'] += int(union[inside].sum())
                codes, counts = np.unique(visibility_bits[inside], return_counts=True)
                for code, count in zip(codes, counts):
                    key = str(int(code))
                    actor['visibility_bit_counts'][key] = actor['visibility_bit_counts'].get(key, 0)+int(count)
                for c, (in_fov, seen) in views.items():
                    actor['fov_points'][c] += int(in_fov[inside].sum())
                    count = int(seen[inside].sum())
                    actor['visible_points'][c] += count
                    actor['visible_frames'][c] += int(count >= 3)
            if index % 20 == 0:
                print('OBSERVATION_AUDIT frame='+str(index), flush=True)
    report = {'status':'complete', 'scene':str(args.scene.resolve()), 'camera_ids':args.cameras,
        'stride':args.stride, 'occlusion_tolerance_m':args.tolerance, 'cameras':stats, 'actors':actors,
        'scope':'LiDAR projection/color support; pointwise depth checks do not prove complete surface coverage'}
    if args.bundle:
        from nexus_3dgs.bundle import load_bundle
        bundle, manifest = load_bundle(args.bundle)
        future = bundle['valid'][manifest['current_index']+1:].any(axis=0)
        generated = {str(token) for i, token in enumerate(bundle['track_tokens']) if i and future[i]}
        for actor in actors.values():
            actor['generated_in_future'] = actor['track_token'] in generated
        support = []
        for count in range(1, min(4, len(args.cameras))+1):
            for cameras in itertools.combinations(args.cameras, count):
                if 0 not in cameras:
                    continue
                bits = sum(1 << c for c in cameras)
                missing = []
                for actor in actors.values():
                    observed = sum(n for code, n in actor['visibility_bit_counts'].items() if int(code) & bits)
                    if actor['generated_in_future'] and observed < 100:
                        missing.append(actor['track_token'])
                support.append({'cameras':list(cameras), 'generated_future_actors_below_100_points':missing})
        report['subset_support'] = support
        report['generated_future_actors'] = sorted(generated)
    nuplan.write_json(args.output, report)
    print('OBSERVATION_AUDIT_OK '+str(args.output), flush=True)


if __name__ == '__main__':
    main()
