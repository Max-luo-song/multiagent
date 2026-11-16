"""Project reference/generated vehicle boxes into a real image to audit alignment.

This is a calibration diagnostic, not a 3DGS rendering.
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from nexus_3dgs.bundle import load_bundle
from nexus_3dgs.geometry import resample_poses
from nexus_3dgs.nuplan import write_json

EDGES = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
         (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=1.)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--output', type=Path, default=REPO/'artifacts/nexus_3dgs/projection_diagnostic.png')
    args = parser.parse_args()
    bundle, meta = load_bundle(args.bundle)
    scene = json.loads((args.scene/'nexus_scene.json').read_text())
    if scene['status'] != 'prepared' or scene['identity']['scenario_token'] != meta['scenario_token'] or Path(scene['identity']['database']) != Path(meta['database']):
        raise ValueError('Diagnostic requires the same prepared scene and trajectory bundle')
    timestamp = bundle['timestamps_us'][meta['current_index']]+round(args.seconds*1e6)
    frame = min(scene['frames'], key=lambda f: abs(f['timestamp_us']-timestamp))
    camera = next(i for i in frame['images'] if i['camera_id'] == args.camera)
    image_time = camera['timestamp_us']
    raw = cv2.imread(str(args.scene/'images'/f'{frame["frame_idx"]:03d}_{args.camera}.jpg'))
    if raw is None:
        raise ValueError('Missing diagnostic camera image')
    intrinsic = np.loadtxt(args.scene/'intrinsics'/f'{args.camera}.txt')
    K = np.array([[intrinsic[0], 0, intrinsic[2]], [0, intrinsic[1], intrinsic[3]], [0, 0, 1]])
    raw = cv2.cvtColor(cv2.undistort(raw, K, intrinsic[4:]), cv2.COLOR_BGR2RGB)
    world_to_camera = np.linalg.inv(np.asarray(camera['ego_to_world']) @ np.loadtxt(args.scene/'extrinsics'/f'{args.camera}.txt'))
    info = json.loads((args.scene/'instances/instances_info.json').read_text())
    sizes = {value['id']: np.median(value['frame_annotations']['box_size'], axis=0) for value in info.values()}
    panels, counts = [], {}
    for key, title, color in (('reference_poses_world', 'Logged trajectories / logged camera', '#4cff69'),
                              ('predicted_poses_world', 'Nexus trajectories / same logged camera', '#ff9b32')):
        poses, valid = resample_poses(bundle['timestamps_us'], bundle[key], bundle['valid'], [image_time])
        panel = Image.fromarray(raw.copy())
        draw = ImageDraw.Draw(panel)
        count = 0
        for agent, token in enumerate(bundle['track_tokens']):
            if agent == 0 or not valid[0, agent] or str(token) not in sizes:
                continue
            # Rigid appearance retains reconstructed dimensions in the renderer.
            corners = np.array([[x, y, z, 1.] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])
            corners[:, :3] *= sizes[str(token)]
            xyz = (world_to_camera @ poses[0, agent] @ corners.T).T[:, :3]
            if np.any(xyz[:, 2] <= .1):
                continue
            uv = xyz @ K.T
            uv = uv[:, :2]/uv[:, 2:]
            if uv[:, 0].max() < 0 or uv[:, 0].min() >= panel.width or uv[:, 1].max() < 0 or uv[:, 1].min() >= panel.height:
                continue
            for a, b in EDGES:
                draw.line([tuple(uv[a]), tuple(uv[b])], fill=color, width=3)
            label = tuple(uv.mean(axis=0))
            draw.text(label, f'slot {bundle["nexus_slots"][agent]}', fill=color, stroke_width=1, stroke_fill='black')
            count += 1
        draw.rectangle((0, 0, panel.width, 65), fill='black')
        draw.text((15, 12), title, fill=color)
        draw.text((15, 35), 'PROJECTION DIAGNOSTIC ONLY - NOT A 3DGS RENDER', fill='white')
        panels.append(panel)
        counts[key] = count
    result = Image.new('RGB', (panels[0].width, panels[0].height*2))
    for row, panel in enumerate(panels):
        result.paste(panel, (0, panel.height*row))
    result.thumbnail((1280, 1440))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)
    write_json(args.output.with_suffix('.json'), {'type': 'projection_diagnostic', 'photorealistic_render': False,
        'scene': str(args.scene.resolve()), 'bundle': str(args.bundle.resolve()), 'frame_idx': frame['frame_idx'],
        'timestamp_us': image_time, 'visible_boxes': counts, 'camera_mode': 'logged', 'lens': 'undistorted_pinhole'})
    print('NEXUS_3DGS_PROJECTION_OK '+str(args.output))


if __name__ == '__main__':
    main()
