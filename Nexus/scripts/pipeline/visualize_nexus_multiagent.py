"""Audit a rendered Nexus bundle with synchronized multi-agent bird's-eye tracks.

Trajectory differences do not establish a conditional interaction response.
"""
import argparse
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from nexus_3dgs.bundle import load_bundle
from nexus_3dgs.geometry import resample_poses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--render', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    bundle, manifest = load_bundle(args.bundle)
    render = json.loads((args.render/'run.json').read_text())
    if render['status'] != 'complete' or render['bundle_sha256'] != manifest['bundle_sha256']:
        raise ValueError('Expected a completed rendering of this exact trajectory bundle')
    current = manifest['current_index']
    source_time = bundle['timestamps_us']
    anchor = bundle['reference_poses_world'][current, 0]
    yaw = np.arctan2(anchor[1, 0], anchor[0, 0])
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    def project(poses):
        local = (poses[..., :2, 3]-anchor[:2, 3]) @ rotation
        return np.stack([-local[..., 1], local[..., 0]], axis=-1)
    ref = bundle['reference_poses_world']
    pred = bundle['predicted_poses_world']
    valid = bundle['valid']
    ref_xy, pred_xy = project(ref), project(pred)
    rows = []
    for agent, token in enumerate(bundle['track_tokens']):
        mask = valid[current:, agent]
        times = np.flatnonzero(mask)+current
        if not len(times):
            continue
        differences = np.linalg.norm(pred_xy[times, agent]-ref_xy[times, agent], axis=-1)
        rows.append({'slot': int(bundle['nexus_slots'][agent]), 'track_token': str(token),
            'valid_future_samples': len(times), 'mean_difference_from_logged_m': float(differences.mean()),
            'maximum_difference_from_logged_m': float(differences.max()),
            'last_valid_difference_from_logged_m': float(differences[-1]),
            'generated_path_length_m': float(np.linalg.norm(np.diff(pred_xy[current:, agent], axis=0), axis=-1)[mask[1:] & mask[:-1]].sum()),
            'logged_path_length_m': float(np.linalg.norm(np.diff(ref_xy[current:, agent], axis=0), axis=-1)[mask[1:] & mask[:-1]].sum())})
    records = [(frame['frame_idx'], image) for frame in render['frames'] for image in frame['images'] if image['camera_id'] == 0]
    times = np.array([image['timestamp_us'] for _, image in records])
    ref_samples, ref_valid = resample_poses(source_time, ref, valid, times)
    pred_samples, pred_valid = resample_poses(source_time, pred, valid, times)
    ref_samples, pred_samples = project(ref_samples), project(pred_samples)
    active_points = np.concatenate([ref_xy[current:][valid[current:]], pred_xy[current:][valid[current:]]])
    low, high = active_points.min(axis=0)-5, active_points.max(axis=0)+5
    colors = plt.get_cmap('tab20')(np.linspace(0, 1, len(bundle['track_tokens'])))
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={'width_ratios': [1, 1, 1.5]})
    artists = []
    titles = ('Logged future trajectories', f'Nexus joint future ({manifest["task"]})')
    for axis, xy, title in zip(axes[:2], (ref_xy, pred_xy), titles):
        axis.set(title=title, xlabel='Right of anchor ego (m)', ylabel='Forward from anchor ego (m)',
            xlim=(low[0], high[0]), ylim=(low[1], high[1]))
        axis.set_aspect('equal', adjustable='box')
        axis.grid(alpha=.2)
        dots, labels = [], []
        for agent in range(len(colors)):
            track = xy[current:, agent].copy()
            track[~valid[current:, agent]] = np.nan
            axis.plot(track[:, 0], track[:, 1], color=colors[agent], alpha=.6, linewidth=1.5)
            dot, = axis.plot([], [], 'o', color=colors[agent], markersize=7 if agent else 10)
            dots.append(dot)
            labels.append(axis.text(0, 0, 'ego' if agent == 0 else str(bundle['nexus_slots'][agent]), fontsize=8))
        artists.append((dots, labels))
    camera_artist = axes[2].imshow(np.asarray(Image.open(args.render/records[0][1]['file'])))
    axes[2].set_title('3DGS camera driven by the same Nexus trajectories')
    axes[2].axis('off')
    stamp = fig.suptitle('')
    conditioned = manifest['task'] == 'intent_conditioned_behavior_pred'
    fig.text(.5, .025, 'Same actor colors / IDs. Lines: full future. Dots: current positions. '+
        ('Given past + logged final states; Nexus generates intermediate futures.' if conditioned
         else 'Given past only; no designed interaction condition.'), ha='center', fontsize=10)
    fig.subplots_adjust(left=.055, right=.99, bottom=.13, top=.86, wspace=.22)
    args.output.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(args.output/'multiagent_and_camera.mp4'), fps=10, macro_block_size=2,
            ffmpeg_params=['-threads', '4']) as writer:
        for index, (frame_index, image) in enumerate(records):
            seconds = (image['timestamp_us']-source_time[current])/1e6
            stamp.set_text(f'Joint multi-agent future and rendered camera | t = {seconds:.1f}s | source task: {manifest["task"]}')
            for (dots, labels), xy, present in zip(artists, (ref_samples[index], pred_samples[index]), (ref_valid[index], pred_valid[index])):
                for agent, (dot, label) in enumerate(zip(dots, labels)):
                    dot.set_data([xy[agent, 0]] if present[agent] else [], [xy[agent, 1]] if present[agent] else [])
                    label.set_visible(bool(present[agent]))
                    if present[agent]:
                        label.set_position(xy[agent]+[.7, .7])
            camera_artist.set_data(np.asarray(Image.open(args.render/image['file'])))
            fig.canvas.draw()
            pixels = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            writer.append_data(pixels)
            if index == len(records)//2:
                Image.fromarray(pixels).save(args.output/'multiagent_preview.png')
    plt.close(fig)
    report = {'source_task': manifest['task'], 'bundle_sha256': manifest['bundle_sha256'],
        'actors_including_ego': len(rows), 'non_ego_actors_with_max_change_over_1m': sum(r['slot'] != 0 and r['maximum_difference_from_logged_m'] > 1 for r in rows),
        'designed_interaction_condition': False, 'counterfactual_response_tested': False,
        'given_logged_endpoints': conditioned,
        'scope': 'visualize joint future output; identity binding and position differences are not evidence of interaction response',
        'frames': len(records), 'actors': rows}
    (args.output/'trajectory_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'actors'}, indent=2))


if __name__ == '__main__':
    main()
