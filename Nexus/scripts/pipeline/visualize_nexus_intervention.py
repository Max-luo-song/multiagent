"""Show a paired bus intervention with generated neighbors and collision flags."""
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
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO/'scripts/pipeline'))
from nexus_3dgs.bundle import decode, load_bundle
from nexus_3dgs.geometry import resample_poses
from collision_audit import rectangle_overlaps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--stop-render', type=Path, required=True)
    parser.add_argument('--go-render', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    experiment = json.loads((args.experiment/'run.json').read_text())
    if experiment['status'] != 'complete':
        raise ValueError('Incomplete intervention')
    names = ('bus_stop', 'bus_follow_logged')
    filtered = 'source_experiment' in experiment
    source_experiment = json.loads((Path(experiment['source_experiment'])/'run.json').read_text()) if filtered else experiment
    variants = []
    for name, directory in zip(names, (args.stop_render, args.go_render)):
        bundle, manifest = load_bundle(args.experiment/name)
        run = json.loads((directory/'run.json').read_text())
        if run['status'] != 'complete' or run['bundle_sha256'] != manifest['bundle_sha256']:
            raise ValueError('Rendering does not match intervention bundle')
        records = {(f['frame_idx'], i['timestamp_us']): i['file'] for f in run['frames']
                   for i in f['images'] if i['camera_id'] == 0}
        variants.append((bundle, manifest, run, records, directory))
    left, right = variants
    if left[3].keys() != right[3].keys():
        raise ValueError('Paired timestamps differ')
    for key in ('checkpoint_sha256', 'camera_mode', 'pose_policy', 'camera_refinement'):
        if left[2][key] != right[2][key]:
            raise ValueError('Paired renderer settings differ: '+key)
    np.testing.assert_array_equal(left[0]['reference_poses_world'], right[0]['reference_poses_world'])
    np.testing.assert_array_equal(left[0]['valid'], right[0]['valid'])
    if source_experiment['variants'][names[0]]['initial_noise_sha256'] != source_experiment['variants'][names[1]]['initial_noise_sha256']:
        raise ValueError('Paired noise differs')
    current = left[1]['current_index']
    bus_slot = experiment['conditioned_slot']
    bus_agent = list(left[0]['nexus_slots']).index(bus_slot)
    anchor = left[0]['reference_poses_world'][current, 0]
    yaw = np.arctan2(anchor[1, 0], anchor[0, 0])
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    def project(poses):
        xy = (poses[..., :2, 3]-anchor[:2, 3]) @ rotation
        return np.stack((-xy[..., 1], xy[..., 0]), axis=-1)
    keys = sorted(left[3])
    times = np.array([key[1] for key in keys])
    samples = []
    collision_steps = []
    for name, (bundle, *_) in zip(names, variants):
        poses, valid = resample_poses(bundle['timestamps_us'], bundle['predicted_poses_world'], bundle['valid'], times)
        samples.append((project(poses), valid))
        # Coarse sampled-state collisions are displayed only at the nearest
        # original sample; no continuous-time collision claim is made.
        if filtered:
            collision_steps.append(None)
        else:
            payload = torch.load(args.experiment/f'{name}.pt', map_location='cpu', weights_only=False)
            states = decode(payload['sampled_tensor'][0].numpy())
            validity = payload['valid_mask'][0].numpy().astype(bool)
            collision_steps.append([bool(rectangle_overlaps(states[:, i], validity[:, i])[0].any()) for i in range(states.shape[1])])
    colors = plt.get_cmap('tab20')(np.linspace(0, 1, len(left[0]['track_tokens'])))
    colors[0] = [0.85, .1, .1, 1]
    colors[bus_agent] = [.1, .1, .1, 1]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), gridspec_kw={'width_ratios': [1, 1.5]})
    artists = []
    for row, ((bundle, _, _, records, directory), title) in enumerate(zip(variants,
            ('A: bus future GIVEN: stopped', 'B: bus future GIVEN: follows logged path'))):
        axis = axes[row, 0]
        axis.set(title=title, xlabel='Right of anchor ego (m)', ylabel='Forward from anchor ego (m)',
                 xlim=(-20, 20), ylim=(-8, 50))
        axis.set_aspect('equal', adjustable='box'); axis.grid(alpha=.2)
        dots, labels = [], []
        xy = project(bundle['predicted_poses_world'])
        for agent in range(len(colors)):
            track = xy[current:, agent].copy(); track[~bundle['valid'][current:, agent]] = np.nan
            axis.plot(track[:, 0], track[:, 1], color=colors[agent], alpha=.5)
            dot, = axis.plot([], [], 's' if agent == bus_agent else 'o', color=colors[agent], markersize=8)
            dots.append(dot)
            labels.append(axis.text(0, 0, 'ego' if agent == 0 else ('BUS GIVEN' if agent == bus_agent else str(bundle['nexus_slots'][agent])), fontsize=8, clip_on=True))
        picture = axes[row, 1].imshow(np.asarray(Image.open(directory/records[keys[0]])))
        axes[row, 1].axis('off')
        artists.append((dots, labels, picture))
    stamp = fig.suptitle('')
    footer = ('Nexus samples + DOWNSTREAM EGO TIMING FILTER. Bus and other actors unchanged.\n'
              'Physical ego clearance checked at 100 Hz; filtered motion is not a learned Nexus reaction.\n'
              'Side/rear actor overlaps remain; this is not a collision-free full-scene simulation.' if filtered else
              'Same history / map / noise. Only bus future supplied. Other vehicles sampled by Nexus.\n'
              'DIAGNOSTIC: collisions detected in both variants; conditional response does not establish safe interaction.')
    if filtered and experiment.get('limit_to_observed_camera_platform_travel'):
        cap=experiment['variants']['bus_follow_logged']['ego_travel_cap_m']
        footer=(f'Nexus + downstream timing filter. Ego travel capped at observed camera-platform travel ({cap:.2f} m).\n'
                'Not a raw learned Nexus reaction or arbitrary novel-view reconstruction. Bus and other actors unchanged.\n'
                'Ego physical clearance checked at 100 Hz. Side/rear actor overlaps remain (see report).')
    fig.text(.5, .025, footer,
             ha='center', fontsize=11)
    fig.subplots_adjust(left=.05, right=.98, top=.91, bottom=.15 if filtered else .11, hspace=.3, wspace=.15)
    args.output.mkdir(parents=True, exist_ok=True)
    ego_response = next(a['max_response_distance_m'] for a in source_experiment['actors'] if a['slot'] == 0)
    with imageio.get_writer(str(args.output/'bus_intervention_diagnostic.mp4'), fps=10, macro_block_size=2,
                            ffmpeg_params=['-threads', '4']) as writer:
        for index, key in enumerate(keys):
            seconds = (key[1]-left[0]['timestamps_us'][current])/1e6
            stamp.set_text(f'Bus conditions + downstream ego timing filter | t={seconds:.1f}s' if filtered else
                           f'Paired bus intervention | t={seconds:.1f}s | generated ego response max {ego_response:.2f} m')
            nearest = int(np.argmin(np.abs(left[0]['timestamps_us']-key[1])))
            for row, ((dots, labels, picture), (xy, present), (_, _, _, records, directory)) in enumerate(zip(artists,
                    [(v[0][index], v[1][index]) for v in samples], variants)):
                for agent, (dot, label) in enumerate(zip(dots, labels)):
                    dot.set_data([xy[agent, 0]] if present[agent] else [], [xy[agent, 1]] if present[agent] else [])
                    label.set_visible(bool(present[agent]))
                    if present[agent]:
                        label.set_position(xy[agent]+[.7, .5])
                picture.set_data(np.asarray(Image.open(directory/records[key])))
                if filtered:
                    verified=experiment['variants'][names[row]]['after_physical_min_separation_m']
                    axes[row,1].set_title(f'Filtered ego vs all actors: min physical clearance {verified:.2f} m (100 Hz)')
                else:
                    axes[row, 1].set_title('Ego overlaps another box at nearest 2 Hz sample' if collision_steps[row][nearest]
                                          else '3DGS driven by this variant; collision audit at 2 Hz', color='red' if collision_steps[row][nearest] else 'black')
            fig.canvas.draw()
            pixels = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            writer.append_data(pixels)
            if index == len(keys)//2:
                Image.fromarray(pixels).save(args.output/'intervention_preview.png')
    plt.close(fig)
    with imageio.get_reader(str(args.output/'bus_intervention_diagnostic.mp4'), input_params=['-threads', '2'],
                           output_params=['-threads', '2']) as reader:
        count = sum(1 for _ in reader)
    if count != len(keys):
        raise ValueError('Comparison decode mismatch')
    (args.output/'verification.json').write_text(json.dumps({'status': 'passed', 'decoded_frames': count,
        'same_history_reference_validity_and_noise': True, 'checkpoint_sha256': left[2]['checkpoint_sha256'],
        'experiment': str(args.experiment.resolve()), 'scope': 'downstream physical timing filter' if filtered else 'conditional response diagnostic with detected collisions'}, indent=2)+'\n')
    print('INTERVENTION_VISUALIZED frames='+str(count))


if __name__ == '__main__':
    main()
