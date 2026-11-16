"""Make an input-checked, synchronized reconstruction ablation video.

This compares appearance checkpoints on fixed generated trajectories. Sky
range diagnostics establish numerical validity, not novel-view image quality.
"""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def records(run, camera):
    return {(f['frame_idx'], i['timestamp_us']): i['file']
            for f in run['frames'] for i in f['images'] if i['camera_id'] == camera}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', action='append', required=True, help='Label=render directory, in ablation order')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--preview-frame', type=int, default=60)
    parser.add_argument('--heading-only', action='store_true',
        help='Allow only audited heading changes; require identical positions, conditions, other arrays and checkpoint')
    args = parser.parse_args()
    stages = []
    for spec in args.stage:
        label, separator, directory = spec.partition('=')
        if not separator:
            parser.error('--stage requires Label=directory')
        directory = Path(directory).resolve()
        run = json.loads((directory/'run.json').read_text())
        if run['status'] != 'complete':
            raise ValueError('Incomplete rendering: '+str(directory))
        items = records(run, args.camera)
        if not items:
            raise ValueError('Requested camera absent')
        stages.append((label, directory, run, items))
    baseline = stages[0][2]
    for _, _, run, items in stages[1:]:
        for key in ('bundle_sha256', 'reference', 'pose_policy', 'camera_mode', 'camera_refinement'):
            if key=='bundle_sha256' and args.heading_only:
                import sys
                sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
                from nexus_3dgs.bundle import load_bundle
                before,before_meta=load_bundle(baseline['bundle']);after,after_meta=load_bundle(run['bundle'])
                audit=after_meta.get('heading_repair',{})
                if audit.get('source_bundle_sha256')!=before_meta['bundle_sha256'] or run['checkpoint_sha256']!=baseline['checkpoint_sha256']:
                    raise ValueError('Expected an audited heading repair of the same checkpoint and source bundle')
                for field in before:
                    if field!='predicted_poses_world':np.testing.assert_array_equal(before[field],after[field])
                np.testing.assert_array_equal(before['predicted_poses_world'][...,:3,3],after['predicted_poses_world'][...,:3,3])
                changed=np.any(before['predicted_poses_world']!=after['predicted_poses_world'],axis=(-1,-2))
                expected={(c['sample_index'],c['agent_index']) for c in audit['changes']}
                if set(map(tuple,np.argwhere(changed)))!=expected:raise ValueError('Unrecorded pose changes in heading-only comparison')
                continue
            if run[key] != baseline[key]:
                raise ValueError('Comparison changes trajectory/render policy: '+key)
        if items.keys() != stages[0][3].keys():
            raise ValueError('Camera timestamps differ')
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 21)
    except OSError:
        try:
            from matplotlib.font_manager import findfont
            font = ImageFont.truetype(findfont('DejaVu Sans'), 21)
        except (ImportError, OSError):
            font = ImageFont.load_default()
    columns = min(2, len(stages))
    rows = (len(stages)+columns-1)//columns
    keys = sorted(stages[0][3])
    width, height = Image.open(stages[0][1]/stages[0][3][keys[0]]).size
    report = {'status': 'started', 'bundle_sha256': baseline['bundle_sha256'],
              'same_trajectory_and_timestamps': not args.heading_only,
              'same_positions_and_timestamps':True,'audited_heading_only_repair':args.heading_only,'camera': args.camera,
              'frames': len(keys), 'stages': [],
              'scope': 'single-scene appearance ablation; sky bounds are not a perceptual-quality score; no novel-view ground truth'}
    with imageio.get_writer(str(args.output/'same_trajectories_stages.mp4'), fps=10,
                            macro_block_size=2, ffmpeg_params=['-threads', '4']) as writer:
        for frame, timestamp in keys:
            panel = Image.new('RGB', (columns*width, rows*(height+52)))
            draw = ImageDraw.Draw(panel)
            for index, (label, directory, _, items) in enumerate(stages):
                picture = Image.open(directory/items[(frame, timestamp)]).convert('RGB')
                if picture.size != (width, height):
                    raise ValueError('Frame sizes differ')
                x, y = index % columns*width, index//columns*(height+52)
                panel.paste(picture, (x, y+52))
                draw.text((x+10, y+4), label, font=font, fill='white')
                caption='Same positions/checkpoint; audited heading repair' if args.heading_only else 'Same Nexus goal-conditioned future'
                draw.text((x+10, y+28), f'{caption} | frame {frame}', font=font, fill='#bbbbbb')
            writer.append_data(np.asarray(panel))
            if frame == args.preview_frame:
                panel.save(args.output/'comparison_preview.png')
    for label, directory, run, items in stages:
        with imageio.get_reader(str(directory/f'camera_{args.camera}.mp4'),
                               input_params=['-threads', '2'], output_params=['-threads', '2']) as reader:
            fps = reader.get_meta_data()['fps']
            count = sum(1 for _ in reader)
        if count != len(keys) or abs(fps-10) > .01:
            raise ValueError('Source video decode mismatch')
        sky_stats = {}
        for path in sorted((directory/'diagnostic_layers').glob(f'*_{args.camera}_stats.json')):
            stats = json.loads(path.read_text())
            sky_stats[path.stem] = {k: stats[k] for k in ('rgb_sky', 'rgb_sky_blend') if k in stats}
        report['stages'].append({'label': label, 'render': str(directory),
                                'checkpoint': run['checkpoint'], 'checkpoint_sha256': run['checkpoint_sha256'],
                                'training_cameras': run.get('training_cameras', [0]),
                                'video_decoded_frames': count, 'fps': fps, 'sky_diagnostics': sky_stats})
    with imageio.get_reader(str(args.output/'same_trajectories_stages.mp4'),
                           input_params=['-threads', '2'], output_params=['-threads', '2']) as reader:
        count = sum(1 for _ in reader)
    if count != len(keys):
        raise ValueError('Comparison video decode mismatch')
    report.update(status='passed', comparison_decoded_frames=count)
    (args.output/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'status': 'passed', 'stages': len(stages), 'frames': count}))


if __name__ == '__main__':
    main()
