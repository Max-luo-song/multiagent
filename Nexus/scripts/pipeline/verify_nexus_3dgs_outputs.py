"""Check actual reference/Nexus renders and create a labeled comparison video."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load(path):
    return json.loads(path.read_text())


def frames(run):
    return {(frame['frame_idx'], image['camera_id']): image for frame in run['frames'] for image in frame['images']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--generated', type=Path, required=True)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    ref, gen = load(args.reference/'run.json'), load(args.generated/'run.json')
    if ref['status'] != 'complete' or gen['status'] != 'complete':
        raise ValueError('Both renders must be complete')
    if not ref['reference'] or ref['camera_mode'] != 'logged' or gen['reference'] or gen['camera_mode'] != 'generated_ego':
        raise ValueError('Expected logged reference and Nexus-controlled ego-camera renders')
    for field in ('checkpoint_sha256', 'bundle_sha256', 'lens'):
        if ref[field] != gen[field]:
            raise ValueError(f'Reference/generated {field} disagree')
    if hashlib.sha256(Path(gen['checkpoint']).read_bytes()).hexdigest() != gen['checkpoint_sha256']:
        raise ValueError('Rendered checkpoint checksum changed')
    if gen['uncontrolled_nexus_tracks']:
        raise ValueError('Some requested Nexus tracks were not controlled')
    ref_frames, gen_frames = frames(ref), frames(gen)
    if not ref_frames or ref_frames.keys() != gen_frames.keys():
        raise ValueError('Reference/generated frames disagree or are empty')
    scene = load(args.scene/'nexus_scene.json')
    bundle_meta = load(Path(gen['bundle'])/'manifest.json')
    if (Path(scene['identity']['database']).resolve() != Path(bundle_meta['database']).resolve()
            or scene['identity']['scenario_token'] != bundle_meta['scenario_token']):
        raise ValueError('Comparison scene does not match the trajectory bundle')
    args.output.mkdir(parents=True, exist_ok=True)
    intrinsic_cache, writers, scores, previews = {}, {}, [], []
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
    except OSError:
        font = ImageFont.load_default()
    keys = sorted(gen_frames)
    preview_key = keys[len(keys)//2]
    video_checks = []
    for camera_id in sorted({key[1] for key in keys}):
        expected = sum(key[1] == camera_id for key in keys)
        for name, directory in (('reference', args.reference), ('generated', args.generated)):
            with imageio.get_reader(str(directory/f'camera_{camera_id}.mp4')) as reader:
                video_fps = reader.get_meta_data()['fps']
                decoded = 0
                for frame in reader:
                    if frame.ndim != 3 or frame.shape[2] != 3:
                        raise ValueError('Invalid decoded video frame')
                    decoded += 1
            if decoded != expected or abs(video_fps-scene['identity']['fps']) > .01:
                raise ValueError('Video duration/frame rate does not match render manifest')
            video_checks.append({'render': name, 'camera_id': camera_id, 'decoded_frames': decoded, 'fps': video_fps})
    try:
        for key in keys:
            index, camera = key
            if ref_frames[key]['timestamp_us'] != gen_frames[key]['timestamp_us']:
                raise ValueError('Camera timestamps disagree')
            reference = np.asarray(Image.open(args.reference/ref_frames[key]['file']).convert('RGB'))
            generated = np.asarray(Image.open(args.generated/gen_frames[key]['file']).convert('RGB'))
            if reference.shape != generated.shape or reference.std() < 1 or generated.std() < 1:
                raise ValueError('Invalid render size or uniform/empty image')
            height, width = reference.shape[:2]
            with Image.open(args.scene/'images'/f'{index:03d}_{camera}.jpg') as source:
                sx, sy = width/source.width, height/source.height
                logged = np.asarray(source.convert('RGB').resize((width, height), Image.Resampling.BILINEAR))
            if ref['lens'] == 'undistorted_pinhole':
                if camera not in intrinsic_cache:
                    intrinsic_cache[camera] = np.loadtxt(args.scene/'intrinsics'/f'{camera}.txt')
                values = intrinsic_cache[camera]
                K = np.array([[values[0]*sx, 0, values[2]*sx], [0, values[1]*sy, values[3]*sy], [0, 0, 1.]])
                logged = cv2.undistort(logged, K, values[4:])
            error = np.mean(((reference.astype(np.float64)-logged)/255.)**2)
            difference = np.abs(reference.astype(np.float64)-generated)
            scores.append({'frame_idx': index, 'camera_id': camera,
                'reference_training_view_psnr_db': float(-10*np.log10(max(error, 1e-12))),
                'nexus_vs_reference_mean_absolute_pixel_difference': float(difference.mean()),
                'nexus_vs_reference_changed_pixel_fraction': float(np.any(difference > 2, axis=-1).mean())})
            panel_width, panel_height = 640, round(height*640/width)
            panel = Image.new('RGB', (panel_width*3, panel_height+42))
            draw = ImageDraw.Draw(panel)
            for column, (array, title) in enumerate(((logged, 'Logged RGB (undistorted)'),
                    (reference, '3DGS / logged trajectories + camera'),
                    (generated, '3DGS / Nexus trajectories + ego camera'))):
                panel.paste(Image.fromarray(array).resize((panel_width, panel_height)), (column*panel_width, 42))
                draw.text((column*panel_width+10, 5), title, font=font, fill='white')
            draw.text((10, 24), f'frame {index} | scene-specific reconstruction, training views', font=font, fill='#cccccc')
            if camera not in writers:
                writers[camera] = imageio.get_writer(str(args.output/f'comparison_camera_{camera}.mp4'),
                    fps=scene['identity']['fps'], macro_block_size=2, ffmpeg_params=['-threads', '4'])
            writers[camera].append_data(np.asarray(panel))
            if key == preview_key:
                panel.save(args.output/'comparison_preview.png')
                previews.append({'frame_idx': index, 'file': 'comparison_preview.png'})
    finally:
        for writer in writers.values():
            writer.close()
    changed = float(np.mean([s['nexus_vs_reference_changed_pixel_fraction'] for s in scores]))
    if changed <= .0001:
        raise ValueError('Nexus-controlled rendering has no visible change from reference')
    report = {'status': 'passed', 'actual_3dgs_render_verified': True,
        'reference_frames': len(ref_frames), 'generated_frames': len(gen_frames),
        'controlled_track_count': gen['controlled_track_count'], 'uncontrolled_tracks': [],
        'checkpoint': gen['checkpoint'], 'checkpoint_sha256': gen['checkpoint_sha256'],
        'bundle_sha256': gen['bundle_sha256'], 'previews': previews,
        'video_checks': video_checks,
        'reference_training_view_mean_psnr_db': float(np.mean([s['reference_training_view_psnr_db'] for s in scores])),
        'mean_changed_pixel_fraction': changed, 'metrics_scope': 'scene reconstruction training views; not held-out Nexus evaluation',
        'per_frame': scores}
    (args.output/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print('NEXUS_3DGS_OUTPUTS_OK '+str(args.output), flush=True)


if __name__ == '__main__':
    main()
