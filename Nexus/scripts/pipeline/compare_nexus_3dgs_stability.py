"""Compare original/grounded rendering of exactly the same Nexus prediction."""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old', type=Path, required=True)
    parser.add_argument('--new', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    old, new = [json.loads((p/'run.json').read_text()) for p in (args.old, args.new)]
    if old['status'] != 'complete' or new['status'] != 'complete':
        raise ValueError('Incomplete render')
    for key in ('checkpoint_sha256', 'bundle_sha256'):
        if old[key] != new[key]:
            raise ValueError('Inputs differ: '+key)
    def records(run):
        return {(f['frame_idx'], i['camera_id']): i for f in run['frames'] for i in f['images']}
    left, right = records(old), records(new)
    if left.keys() != right.keys():
        raise ValueError('Camera frames differ')
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    except OSError:
        font = ImageFont.load_default()
    keys = sorted(left)
    shapes = {}
    with imageio.get_writer(str(args.output/'same_prediction_before_after.mp4'), fps=10,
            macro_block_size=2, ffmpeg_params=['-threads', '4']) as writer:
        for n, key in enumerate(keys):
            images = [Image.open(directory/items[key]['file']).convert('RGB') for directory, items in
                ((args.old, left), (args.new, right))]
            if images[0].size != images[1].size:
                raise ValueError('Image sizes differ')
            width, height = images[0].size
            panel = Image.new('RGB', (2*width, height+50))
            draw = ImageDraw.Draw(panel)
            for column, (picture, label) in enumerate(zip(images,
                    ('Original adapter: sampled annotation height / camera',
                     'LiDAR ground + fixed rigid/camera registration'))):
                panel.paste(picture, (column*width, 50))
                draw.text((column*width+10, 5), label, fill='white', font=font)
            draw.text((10, 28), f'SAME Nexus BP prediction + SAME checkpoint | frame {key[0]} | no XY/yaw trajectory smoothing',
                fill='#bbbbbb', font=font)
            writer.append_data(np.asarray(panel))
            if n == len(keys)//2:
                panel.save(args.output/'same_prediction_preview.png')
    for name, directory in (('old', args.old), ('grounded', args.new)):
        with imageio.get_reader(str(directory/'camera_0.mp4'), input_params=['-threads', '2'],
                output_params=['-threads', '2']) as reader:
            count = sum(1 for _ in reader)
            if count != len(keys):
                raise ValueError('Decoded video frame count differs')
            shapes[name] = {'decoded_frames': count, 'fps': reader.get_meta_data()['fps']}
    audit = json.loads(args.audit.read_text())
    metrics = audit['bus_center_projection_pixels']
    old_v, new_v = metrics['old_generated']['v'], metrics['grounded_generated']['v']
    report = {'status': 'passed', 'bundle_sha256': new['bundle_sha256'],
        'checkpoint_sha256': new['checkpoint_sha256'], 'inputs_unchanged': True, 'video_checks': shapes,
        'old_bus_projected_vertical_range_px': old_v['range'], 'grounded_bus_projected_vertical_range_px': new_v['range'],
        'old_bus_projected_vertical_second_difference_rms_px': old_v['second_difference_rms'],
        'grounded_bus_projected_vertical_second_difference_rms_px': new_v['second_difference_rms'],
        'metric_scope': 'rigid bus center geometric projection; not a full image-quality or flicker metric',
        'ground_registration': new['ground_registration']['policy'],
        'remaining_limitations': '3DGS appearance artifacts and unobserved surfaces are not repaired by geometric grounding'}
    (args.output/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
