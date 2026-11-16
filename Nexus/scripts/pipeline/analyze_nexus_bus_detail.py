"""Compare bus detail against measured RGB, including untouched-image controls."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['evaluation', 'baseline', 'checkpoint', 'scene', 'output']:
        p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args()
    import cv2
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageFont
    torch.set_num_threads(8)
    cv2.setNumThreads(4)
    a.output.mkdir(parents=True, exist_ok=False)
    weights = torch.load(a.checkpoint, map_location='cpu', weights_only=False)['models']['RigidNodes']
    size = weights['instances_size'][6].numpy()
    evaluation = json.loads((a.evaluation / 'validation.json').read_text())
    poses = {r['frame']: r for r in evaluation['actual_camera_and_actor_poses']}
    intr = np.loadtxt(a.scene / 'intrinsics/0.txt')
    K = np.array([[intr[0]/2, 0, intr[2]/2], [0, intr[1]/2, intr[3]/2], [0, 0, 1]])
    signs = np.array([[x, y, z] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]])
    root = Path(__file__).resolve().parents[2]
    font = ImageFont.truetype(str(root / 'artifacts/nexus_3dgs/cache/fonts/NotoSansSC-Medium.ttf'), 22)
    frames = [r['frame'] for r in evaluation['rows']]
    selected = [f for f in [20, 55, 70, 80, 90, 95, 99] if f in frames]
    contact = Image.new('RGB', (1620, len(selected)*405), '#142136')
    rows = []
    for f in frames:
        photo = np.array(Image.open(a.evaluation / f'{f:03d}_photo.png'))
        old = np.array(Image.open(a.baseline / 'new' / f'{f:03d}_0.png'))
        new = np.array(Image.open(a.evaluation / 'new' / f'{f:03d}_0.png'))
        C = np.array(poses[f]['camera_world'])
        B = np.array(poses[f]['actors_world'][6])
        q = ((signs*size/2) @ B[:3, :3].T + B[:3, 3] - C[:3, 3]) @ C[:3, :3]
        assert q[:, 2].min() > .1
        uv = q[:, :2]/q[:, 2, None] @ K[:2, :2].T + K[:2, 2]
        mask = np.zeros((540, 960), np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(np.rint(uv).astype(np.int32)), 1)
        mask = cv2.erode(mask, np.ones((7, 7), np.uint8)).astype(bool)
        metrics = {}
        for name, im in [('v8', old), ('candidate', new)]:
            error = ((im.astype(float)-photo)/255)**2
            metrics[name] = {
                'bus_box_psnr': float(-10*np.log10(max(error[mask].mean(), 1e-12))),
                'full_psnr': float(-10*np.log10(max(error.mean(), 1e-12))),
            }
        oldmask = np.array(Image.open(a.baseline / f'{f:03d}_static_mask.png')) > 0
        newmask = np.array(Image.open(a.evaluation / f'{f:03d}_static_mask.png')) > 0
        quiet = oldmask & newmask
        delta = abs(new.astype(np.int16)-old.astype(np.int16))
        row = {'frame': f, 'metrics': metrics, 'bus_box_pixels': int(mask.sum()),
               'common_static_max_rgb_difference_8bit': int(delta[quiet].max()),
               'black_car_crop_max_rgb_difference_8bit': int(delta[215:335, 690:850].max())}
        rows.append(row)
        if f in selected:
            y = selected.index(f)*405
            for j, (label, im) in enumerate([('真实照片', photo), ('v8 基线', old), ('公交资产候选', new)]):
                ImageDraw.Draw(contact).text((j*540+10, y+5), f'帧 {f}  {label}', font=font, fill='white')
                crop = Image.fromarray(im).crop((300, 90, 660, 450)).resize((360, 360), Image.Resampling.BILINEAR)
                contact.paste(crop, (j*540+90, y+40))
    contact.save(a.output / 'bus_contact.jpg', quality=95)
    means = {name: {metric: float(np.mean([row['metrics'][name][metric] for row in rows]))
                    for metric in ['bus_box_psnr', 'full_psnr']} for name in ['v8', 'candidate']}
    report = {'status': 'complete', 'frames': frames, 'mean_per_frame_metrics': means,
              'metric_scope': 'eroded projected bus 3D-box silhouette; not an exact semantic mask; measured camera only',
              'common_static_max_rgb_difference_8bit': max(r['common_static_max_rgb_difference_8bit'] for r in rows),
              'black_car_crop_max_rgb_difference_8bit': max(r['black_car_crop_max_rgb_difference_8bit'] for r in rows),
              'rows': rows}
    (a.output / 'bus_detail_metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
