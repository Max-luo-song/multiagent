"""Track bus rear texture in actual old/new rendered images (diagnostic ROI)."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def track(directory, roi):
    run = json.loads((directory/'run.json').read_text())
    records = [image for frame in run['frames'] for image in frame['images'] if image['camera_id'] == 0]
    images = [cv2.imread(str(directory/image['file']), cv2.IMREAD_GRAYSCALE) for image in records]
    mask = np.zeros_like(images[0]); x0, y0, x1, y1 = roi; mask[y0:y1, x0:x1] = 255
    points = cv2.goodFeaturesToTrack(images[0], maxCorners=200, qualityLevel=.005, minDistance=5, mask=mask)
    if points is None or len(points) < 15:
        raise ValueError('Insufficient initial texture tracks in the specified ROI')
    initial_count = len(points)
    counts, vertical = [], []
    for before, after in zip(images, images[1:]):
        forward, status, _ = cv2.calcOpticalFlowPyrLK(before, after, points, None,
            winSize=(21, 21), maxLevel=3)
        backward, reverse_status, _ = cv2.calcOpticalFlowPyrLK(after, before, forward, None,
            winSize=(21, 21), maxLevel=3)
        keep = (status[:, 0] == 1) & (reverse_status[:, 0] == 1) & (np.linalg.norm((backward-points)[:, 0], axis=-1) < 1.)
        if keep.sum() < 15:
            raise ValueError('Insufficient persistent bus texture tracks; no stability claim emitted')
        displacement = forward[keep, 0]-points[keep, 0]
        vertical.append(float(np.median(displacement[:, 1])))
        counts.append(int(keep.sum()))
        points = forward[keep]
    positions = np.r_[0., np.cumsum(vertical)]
    return {'initial_tracks': initial_count, 'minimum_persistent_tracks': min(counts),
        'median_texture_vertical_range_px': float(np.ptp(positions)),
        'vertical_frame_step_rms_px': float(np.sqrt(np.mean(np.array(vertical)**2))),
        'vertical_second_difference_rms_px': float(np.sqrt(np.mean(np.diff(vertical)**2))),
        'vertical_positions_px': positions.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old', type=Path, required=True)
    parser.add_argument('--new', type=Path, required=True)
    parser.add_argument('--roi', type=int, nargs=4, default=[390, 155, 620, 355])
    parser.add_argument('--label', default='bus rear')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(4)
    result = {'scope': f'forward/backward-consistent {args.label} texture tracking; not a full video flicker score',
        'roi_first_frame_xyxy': args.roi, 'old': track(args.old, args.roi), 'grounded': track(args.new, args.roi)}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('old', 'grounded')}, indent=2))
    for key in ('old', 'grounded'):
        print(key, {k:v for k,v in result[key].items() if k != 'vertical_positions_px'})


if __name__ == '__main__':
    main()
