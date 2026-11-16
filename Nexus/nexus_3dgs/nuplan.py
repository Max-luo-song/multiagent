"""Read nuPlan annotation/calibration and locate unpacked sensor blobs."""
import json
from pathlib import Path
import pickle
import sqlite3

import numpy as np

from .geometry import pose, quaternion_matrix

CAMERAS = ('CAM_F0', 'CAM_L0', 'CAM_R0', 'CAM_L1', 'CAM_R1', 'CAM_L2', 'CAM_R2', 'CAM_B0')


def connect(database):
    path = Path(database).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    db = sqlite3.connect(path.as_uri()+'?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    return db


def ego_pose(row):
    result = np.eye(4)
    result[:3, :3] = quaternion_matrix([row['qw'], row['qx'], row['qy'], row['qz']])
    result[:3, 3] = [row['x'], row['y'], row['z']]
    return result


def ego_yaw(row):
    """Same intrinsic yaw as pyquaternion.yaw_pitch_roll used by nuPlan."""
    q = np.array([row['qw'], row['qx'], row['qy'], row['qz']], dtype=float)
    w, x, y, z = q/np.linalg.norm(q)
    return float(np.arctan2(2*(w*z-x*y), 1-2*(y*y+z*z)))


def lidar_frames(db, start_us, end_us):
    return [dict(r) for r in db.execute(
        'SELECT lp.token AS lidar_token, lp.timestamp AS lidar_timestamp, lp.filename, ep.* '
        'FROM lidar_pc lp JOIN ego_pose ep ON ep.token=lp.ego_pose_token '
        'WHERE lp.timestamp BETWEEN ? AND ? ORDER BY lp.timestamp, lp.token',
        (int(start_us), int(end_us)))]


def anchor_frame(db, token):
    row = db.execute('SELECT timestamp FROM lidar_pc WHERE token=?', (bytes.fromhex(token),)).fetchone()
    if row is None:
        raise ValueError(f'Scenario token not found: {token}')
    frames = lidar_frames(db, row['timestamp'], row['timestamp'])
    if len(frames) != 1:
        raise ValueError('Scenario does not identify a unique LiDAR frame')
    return frames[0]


def nearest_frames(db, targets, tolerance_us=35000):
    rows = lidar_frames(db, min(targets)-tolerance_us, max(targets)+tolerance_us)
    if not rows:
        raise ValueError('No LiDAR frames in requested interval')
    times = np.array([r['lidar_timestamp'] for r in rows], dtype=np.int64)
    result = []
    for target in targets:
        index = int(np.argmin(np.abs(times-int(target))))
        if abs(int(times[index])-int(target)) > tolerance_us:
            raise ValueError(f'No LiDAR frame within {tolerance_us} us of {target}')
        result.append(rows[index])
    if len({r['lidar_token'] for r in result}) != len(result):
        raise ValueError('Sample rate maps multiple frames to one LiDAR sample')
    return result


def boxes(db, lidar_token):
    rows = db.execute(
        'SELECT lb.*, c.name AS category_name FROM lidar_box lb '
        'JOIN track t ON t.token=lb.track_token JOIN category c ON c.token=t.category_token '
        'WHERE lb.lidar_pc_token=? ORDER BY lb.track_token', (lidar_token,))
    return [{**dict(row), 'track_token': row['track_token'].hex()} for row in rows]


def box_pose(row):
    return pose(row['x'], row['y'], row['z'], row['yaw'])


class SensorLocator:
    """Resolve direct, sensor_blobs/, and shared package/log/channel layouts.

    Only directory names are indexed; shared datasets are never copied/scanned
    recursively. Return None for missing data, never a fabricated placeholder.
    """
    def __init__(self, roots):
        self.roots = [Path(r).resolve() for r in roots]
        self.cache = {}

    def resolve(self, filename):
        relative = Path(filename)
        if relative.is_absolute() or '..' in relative.parts or not relative.parts:
            raise ValueError(f'Invalid sensor-relative path: {filename}')
        log = relative.parts[0]
        if log not in self.cache:
            candidates = []
            for root in self.roots:
                candidates.extend((root, root/'sensor_blobs'))
                if root.is_dir():
                    for entry in sorted(root.iterdir()):
                        if entry.name.startswith('nuplan-v1.1') and entry.is_dir():
                            candidates.extend((entry, entry/'sensor_blobs'))
            self.cache[log] = [p for p in candidates if (p/log).is_dir()]
        for parent in self.cache[log]:
            candidate = parent/relative
            if candidate.is_file():
                return candidate
        return None


def cameras(db, channels=CAMERAS):
    result = {}
    for row in db.execute('SELECT * FROM camera'):
        if row['channel'] not in channels:
            continue
        # Official nuPlan calibration columns are pickles in trusted local DBs.
        translation = np.asarray(pickle.loads(row['translation']), dtype=float)
        rotation = np.asarray(pickle.loads(row['rotation']), dtype=float)
        intrinsics = np.asarray(pickle.loads(row['intrinsic']), dtype=float)
        distortion = np.asarray(pickle.loads(row['distortion']), dtype=float)
        extrinsics = np.eye(4)
        extrinsics[:3, :3] = quaternion_matrix(rotation)
        extrinsics[:3, 3] = translation
        result[row['channel']] = {'token': row['token'], 'id': CAMERAS.index(row['channel']),
            'extrinsics': extrinsics, 'intrinsics': intrinsics, 'distortion': distortion,
            'width': row['width'], 'height': row['height']}
    missing = set(channels)-set(result)
    if missing:
        raise ValueError(f'Missing camera calibration: {sorted(missing)}')
    return result


def nearest_image(db, camera_token, timestamp, tolerance_us=50000):
    rows = [dict(r) for r in db.execute(
        'SELECT im.token AS image_token, im.filename_jpg, im.timestamp AS image_timestamp, ep.* '
        'FROM image im JOIN ego_pose ep ON ep.token=im.ego_pose_token '
        'WHERE im.camera_token=? AND im.timestamp BETWEEN ? AND ? ORDER BY im.timestamp',
        (camera_token, int(timestamp)-tolerance_us, int(timestamp)+tolerance_us))]
    if not rows:
        raise ValueError(f'No camera image within {tolerance_us} us of {timestamp}')
    return min(rows, key=lambda row: (abs(row['image_timestamp']-timestamp), row['image_timestamp']))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
