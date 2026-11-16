"""Coordinate and time conversions shared by export and rendering."""
import numpy as np


def yaw_matrix(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def pose(x, y, z=0., yaw=0.):
    result = np.eye(4)
    result[:3, :3] = yaw_matrix(yaw)
    result[:3, 3] = (x, y, z)
    return result


def quaternion_matrix(q):
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError('Invalid wxyz quaternion')
    w, x, y, z = q / norm
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def matrix_quaternion(matrix):
    """Proper rotation -> unit quaternion in Omniscenegraph's wxyz order."""
    m = np.asarray(matrix, dtype=np.float64)
    trace = np.trace(m)
    if trace > 0:
        s = np.sqrt(trace + 1.) * 2
        q = np.array([s/4, (m[2, 1]-m[1, 2])/s, (m[0, 2]-m[2, 0])/s, (m[1, 0]-m[0, 1])/s])
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i+1) % 3, (i+2) % 3
        s = np.sqrt(max(0., 1+m[i, i]-m[j, j]-m[k, k])) * 2
        q = np.zeros(4)
        q[0] = (m[k, j]-m[j, k])/s
        q[i+1] = s/4
        q[j+1] = (m[j, i]+m[i, j])/s
        q[k+1] = (m[k, i]+m[i, k])/s
    return q / np.linalg.norm(q)


def interpolate_rotation(a, b, alpha):
    qa, qb = matrix_quaternion(a), matrix_quaternion(b)
    dot = float(qa @ qb)
    if dot < 0:
        qb, dot = -qb, -dot
    if dot > .9995:
        q = (1-alpha)*qa + alpha*qb
    else:
        angle = np.arccos(np.clip(dot, -1., 1.))
        q = (np.sin((1-alpha)*angle)*qa + np.sin(alpha*angle)*qb) / np.sin(angle)
    return quaternion_matrix(q)


def resample_poses(timestamps, poses, valid, targets):
    """Interpolate consecutive valid samples; never fill gaps or extrapolate.

    poses: [time, agent, 4, 4]. Invalid output poses are identity matrices.
    """
    timestamps = np.asarray(timestamps, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    poses, valid = np.asarray(poses), np.asarray(valid, dtype=bool)
    if timestamps.ndim != 1 or not len(timestamps) or np.any(np.diff(timestamps) <= 0):
        raise ValueError('Timestamps must be strictly increasing')
    if valid.ndim != 2 or poses.shape != (*valid.shape, 4, 4) or valid.shape[0] != len(timestamps):
        raise ValueError('Pose/validity dimensions disagree')
    output = np.broadcast_to(np.eye(4), (len(targets), valid.shape[1], 4, 4)).copy()
    mask = np.zeros((len(targets), valid.shape[1]), dtype=bool)
    for t, timestamp in enumerate(targets):
        right = int(np.searchsorted(timestamps, timestamp))
        if right < len(timestamps) and timestamps[right] == timestamp:
            mask[t] = valid[right]
            output[t, mask[t]] = poses[right, mask[t]]
        elif 0 < right < len(timestamps):
            left = right-1
            alpha = float(timestamp-timestamps[left]) / float(timestamps[right]-timestamps[left])
            mask[t] = valid[left] & valid[right]
            for agent in np.flatnonzero(mask[t]):
                output[t, agent, :3, 3] = (1-alpha)*poses[left, agent, :3, 3] + alpha*poses[right, agent, :3, 3]
                output[t, agent, :3, :3] = interpolate_rotation(poses[left, agent, :3, :3], poses[right, agent, :3, :3], alpha)
    return output, mask
