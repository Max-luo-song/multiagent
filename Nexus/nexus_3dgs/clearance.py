"""Oriented physical vehicle-box clearance with rear-axle ego support."""
import numpy as np


def box_separation(a, a_size, b, b_size):
    """Largest SAT separation; positive means disjoint, zero means contact.

    Inputs broadcast, with poses represented as [center_x, center_y, yaw].
    Dimensions are physical length and width, independent of predicted size.
    """
    a, b = np.asarray(a), np.asarray(b)
    ah, bh = a[..., 2], b[..., 2]
    af = np.stack((np.cos(ah), np.sin(ah)), axis=-1)
    al = np.stack((-np.sin(ah), np.cos(ah)), axis=-1)
    bf = np.stack((np.cos(bh), np.sin(bh)), axis=-1)
    bl = np.stack((-np.sin(bh), np.cos(bh)), axis=-1)
    delta = b[..., :2]-a[..., :2]
    a_size, b_size = np.asarray(a_size)/2, np.asarray(b_size)/2
    separations = []
    for axis in (af, al, bf, bl):
        projection = np.abs((delta*axis).sum(axis=-1))
        ar = np.abs((af*axis).sum(axis=-1))*a_size[..., 0] + np.abs((al*axis).sum(axis=-1))*a_size[..., 1]
        br = np.abs((bf*axis).sum(axis=-1))*b_size[..., 0] + np.abs((bl*axis).sum(axis=-1))*b_size[..., 1]
        separations.append(projection-ar-br)
    return np.maximum.reduce(separations)


def rear_axle_to_box(poses_xy_yaw, offset):
    result = np.array(poses_xy_yaw, copy=True)
    result[..., :2] += offset*np.stack((np.cos(result[..., 2]), np.sin(result[..., 2])), axis=-1)
    return result
