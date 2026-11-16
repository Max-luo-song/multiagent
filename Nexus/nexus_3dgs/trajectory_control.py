"""Pair measured and generated planar motion under identical sensor calibration."""
import numpy as np
from .geometry import pose
from .replay import road_rotation


def planar(T):
    return pose(T[0,3],T[1,3],yaw=np.arctan2(T[1,0],T[0,0]))


def paired_cameras(measured_ego,extrinsic,predicted_ego,correction=None):
    """Preserve measured height/tilt and the same right-multiplied calibration.

    Only generated XY/yaw replace measured XY/yaw. No learned correction is
    allowed to appear on just one side of a paired comparison.
    """
    real=np.asarray(measured_ego)@np.asarray(extrinsic)
    if correction is not None:real=real@correction
    generated=planar(predicted_ego)@np.linalg.inv(planar(measured_ego))@real
    return real,generated


def recover_generated_ego(camera_world,camera_mount,yaw):
    """Saved grounded camera and its fixed mount retain the exact ego XY."""
    road=np.asarray(camera_world)@np.linalg.inv(camera_mount)
    return pose(road[0,3],road[1,3],yaw=yaw)


def register_measured_actor(annotation,mount,world_plane,flat=False):
    """Use measured XY/yaw with the same fixed asset mounting as Nexus.

    Height and tilt use the common reconstruction road assumption, rather than
    comparing unregistered annotation boxes with registered generated bodies.
    """
    T=np.eye(4);x,y=annotation[:2,3];yaw=np.arctan2(annotation[1,0],annotation[0,0])
    n=np.array([0.,0.,1.]) if flat else np.asarray(world_plane)[:3]
    T[:3,:3]=road_rotation(yaw,n)
    T[:3,3]=[x,y,0. if flat else -(world_plane[0]*x+world_plane[1]*y+world_plane[3])/world_plane[2]]
    return T@mount
