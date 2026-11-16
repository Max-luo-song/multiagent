"""Behavior checks for the measured/generated trajectory experiment."""
import sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from nexus_3dgs.geometry import pose
from nexus_3dgs.trajectory_control import paired_cameras,recover_generated_ego,register_measured_actor
from nexus_3dgs.replay import road_rotation


class PairedTrajectoryChecks(unittest.TestCase):
    def test_identical_motion_keeps_sensor_camera_and_learned_calibration(self):
        ego=pose(100,200,3.,.7);tilt=pose(0,0);a=.08;tilt[:3,:3]=[[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]];ego=ego@tilt
        extrinsic=pose(1.6,-.1,1.7,.2);correction=pose(.003,.004,.002,.01)
        real,new=paired_cameras(ego,extrinsic,ego,correction)
        np.testing.assert_allclose(real,ego@extrinsic@correction,atol=1e-12)
        np.testing.assert_allclose(new,real,atol=1e-12)

    def test_planar_change_keeps_mount_height_tilt_and_exposure_correction(self):
        ego=pose(100,200,3.,.2);target=pose(103,204,999.,1.2);mount=pose(1.7,.1,1.8,.3);D=pose(.001,-.003,.002,.01)
        real,new=paired_cameras(ego,mount,target,D)
        expected=pose(103,204,3.,1.2)@mount@D
        np.testing.assert_allclose(new,expected,atol=1e-12)

    def test_generated_ego_recovered_from_camera_with_mount_offset(self):
        road=pose(365000.,143900.,25.,.4);road[:3,:3]=road_rotation(.4,np.array([.01,-.02,1.]));mount=pose(1.68,-.01,1.87,.02)
        recovered=recover_generated_ego(road@mount,mount,.4)
        np.testing.assert_allclose(recovered,pose(365000.,143900.,yaw=.4),atol=1e-10)

    def test_measured_actor_uses_same_slope_and_fixed_mount_as_generated(self):
        plane=np.array([-.02,.01,1.,-25.]);ann=pose(10.,20.,999.,.5);mount=pose(.02,.03,1.5,.01)
        actual=register_measured_actor(ann,mount,plane)
        road=pose(10,20,25.,.5);road[:3,:3]=road_rotation(.5,plane[:3])
        np.testing.assert_allclose(actual,road@mount,atol=1e-12)
        flat=register_measured_actor(ann,mount,plane,flat=True)
        np.testing.assert_allclose(flat,pose(10,20,yaw=.5)@mount,atol=1e-12)


if __name__=='__main__':unittest.main()
