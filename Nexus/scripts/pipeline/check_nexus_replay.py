"""Check full rotation recovery independently over slopes, mounts and yaw wrap."""
import sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from nexus_3dgs.replay import recover_rotation,road_rotation
from nexus_3dgs.geometry import yaw_matrix,quaternion_matrix,matrix_quaternion


class ReplayChecks(unittest.TestCase):
    def test_grounded_body_yaw_recovers_full_sloped_rotation(self):
        rng=np.random.default_rng(19)
        for _ in range(100):
            n=np.r_[rng.uniform(-.04,.04,2),1.];n/=np.linalg.norm(n)
            axis=rng.normal(size=3);axis/=np.linalg.norm(axis);a=rng.uniform(-.08,.08)
            q=np.r_[np.cos(a/2),axis*np.sin(a/2)];mount=np.eye(4);mount[:3,:3]=quaternion_matrix(q)
            expected=road_rotation(rng.uniform(-np.pi,np.pi),n)@mount[:3,:3]
            actual=recover_rotation(np.arctan2(expected[1,0],expected[0,0]),mount,n)
            np.testing.assert_allclose(actual,expected,atol=1e-10)
            np.testing.assert_allclose(quaternion_matrix(matrix_quaternion(actual)),expected,atol=1e-10)

    def test_flat_track_retains_mount_tilt_outside_ground_coverage(self):
        n=np.array([.02,-.03,1.]);mount=np.eye(4);mount[:3,:3]=quaternion_matrix([.999,.02,-.01,.03])
        for theta in [-np.pi,-3.,-.001,0.,3.,np.pi]:
            expected=yaw_matrix(theta)@mount[:3,:3]
            recovered=recover_rotation(np.arctan2(expected[1,0],expected[0,0]),mount,n,flat=True)
            np.testing.assert_allclose(recovered,expected,atol=1e-10)

if __name__=='__main__':unittest.main()
