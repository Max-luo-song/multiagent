"""Checks that 3D stabilization preserves motion, locks and visibility gaps."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from nexus_3dgs.motion_stability import bounded_curve_correction,regularize_observed_bodies
from nexus_3dgs.geometry import pose


class MotionStabilityChecks(unittest.TestCase):
    def test_constant_velocity_is_not_stopped(self):
        y=np.linspace(3,12,31);np.testing.assert_allclose(bounded_curve_correction(y),y,atol=1e-12)
    def test_locks_and_trust_region(self):
        y=np.zeros(21);y[10]=1;fixed=np.zeros(21,bool);fixed[[0,5,-1]]=True
        out=bounded_curve_correction(y,20,.05,fixed)
        np.testing.assert_array_equal(out[fixed],y[fixed]);self.assertLessEqual(abs(out-y).max(),.05000001);self.assertLess(np.linalg.norm(np.diff(out,2)),np.linalg.norm(np.diff(y,2)))
    def test_real_turn_preserved_and_gaps_not_bridged(self):
        ts=np.arange(31)*.1;P=np.array([[pose(5*np.sin(.1*t),5*(1-np.cos(.1*t)),yaw=.1*t)] for t in ts]);valid=np.ones((31,1),bool);valid[15]=False
        out,_=regularize_observed_bodies(P,valid)
        np.testing.assert_array_equal(out[15],P[15]);np.testing.assert_array_equal(out[[0,14,16,30]],P[[0,14,16,30]])
        self.assertGreater(np.linalg.norm(out[-1,0,:2,3]-out[0,0,:2,3]),1.)
        self.assertLess(np.max(np.linalg.norm(out-P,axis=(-2,-1))),.02)
    def test_lateral_noise_reduced_with_longitudinal_and_height_exact(self):
        P=np.array([[pose(.2*i,.04*(-1)**i,z=.4,yaw=np.deg2rad(.4*(-1)**i))] for i in range(32)]);v=np.ones((32,1),bool)
        out,records=regularize_observed_bodies(P,v)
        np.testing.assert_allclose(out[:,0,0,3],P[:,0,0,3],atol=1e-4);np.testing.assert_array_equal(out[:,:,2,3],P[:,:,2,3]);self.assertLess(np.std(out[3:-3,0,1,3]),np.std(P[3:-3,0,1,3])*.3);self.assertTrue(records)


if __name__=='__main__':unittest.main()
