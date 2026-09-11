"""Deterministic software checks; these are not road or participant experiments."""
import unittest, math, sys, random
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prism.arbitration import Arbiter, ArbiterConfig
from prism.geometry import Camera, TTCEstimator, range_from_height
from prism.risk import TrackState, risk_score, rationale

def candidate(i,r=.4,u=.1):
    return dict(tid=i,risk=r,uncertainty=u,box=(10+100*i,10,50+100*i,90))

class CoreTests(unittest.TestCase):
    def test_dwell_cannot_bypass_ink(self):
        a=Arbiter(ArbiterConfig(ink_budget=.01,use_foveal=False))
        self.assertEqual(len(a.step([candidate(1)],960,720)),1)
        selected=a.step([candidate(1,.8)],960,720)
        self.assertLessEqual(sum(c['ink'] for c in selected),.01)
    def test_urgent_preempts_dwell(self):
        a=Arbiter(ArbiterConfig(max_cues=1,use_foveal=False))
        a.step([candidate(1)],960,720)
        self.assertEqual([c['tid'] for c in a.step([candidate(1),candidate(2,.9)],960,720)],[2])
    def test_invalid_candidate_suppressed(self):
        self.assertEqual(Arbiter().step([candidate(1,float('nan'))],960,720),[])
    def test_duplicate_id_rejected(self):
        with self.assertRaises(ValueError): Arbiter().step([candidate(1),candidate(1)],960,720)
    def test_budget_invariants_seeded(self):
        rng=random.Random(20260910)
        for budget in [.0,.01,.055]:
            a=Arbiter(ArbiterConfig(ink_budget=budget,max_cues=5))
            for _ in range(200):
                out=a.step([candidate(i,rng.random(),rng.random()) for i in range(9)],960,720)
                self.assertLessEqual(len(out),5)
                self.assertLessEqual(sum(x['ink'] for x in out),budget+1e-12)
    def test_constant_velocity_ttc(self):
        for z,v in [(20,5),(10,5),(50,2)]:
            a=TTCEstimator()
            for i in range(9):
                t=i*.05;a.update(1,t,1000/(z-v*t))
            self.assertAlmostEqual(a.ttc(1)[0],z/v-.4,places=10)
    def test_nonclosing_unavailable(self):
        for direction in [0,1]:
            a=TTCEstimator()
            for i in range(9): a.update(1,i*.05,1000/(20+direction*i*.05))
            self.assertTrue(math.isinf(a.ttc(1)[0]))
    def test_timestamps_and_height_validated(self):
        a=TTCEstimator();a.update(1,.1,10)
        with self.assertRaises(ValueError): a.update(1,.1,11)
        with self.assertRaises(ValueError): a.update(1,.2,0)
    def test_range_prior_sensitivity(self):
        cam=Camera(960,720)
        h=cam.fx*1.2/20
        self.assertAlmostEqual(range_from_height('person',h,cam)[0]/20,1.7/1.2)
    def test_risk_bounded(self):
        for z in [1,8,20,45,100]:
            for t in [float('inf'),1,3,6]:
                s=TrackState(1,'person',.5,(0,0,10,10),z,2,0,t,.9,5)
                r=risk_score(s)['risk'];self.assertGreaterEqual(r,0);self.assertLessEqual(r,1)
    def test_description_does_not_invent_lateral_motion(self):
        s=TrackState(1,'person',.9,(0,0,10,10),10,2,3.5,3,.9,5)
        self.assertNotIn('entering',rationale(s,{'corridor':.3}))
    def test_reset_clears_state(self):
        a=Arbiter();a.step([candidate(1)],960,720);a.reset()
        self.assertEqual(a.shown,{});self.assertEqual(a.frame,0)

if __name__=='__main__': unittest.main(verbosity=2)
