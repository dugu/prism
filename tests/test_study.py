"""Independent behavioral examples for extended comparators and metrics."""
import unittest
from prism.study import RiskSelector,temporal_metrics,make_policy
from prism.arbitration import Arbiter

def cue(tid,risk,box=(0,0,20,30)):
    return dict(tid=tid,risk=risk,uncertainty=.1,box=box)
class StudyTests(unittest.TestCase):
    def test_hysteresis_keeps_low_score_but_threshold_does_not(self):
        a=RiskSelector('threshold');b=RiskSelector('hysteresis')
        for p in [a,b]:self.assertEqual(len(p.step([cue(1,.3)],960,720)),1)
        self.assertEqual(a.step([cue(1,.1)],960,720),[])
        self.assertEqual(len(b.step([cue(1,.1)],960,720)),1)
        self.assertEqual(b.step([cue(1,.08)],960,720),[])
    def test_dwell_is_preemptible_by_urgent_object(self):
        p=RiskSelector('dwell',count=1)
        p.step([cue(1,.3)],960,720)
        selected=p.step([cue(1,.05),cue(2,.8,(100,100,140,150))],960,720)
        self.assertEqual([x['tid'] for x in selected],[2])
    def test_infeasible_urgent_not_admitted(self):
        p=RiskSelector('dwell',budget=.01)
        self.assertEqual(p.step([cue(1,.8)],960,720),[])
    def test_default_factory_matches_original_on_adversarial_history(self):
        a=Arbiter();b=make_policy('PRISM')
        for i in range(30):
            tracks=[cue(1,.35 if i<3 else .08),cue(2,.9 if i==5 else .25,(400,400,500,500))]
            self.assertEqual(a.step(tracks,960,720),b.step(tracks,960,720))
    def test_missing_and_switching_split_conditional_bouts(self):
        rows=[dict(eligible=e,reference_tid=t,hit=h) for e,t,h in
              [(True,1,False),(True,1,True),(True,None,False),(True,2,False),(True,2,False),(False,None,False)]]
        m=temporal_metrics(rows,20)
        self.assertEqual(m['bouts'],2);self.assertEqual(m['never_displayed_bouts'],1)
        self.assertEqual(m['conditional_first_display_delays_s'],[.05])
        self.assertEqual(m['longest_eligible_omission_s'],.15)
if __name__=='__main__':unittest.main()
