"""Detector-independent integration tests; optional renderer checks require OpenCV."""
import unittest,sys,importlib.util,math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prism.geometry import Camera
from prism.replay import StateReplay,RankedBudget
from prism.metrics import match_tracks_to_gt

def track(tid=1,box=(20,20,80,140)):
    return dict(tid=tid,cls='person',conf=.8,box=box,age=5)
class ReplayTests(unittest.TestCase):
    def test_expired_identity_restarts_ttc(self):
        replay=StateReplay(Camera(960,720))
        for i in range(5):replay.step([track(box=(20,20,80,140+i))],i*.05)
        self.assertTrue(math.isfinite(replay.step([track(box=(20,20,80,146))],.25)[0]['ttc']))
        self.assertTrue(math.isinf(replay.step([track(box=(20,20,80,147))],2)[0]['ttc']))
    def test_budget_comparator_rejects_infeasible_full_cue(self):
        from prism.arbitration import ArbiterConfig
        tr={**track(),'risk':.9,'uncertainty':.1}
        self.assertEqual(RankedBudget('risk',ArbiterConfig(ink_budget=.01)).step([tr],960,720),[])
    def test_class_aware_match_rejects_category_error(self):
        gt=[dict(cls='car',box=(20,20,80,140))]
        self.assertEqual(match_tracks_to_gt([track()],gt),[])
        self.assertEqual(len(match_tracks_to_gt([track()],gt,class_aware=False)),1)

@unittest.skipUnless(importlib.util.find_spec('cv2'),'OpenCV required for renderer integration')
class RenderingTests(unittest.TestCase):
    def test_bracket_does_not_draw_label(self):
        from prism.render import render_frame
        frame=np.zeros((240,320,3),dtype=np.uint8)
        cue=dict(box=(80,80,140,180),risk=.4,uncertainty=.1,level='bracket',text='MUST NOT APPEAR')
        out=render_frame(frame,[cue],panel=False,ribbon=False)
        self.assertEqual(int(out[:60].sum()),0)
        self.assertGreater(int(out.sum()),0)
    def test_pipeline_uses_supplied_time_and_shared_state(self):
        from prism.pipeline import Pipeline
        class Array:
            def __init__(self,x):self.x=np.array(x)
            def cpu(self):return self
            def numpy(self):return self.x
        boxes=SimpleNamespace(xyxy=Array([[20,20,80,140]]),id=Array([1]),cls=Array([0]),conf=Array([.8]))
        model=SimpleNamespace(track=lambda *a,**kw:[SimpleNamespace(boxes=boxes)])
        pipe=Pipeline(model,Camera(320,240));frame=np.zeros((240,320,3),np.uint8)
        out=pipe.step(frame,timestamp=3.5)
        self.assertEqual(out['t'],3.5);self.assertEqual(out['tracks'][0]['age'],1)
        self.assertEqual(pipe.draw(frame,out).shape,frame.shape)
        with self.assertRaises(ValueError):pipe.step(frame,timestamp=3.5)
        pipe.reset();self.assertEqual(pipe.step(frame)['t'],0)
if __name__=='__main__':unittest.main()
