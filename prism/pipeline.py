"""CPU perception, shared temporal state, cue selection, and image-space rendering."""
from __future__ import annotations
import math
import time
from .geometry import Camera
from .arbitration import Arbiter
from .replay import StateReplay
from . import render as R

COCO_NAMES={0:'person',1:'bicycle',2:'car',3:'motorcycle',5:'bus',7:'truck',9:'traffic light',11:'stop sign'}
KEEP=list(COCO_NAMES)

class Pipeline:
    def __init__(self, model, cam:Camera, fps=20, conf=.25, imgsz=640, arbiter=None, tracker='bytetrack.yaml'):
        self.model,self.cam,self.fps=model,cam,fps
        self.conf,self.imgsz,self.tracker=conf,imgsz,tracker
        self.arbiter=arbiter if arbiter is not None else Arbiter()
        self.state=StateReplay(cam);self.ages={};self.k=0;self.last_timestamp=None

    def reset(self):
        self.state=StateReplay(self.cam);self.ages.clear();self.arbiter.reset()
        self.k=0;self.last_timestamp=None
        if getattr(self.model,'predictor',None) is not None:
            for tracker in getattr(self.model.predictor,'trackers',[]):tracker.reset()

    def step(self,frame,timestamp=None):
        """Use acquisition timestamp for live input; omitted timestamp means offline nominal FPS."""
        t=self.k/self.fps if timestamp is None else float(timestamp)
        if not math.isfinite(t) or (self.last_timestamp is not None and t<=self.last_timestamp):
            raise ValueError('Frame timestamps must be finite and strictly increasing')
        self.last_timestamp=t;self.k+=1;timings={}
        start=time.perf_counter()
        result=self.model.track(frame,persist=True,verbose=False,conf=self.conf,imgsz=self.imgsz,
                                classes=KEEP,device='cpu',tracker=self.tracker)[0]
        timings['perception_track_ms']=(time.perf_counter()-start)*1000
        start=time.perf_counter();raw=[]
        for tid in list(self.ages):
            if t-self.ages[tid][1]>1:self.ages.pop(tid)
        boxes=result.boxes
        if boxes is not None and boxes.id is not None:
            xyxy=boxes.xyxy.cpu().numpy();ids=boxes.id.cpu().numpy().astype(int)
            classes=boxes.cls.cpu().numpy().astype(int);conf=boxes.conf.cpu().numpy()
            for box,tid,cls,p in zip(xyxy,ids,classes,conf):
                age=self.ages.get(int(tid),(0,t))[0]+1;self.ages[int(tid)]=(age,t)
                raw.append(dict(tid=int(tid),cls=COCO_NAMES.get(int(cls),'object'),conf=float(p),
                                box=tuple(float(v) for v in box),age=age))
        tracks=self.state.step(raw,t)
        timings['risk_ms']=(time.perf_counter()-start)*1000
        start=time.perf_counter();cues=self.arbiter.step(tracks,self.cam.width,self.cam.height)
        timings['arbitration_ms']=(time.perf_counter()-start)*1000
        return dict(t=t,tracks=tracks,cues=cues,timings=timings)

    def draw(self,frame,out):
        cues=out['cues'];top=max(cues,key=lambda c:c['risk'])['text'] if cues else 'no selected cues'
        budget=self.arbiter.cfg.ink_budget if hasattr(self.arbiter,'cfg') else .055
        load=sum(c.get('ink',0) for c in cues)/budget if budget else 0
        start=time.perf_counter()
        image=R.render_frame(frame,cues,n_objects=len(out['tracks']),load=load,top_text=top)
        out['timings']['render_ms']=(time.perf_counter()-start)*1000
        return image
