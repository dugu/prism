"""Transparent comparison policies and temporal metrics for exploratory replay."""
from dataclasses import dataclass
from .arbitration import Arbiter, ArbiterConfig, CUE_INK, _iou

class RiskSelector:
    """Risk ranking with optional eligibility hysteresis and preemptible dwell.

    Same presentation mapping and geometric overlap/capacity checks as PRISM.
    Hysteresis alone affects eligibility. Dwell also adds persistence ordering
    and the specified bonus; neither uses central-image attenuation.
    """
    def __init__(self, mode='threshold', on=.24, budget=.055, count=5,
                 dwell=8, bonus=.06):
        if mode not in ('threshold','hysteresis','dwell'): raise ValueError(mode)
        self.mode,self.on,self.off=mode,on,max(0,on-.15)
        self.budget,self.count,self.dwell,self.bonus=budget,count,dwell,bonus
        self.helper=Arbiter();self.shown={};self.frame=0
    def step(self, tracks, width, height):
        self.frame+=1;scored=[]
        for t in tracks:
            prev=t['tid'] in self.shown
            locked=self.mode=='dwell' and prev and self.frame-self.shown[t['tid']]<self.dwell
            threshold=self.off if prev and self.mode!='threshold' else self.on
            if t['risk']<threshold and not locked:continue
            priority=min(1,t['risk']+(self.bonus if prev and self.mode=='dwell' else 0))
            level=self.helper._level(t['risk'],t['uncertainty'])
            scored.append(({**t,'level':level,'ink':CUE_INK[level]},locked,priority))
        scored.sort(key=lambda x:(x[0]['risk']<.55,not x[1],-x[2],x[0]['tid']))
        selected=[];used=0
        for t,_,_ in scored:
            if len(selected)>=self.count:break
            if used+t['ink']>self.budget+1e-12:continue
            if any(_iou(t['box'],s['box'])>.55 for s in selected):continue
            selected.append(t);used+=t['ink']
        self.shown={t['tid']:self.shown.get(t['tid'],self.frame) for t in selected}
        return selected

def make_policy(family,on=.24,budget=.055,count=5):
    if family=='PRISM':
        return Arbiter(ArbiterConfig(r_on=on,r_off=max(0,on-.15)+.06,
                                    ink_budget=budget,max_cues=count))
    return RiskSelector(family,on,budget,count)

def temporal_metrics(records,fps):
    """Bout = consecutive eligible detected frames with identical matched tid.

    Missing detection, ineligibility, or a different matched identity ends a bout.
    Delay is conditional on display within the bout. No collision-event meaning.
    """
    longest=run=0;bouts=[];current=[];last=None
    for r in records:
        run=run+1 if r['eligible'] and not r['hit'] else 0;longest=max(longest,run)
        tid=r['reference_tid'] if r['eligible'] else None
        if tid is None or tid!=last:
            if current:bouts.append(current)
            current=[]
        if tid is not None:current.append(bool(r['hit']))
        last=tid
    if current:bouts.append(current)
    delays=[b.index(True)/fps for b in bouts if any(b)]
    return dict(longest_eligible_omission_s=longest/fps,bouts=len(bouts),
                never_displayed_bouts=sum(not any(b) for b in bouts),
                conditional_first_display_delays_s=delays)
