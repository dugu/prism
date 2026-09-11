"""State reconstruction and reproducible policy comparators for fixed detections."""
from dataclasses import replace
from .arbitration import Arbiter, ArbiterConfig, CUE_INK, _iou
from .geometry import Camera, TTCEstimator, range_from_height, lateral_offset
from .risk import TrackState, risk_score, uncertainty, rationale

class StateReplay:
    def __init__(self, camera):
        self.camera = camera
        self.ttc = TTCEstimator()
        self.history = {}

    def step(self, tracks, timestamp):
        # Expire before accepting reappearing identifiers.
        for tid in list(self.history):
            if timestamp-self.history[tid]['last'] > 1.0:
                del self.history[tid]
                self.ttc.hist.pop(tid, None)
        output = []
        for track in tracks:
            tid, box = track['tid'], track['box']
            h = self.history.setdefault(tid, dict(conf=[], iou=[], box=None, last=timestamp))
            h['conf'] = (h['conf']+[track['conf']])[-8:]
            if h['box'] is not None:
                h['iou'] = (h['iou']+[_iou(h['box'], box)])[-8:]
            h['box'], h['last'] = box, timestamp
            depth, sigma = range_from_height(track['cls'], box[3]-box[1], self.camera)
            offset = lateral_offset((box[0]+box[2])/2, depth, self.camera)
            self.ttc.update(tid, timestamp, box[3]-box[1])
            ttc, fit = self.ttc.ttc(tid)
            state = TrackState(tid, track['cls'], track['conf'], box, depth, sigma,
                               offset, ttc, fit, track['age'], h['conf'], h['iou'])
            score = risk_score(state)
            output.append({**track, 'Z':depth, 'X':offset, 'sigma_Z':sigma, 'ttc':ttc,
                           'ttc_r2':fit, 'risk':score['risk'],
                           'uncertainty':uncertainty(state)['uncertainty'],
                           'text':rationale(state,score)})
        return output

class RankedBudget:
    """Stateless ranking with PRISM's cue costs, overlap limit and capacities.

    No activation threshold or persistence; all candidates may compete.
    """
    def __init__(self, key, config=None):
        self.key = key
        self.helper = Arbiter(config or ArbiterConfig())
    def step(self, tracks, width, height):
        cfg = self.helper.cfg
        key = {'confidence':lambda t:-t['conf'], 'risk':lambda t:-t['risk'],
               'nearest':lambda t:t['Z'], 'ttc':lambda t:t['ttc']}[self.key]
        selected, used = [], 0.0
        for tr in sorted(tracks, key=lambda t:(key(t),t['tid'])):
            level = self.helper._level(tr['risk'], tr['uncertainty'])
            cost = CUE_INK[level]
            if len(selected) >= cfg.max_cues: break
            if used+cost > cfg.ink_budget+1e-12: continue
            if any(_iou(tr['box'], x['box']) > cfg.overlap_iou for x in selected): continue
            used += cost
            selected.append({**tr,'level':level,'ink':cost})
        return selected

def policy_set():
    return {'PRISM':Arbiter(),
            'Confidence budget':RankedBudget('confidence'),
            'Risk budget':RankedBudget('risk'),
            'Nearest budget':RankedBudget('nearest'),
            'TTC budget':RankedBudget('ttc'),
            'No persistence':Arbiter(ArbiterConfig(use_hysteresis=False)),
            'No centre attenuation':Arbiter(ArbiterConfig(use_foveal=False)),
            'No quality gate':Arbiter(ArbiterConfig(use_uncertainty_gate=False))}
