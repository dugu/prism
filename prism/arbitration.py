"""Heuristic cue selection with hard surrogate-cost and count limits."""
from __future__ import annotations
from dataclasses import dataclass, field
import math

# Fixed surrogate cost for each detail level; not measured pixel occupancy
CUE_INK = {"full": 0.0165, "bracket": 0.0072, "tick": 0.0022}


@dataclass
class ArbiterConfig:
    ink_budget: float = 0.055      # maximum summed surrogate cost
    max_cues: int = 5              # max simultaneous cues
    r_on: float = 0.24             # risk needed to switch a cue on
    r_off: float = 0.15            # risk below which a shown cue is switched off
    dwell_frames: int = 8          # preemptible persistence preference (frames)
    hysteresis_bonus: float = 0.06
    foveal_box: tuple = (0.34, 0.40, 0.66, 0.78)   # protected road region (x1,y1,x2,y2)
    foveal_penalty: float = 0.45   # multiplier on value density inside the central image region
    overlap_iou: float = 0.55      # cue-level declutter threshold
    use_hysteresis: bool = True
    use_foveal: bool = True
    use_uncertainty_gate: bool = True
    u_gate: float = 0.72           # cues above this uncertainty are demoted


@dataclass
class Shown:
    tid: int
    since: int
    level: str


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class Arbiter:
    """Stateful frame-by-frame cue selector."""

    def __init__(self, cfg: ArbiterConfig | None = None):
        self.cfg = cfg or ArbiterConfig()
        if not math.isfinite(self.cfg.ink_budget) or self.cfg.ink_budget < 0 or self.cfg.max_cues < 0:
            raise ValueError("Budgets must be finite and non-negative")
        self.shown: dict[int, Shown] = {}
        self.frame = 0

    def reset(self):
        self.shown.clear()
        self.frame = 0

    def _level(self, risk: float, unc: float) -> str:
        c = self.cfg
        if c.use_uncertainty_gate and unc > c.u_gate and risk < 0.6:
            return "tick"
        if risk >= 0.55:
            return "full"
        if risk >= 0.30:
            return "bracket"
        return "tick"

    def step(self, cands: list[dict], w: int, h: int) -> list[dict]:
        """cands: dicts with tid, risk, uncertainty, box(px). Returns selected cues."""
        c = self.cfg
        self.frame += 1
        fx1, fy1, fx2, fy2 = (c.foveal_box[0] * w, c.foveal_box[1] * h,
                              c.foveal_box[2] * w, c.foveal_box[3] * h)
        area = float(w * h)

        scored = []
        seen = set()
        for d in cands:
            if d["tid"] in seen:
                raise ValueError("Duplicate track ID")
            seen.add(d["tid"])
            if not all(math.isfinite(d[k]) and 0 <= d[k] <= 1 for k in ("risk", "uncertainty")):
                continue
            bx = d["box"]
            if len(bx) != 4 or not all(math.isfinite(v) for v in bx) or bx[2] <= bx[0] or bx[3] <= bx[1]:
                continue
            prev = self.shown.get(d["tid"])
            bonus = c.hysteresis_bonus if (prev and c.use_hysteresis) else 0.0
            r_eff = min(1.0, d["risk"] + bonus)
            thr = c.r_off if (prev and c.use_hysteresis) else c.r_on
            locked = bool(prev and c.use_hysteresis and
                          self.frame - prev.since < c.dwell_frames)
            if r_eff < thr and not locked:
                continue
            level = self._level(d["risk"], d["uncertainty"])
            ink = CUE_INK[level]
            pen = 1.0
            if c.use_foveal:
                bx = d["box"]
                if _iou(bx, (fx1, fy1, fx2, fy2)) > 0.10 and d["risk"] < 0.55:
                    pen = c.foveal_penalty
            scored.append({**d, "r_eff": r_eff, "level": level, "ink": ink,
                           "priority": r_eff * pen, "locked": locked,
                           "norm_area": ((d["box"][2] - d["box"][0]) *
                                         (d["box"][3] - d["box"][1])) / area})

        # Urgent cues first, then dwell preference, then priority. Dwell never
        # overrides resource limits and may be preempted by urgent cues. Priority is
        # the effective risk, optionally attenuated inside the protected foveal
        # region; the ink cost enters only as a feasibility constraint, so that
        # a cheap low-risk cue can never outrank an expensive high-risk one.
        scored.sort(key=lambda d: (d["risk"] < 0.55, not d["locked"], -d["priority"], d["tid"]))

        sel, used_ink, boxes = [], 0.0, []
        for d in scored:
            if len(sel) >= c.max_cues:
                break
            if used_ink + d["ink"] > c.ink_budget + 1e-12:
                continue
            if any(_iou(d["box"], b) > c.overlap_iou for b in boxes):
                continue          # declutter: do not stack cues on the same region
            sel.append(d)
            boxes.append(d["box"])
            used_ink += d["ink"]

        new_shown = {}
        for d in sel:
            prev = self.shown.get(d["tid"])
            new_shown[d["tid"]] = Shown(d["tid"],
                                        prev.since if prev else self.frame,
                                        d["level"])
        self.shown = new_shown
        for d in sel:
            d["ink_used"] = used_ink
        return sel


class BaselineAll:
    """B1 - render every detection (the naive 'show everything' interface)."""
    name = "render-all"

    def reset(self):
        pass

    def step(self, cands, w, h):
        out = []
        for d in cands:
            out.append({**d, "level": "full", "ink": CUE_INK["full"],
                        "ink_used": len(cands) * CUE_INK["full"]})
        return out


class BaselineTopKConf:
    """B2 - confidence-ranked top-k, the common ADAS heuristic."""
    name = "top-k-conf"

    def __init__(self, k: int = 5):
        self.k = k

    def reset(self):
        pass

    def step(self, cands, w, h):
        s = sorted(cands, key=lambda d: -d["conf"])[: self.k]
        return [{**d, "level": "full", "ink": CUE_INK["full"],
                 "ink_used": len(s) * CUE_INK["full"]} for d in s]


class BaselineNearest:
    """B3 - proximity-ranked top-k, ignoring class and path membership."""
    name = "top-k-near"

    def __init__(self, k: int = 5):
        self.k = k

    def reset(self):
        pass

    def step(self, cands, w, h):
        s = sorted(cands, key=lambda d: d["Z"])[: self.k]
        return [{**d, "level": "full", "ink": CUE_INK["full"],
                 "ink_used": len(s) * CUE_INK["full"]} for d in s]
