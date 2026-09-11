"""Heuristic priority and uncalibrated evidence-quality scoring for tracked objects."""
from __future__ import annotations
import math
from dataclasses import dataclass, field

# Vulnerability weight of each class: unprotected road users dominate.
VULNERABILITY = {
    "person": 1.00, "bicycle": 0.95, "motorcycle": 0.90,
    "car": 0.62, "truck": 0.66, "bus": 0.62,
    "traffic light": 0.30, "stop sign": 0.30,
}
DEFAULT_VULN = 0.5

TTC_CRIT = 1.5     # s - full urgency at or below this value
TTC_HORIZON = 6.0  # s - no urgency contribution beyond this value
D_CRIT = 8.0       # m - proximity term saturates below this range
D_HORIZON = 45.0   # m
CORRIDOR_HALF_W = 1.6   # m - half width of the ego travel corridor
CORRIDOR_SOFT = 1.5     # m - soft margin outside the corridor


@dataclass
class TrackState:
    tid: int
    cls: str
    conf: float
    box: tuple            # x1, y1, x2, y2 in pixels
    Z: float              # range, m
    sigma_Z: float
    X: float              # lateral offset, m
    ttc: float
    ttc_r2: float
    age: int
    conf_hist: list = field(default_factory=list)
    iou_hist: list = field(default_factory=list)


def _clip01(v):
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def urgency(ttc: float, Z: float) -> tuple:
    """Return (u_ttc, u_dist, u) where u is the dominant temporal/spatial term."""
    if math.isinf(ttc) or ttc <= 0:
        u_ttc = 0.0
    else:
        u_ttc = _clip01((TTC_HORIZON - ttc) / (TTC_HORIZON - TTC_CRIT))
    u_dist = _clip01((D_HORIZON - Z) / (D_HORIZON - D_CRIT))
    return u_ttc, u_dist, max(u_ttc, 0.55 * u_dist)


def corridor_factor(X: float, Z: float) -> float:
    """Membership of the ego travel corridor, softened with range."""
    half = CORRIDOR_HALF_W + 0.02 * Z          # corridor widens with lookahead error
    excess = max(abs(X) - half, 0.0)
    return math.exp(-(excess ** 2) / (2.0 * CORRIDOR_SOFT ** 2))


def risk_score(ts: TrackState) -> dict:
    v = VULNERABILITY.get(ts.cls, DEFAULT_VULN)
    u_ttc, u_dist, u = urgency(ts.ttc, ts.Z)
    c = corridor_factor(ts.X, ts.Z)
    # evidence term: a young or low-confidence track should not dominate the display
    ev = _clip01(0.45 + 0.55 * ts.conf) * _clip01(0.55 + 0.15 * min(ts.age, 3))
    r = v * u * (0.35 + 0.65 * c) * ev
    return {"risk": _clip01(r), "v": v, "u_ttc": u_ttc, "u_dist": u_dist,
            "u": u, "corridor": c, "evidence": ev}


def uncertainty(ts: TrackState) -> dict:
    """Aggregate uncalibrated evidence-quality index of the state estimate, in [0, 1]."""
    # 1. detection confidence
    u_conf = _clip01(1.0 - ts.conf)
    # 2. temporal instability of the confidence
    ch = ts.conf_hist[-8:]
    if len(ch) >= 3:
        m = sum(ch) / len(ch)
        sd = math.sqrt(sum((x - m) ** 2 for x in ch) / (len(ch) - 1))
        u_var = _clip01(sd / 0.20)
    else:
        u_var = 0.5
    # 3. geometric instability: mean IoU between successive boxes of the track
    ih = ts.iou_hist[-8:]
    u_geo = _clip01((0.92 - (sum(ih) / len(ih))) / 0.35) if ih else 0.5
    # 4. range uncertainty
    u_rng = _clip01((ts.sigma_Z / max(ts.Z, 1e-3)) / 0.30)
    # 5. quality of the looming fit driving the TTC
    u_ttc = _clip01(1.0 - ts.ttc_r2) if not math.isinf(ts.ttc) else 0.85
    # 6. track immaturity
    u_age = _clip01((5 - min(ts.age, 5)) / 5.0)
    w = (0.22, 0.14, 0.16, 0.18, 0.18, 0.12)
    parts = (u_conf, u_var, u_geo, u_rng, u_ttc, u_age)
    return {"uncertainty": _clip01(sum(a * b for a, b in zip(w, parts))),
            "u_conf": u_conf, "u_var": u_var, "u_geo": u_geo,
            "u_rng": u_rng, "u_ttcfit": u_ttc, "u_age": u_age}


def rationale(ts: TrackState, r: dict) -> str:
    """One-line natural-language justification shown next to a high-risk cue."""
    who = ts.cls.replace("traffic light", "signal")
    if r["corridor"] > 0.7:
        where = "in corridor"
    elif abs(ts.X) < 4.0:
        where = "near corridor"
    else:
        where = "off path"
    if not math.isinf(ts.ttc) and ts.ttc < TTC_HORIZON:
        when = f"{ts.ttc:.1f}s"
    else:
        when = f"{ts.Z:.0f}m"
    return f"{who}, {where}, {when}"
