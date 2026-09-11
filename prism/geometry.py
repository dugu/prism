"""M2b - monocular geometry: range, lateral offset and time-to-collision.

The proof-of-concept uses a single forward-facing camera, so metric quantities
are recovered from image measurements with class-conditional size priors.  All
estimates are returned together with a first-order uncertainty so that the
interface layer can encode what the perception stack does *not* know.
"""
from __future__ import annotations
import math
from dataclasses import dataclass

# Assumed physical height (m) and dispersion for each supported class.
# Engineering priors, not measured distributions or validated confidence bounds.
SIZE_PRIOR = {
    "person":        (1.70, 0.10),
    "bicycle":       (1.70, 0.12),
    "motorcycle":    (1.60, 0.15),
    "car":           (1.50, 0.15),
    "bus":           (3.20, 0.35),
    "truck":         (3.40, 0.55),
    "traffic light": (0.90, 0.20),
    "stop sign":     (0.75, 0.10),
}
DEFAULT_PRIOR = (1.60, 0.40)


@dataclass
class Camera:
    width: int
    height: int
    hfov_deg: float = 60.0
    pitch_deg: float = 0.0

    @property
    def fx(self) -> float:
        return self.width / (2.0 * math.tan(math.radians(self.hfov_deg) / 2.0))

    @property
    def cx(self) -> float:
        return self.width / 2.0

    @property
    def cy(self) -> float:
        return self.height / 2.0 + self.height * math.tan(math.radians(self.pitch_deg))


def range_from_height(cls_name: str, box_h_px: float, cam: Camera):
    """Return (Z, sigma_Z) in metres from the apparent height of a box."""
    H, sH = SIZE_PRIOR.get(cls_name, DEFAULT_PRIOR)
    box_h_px = max(box_h_px, 1e-3)
    Z = cam.fx * H / box_h_px
    # localisation noise on the box height, ~2 px plus 3 % of the height
    sh = 2.0 + 0.03 * box_h_px
    rel = math.sqrt((sH / H) ** 2 + (sh / box_h_px) ** 2)
    return Z, Z * rel


def lateral_offset(u_center: float, Z: float, cam: Camera) -> float:
    """Signed lateral distance (m) of the object from the ego longitudinal axis."""
    return (u_center - cam.cx) * Z / cam.fx


class TTCEstimator:
    """Time-to-collision from the looming (scale-expansion) cue.

    For a rigid object approached along the optical axis the apparent height h
    obeys h(t) ~ 1 / (Z0 - v t), hence d(log h)/dt = 1 / TTC.  A least-squares
    fit of reciprocal height is exactly linear under constant closing speed.
    TTC is -q(t_latest)/q_dot, where q=1/h, independent of absolute size.
    This revision replaces the original finite-window log-height fit.
    """

    def __init__(self, window: int = 9, min_samples: int = 4):
        self.window = window
        self.min_samples = min_samples
        self.hist: dict[int, list[tuple[float, float]]] = {}

    def update(self, track_id: int, t: float, box_h_px: float):
        buf = self.hist.setdefault(track_id, [])
        if not math.isfinite(t) or not math.isfinite(box_h_px) or box_h_px <= 0:
            raise ValueError("Timestamp and positive height must be finite")
        if buf and t <= buf[-1][0]:
            raise ValueError("Track timestamps must increase strictly")
        buf.append((t, 1.0 / box_h_px))
        if len(buf) > self.window:
            buf.pop(0)

    def ttc(self, track_id: int):
        """Return (ttc_seconds, r2). ttc is +inf when the object is not closing."""
        buf = self.hist.get(track_id, [])
        n = len(buf)
        if n < self.min_samples:
            return float("inf"), 0.0
        ts = [p[0] for p in buf]
        ys = [p[1] for p in buf]
        mt, my = sum(ts) / n, sum(ys) / n
        sxx = sum((t - mt) ** 2 for t in ts)
        if sxx <= 1e-9:
            return float("inf"), 0.0
        slope = sum((t - mt) * (y - my) for t, y in buf) / sxx
        ss_tot = sum((y - my) ** 2 for y in ys)
        ss_res = sum((y - (my + slope * (t - mt))) ** 2 for t, y in buf)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        if slope >= -1e-12:
            return float("inf"), max(r2, 0.0)
        latest_fit = my + slope * (ts[-1] - mt)
        if latest_fit <= 0:
            return float("inf"), max(r2, 0.0)
        return -latest_fit / slope, max(0.0, min(1.0, r2))

    def reset(self):
        self.hist.clear()
