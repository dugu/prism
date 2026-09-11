"""Projective object compositing over cropped/resized COCO source photographs.

Generative states use the same class-height priors as the estimator. They are
exact within the synthesis model, not independently measured physical truth.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
import numpy as np
import cv2

from .geometry import Camera, SIZE_PRIOR, DEFAULT_PRIOR

COCO_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus",
              7: "truck", 9: "traffic light", 11: "stop sign"}
VRU = {"person", "bicycle", "motorcycle"}


@dataclass
class ObjectState:
    oid: int
    cls: str
    Z0: float
    X0: float
    Hc: float           # height of the camera above this object's ground anchor
    H: float            # physical height (m)
    aspect: float
    patch: np.ndarray
    alpha: np.ndarray
    box0: tuple
    v_long: float = 0.0    # object speed along the ego axis (+ = same direction)
    v_lat: float = 0.0

    def state_at(self, t, v_ego):
        return self.Z0 - (v_ego - self.v_long) * t, self.X0 + self.v_lat * t


@dataclass
class Scenario:
    name: str
    title: str
    source_image: str
    v_ego: float
    duration: float
    fps: int
    degradation: str
    cam: Camera
    objects: list = field(default_factory=list)
    Zbg: float = 70.0
    primary: int = -1
    meta: dict = field(default_factory=dict)


def _feather(h, w, pad=0.08):
    a = np.ones((h, w), np.float32)
    py, px = max(1, int(h * pad)), max(1, int(w * pad))
    ry = np.linspace(0, 1, py, dtype=np.float32)
    rx = np.linspace(0, 1, px, dtype=np.float32)
    a[:py, :] *= ry[:, None]; a[-py:, :] *= ry[::-1][:, None]
    a[:, :px] *= rx[None, :]; a[:, -px:] *= rx[::-1][None, :]
    return a


def build_scenario(img_path, label_path, name, title, duration=None, fps=20,
                   degradation="none", motion=None, hfov=52.0, max_objects=14,
                   out_w=960, v_ego=None, end_ttc=0.7, primary_id=None):
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(img_path)
    # normalise every source frame to a common 4:3 windshield aspect ratio
    H0, W0 = img.shape[:2]
    ar = 4.0 / 3.0
    if W0 / H0 > ar:                                   # too wide -> crop sides
        cw = int(round(H0 * ar)); ox, oy, ch = (W0 - cw) // 2, 0, H0
    else:                                              # too tall -> crop top/bottom
        ch = int(round(W0 / ar)); cw, ox, oy = W0, 0, (H0 - ch) // 2
    img = img[oy:oy + ch, ox:ox + cw]
    scl = out_w / cw
    img = cv2.resize(img, (out_w, int(round(ch * scl))), interpolation=cv2.INTER_CUBIC)
    h, w = img.shape[:2]
    cam = Camera(w, h, hfov_deg=hfov)

    rows = [r.split() for r in open(label_path).read().strip().splitlines() if r.strip()]
    boxes = []
    for r in rows:
        c = int(r[0])
        xc, yc, bw, bh = (float(v) for v in r[1:5])
        bx1, by1 = ((xc - bw / 2) * W0 - ox) * scl, ((yc - bh / 2) * H0 - oy) * scl
        bx2, by2 = ((xc + bw / 2) * W0 - ox) * scl, ((yc + bh / 2) * H0 - oy) * scl
        bx1, by1 = max(bx1, 0.0), max(by1, 0.0)
        bx2, by2 = min(bx2, w - 1.0), min(by2, h - 1.0)
        if bx2 - bx1 < 6 or by2 - by1 < 6:
            continue
        boxes.append((c, bx1, by1, bx2, by2))
    boxes.sort(key=lambda b: -(b[4] - b[2]))
    boxes = boxes[:max_objects]

    mask = np.zeros((h, w), np.uint8)
    for _, x1, y1, x2, y2 in boxes:
        cv2.rectangle(mask, (int(x1) - 1, int(y1) - 1), (int(x2) + 1, int(y2) + 1),
                      255, -1)
    inpainted = cv2.inpaint(img, mask, 4, cv2.INPAINT_TELEA)

    motion = motion or {}
    objs = []
    for i, (c, x1, y1, x2, y2) in enumerate(boxes):
        cls = COCO_NAMES.get(c, "object")
        bh_px, bw_px = max(y2 - y1, 2.0), max(x2 - x1, 2.0)
        H = SIZE_PRIOR.get(cls, DEFAULT_PRIOR)[0]
        Z0 = cam.fx * H / bh_px
        if not (2.5 <= Z0 <= 200.0):
            continue
        X0 = ((x1 + x2) / 2 - cam.cx) * Z0 / cam.fx
        Hc = (y2 - cam.cy) * Z0 / cam.fx
        px1, py1 = int(max(x1, 0)), int(max(y1, 0))
        px2, py2 = int(min(x2, w)), int(min(y2, h))
        patch = img[py1:py2, px1:px2]
        if patch.size == 0 or patch.shape[0] < 4 or patch.shape[1] < 4:
            continue
        m = motion.get(i, {})
        objs.append(ObjectState(i, cls, Z0, X0, Hc, H, bw_px / bh_px, patch.copy(),
                                _feather(*patch.shape[:2]), (x1, y1, x2, y2),
                                m.get("v_long", 0.0), m.get("v_lat", 0.0)))

    # Designate the primary hazard: an explicit choice, otherwise the nearest
    # vulnerable road user inside the useful warning band.
    if primary_id is not None and any(o.oid == primary_id for o in objs):
        p = next(o for o in objs if o.oid == primary_id)
    else:
        def _pick(pool):
            band = [o for o in pool if 8.0 <= o.Z0 <= 70.0]
            return min(band or pool, key=lambda o: o.Z0) if pool else None
        dyn = [o for o in objs if o.cls not in ("traffic light", "stop sign")]
        p = _pick([o for o in dyn if o.cls in VRU]) or _pick(dyn) or _pick(objs)
    primary = p.oid if p is not None else -1
    v_ego = float(v_ego if v_ego is not None else 8.0)

    # Clip length: stop the clip when the primary hazard reaches `end_ttc`.
    if p is not None and duration is None:
        v_cl = max(v_ego - p.v_long, 0.3)
        duration = min(4.5, max(2.5, (p.Z0 - end_ttc * v_cl) / v_cl))
    duration = float(duration if duration is not None else 3.0)

    sc = Scenario(name, title, str(img_path), v_ego, duration, fps, degradation,
                  cam, objs, primary=primary)
    # background parallax: place the far plane so that the plate never zooms by
    # more than ~1.4x over the clip, which keeps interpolation artefacts small
    sc.Zbg = max(80.0, v_ego * duration / 0.28)
    sc._bg = inpainted
    sc._img = img
    sc.meta = {"width": w, "height": h, "duration_s": round(duration, 2),
               "frames": int(round(duration * fps)), "fps": fps, "hfov_deg": hfov, "fx_px": round(cam.fx, 1),
               "n_objects": len(objs), "v_ego_ms": round(v_ego, 2),
               "v_ego_kmh": round(v_ego * 3.6, 1),
               "primary_id": primary,
               "primary_cls": next((o.cls for o in objs if o.oid == primary), None),
               "primary_Z0": round(next((o.Z0 for o in objs if o.oid == primary), 0), 2)}
    return sc


# ----------------------------------------------------------------- degradations
def degrade(img, kind, rng):
    if kind == "none":
        return img
    f = img.astype(np.float32) / 255.0
    h, w = img.shape[:2]
    if kind == "night":
        f = np.clip(f, 0, 1) ** 2.1 * 0.52
        bright = cv2.GaussianBlur((f.mean(axis=2) > 0.30).astype(np.float32), (0, 0), 11)
        f = np.clip(f + 0.16 * bright[..., None], 0, 1)
        f = np.clip(f + rng.normal(0, 0.020, f.shape).astype(np.float32), 0, 1)
        f = np.clip(f * np.array([1.12, 0.98, 0.88], np.float32), 0, 1)
    elif kind == "fog":
        rows = np.linspace(0.30, 1.0, h, dtype=np.float32)[:, None]
        d = (1.0 - rows) * 60.0 + 6.0
        tr = np.exp(-0.032 * d)
        f = f * tr[..., None] + 0.80 * (1.0 - tr[..., None])
        f = cv2.GaussianBlur(f, (0, 0), 0.9)
    elif kind == "rain":
        f = f * 0.82 + 0.05
        layer = np.zeros((h, w), np.float32)
        for _ in range(int(w * h / 1200)):
            x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
            L = int(rng.integers(10, 26))
            cv2.line(layer, (x, y), (x - L // 4, y + L), 1.0, 1)
        layer = cv2.GaussianBlur(layer, (0, 0), 0.8)
        f = np.clip(f + 0.32 * layer[..., None], 0, 1)
        k = np.zeros((7, 7), np.float32); k[3, :] = 1 / 7.0
        f = cv2.filter2D(f, -1, k)
    return np.clip(f * 255.0, 0, 255).astype(np.uint8)


# --------------------------------------------------------------- frame synthesis
def _warp(img, s, cam):
    M = np.array([[s, 0, cam.cx * (1 - s)], [0, s, cam.cy * (1 - s)]], np.float32)
    return cv2.warpAffine(img, M, (cam.width, cam.height), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE), M


def render_scenario(sc: Scenario, seed=7, degradation=None):
    """Yield (frame_bgr, ground_truth) for each time step of the scenario."""
    rng = np.random.default_rng(seed)
    deg = sc.degradation if degradation is None else degradation
    cam = sc.cam
    h, w = cam.height, cam.width
    n = int(round(sc.duration * sc.fps))
    for k in range(n):
        t = k / sc.fps
        s_bg = sc.Zbg / max(sc.Zbg - sc.v_ego * t, 3.0)
        base, M = _warp(sc._img, s_bg, cam)
        plate, _ = _warp(sc._bg, s_bg, cam)

        placements, gt = [], []
        for o in sorted(sc.objects, key=lambda o: -o.state_at(t, sc.v_ego)[0]):
            Z, X = o.state_at(t, sc.v_ego)
            if Z < 2.0:
                continue
            hp = cam.fx * o.H / Z
            wp = hp * o.aspect
            if hp < 7 or hp > 3.2 * h or wp > 3.2 * w:
                continue
            u = cam.cx + cam.fx * X / Z
            vb = cam.cy + cam.fx * o.Hc / Z
            nb = (u - wp / 2, vb - hp, u + wp / 2, vb)
            ob = (o.box0[0] * s_bg + cam.cx * (1 - s_bg),
                  o.box0[1] * s_bg + cam.cy * (1 - s_bg),
                  o.box0[2] * s_bg + cam.cx * (1 - s_bg),
                  o.box0[3] * s_bg + cam.cy * (1 - s_bg))
            placements.append((o, nb, ob, Z, X))

        # erase the ghosts left where an object no longer covers its old footprint
        for o, nb, ob, Z, X in placements:
            ox1, oy1 = int(max(ob[0], 0)), int(max(ob[1], 0))
            ox2, oy2 = int(min(ob[2], w)), int(min(ob[3], h))
            if ox2 <= ox1 or oy2 <= oy1:
                continue
            m = np.ones((oy2 - oy1, ox2 - ox1), bool)
            nx1, ny1 = int(max(nb[0], ox1)), int(max(nb[1], oy1))
            nx2, ny2 = int(min(nb[2], ox2)), int(min(nb[3], oy2))
            if nx2 > nx1 and ny2 > ny1:
                m[ny1 - oy1:ny2 - oy1, nx1 - ox1:nx2 - ox1] = False
            reg = base[oy1:oy2, ox1:ox2]
            reg[m] = plate[oy1:oy2, ox1:ox2][m]

        for o, nb, ob, Z, X in placements:
            ix1, iy1 = int(round(nb[0])), int(round(nb[1]))
            pw = max(3, int(round(nb[2] - nb[0])))
            ph = max(3, int(round(nb[3] - nb[1])))
            ix2, iy2 = ix1 + pw, iy1 + ph
            if ix2 <= 0 or iy2 <= 0 or ix1 >= w or iy1 >= h:
                continue
            patch = cv2.resize(o.patch, (pw, ph), interpolation=cv2.INTER_LINEAR)
            alpha = cv2.resize(o.alpha, (pw, ph), interpolation=cv2.INTER_LINEAR)
            sx1, sy1 = max(ix1, 0), max(iy1, 0)
            sx2, sy2 = min(ix2, w), min(iy2, h)
            if sx2 <= sx1 or sy2 <= sy1:
                continue
            a = alpha[sy1 - iy1:sy2 - iy1, sx1 - ix1:sx2 - ix1][..., None]
            src = patch[sy1 - iy1:sy2 - iy1, sx1 - ix1:sx2 - ix1].astype(np.float32)
            dst = base[sy1:sy2, sx1:sx2].astype(np.float32)
            base[sy1:sy2, sx1:sx2] = (a * src + (1 - a) * dst).astype(np.uint8)
            v_close = sc.v_ego - o.v_long
            gt.append({"oid": o.oid, "cls": o.cls, "Z": Z, "X": X,
                       "ttc": Z / v_close if v_close > 0.05 else float("inf"),
                       "box": (float(sx1), float(sy1), float(sx2), float(sy2)),
                       "is_primary": o.oid == sc.primary})
        yield degrade(base, deg, rng), gt
