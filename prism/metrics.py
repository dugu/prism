"""Geometric matching and image-space occupancy utilities.

The final replay retains the recorded class-agnostic reference associations.
The matcher supports category gating for future independently annotated studies.
"""
import numpy as np
from .arbitration import _iou as iou

def match_tracks_to_gt(tracks, ground_truth, thr=.4, class_aware=True):
    pairs, used = [], set()
    for i in sorted(range(len(tracks)),key=lambda i:(-tracks[i]['conf'],tracks[i]['tid'])):
        best, index = thr, -1
        for j, target in enumerate(ground_truth):
            if j in used or (class_aware and tracks[i]['cls']!=target['cls']):continue
            overlap=iou(tracks[i]['box'],target['box'])
            if overlap>best:best,index=overlap,j
        if index>=0:used.add(index);pairs.append((i,index,best))
    return pairs

def changed_pixel_fraction(base,overlay,threshold=12):
    """Fraction whose summed absolute channel difference exceeds threshold."""
    if base.shape!=overlay.shape:raise ValueError('Image shapes must agree')
    return float((np.abs(base.astype(np.int16)-overlay.astype(np.int16)).sum(axis=2)>threshold).mean())
