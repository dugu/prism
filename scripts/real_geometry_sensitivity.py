"""Deterministic geometry sensitivity on saved real-video candidates, not accuracy.

Scenarios are prescribed perturbations, not empirically calibrated error draws.
All candidates are perturbed together; boxes, confidence, identities, ages and
quality remain fixed. Each branch evolves its own allocation history.
"""
from pathlib import Path
import gzip
import hashlib
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prism.arbitration import Arbiter
from prism.risk import TrackState, risk_score

SOURCE = ROOT / 'demos/real_driving/recorded/frame_records.jsonl.gz'
OUT = ROOT / 'results/revision_20261008'
SCENARIOS = {
    'recorded': (1.0, 1.0, False),
    'depth_0.8': (0.8, 1.0, False),
    'depth_1.2': (1.2, 1.0, False),
    'ttc_0.8': (1.0, 0.8, False),
    'ttc_1.2': (1.0, 1.2, False),
    'joint_0.8': (0.8, 0.8, False),
    'joint_1.2': (1.2, 1.2, False),
    'ttc_unavailable': (1.0, 1.0, True),
}


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    with gzip.open(SOURCE, 'rt') as f:
        frames = [json.loads(line) for line in f]
    records = []
    for name, (depth_scale, ttc_scale, remove_ttc) in SCENARIOS.items():
        arbiter = Arbiter()
        previous = set()
        for frame, row in enumerate(frames):
            tracks = []
            for saved in row['tracks']:
                t = dict(saved)
                # A scale change in assumed physical height scales Z and X together.
                t['Z'] *= depth_scale
                t['X'] *= depth_scale
                t['ttc'] = math.inf if remove_ttc or t['ttc'] is None else t['ttc'] * ttc_scale
                state = TrackState(t['tid'], t['cls'], t['conf'], t['box'],
                                   t['Z'], t['sigma_Z'], t['X'], t['ttc'],
                                   t['ttc_r2'], t['age'])
                t['risk'] = risk_score(state)['risk']
                if name == 'recorded':
                    assert math.isclose(t['risk'], saved['risk'], abs_tol=1e-12)
                tracks.append(t)
            cues = arbiter.step(tracks, 1164, 874)
            ids = {t['tid'] for t in cues}
            baseline = {t['tid'] for t in row['cues']}
            levels = sorted((t['tid'], t['level']) for t in cues)
            baseline_levels = sorted((t['tid'], t['level']) for t in row['cues'])
            order = [t['tid'] for t in sorted(tracks, key=lambda t: (-t['risk'], t['tid']))]
            baseline_order = [t['tid'] for t in sorted(row['tracks'], key=lambda t: (-t['risk'], t['tid']))]
            if name == 'recorded':
                assert levels == baseline_levels
            records.append(dict(scenario=name, frame=frame, t=row['t'],
                rank_changed=order != baseline_order, set_changed=ids != baseline,
                representation_changed=levels != baseline_levels,
                selected_ids=sorted(ids), selected_levels=levels,
                extra_ids=sorted(ids-baseline), omitted_ids=sorted(baseline-ids),
                cue_count=len(ids), transitions=len(ids ^ previous)))
            previous = ids
    summary = []
    for name in SCENARIOS:
        rr = [r for r in records if r['scenario'] == name]
        summary.append(dict(scenario=name, frames=len(rr),
            **{key: sum(r[key] for r in rr) for key in
               ('rank_changed', 'set_changed', 'representation_changed', 'cue_count', 'transitions')},
            extra_cue_frames=sum(len(r['extra_ids']) for r in rr),
            omitted_cue_frames=sum(len(r['omitted_ids']) for r in rr)))
    payload = dict(source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                   scenarios=SCENARIOS, summary=summary,
                   scope='Sensitivity to prescribed common scaling or removal of geometric inputs on one unlabelled video. Recorded estimates are not ground truth. No empirical error probabilities, safety, accuracy or generalisation claims.')
    with gzip.open(OUT/'real_geometry_rows.jsonl.gz', 'wt') as f:
        for row in records:
            f.write(json.dumps(row, allow_nan=False)+'\n')
    (OUT/'real_geometry_summary.json').write_text(json.dumps(payload, indent=2)+'\n')
    for r in summary:
        print(r['scenario'], 'rank/set/representation %:',
              *[round(100*r[k]/r['frames'], 2) for k in ('rank_changed','set_changed','representation_changed')],
              'cues:', round(r['cue_count']/r['frames'], 3))
    print(f'Unmodified branch reproduces all {len(frames)} saved identity and detail decisions.')


if __name__ == '__main__':
    run()
