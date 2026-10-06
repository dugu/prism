# PRISM

**Perception-to-Risk Interface for Situational Mediation** is a research prototype connecting object detection, temporal monocular state, heuristic priority, and selective image-aligned cues for vehicle interfaces.

This repository contains implementation, experiment records, tests, and reproducible evaluation.

## Real driving video

[![Watch PRISM on real driving video](demos/real_driving/assets/preview.jpg)](https://youtu.be/pXq9bM1Bdxk)

**[▶ Watch the demonstration on YouTube](https://youtu.be/pXq9bM1Bdxk)** · [Download the comparison MP4](https://github.com/dugu/prism/releases/download/real-driving-v1/PRISM_real_driving_comparison.mp4) · [PRISM overlay](https://github.com/dugu/prism/releases/download/real-driving-v1/PRISM_real_driving.mp4) · [Data and reproduction instructions](demos/real_driving/README.md)

A real 60-second comma2k19 recording is processed through YOLO11n, ByteTrack, temporal state, and PRISM. The demonstration preserves the default algorithm's behaviour, including false detections on the recording vehicle. It contains 1200 frames and compares identical observations across displays; it is not a labelled safety benchmark. Recorded per-frame tracks, decisions, camera metadata, timing summaries, and verification scripts are included in `demos/real_driving`.

## Reproduce the results

Python 3.12 is recommended. Cached evaluation requires NumPy only; it does not download images, load detector weights, or run inference.

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/verify_inputs.py
python -m unittest discover -s tests -v
python scripts/evaluate.py
python scripts/check_results.py
```

To run the two renderer/pipeline integration tests as well, install the rendering dependencies:

```sh
pip install -r requirements-render.txt
python -m unittest discover -s tests -v
```

The complete suite has **22 tests**: 12 core tests, three replay/matching tests, five comparator/temporal-metric tests, and two OpenCV integration tests. Without OpenCV, the latter two are skipped explicitly. The pipeline test uses a mock detector and does not establish detector performance.

## What the evaluation computes

The fixed perception cache contains 2580 frames: eight source scenes under four correlated image conditions. Temporal depth, TTC, priority, quality, and cue selection are reconstructed from boxes, confidence, identities, and timestamps.

The policies are PRISM, render-all, and budget-constrained confidence, risk, nearest-object, and TTC rankings. Three component ablations disable persistence, central attenuation, or the quality gate individually. Two matched-count comparisons are diagnostics supplied with PRISM's per-frame output count; they are not independent deployment policies.

All budget-constrained policies use a cue-count limit of five, a surrogate-cost limit of 0.055, identical detail-level rules, and box-overlap rejection. The cost limit is not a measured pixel-occupancy limit.

| Clear-condition policy | Mean cues | Reference shown / eligible | Transitions/s | Painted pixels % |
|---|---:|---:|---:|---:|
| Risk budget | 4.55 | 218 / 325 | 18.80 | 1.210 |
| Risk + threshold | 1.73 | 174 / 325 | 4.97 | 0.814 |
| Risk + hysteresis | 2.03 | 213 / 325 | 3.68 | 0.867 |
| Risk + dwell | 2.06 | 213 / 325 | 3.24 | 0.868 |
| PRISM | 2.06 | 213 / 325 | 3.46 | 0.846 |

Persistence explains much of the change. Simpler dwell ranking matches clear-scene retention and cue count with fewer transitions; PRISM paints slightly less area. A 360-configuration exploratory study finds preference-dependent trade-offs, not consistent superiority. See [extended evaluation](results/extended/README.md) for grouped scene selection, temporal omissions, occupancy and geometry stress tests.

A [paired geometry-to-selection diagnostic](results/revision_20261006/README.md) substitutes true versus estimated geometry while holding other candidate attributes fixed. It measures selection consistency under prescribed competition, not road-hazard accuracy.

The reference is the largest heuristic priority on generative states, with eligibility threshold 0.35. Coverage includes missed detections; it is not independently labelled collision-hazard recall. Finite TTC estimates have a median error of 0.455 s for 840 of 1449 eligible matched observations (58.0% availability).

## Layout

- `prism/`: geometry, scoring, allocation, shared temporal replay, pipeline adapter, renderer, and evaluation utilities.
- `scripts/evaluate.py`: deterministic final state/policy evaluation and scene bootstrap.
- `scripts/verify_inputs.py`: input hashes, dataset counts, model export checksums, and recorded-run consistency.
- `scripts/check_results.py`: numerical regression against the frozen final summary.
- `scripts/acquisition/`: detector/subset/backend drivers; require original inference dependencies, images, and weights.
- `tests/`: deterministic unit and integration tests.
- `data/experiment/`: supplied raw results, traces, cached detections, labels, manifests, environment capture, selected model exports, and logs.
- `results/`: final summary, per-scene/per-frame results, matched-state observations, and local test logs.
- `docs/`: protocol, provenance, and dependency/source notices.
- `demos/real_driving/`: real-video records, source metadata, generation and verification scripts.

## Results and provenance

Run `scripts/evaluate.py` from any directory. It writes `results/final_summary.json`, `scene_metrics.json`, `frame_metrics.json`, `matched_states.json`, and `figure_cues.json`. Non-finite newly computed values are written as JSON null. `results/reference_summary.json` is the frozen regression reference; the evaluation does not overwrite it.

Recorded detector timings come from the virtualised two-core Xeon described in `data/experiment/environment/`. They are **not** timings of this checkout on your machine. The detector benchmark and INT8 records are retained for traceability. INT8 calibration overlaps the selected evaluation population; the accuracy comparison is exploratory.

See [the protocol](docs/protocol.md) and [data provenance](docs/data_provenance.md) for denominators, bootstrap units, matching limitations, and the distinction between recorded inference runs and current replay.

## Scope

This is an offline research prototype. Synthetic depth uses the same class-height assumptions as the estimator. Camera-image alignment does not establish physical head-up-display registration. No participant study or complete target-device capture-to-display timing is supplied. No claim of calibrated collision probability or validated on-road warning performance is made.

Third-party images and pretrained weights retain their own terms; see [source notices](docs/third_party.md). Original media are referenced through source URLs and checksums.
