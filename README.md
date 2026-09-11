# PRISM

**Perception-to-Risk Interface for Situational Mediation** is a research prototype connecting object detection, temporal monocular state, heuristic priority, and selective image-aligned cues for vehicle interfaces.

This repository contains implementation, experiment records, tests, and reproducible evaluation. The manuscript and publication figures are intentionally not hosted here.

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

The complete suite has **17 tests**: 12 core tests, three replay/matching tests, and two OpenCV integration tests. Without OpenCV, the latter two are skipped explicitly. The pipeline test uses a mock detector and does not establish detector performance.

## What the evaluation computes

The fixed perception cache contains 2580 frames: eight source scenes under four correlated image conditions. Temporal depth, TTC, priority, quality, and cue selection are reconstructed from boxes, confidence, identities, and timestamps. Previously stored TTC or priority estimates are not reused for the final comparison.

The policies are PRISM, render-all, and budget-constrained confidence, risk, nearest-object, and TTC rankings. Three component ablations disable persistence, central attenuation, or the quality gate individually. Two matched-count comparisons are diagnostics supplied with PRISM's per-frame output count; they are not independent deployment policies.

All budget-constrained policies use a cue-count limit of five, a surrogate-cost limit of 0.055, identical detail-level rules, and box-overlap rejection. The cost limit is not a measured pixel-occupancy limit.

| Clear-condition policy | Mean cues | Reference displayed / eligible | End-to-end coverage |
|---|---:|---:|---:|
| PRISM | 2.06 | 213 / 325 | 65.5% |
| Risk budget | 4.55 | 218 / 325 | 67.1% |
| Confidence budget | 4.55 | 170 / 325 | 52.3% |
| Render all | 7.78 | 227 / 325 | 69.8% |

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

## Results and provenance

Run `scripts/evaluate.py` from any directory. It writes `results/final_summary.json`, `scene_metrics.json`, `frame_metrics.json`, `matched_states.json`, and `figure_cues.json`. Non-finite newly computed values are written as JSON null. `results/reference_summary.json` is the frozen regression reference; the evaluation does not overwrite it.

Recorded detector timings come from the virtualised two-core Xeon described in `data/experiment/environment/`. They are **not** timings of this checkout on your machine. The detector benchmark and INT8 records are retained for traceability. INT8 calibration overlaps the selected evaluation population; the accuracy comparison is exploratory.

See [the protocol](docs/protocol.md) and [data provenance](docs/data_provenance.md) for denominators, bootstrap units, matching limitations, and the distinction between recorded inference runs and current replay.

## Scope

This is an offline research prototype. Synthetic depth uses the same class-height assumptions as the estimator. Camera-image alignment does not establish physical head-up-display registration. No participant study or complete target-device capture-to-display timing is supplied. No claim of calibrated collision probability or validated on-road warning performance is made.

Third-party images and pretrained weights retain their own terms; see [source notices](docs/third_party.md). Source photographs and manuscript files are excluded from Git.
