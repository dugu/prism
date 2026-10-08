# Real-video geometry sensitivity (8 October 2026)

This replay uses all 1200 saved frames from the existing comma2k19 demonstration. It adds no new road sequences or independent relevance labels. Recorded estimates are the comparison baseline, not ground truth.

Run from the repository or extracted supplement root with Python 3.10 or later (standard library only):

```sh
python3 scripts/real_geometry_sensitivity.py
```

The control recomputes priority and asserts that every recorded cue identity and detail level is reproduced. Each perturbation starts a fresh allocator and evolves its own persistent history. Depth factors 0.8/1.2 scale both Z and X, preserving bearing; TTC factors scale finite values only. Joint scenarios apply both factors; the unavailable scenario removes all TTC. Boxes, classes, confidence, identities, age and quality are fixed. These are prescribed common perturbations, not measured error distributions; they do not jointly model detection, association and quality errors.

`real_geometry_summary.json` records the input SHA256 and integer aggregates. `real_geometry_rows.jsonl.gz` contains 9600 scenario/frame records with candidate-order changes, selected identity and representation changes, extra/omitted identities, cue counts and transitions. Percentages use all 1200 frames, including empty frames. Mean cue count is summed cue count / 1200. Identity changes ignore order and representation; representation changes compare identity/detail pairs. Transition counts include entry from the initially empty display. These metrics quantify sensitivity, not correctness, safety or generalizability.

| Perturbation | Priority order changed (%) | Selected identities changed (%) | Mean cues |
|---|---:|---:|---:|
| Recorded control | 0.0 | 0.0 | 0.780 |
| Depth × 0.8 | 60.9 | 61.8 | 1.630 |
| Depth × 1.2 | 57.0 | 19.8 | 0.572 |
| TTC × 0.8 | 43.8 | 26.0 | 1.046 |
| TTC × 1.2 | 38.9 | 19.4 | 0.578 |
| Joint × 0.8 | 65.5 | 76.6 | 1.957 |
| Joint × 1.2 | 63.0 | 30.5 | 0.435 |
| TTC unavailable | 85.7 | 51.1 | 0.132 |

Input: `demos/real_driving/recorded/frame_records.jsonl.gz`. Source attribution and license are in `demos/real_driving/source/`. No video pixels are needed for this replay. Outputs are written under `results/revision_20261008/`. Gzip container timestamps may differ on regeneration; compare decompressed JSONL and parsed summary JSON.
