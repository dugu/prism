# Geometry-to-selection diagnostic

This controlled diagnostic propagates the archived analytic geometry trajectories into the unchanged priority function and allocator. Reproduce from the repository root:

```sh
python scripts/selection_stress.py
```

The source is `results/extended/geometry_rows.jsonl.gz`; its SHA-256 is recorded in `selection_stress_summary.json`. The output includes `selection_stress_rows.jsonl.gz`, with every paired decision comparison, and the summary. This script does not run a detector or modify PRISM defaults.

Each pair has six person candidates. The focal candidate receives either true or estimated depth/TTC. Five fixed competitors have (depth in metres, TTC in seconds) of (12,2.5), (20,4), (28,5.5), (36,infinite), (44,infinite). Confidence is 0.9, age 10, X=0, and quality index 0.2 in both branches. Boxes are identical and disjoint: competitor i=0..4 uses (20+180i,100,60+180i,180); focal uses (450,400,490,480). These are scoring fixtures, not a physically consistent traffic scene. Both selectors have B=0.055; K is separately 1 or 5. Persistent histories evolve independently in the two branches.

Rank change compares the complete ordering by raw priority. Set change compares selected identities, excluding within-set order and representation changes. Extra admission and omission refer to the focal candidate relative to the true-geometry branch; they are not false-warning rates against human labels. Dropout removes the sample from both branches. Identity-reset effects enter only through the archived TTC, not changed selector identity or age. This isolates geometry substitution and does not measure total perception, quality or association errors.

Noise changes selected sets in 12.3% of observations at K=1 and 64.3% at K=5. Among 1718 finite TTC errors above 10 seconds, zero changes the selected set at K=1 and 970 changes it at K=5. Thus numerical error alone does not determine selection error. Results depend on the prescribed competitors and fixed evidence quality. All eight conditions are archived, including negative results. Noiseless repeats are identical trajectories, not independent evidence.
