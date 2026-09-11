# Evaluation protocol

## Inputs and state

Each cached frame provides timestamp, track identity, class, confidence, box, and age. `StateReplay` recomputes depth from box height, offset, reciprocal-height TTC, quality, and priority. All final policies receive the same reconstructed candidates. The live pipeline adapter shares this state implementation and accepts acquisition timestamps; omitted timestamps use nominal offline FPS.

The cached reference identity and reference score come from the generative state and the recorded confidence-ordered one-to-one matcher with strict IoU > 0.4, without category gating. The public metrics helper defaults to category-aware matching for future experiments, but the supplied experiment retains its recorded associations. Changing that association protocol would define a different experiment.

For state statistics, recorded matched rows are connected to cached observations by sequence, timestamp, confidence, and cached depth. The evaluation asserts uniqueness and complete linkage. Depth remains a longitudinal pinhole estimate with engineering class-size priors. TTC is a closing-depth-plane quantity, not a general collision prediction.

## Metrics

An eligible reference frame has generative priority >= 0.35. H counts eligible frames; D counts those with a matched reference detection; S counts those in which that detection is displayed. D/H, S/D and S/H are distinct quantities. The instantaneous reference can change identity; these statistics do not measure persistent hazard events.

Per-condition coverage pools frame counts. Cue counts and additions/removals per second first average within each sequence, then equally across eight source scenes. A transition counts each identity added or removed, including initial entry from an empty display; it is not a participant-measured flicker score. Track switches can contribute transitions.

TTC eligibility is finite positive generative TTC <= 4 seconds among matched object-frame observations. Both finite estimate availability and error conditional on availability are reported. Repeated tracks are not independent samples.

Bootstrap intervals resample paired source-scene differences, with seven eligible scenes in the clear-condition end-to-end comparisons, 10,000 draws and seed 20260911. They are exploratory, with limited power and external validity. No multiplicity-adjusted confirmatory claims are made.

## Comparators

Budget rankings consider all candidates in their ranking order, with B=0.055, K=5, PRISM's detail rules and overlap constraint. They have no PRISM activation or persistence mechanism. Equal capacity does not imply equal cue count or painted area.

The persistence ablation disables the lower continuation threshold, bonus and dwell preference as a group. Other ablations change only their named mechanism. Matched-count rankings are supplied PRISM's per-frame cue count; they are counterfactual diagnostics, not standalone operational policies.

## Execution boundary

The final replay runs no detector inference. Component benchmark results are recorded observations with their own environment. OpenCV tests verify rendering and a pipeline adapter with a mock detector. They are not camera-to-display performance measurements. Photographic scenario synthesis is available in `prism/scenario.py`, with its documented shared-prior and compositing limitations.
