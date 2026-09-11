# Recorded experiment inputs

This directory preserves detector run results, synthetic scenario traces, fixed perception caches, matched-state records, dataset labels/manifests, selected model exports, environment records, and console logs.

Start with the repository root README and `docs/data_provenance.md`. Final downstream results are regenerated into the top-level `results/` directory by `scripts/evaluate.py`; the derived presentation/state summaries here are retained acquisition records, not interchangeable with those final results.

- `dataset/`: 554-image filename manifest, 5644 filtered labels, source archive checksums and selection script.
- `environment/`: exact acquisition software versions and static virtualised CPU capture.
- `models/`: supplied derived exports and source/export checksum manifest.
- `results/`: recorded E1/E2 benchmarks, matched object-frame rows, 32 traces, 32 perception caches, acquisition summaries and sweep records.
- `logs/`: console records of the same recorded experiment runs.

No physical automotive latency bound, participant measurements, calibrated risk probability, or independent metric depth validation follows from these inputs. Source photographs are not redistributed.
