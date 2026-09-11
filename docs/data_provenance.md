# Data provenance

`data/experiment/` contains the supplied experiment records. SHA-256 values for the included input files are frozen in `data/input_sha256.json` and checked by `scripts/verify_inputs.py`. The detector data contain 14 validation runs and 16 backend runs. Console-emitted fields reconcile with their JSON counterparts; per-class AP is JSON-only and is not independently present in console run dictionaries.

The dataset manifest identifies 554 selected COCO val2017 images with 5644 traffic-category labels. It includes the selection script and download archive checksums. Source images are not in Git. The 32 perception caches contain 2580 frames from eight source photographs and four image conditions. The 12,222 matched rows are object-frame observations, not independent physical objects.

Files under `data/experiment/results/` also contain acquisition-era derived state and presentation summaries. These are retained as evidence, but the final study's downstream state and policy results are those generated into the top-level `results/` directory. In particular, acquisition-era search-time proxies and warning lead-time summaries are not final endpoints.

The supplied ONNX and INT8 exports are preserved with their manifests and hashes. Public pretrained weights and full COCO image archives were not downloaded or rehashed for the replay. INT8 calibration uses the same selected population as evaluation; the exact historical accuracy numbers are not a held-out assessment.

## Acquisition drivers

The subset and detector/backend scripts in `scripts/acquisition/` use `PRISM_WORK_DIR` (default: this repository's `work/`). They require downloaded images, model weights, and the inference dependencies in the supplied environment record. These drivers were inspected for provenance; detector/export runs were not repeated in preparing this checkout.

E2 requires an explicit existing `PRISM_CALIBRATION_YAML`, different from the evaluation YAML. Different YAML filenames do not prove disjoint images: verify image manifests before calibration. A disjoint calibration run intentionally differs from the historical overlapping-population experiment. Drivers write outputs and may replace export directories; use a separate work directory, never `data/experiment/`.

Static platform metadata and recorded wall-clock timing are available. Per-run power, thermal, or frequency telemetry is not available.
