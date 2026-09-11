"""E2 - edge deployment study.

The same detector is exported to three inference backends that are representative
of what is actually deployed on automotive edge hardware, and each is measured
for latency (on the scenario imagery) and for accuracy (on the COCO-Traffic
subset).  This quantifies the cost of the runtime optimisations that make
on-board inference feasible.
"""
from __future__ import annotations
import json, time, statistics, sys
from pathlib import Path
import os
WORK = Path(os.environ.get("PRISM_WORK_DIR", str(Path(__file__).resolve().parents[2] / "work")))
import numpy as np
import cv2
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))

MODELS = Path(str(WORK) + "/models")
OUT = Path(str(WORK) + "/results"); OUT.mkdir(parents=True, exist_ok=True)
DATA = str(WORK) + "/data/coco_traffic/coco_traffic.yaml"
IMGS = sorted(Path(str(WORK) + "/data/coco_traffic/images").glob("*.jpg"))
TRAFFIC = [0, 1, 2, 3, 5, 7, 9, 11]
N_WARM, N_TIME = 8, 60


def timed(model, frames, imgsz):
    lat = []
    for f in frames[:N_WARM]:
        model.predict(f, imgsz=imgsz, verbose=False, device="cpu", classes=TRAFFIC)
    for f in frames[:N_TIME]:
        t0 = time.perf_counter()
        model.predict(f, imgsz=imgsz, verbose=False, device="cpu", classes=TRAFFIC)
        lat.append((time.perf_counter() - t0) * 1e3)
    lat.sort()
    return {"mean_ms": round(statistics.mean(lat), 2),
            "median_ms": round(statistics.median(lat), 2),
            "p95_ms": round(lat[int(0.95 * (len(lat) - 1))], 2),
            "fps_mean": round(1000.0 / statistics.mean(lat), 2)}


def accuracy(model, imgsz):
    r = model.val(data=DATA, imgsz=imgsz, batch=1, device="cpu", classes=TRAFFIC,
                  conf=0.001, iou=0.7, plots=False, verbose=False,
                  project=str(OUT / "val_e2"), name=f"tmp{time.time_ns()}",
                  exist_ok=True)
    return {"mAP50": round(float(r.box.map50), 4),
            "mAP50_95": round(float(r.box.map), 4)}


def main():
    calibration_yaml = os.environ.get("PRISM_CALIBRATION_YAML")
    if not calibration_yaml or not Path(calibration_yaml).is_file():
        raise ValueError("Set PRISM_CALIBRATION_YAML to an existing disjoint calibration dataset YAML before running E2.")
    if Path(calibration_yaml).resolve() == Path(DATA).resolve():
        raise ValueError("Calibration YAML must differ from evaluation YAML; also verify the actual image manifests are disjoint.")
    frames = [cv2.imread(str(p)) for p in IMGS[:N_TIME + N_WARM]]
    rows = []
    for base in ["yolo11n", "yolov8n"]:
        for imgsz in (640, 416):
            # ---- PyTorch FP32 reference
            m = YOLO(str(MODELS / f"{base}.pt"))
            rows.append({"model": base, "imgsz": imgsz, "backend": "PyTorch FP32",
                         **timed(m, frames, imgsz), **accuracy(m, imgsz)})
            print(rows[-1], flush=True)

            # ---- ONNX Runtime FP32
            try:
                p = YOLO(str(MODELS / f"{base}.pt")).export(
                    format="onnx", imgsz=imgsz, opset=17, simplify=False,
                    dynamic=False, device="cpu")
                p2 = MODELS / f"{base}_{imgsz}.onnx"
                Path(p).replace(p2)
                mo = YOLO(str(p2), task="detect")
                rows.append({"model": base, "imgsz": imgsz, "backend": "ONNX Runtime FP32",
                             **timed(mo, frames, imgsz), **accuracy(mo, imgsz)})
                print(rows[-1], flush=True)
            except Exception as e:
                print("onnx failed", e, flush=True)

            # ---- OpenVINO FP32 and INT8 post-training quantisation
            for label, kw in (("OpenVINO FP32", {}), ("OpenVINO INT8", {"int8": True})):
                try:
                    d = YOLO(str(MODELS / f"{base}.pt")).export(
                        format="openvino", imgsz=imgsz, device="cpu",
                        data=(calibration_yaml if kw else DATA), **kw)
                    d2 = MODELS / (f"{base}_{imgsz}" + ("_int8" if kw else "")
                                   + "_openvino_model")
                    if d2.exists():
                        import shutil; shutil.rmtree(d2)
                    Path(d).replace(d2)
                    mv = YOLO(str(d2), task="detect")
                    rows.append({"model": base, "imgsz": imgsz, "backend": label,
                                 **timed(mv, frames, imgsz), **accuracy(mv, imgsz)})
                    print(rows[-1], flush=True)
                except Exception as e:
                    print(label, "failed", repr(e)[:200], flush=True)
            (OUT / "e2_backends.json").write_text(json.dumps(rows, indent=2))
    print("done")


if __name__ == "__main__":
    main()
