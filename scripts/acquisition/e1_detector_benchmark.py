"""E1: accuracy of candidate edge detectors on the COCO-Traffic subset.

Every model is evaluated with the identical validation protocol so that the
accuracy/complexity trade-off can be read directly from the resulting table.
"""
import json, time, csv
from pathlib import Path
import os
WORK = Path(os.environ.get("PRISM_WORK_DIR", str(Path(__file__).resolve().parents[2] / "work")))
from ultralytics import YOLO

DATA = str(WORK) + "/data/coco_traffic/coco_traffic.yaml"
MODELS = Path(str(WORK) + "/models")
OUT = Path(str(WORK) + "/results"); OUT.mkdir(parents=True, exist_ok=True)
TRAFFIC = [0, 1, 2, 3, 5, 7, 9, 11]
NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus",
         7: "truck", 9: "traffic light", 11: "stop sign"}

RUNS = [(m, 640) for m in ["yolov5nu", "yolov8n", "yolov8s", "yolov10n",
                           "yolov10s", "yolo11n", "yolo11s", "yolo11m"]]
RUNS += [("yolo11n", s) for s in (320, 416, 512, 960)]
RUNS += [("yolo11s", s) for s in (416, 512)]


def gflops_params(model):
    try:
        from ultralytics.utils.torch_utils import get_flops, get_num_params
        return get_flops(model.model), get_num_params(model.model) / 1e6
    except Exception:
        return float("nan"), float("nan")


def main():
    rows = []
    for name, imgsz in RUNS:
        w = MODELS / f"{name}.pt"
        print(f"=== {name} @ {imgsz}", flush=True)
        m = YOLO(str(w))
        fl, pa = gflops_params(m)
        t0 = time.time()
        r = m.val(data=DATA, imgsz=imgsz, batch=1, device="cpu", classes=TRAFFIC,
                  conf=0.001, iou=0.7, plots=False, verbose=False,
                  project=str(OUT / "val"), name=f"{name}_{imgsz}", exist_ok=True)
        wall = time.time() - t0
        row = {"model": name, "imgsz": imgsz, "params_M": round(pa, 2),
               "GFLOPs": round(fl, 1), "mAP50": round(float(r.box.map50), 4),
               "mAP50_95": round(float(r.box.map), 4),
               "precision": round(float(r.box.mp), 4),
               "recall": round(float(r.box.mr), 4),
               "val_wall_s": round(wall, 1),
               "speed_ms": {k: round(v, 2) for k, v in r.speed.items()}}
        for i, c in enumerate(r.box.ap_class_index):
            row[f"AP50_{NAMES.get(int(c), int(c))}"] = round(float(r.box.ap50[i]), 4)
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if not k.startswith("AP50_")}), flush=True)
        (OUT / "e1_detector_benchmark.json").write_text(json.dumps(rows, indent=2))

    keys = sorted({k for r in rows for k in r})
    with open(OUT / "e1_detector_benchmark.csv", "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=keys)
        wcsv.writeheader()
        for r in rows:
            wcsv.writerow({k: (json.dumps(v) if isinstance(v, dict) else v)
                           for k, v in r.items()})
    print("done")


if __name__ == "__main__":
    main()
