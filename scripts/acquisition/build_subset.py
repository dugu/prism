"""Build the COCO-Traffic evaluation subset used by the PRISM proof-of-concept.

Selects COCO val2017 images that contain a sufficient number of traffic-relevant
object instances, so that the benchmark reflects road-scene conditions rather
than generic COCO imagery.
"""
import json, os, random, shutil
from pathlib import Path
import os
WORK = Path(os.environ.get("PRISM_WORK_DIR", str(Path(__file__).resolve().parents[2] / "work")))
from collections import Counter

ROOT = Path(str(WORK) + "/data")
IMG = ROOT / "cocoimg" / "coco" / "images" / "val2017"
LBL = ROOT / "cocolab" / "coco" / "labels" / "val2017"
OUT = Path(str(WORK) + "/data/coco_traffic")

# COCO class ids (0-based, YOLO convention) that are relevant to road scenes
TRAFFIC = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus",
           7: "truck", 9: "traffic light", 11: "stop sign"}
ROAD_CTX = {1, 2, 3, 5, 7, 9, 11}   # vehicles and traffic infrastructure
MIN_INST = 3          # minimum traffic instances for an image to be a "road scene"
MIN_LARGE = 1         # at least one instance with normalised area >= 0.005

def main():
    if not IMG.exists():
        raise SystemExit(f"missing {IMG}")
    random.seed(20260910)
    keep, stats = [], Counter()
    for lp in sorted(LBL.glob("*.txt")):
        rows = [r.split() for r in lp.read_text().strip().splitlines() if r.strip()]
        traf = [r for r in rows if int(r[0]) in TRAFFIC]
        if len(traf) < MIN_INST:
            continue
        if sum(1 for r in traf if float(r[3]) * float(r[4]) >= 0.005) < MIN_LARGE:
            continue
        # require road context: at least two instances of a vehicle or an item of
        # traffic infrastructure, so that generic crowd photographs are excluded
        if sum(1 for r in traf if int(r[0]) in ROAD_CTX) < 2:
            continue
        ip = IMG / (lp.stem + ".jpg")
        if not ip.exists():
            continue
        keep.append((ip, lp, traf))
        for r in traf:
            stats[TRAFFIC[int(r[0])]] += 1

    print(f"road-scene images found: {len(keep)}")
    print("instances:", dict(stats))

    for sub in ("images", "labels"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    for ip, lp, traf in keep:
        shutil.copy(ip, OUT / "images" / ip.name)
        (OUT / "labels" / lp.name).write_text(
            "\n".join(" ".join(r) for r in traf) + "\n")

    yaml = OUT / "coco_traffic.yaml"
    names = {i: n for i, n in sorted(TRAFFIC.items())}
    lines = ["path: %s" % OUT, "train: images", "val: images", "names:"]
    # keep original COCO indices so pretrained weights map directly
    full = json.loads((ROOT / "cocolab" / "coco" / "LICENSE").exists() and "{}" or "{}")
    for i in range(80):
        lines.append(f"  {i}: {names.get(i, 'class_%d' % i)}")
    yaml.write_text("\n".join(lines) + "\n")
    print("wrote", yaml)

    meta = {"n_images": len(keep), "instances": dict(stats),
            "min_instances": MIN_INST, "classes": TRAFFIC}
    (OUT / "subset_meta.json").write_text(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
