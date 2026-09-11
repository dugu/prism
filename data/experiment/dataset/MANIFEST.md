# COCO-Traffic — dataset manifest

554 images, 5644 annotated instances of the eight traffic classes, selected from the
COCO 2017 validation split. The images themselves are not redistributed here: they
are unmodified COCO files, and the subset is fully reproducible from the two public
archives below plus `build_subset.py`.

## Sources (exact archives used)

| file | bytes | sha256 |
|---|---|---|
| coco2017val.zip | 818322941 | a9291310cd3ecd93f05e8c5815ba62a64215646ebab0bb39b7a10ef4af500b1f |
| coco2017labels.zip | 48639045 | 51a5175c894a7a1010f90eb4cba613473445f02633b684ed46c0292a997d0234 |

    curl -L -o coco2017val.zip    https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017val.zip
    curl -L -o coco2017labels.zip https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip

## Selection criteria (implemented in build_subset.py)

An image from val2017 is retained when all three hold:
1. it contains at least 3 instances of {person, bicycle, car, motorcycle, bus,
   truck, traffic light, stop sign};
2. at least 1 of those instances has normalised area >= 0.005;
3. at least 2 of those instances belong to {bicycle, car, motorcycle, bus, truck,
   traffic light, stop sign} — this excludes crowd photographs with pedestrians
   but no road context.

## Contents of this directory

- `subset_meta.json` — image count, per-class instance counts, criteria
- `image_list.txt` — the 554 selected COCO filenames (this IS the subset definition)
- `labels/` — the YOLO-format label files as used, filtered to the eight classes
- `coco_traffic.yaml` — the Ultralytics dataset descriptor
- `build_subset.py` — regenerates everything above from the two archives
