"""Build a YOLO-seg dataset (classes: road, intersection) from OSM for the
train/val tiles. Test tiles are never written.

Windows of WIN native px are cut with OVERLAP and downsampled by SCALE, so the
model sees 0.4 m/px images of IMGSZ px. detect.py must use the same SCALE.

Usage: uv run make_dataset.py [--out datasets/roads]
"""
import argparse
import random
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image
from shapely.affinity import affine_transform
from shapely.geometry import box

from osm import load_layers
from tiles import ROOT, load_rgb, splits, tile_origin, tile_path, windows

CLASSES = ["road", "intersection"]
WIN, OVERLAP, SCALE = 1280, 160, 0.5
IMGSZ = int(WIN * SCALE)
INTERSECTION_RADIUS_M = 8
NEGATIVE_KEEP = 0.1

_layers = None


def _init():
    global _layers
    roads, inter, _ = load_layers()
    roads = roads.copy()
    roads["geometry"] = [g.buffer(w / 2, cap_style="flat", join_style="round").simplify(0.3)
                         for g, w in zip(roads.geometry, roads.width_m)]
    inter = inter.copy()
    inter["geometry"] = inter.geometry.buffer(INTERSECTION_RADIUS_M, quad_segs=4)
    _layers = (roads, inter)


def _polys(g):
    if g.is_empty:
        return []
    if g.geom_type == "Polygon":
        return [g]
    return [p for p in getattr(g, "geoms", []) if p.geom_type == "Polygon"]


def window_labels(e0, n0, gsd, r0, c0):
    """YOLO-seg label lines for the window whose native top-left pixel is (r0, c0)."""
    roads, inter = _layers
    we, wn = e0 + c0 * gsd, n0 - r0 * gsd
    span = WIN * gsd
    wbox = box(we, wn - span, we + span, wn)
    # world -> normalized window coords: u = (x - we)/span, v = (wn - y)/span
    m = [1 / span, 0, 0, -1 / span, -we / span, wn / span]
    lines = []
    for cls, layer in enumerate((roads, inter)):
        for g in layer.geometry.iloc[layer.sindex.query(wbox, predicate="intersects")]:
            for p in _polys(g.intersection(wbox)):
                if p.area < (1.0 if cls == 0 else 20.0):
                    continue
                xy = np.clip(np.asarray(affine_transform(p, m).exterior.coords)[:-1], 0, 1)
                if len(xy) >= 3:
                    lines.append(f"{cls} " + " ".join(f"{v:.5f}" for v in xy.ravel()))
    return lines


def process(job):
    name, split, out = job
    rng = random.Random(name)
    e0, n0, gsd = tile_origin(tile_path(name))
    img = load_rgb(tile_path(name))
    h, w = img.shape[:2]
    n_pos = n_neg = 0
    for r0 in windows(h, WIN, OVERLAP):
        for c0 in windows(w, WIN, OVERLAP):
            lines = window_labels(e0, n0, gsd, r0, c0)
            if not lines and rng.random() > NEGATIVE_KEEP:
                continue
            stem = f"{name}_{r0}_{c0}"
            crop = Image.fromarray(img[r0:r0 + WIN, c0:c0 + WIN]).resize((IMGSZ, IMGSZ), Image.LANCZOS)
            crop.save(out / "images" / split / f"{stem}.jpg", quality=92)
            (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))
            n_pos += bool(lines)
            n_neg += not lines
    return name, split, n_pos, n_neg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "datasets" / "roads")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    sp = splits()
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            (a.out / sub / split).mkdir(parents=True, exist_ok=True)
    jobs = [(t, s, a.out) for s in ("train", "val") for t in sp[s]]
    with Pool(a.workers, initializer=_init) as pool:
        for name, split, p, n in pool.imap_unordered(process, jobs):
            print(f"{split:5s} {name}: {p} labelled, {n} negative")
    (a.out / "roads.yaml").write_text(
        f"path: {a.out}\ntrain: images/train\nval: images/val\n"
        + "names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    print(a.out / "roads.yaml")


if __name__ == "__main__":
    main()
