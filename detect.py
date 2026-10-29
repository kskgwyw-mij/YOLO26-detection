"""Run both YOLO26 models over whole tiles and write pixel-space detections.

Nothing georeferenced is written, so locate.py cannot read the answer.

Outputs per tile in outputs/det/:
  <tile>.json          image size, gsd, DOTA OBB objects, predicted intersections
  <tile>_roadmask.png  predicted road mask at ROADMASK_GSD m/px (0-255), U-Net by default
  <tile>_preview.jpg   overlay for a visual check

Usage: uv run detect.py <tif-or-folder> [--seg runs/roads/weights/best.pt]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from ultralytics import YOLO
from ultralytics.utils.metrics import batch_probiou

import make_dataset as ds
import roadseg
from tiles import OUT, ROOT, load_rgb, windows

GSD = 0.2                  # native DOP20 pixel size (m)
ROADMASK_GSD = 1.0         # resolution of the saved road mask (m/px)
OBB_WIN, OBB_OVERLAP = 1024, 200
OBB_SCALES = (1.0, 0.5)    # x0.5 lets large facilities (running tracks, pitches) fit one window
SEG_OVERLAP = 320
VEHICLES = {"small vehicle", "large vehicle"}
INTERSECTION_MERGE_M = 8


def obb_nms(xywhr, conf, cls, iou=0.3):
    """Class-wise greedy rotated NMS (probiou) over detections from all windows."""
    keep = []
    for c in np.unique(cls):
        idx = np.where(cls == c)[0]
        idx = idx[np.argsort(-conf[idx])]
        ious = batch_probiou(torch.tensor(xywhr[idx]), torch.tensor(xywhr[idx])).numpy()
        alive = np.ones(len(idx), bool)
        for i in range(len(idx)):
            if alive[i]:
                keep.append(idx[i])
                alive[i + 1:] &= ious[i, i + 1:] < iou
    return np.array(keep, int)


def run_obb(model, img, conf, scales=OBB_SCALES):
    polys, xywhr, cf, cl = [], [], [], []
    for s in scales:
        im = img if s == 1 else np.asarray(Image.fromarray(img).resize(
            (int(img.shape[1] * s), int(img.shape[0] * s)), Image.LANCZOS))
        h, w = im.shape[:2]
        for r0 in windows(h, OBB_WIN, OBB_OVERLAP):
            for c0 in windows(w, OBB_WIN, OBB_OVERLAP):
                res = model.predict(im[r0:r0 + OBB_WIN, c0:c0 + OBB_WIN], imgsz=OBB_WIN,
                                    conf=conf, verbose=False)[0]
                if res.obb is None or len(res.obb) == 0:
                    continue
                off = np.array([c0, r0], np.float32)
                polys.append((res.obb.xyxyxyxy.cpu().numpy() + off) / s)
                b = res.obb.xywhr.cpu().numpy().copy()
                b[:, :2] += off
                b[:, :4] /= s
                xywhr.append(b)
                cf.append(res.obb.conf.cpu().numpy())
                cl.append(res.obb.cls.cpu().numpy().astype(int))
    if not polys:
        return []
    polys, xywhr, cf, cl = map(np.concatenate, (polys, xywhr, cf, cl))
    keep = obb_nms(xywhr, cf, cl)
    out = []
    for i in keep:
        name = model.names[int(cl[i])]
        out.append({"class": name, "conf": round(float(cf[i]), 3),
                    "poly_px": np.round(polys[i], 1).tolist(),
                    "center_px": np.round(xywhr[i, :2], 1).tolist(),
                    "use_for_locate": name not in VEHICLES})
    return out


def run_seg(model, img, conf, road_model=None):
    """Road mask and intersection centres for a tile.

    Streets come from the U-Net `road_model` when given, otherwise from the
    union of YOLO26-seg road instances. Intersections always come from YOLO26-seg.
    """
    h, w = img.shape[:2]
    sh, sw = int(h * ds.SCALE), int(w * ds.SCALE)
    small = np.asarray(Image.fromarray(img).resize((sw, sh), Image.LANCZOS))
    win = ds.IMGSZ
    mask = np.zeros((sh, sw), np.uint8)
    pts, pconf = [], []
    for r0 in windows(sh, win, int(SEG_OVERLAP * ds.SCALE)):
        for c0 in windows(sw, win, int(SEG_OVERLAP * ds.SCALE)):
            res = model.predict(small[r0:r0 + win, c0:c0 + win], imgsz=win, conf=conf,
                                retina_masks=True, verbose=False)[0]
            if res.boxes is None or len(res.boxes) == 0:
                continue
            cls = res.boxes.cls.cpu().numpy().astype(int)
            if res.masks is not None and (cls == 0).any():
                m = res.masks.data[torch.from_numpy(cls == 0)].any(0).cpu().numpy()
                mask[r0:r0 + win, c0:c0 + win] |= m.astype(np.uint8)
            xywh = res.boxes.xywh.cpu().numpy()
            for (x, y, *_), c, k in zip(xywh, res.boxes.conf.cpu().numpy(), cls):
                if k == 1:
                    pts.append(((c0 + x) / ds.SCALE, (r0 + y) / ds.SCALE))
                    pconf.append(float(c))
    inter = merge_points(np.array(pts).reshape(-1, 2), np.array(pconf), INTERSECTION_MERGE_M / GSD)
    if road_model is not None:
        mask = (roadseg.predict(road_model, small) > roadseg.THRESHOLD).astype(np.uint8)
    size = (int(round(w * GSD / ROADMASK_GSD)), int(round(h * GSD / ROADMASK_GSD)))
    coarse = Image.fromarray(mask * 255).resize(size, Image.BOX)
    return np.asarray(coarse), inter


def merge_points(xy, conf, radius):
    order = np.argsort(-conf)
    kept = []
    for i in order:
        if all(np.hypot(*(xy[i] - xy[j])) > radius for j in kept):
            kept.append(i)
    return [{"center_px": np.round(xy[i], 1).tolist(), "conf": round(float(conf[i]), 3)} for i in kept]


def preview(img, det, roadmask, path):
    s = 0.25
    im = Image.fromarray(img).resize((int(img.shape[1] * s), int(img.shape[0] * s)))
    road = Image.fromarray(roadmask).resize(im.size)
    red = Image.new("RGB", im.size, (255, 0, 0))
    im = Image.composite(red, im, road.point(lambda v: int(v * 0.45)))
    d = ImageDraw.Draw(im)
    for o in det["objects"]:
        color = (120, 120, 120) if not o["use_for_locate"] else (0, 255, 0)
        d.polygon([(x * s, y * s) for x, y in o["poly_px"]], outline=color, width=2)
    for p in det["intersections"]:
        x, y = p["center_px"][0] * s, p["center_px"][1] * s
        d.ellipse((x - 8, y - 8, x + 8, y + 8), outline=(0, 255, 255), width=3)
    im.save(path, quality=85)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("--obb", default=str(ROOT / "weights" / "yolo26s-obb.pt"))
    ap.add_argument("--seg", default=str(ROOT / "runs" / "roads" / "weights" / "best.pt"))
    ap.add_argument("--roads", choices=["unet", "yolo"], default="unet",
                    help="street mask from the U-Net (default) or from YOLO26-seg road instances")
    ap.add_argument("--unet", type=Path, default=roadseg.WEIGHTS)
    ap.add_argument("--obb-conf", type=float, default=0.3)
    ap.add_argument("--seg-conf", type=float, default=0.25)
    ap.add_argument("--out", type=Path, default=OUT / "det")
    a = ap.parse_args()
    tifs = sorted(a.src.glob("*.tif")) if a.src.is_dir() else [a.src]
    a.out.mkdir(parents=True, exist_ok=True)
    obb, seg = YOLO(a.obb), YOLO(a.seg)  # each model loaded once, reused for every tile
    unet = roadseg.load(a.unet) if a.roads == "unet" else None
    for tif in tifs:
        img = load_rgb(tif)
        roadmask, inter = run_seg(seg, img, a.seg_conf, unet)
        det = {"image": tif.stem, "width": img.shape[1], "height": img.shape[0], "gsd": GSD,
               "roadmask_gsd": ROADMASK_GSD, "objects": run_obb(obb, img, a.obb_conf),
               "intersections": inter}
        (a.out / f"{tif.stem}.json").write_text(json.dumps(det))
        Image.fromarray(roadmask).save(a.out / f"{tif.stem}_roadmask.png")
        preview(img, det, roadmask, a.out / f"{tif.stem}_preview.jpg")
        counts = {}
        for o in det["objects"]:
            counts[o["class"]] = counts.get(o["class"], 0) + 1
        print(f"{tif.stem}: road {roadmask.mean() / 2.55:.1f}%  intersections {len(inter)}  {counts}")


if __name__ == "__main__":
    main()
