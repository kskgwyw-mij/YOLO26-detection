"""Run both YOLO26 models over whole tiles and write pixel-space detections.

Nothing georeferenced is written, so locate.py cannot read the answer.

Outputs per tile in outputs/det/:
  <tile>.json          image size, gsd, DOTA OBB objects, predicted intersections
  <tile>_roadmask.png  predicted road mask at ROADMASK_GSD m/px (0-255), U-Net by default
  <tile>_railmask.png  predicted railway mask, same grid (U-Net only)
  <tile>_forestmask.png  predicted forest mask, same grid (U-Net only)
  <tile>_buildingmask.png  predicted building mask, same grid (U-Net only)
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


def run_obb(model, img, conf, scales=OBB_SCALES, on_step=None):
    polys, xywhr, cf, cl = [], [], [], []
    for s in scales:
        im = img if s == 1 else np.asarray(Image.fromarray(img).resize(
            (int(img.shape[1] * s), int(img.shape[0] * s)), Image.LANCZOS))
        h, w = im.shape[:2]
        for r0 in windows(h, OBB_WIN, OBB_OVERLAP):
            for c0 in windows(w, OBB_WIN, OBB_OVERLAP):
                if on_step is not None:  # boxes so far, before NMS
                    on_step("facilities", window_px(r0, c0, OBB_WIN, s), {"boxes": [
                        (p, model.names[int(c)]) for P, C in zip(polys, cl) for p, c in zip(P, C)]})
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
    if on_step is not None:  # stage finished (window None)
        on_step("facilities", None, {"boxes": [
            (p, model.names[int(c)]) for P, C in zip(polys, cl) for p, c in zip(P, C)]})
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


def window_px(r0, c0, win, scale):
    """Window (x0, y0, x1, y1) in full-resolution pixels."""
    return c0 / scale, r0 / scale, (c0 + win) / scale, (r0 + win) / scale


def run_seg(model, img, conf, unets, on_step=None):
    """Masks {"road", "rail", "forest", "building"} at ROADMASK_GSD and intersection centres.

    `unets` maps a class to the U-Net whose channel of that name is used.
    Streets fall back to the union of YOLO26-seg road instances when there is
    no "road" U-Net; any other class without a U-Net gives a None mask.
    Intersections always come from YOLO26-seg.
    on_step(stage, window, partial) is called before each window, for live display,
    and once more with window None when a stage is finished.
    """
    h, w = img.shape[:2]
    sh, sw = int(h * ds.SCALE), int(w * ds.SCALE)
    small = np.asarray(Image.fromarray(img).resize((sw, sh), Image.LANCZOS))
    win = ds.IMGSZ
    mask = np.zeros((sh, sw), np.uint8)
    pts, pconf = [], []
    for r0 in windows(sh, win, int(SEG_OVERLAP * ds.SCALE)):
        for c0 in windows(sw, win, int(SEG_OVERLAP * ds.SCALE)):
            if on_step is not None:
                on_step("intersections", window_px(r0, c0, win, ds.SCALE), {"points": list(pts)})
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
    if on_step is not None:
        on_step("intersections", None, {"points": list(pts)})
    inter = merge_points(np.array(pts).reshape(-1, 2), np.array(pconf), INTERSECTION_MERGE_M / GSD)
    masks = {**dict.fromkeys(roadseg.CLASSES), "road": mask}
    for name, unet in unets.items():
        if unet is not None:
            k = roadseg.channel(unet, name)
            live = None if on_step is None else (lambda r0, c0, win, p, name=name, k=k: on_step(
                name, window_px(r0, c0, win, ds.SCALE), {"mask": p[k] > roadseg.THRESHOLD}))
            prob = roadseg.predict(unet, small, live)[k]
            masks[name] = (prob > roadseg.THRESHOLD).astype(np.uint8)
            if on_step is not None:
                on_step(name, None, {"mask": masks[name] > 0})
    size = (int(round(w * GSD / ROADMASK_GSD)), int(round(h * GSD / ROADMASK_GSD)))
    coarse = lambda m: np.asarray(Image.fromarray(m * 255).resize(size, Image.BOX))
    return {k: None if m is None else coarse(m) for k, m in masks.items()}, inter


def merge_points(xy, conf, radius):
    order = np.argsort(-conf)
    kept = []
    for i in order:
        if all(np.hypot(*(xy[i] - xy[j])) > radius for j in kept):
            kept.append(i)
    return [{"center_px": np.round(xy[i], 1).tolist(), "conf": round(float(conf[i]), 3)} for i in kept]


PREVIEW_COLORS = {"forest": (0, 200, 0), "building": (255, 140, 0), "road": (255, 0, 0), "rail": (0, 90, 255)}


def preview(img, det, masks, path=None):
    s = 0.25
    im = Image.fromarray(img).resize((int(img.shape[1] * s), int(img.shape[0] * s)))
    for name, color in PREVIEW_COLORS.items():
        m = masks.get(name)
        if m is not None:
            a = Image.fromarray(m).resize(im.size).point(lambda v: int(v * 0.45))
            im = Image.composite(Image.new("RGB", im.size, color), im, a)
    d = ImageDraw.Draw(im)
    for o in det["objects"]:
        color = (120, 120, 120) if not o["use_for_locate"] else (0, 255, 0)
        d.polygon([(x * s, y * s) for x, y in o["poly_px"]], outline=color, width=2)
    for p in det["intersections"]:
        x, y = p["center_px"][0] * s, p["center_px"][1] * s
        d.ellipse((x - 8, y - 8, x + 8, y + 8), outline=(0, 255, 255), width=3)
    if path is not None:
        im.save(path, quality=85)
    return im


def load_models(obb=ROOT / "weights" / "yolo26s-obb.pt", seg=ROOT / "runs" / "roads" / "weights" / "best.pt",
                roads="unet", unet=roadseg.STREET_WEIGHTS, rail_unet=roadseg.WEIGHTS,
                forest_unet=roadseg.FOREST_WEIGHTS, building_unet=roadseg.BUILDING_WEIGHTS):
    """OBB model, seg model and the U-Nets per mask class ('none' skips rail / forest / building)."""
    opt = lambda p: None if str(p) == "none" else roadseg.load(p)
    unets = {"road": roadseg.load(unet) if roads == "unet" else None,
             "rail": opt(rail_unet), "forest": opt(forest_unet), "building": opt(building_unet)}
    return YOLO(str(obb)), YOLO(str(seg)), unets


def detect(img, name, obb, seg, unets, obb_conf=0.3, seg_conf=0.25, on_step=None):
    """Detection dict and {road, rail, forest, building} masks (uint8 0-255 at ROADMASK_GSD) for one image.
    on_step(stage, window, partial) reports progress per window (see run_seg, run_obb)."""
    masks, inter = run_seg(seg, img, seg_conf, unets, on_step)
    det = {"image": name, "width": img.shape[1], "height": img.shape[0], "gsd": GSD,
           "roadmask_gsd": ROADMASK_GSD, "objects": run_obb(obb, img, obb_conf, on_step=on_step),
           "intersections": inter}
    return det, masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("--obb", default=str(ROOT / "weights" / "yolo26s-obb.pt"))
    ap.add_argument("--seg", default=str(ROOT / "runs" / "roads" / "weights" / "best.pt"))
    ap.add_argument("--roads", choices=["unet", "yolo"], default="unet",
                    help="street mask from the U-Net (default) or from YOLO26-seg road instances")
    ap.add_argument("--unet", type=Path, default=roadseg.STREET_WEIGHTS, help="U-Net for streets")
    ap.add_argument("--rail-unet", type=Path, default=roadseg.WEIGHTS,
                    help="two-class U-Net for railways ('none' to skip)")
    ap.add_argument("--forest-unet", type=Path, default=roadseg.FOREST_WEIGHTS,
                    help="U-Net for forest ('none' to skip)")
    ap.add_argument("--building-unet", type=Path, default=roadseg.BUILDING_WEIGHTS,
                    help="U-Net for buildings ('none' to skip)")
    ap.add_argument("--obb-conf", type=float, default=0.3)
    ap.add_argument("--seg-conf", type=float, default=0.25)
    ap.add_argument("--out", type=Path, default=OUT / "det")
    a = ap.parse_args()
    tifs = sorted(a.src.glob("*.tif")) if a.src.is_dir() else [a.src]
    a.out.mkdir(parents=True, exist_ok=True)
    # each model loaded once, reused for every tile
    obb, seg, unets = load_models(a.obb, a.seg, a.roads, a.unet, a.rail_unet, a.forest_unet, a.building_unet)
    for tif in tifs:
        img = load_rgb(tif)
        det, masks = detect(img, tif.stem, obb, seg, unets, a.obb_conf, a.seg_conf)
        inter = det["intersections"]
        (a.out / f"{tif.stem}.json").write_text(json.dumps(det))
        for name, m in masks.items():
            if m is not None:
                Image.fromarray(m).save(a.out / f"{tif.stem}_{name}mask.png")
        preview(img, det, masks, a.out / f"{tif.stem}_preview.jpg")
        counts = {}
        for o in det["objects"]:
            counts[o["class"]] = counts.get(o["class"], 0) + 1
        cover = "  ".join(f"{k} {m.mean() / 2.55:.2f}%" for k, m in masks.items() if m is not None)
        print(f"{tif.stem}: {cover}  intersections {len(inter)}  {counts}")


if __name__ == "__main__":
    main()
