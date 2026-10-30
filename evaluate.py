"""Score locate.py against the true tile positions (GeoTIFF tags, used only here).

  uv run evaluate.py --oracle          # OSM-made "perfect detections": checks locator logic
  uv run evaluate.py                   # real detect.py output in outputs/det, test split
  uv run evaluate.py --split val
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from shapely.geometry import box

import osm
from locate import load_detections, locate, rasterize_areas, rasterize_roads
from tiles import OUT, splits, tile_origin, tile_path

SUCCESS_M = 10
STREET_GROUPS = {
    "major": ["motorway", "trunk", "primary", "secondary", "tertiary", "motorway_link",
              "trunk_link", "primary_link", "secondary_link", "tertiary_link"],
    "residential": ["residential", "unclassified", "living_street"],
    "service": ["service"],
    "track": ["track"],
}


def street_quality(tiles, layers, det_dir, tol_m=3):
    """Road-mask precision/recall vs OSM per street type, and the same for the
    railway mask (tol_m tolerance, 1 m grid). Predicted rail on abandoned or
    disused lines is not counted as false. Forest mask: area IoU per tile (1 m, no tolerance)."""
    from scipy.ndimage import binary_dilation
    roads, rails, forest = layers[0], layers[3], layers[4]
    f_tp = f_fp = f_fn = 0
    gt_n, hit_n, pred_n, pred_ok = {k: 0 for k in STREET_GROUPS}, {k: 0 for k in STREET_GROUPS}, 0, 0
    rail_gt_n = rail_hit = rail_pred_n = rail_pred_ok = 0
    for tile in tiles:
        e0, n0, _ = tile_origin(tile_path(tile))
        area = (e0, n0 - 1000, e0 + 1000, n0)
        rail_path = det_dir / f"{tile}_railmask.png"
        if rail_path.exists():
            rp = np.asarray(Image.open(rail_path)) > 127
            gt = rasterize_roads(rails[rails.active], area, 1.0) > 0.5
            ok = binary_dilation(rasterize_roads(rails, area, 1.0) > 0.5, iterations=tol_m)
            hit = (gt & binary_dilation(rp, iterations=tol_m)).sum()
            if gt.any():
                print(f"  {tile}: railway recall {hit / gt.sum():.2f} ({gt.sum()} OSM rail px)")
            rail_gt_n += gt.sum()
            rail_hit += hit
            rail_pred_n += rp.sum()
            rail_pred_ok += (rp & ok).sum()
        forest_path = det_dir / f"{tile}_forestmask.png"
        if forest_path.exists():
            fp_ = np.asarray(Image.open(forest_path)) > 127
            gt = rasterize_areas(forest, area, 1.0) > 0.5
            tp, fp, fn = (fp_ & gt).sum(), (fp_ & ~gt).sum(), (~fp_ & gt).sum()
            print(f"  {tile}: forest OSM {gt.mean():4.0%} predicted {fp_.mean():4.0%}  IoU {tp / max(tp + fp + fn, 1):.2f}")
            f_tp, f_fp, f_fn = f_tp + tp, f_fp + fp, f_fn + fn
        pred = np.asarray(Image.open(det_dir / f"{tile}_roadmask.png")) > 127
        near_pred = binary_dilation(pred, iterations=tol_m)
        gt_all = rasterize_roads(roads, area, 1.0) > 0.5
        pred_n += pred.sum()
        pred_ok += (pred & binary_dilation(gt_all, iterations=tol_m)).sum()
        for k, hw in STREET_GROUPS.items():
            gt = rasterize_roads(roads[roads.highway.isin(hw)], area, 1.0) > 0.5
            gt_n[k] += gt.sum()
            hit_n[k] += (gt & near_pred).sum()
    total = sum(gt_n.values())
    print(f"street mask vs OSM ({tol_m} m tolerance): precision {pred_ok / max(pred_n, 1):.2f}  "
          f"recall {sum(hit_n.values()) / max(total, 1):.2f}")
    for k in STREET_GROUPS:
        print(f"  {k:12s} {gt_n[k] / total:4.0%} of street px   recall {hit_n[k] / max(gt_n[k], 1):.2f}")
    if rail_pred_n or rail_gt_n:
        print(f"railway mask vs OSM: precision {rail_pred_ok / max(rail_pred_n, 1):.2f}  "
              f"recall {rail_hit / max(rail_gt_n, 1):.2f}  ({rail_gt_n} OSM rail px, "
              f"{rail_pred_n} predicted, {rail_pred_n - rail_pred_ok} outside OSM rail)")
    if f_tp + f_fp + f_fn:
        print(f"forest mask vs OSM: IoU {f_tp / (f_tp + f_fp + f_fn):.2f}  precision {f_tp / max(f_tp + f_fp, 1):.2f}  "
              f"recall {f_tp / max(f_tp + f_fn, 1):.2f}")


def make_oracle(tile, layers, out_dir, degrade=False):
    """Detections built from OSM at the true position, in the same format as detect.py.

    degrade=True simulates an imperfect detector: drops 40 % of streets and
    intersections, jitters intersections by 2 m, adds 30 % false intersections
    and speckle noise to the road mask.
    """
    roads, inter, fac, rails, forest = layers
    rails = rails[rails.active]
    rng = np.random.default_rng(int(tile.replace("_", "")))
    e0, n0, gsd = tile_origin(tile_path(tile))
    size = 5000
    ext = size * gsd
    area = (e0, n0 - ext, e0 + ext, n0)
    b = box(*area)
    pts = np.array([[g.x, g.y] for g in inter.geometry if b.contains(g)]).reshape(-1, 2)
    if degrade:
        roads = roads[rng.random(len(roads)) > 0.4]
        rails = rails[rng.random(len(rails)) > 0.4]
        pts = pts[rng.random(len(pts)) > 0.4]
        pts = pts + rng.normal(0, 2, pts.shape)
        fake = rng.uniform([area[0], area[1]], [area[2], area[3]], (int(0.3 * len(pts)), 2))
        pts = np.vstack([pts, fake])
    mask, rail = rasterize_roads(roads, area, 1.0), rasterize_roads(rails, area, 1.0)
    if degrade:
        mask = np.clip(mask * (rng.random(mask.shape) > 0.2) + (rng.random(mask.shape) > 0.97), 0, 1)
        rail = np.clip(rail * (rng.random(rail.shape) > 0.2) + (rng.random(rail.shape) > 0.999), 0, 1)
        forest = forest[rng.random(len(forest)) > 0.4]  # drawn last: earlier noise stays as before
    wood = rasterize_areas(forest, area, 1.0)
    to_px = lambda x, y: [round((x - e0) / gsd, 1), round((n0 - y) / gsd, 1)]
    det = {"image": tile, "width": size, "height": size, "gsd": gsd, "roadmask_gsd": 1.0,
           "intersections": [{"center_px": to_px(x, y), "conf": 1.0} for x, y in pts],
           "objects": [{"class": c, "conf": 1.0, "center_px": to_px(g.centroid.x, g.centroid.y),
                        "use_for_locate": True}
                       for g, c in zip(fac.geometry, fac.dota_class) if b.contains(g.centroid)]}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{tile}.json").write_text(json.dumps(det))
    Image.fromarray((mask * 255).astype(np.uint8)).save(out_dir / f"{tile}_roadmask.png")
    Image.fromarray((rail * 255).astype(np.uint8)).save(out_dir / f"{tile}_railmask.png")
    Image.fromarray((wood * 255).astype(np.uint8)).save(out_dir / f"{tile}_forestmask.png")
    return out_dir / f"{tile}.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--degrade", action="store_true", help="oracle with simulated detector errors")
    ap.add_argument("--split", default="test")
    ap.add_argument("--quality", action="store_true", help="street-mask precision/recall per street type")
    ap.add_argument("--det-dir", type=Path, default=OUT / "det")
    a = ap.parse_args()
    layers = osm.load_layers()
    if a.quality:
        street_quality(splits()[a.split], layers, a.det_dir)
        return
    mode = ("oracle_degraded" if a.degrade else "oracle") if a.oracle else "real"
    det_dir = a.det_dir if mode == "real" else OUT / mode
    rows = []
    for tile in splits()[a.split]:
        path = make_oracle(tile, layers, det_dir, a.degrade) if a.oracle else det_dir / f"{tile}.json"
        if not path.exists():
            print(f"{tile}: no detections, run detect.py first")
            continue
        res = locate(*load_detections(path), osm.AREA, layers)
        e0, n0, _ = tile_origin(tile_path(tile))
        err = float(np.hypot(res["ul_e"] - e0, res["ul_n"] - n0)) if "ul_e" in res else None
        row = {"tile": tile, "true_e": e0, "true_n": n0, "error_m": None if err is None else round(err, 2),
               **{k: v for k, v in res.items() if k not in ("center_lat", "center_lon")}}
        rows.append(row)
        print(f"{tile}: {res['status']:16s} err {row['error_m']} m  score {res.get('score')} "
              f"margin {res.get('margin')}  inter {res.get('matched_intersections')}/"
              f"{res.get('n_det_intersections')}  fac {res.get('matched_facilities')}")
    name = f"eval_{mode}_{a.split}.csv"
    with open(OUT / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if r["status"] == "ok"]
    good = [r for r in ok if r["error_m"] < SUCCESS_M]
    wrong = [r["tile"] for r in ok if r["error_m"] >= SUCCESS_M]
    errs = [r["error_m"] for r in good]
    print(f"\n{len(rows)} tiles | localized within {SUCCESS_M} m: {len(good)} "
          f"({100 * len(good) / max(len(rows), 1):.0f} %) | median error {np.median(errs) if errs else '-'} m")
    print(f"not localizable: {[r['tile'] for r in rows if r['status'] != 'ok']}")
    print(f"WRONG but confident: {wrong}")
    print(OUT / name)


if __name__ == "__main__":
    main()
