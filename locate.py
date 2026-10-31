"""Locate an aerial image (north-up, known GSD) from its detections + OSM.

Reads only detect.py output (pixel space) and OSM. Never opens GeoTIFF tags.

  (a) road correlation: NCC of the predicted road mask against rasterized OSM
      streets over the whole search area (FFT); the same for the railway mask
      against OSM railways, and for a band around the predicted forest edges
      against the band around OSM forest edges, and for the building mask
      against OSM building footprints, when the image has any
  (b) votes: every (detected, OSM) intersection pair and same-class DOTA
      facility pair votes for an image upper-left corner
  (c) combined map -> best / runner-up peak -> local 1 m refinement ->
      least-squares shift over matched intersections

Usage:
  uv run locate.py outputs/det/32732_5306.json [--search minE,minN,maxE,maxN | --place Rosenheim]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw
from pyproj import Transformer
from scipy.ndimage import binary_dilation, binary_erosion, binary_opening, gaussian_filter, maximum_filter
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree
from shapely.geometry import box

import osm
from tiles import CRS, OUT

RES = 2.0                 # coarse search grid (m/px)
BLUR_M = 2.0              # tolerance for road-width / alignment differences
VOTE_SIGMA_M = 3.0
W_INTER, W_FAC = 0.5, 0.3  # weight of vote maps relative to road NCC (range ~[-1, 1])
# Set a priori (no val tile has a railway): a straight track barely fixes the
# position along itself, so rails count less than streets.
W_RAIL = 0.5
# Forest, set a priori: only edges are matched (filled areas give broad peaks
# that barely separate the best position from one 100-200 m away).
W_FOREST = 0.5
FOREST_EDGE_M = 5.0       # half-width of the band around forest edges
FOREST_MIN_M = 4.0        # opening radius: drops single trees and hedges from the mask
W_BUILDING = 0.5          # building-footprint NCC, set a priori like W_RAIL / W_FOREST
MIN_SIGNAL = 0.002        # mask coverage below this: the mask is treated as empty
PEAK_SEPARATION_M = 100
MATCH_RADIUS_M = 6
REFINE_RADIUS_M = 30
MIN_MARGIN = 0.05         # best - runner-up below this -> not localizable
MIN_SCORE = 0.25
# Large search areas are scanned block by block (bounded memory, same result as one map).
BLOCK_M = 10000           # upper-left positions per block, each direction
PAD_M = 50                # rasters and vote maps are built this much larger, so blur and
                          # edge bands don't see block borders
RASTER_CACHE_BYTES = 6e9


_raster_cache = {}


def padded(area, res):
    p = round(PAD_M / res)
    return p, (area[0] - p * res, area[1] - p * res, area[2] + p * res, area[3] + p * res)


def search_raster(name, area, res, build):
    """Blurred raster of one OSM layer over area, built PAD_M larger; cached per
    (layer, area, resolution) up to RASTER_CACHE_BYTES, oldest dropped first."""
    key = (name, tuple(area), res)
    if key not in _raster_cache:
        p, big = padded(area, res)
        _raster_cache[key] = np.ascontiguousarray(gaussian_filter(build(big), BLUR_M / res)[p:-p, p:-p])
        while sum(a.nbytes for a in _raster_cache.values()) > RASTER_CACHE_BYTES and len(_raster_cache) > 1:
            _raster_cache.pop(next(iter(_raster_cache)))
    return _raster_cache[key]


def edge_band(m, res=RES):
    """Band of +-FOREST_EDGE_M around the edges of boolean mask m; raster borders are not edges."""
    m = binary_opening(m, iterations=max(1, round(FOREST_MIN_M / res)))
    r = max(1, round(FOREST_EDGE_M / res))
    return (binary_dilation(m, iterations=r) & ~binary_erosion(m, iterations=r, border_value=1)).astype(np.float32)


def rasterize_roads(roads, area, res):
    e0, n0, e1, n1 = area
    im = Image.new("L", (int(round((e1 - e0) / res)), int(round((n1 - n0) / res))), 0)
    d = ImageDraw.Draw(im)
    sub = roads.iloc[roads.sindex.query(box(*area))]
    for g, w in zip(sub.geometry, sub.width_m):
        px = [((x - e0) / res, (n1 - y) / res) for x, y in g.coords]
        d.line(px, fill=255, width=max(1, int(round(w / res))), joint="curve")
    return np.asarray(im, np.float32) / 255


def rasterize_areas(polys, area, res):
    """Filled polygons with holes; polys must not overlap (osm.build_forest dissolves them)."""
    e0, n0, e1, n1 = area
    im = Image.new("L", (int(round((e1 - e0) / res)), int(round((n1 - n0) / res))), 0)
    d = ImageDraw.Draw(im)
    to_px = lambda ring: [((x - e0) / res, (n1 - y) / res) for x, y in ring.coords]
    for g in polys.geometry.iloc[polys.sindex.query(box(*area))]:
        d.polygon(to_px(g.exterior), fill=255)
        for hole in g.interiors:
            d.polygon(to_px(hole), fill=0)
    return np.asarray(im, np.float32) / 255


def ncc_map(R, T):
    """Normalized cross-correlation of template T at every 'valid' offset in R."""
    th, tw = T.shape
    Tz = T - T.mean()
    tnorm = np.sqrt((Tz ** 2).sum())
    num = fftconvolve(R, Tz[::-1, ::-1], mode="valid")
    def box_sum(a):
        c = np.pad(a.astype(np.float64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        return c[th:, tw:] - c[:-th, tw:] - c[th:, :-tw] + c[:-th, :-tw]
    s1, s2 = box_sum(R), box_sum(R * R)
    var = np.maximum(s2 - s1 ** 2 / T.size, 0)
    return (num / (np.sqrt(var) * tnorm + 1e-3 * T.size)).astype(np.float32)


def vote_map(shape, area, res, det_xy_m, det_w, osm_xy, osm_cls=None, det_cls=None):
    """Each (detection, same-class OSM feature) pair votes for an upper-left corner."""
    e0, _, _, n1 = area
    V = np.zeros(shape, np.float32)
    if len(det_xy_m) == 0 or len(osm_xy) == 0:
        return V
    for k, ((dx, dy), w) in enumerate(zip(det_xy_m, det_w)):
        tgt = osm_xy if osm_cls is None else osm_xy[osm_cls == det_cls[k]]
        if len(tgt) == 0:
            continue
        j = np.round((tgt[:, 0] - dx - e0) / res).astype(int)
        i = np.round((n1 - (tgt[:, 1] + dy)) / res).astype(int)
        ok = (i >= 0) & (i < shape[0]) & (j >= 0) & (j < shape[1])
        np.add.at(V, (i[ok], j[ok]), w)
    sig = VOTE_SIGMA_M / res
    return gaussian_filter(V, sig) * (2 * np.pi * sig ** 2) / max(sum(det_w), 1e-6)


def peaks(C, res, k=5, keep=None):
    """Top-k local maxima (window +-PEAK_SEPARATION_M); keep=(i0, i1, j0, j1) limits them to a sub-window."""
    sep = int(PEAK_SEPARATION_M / res)
    loc = (C == maximum_filter(C, size=2 * sep + 1)) & np.isfinite(C)
    if keep is not None:
        i0, i1, j0, j1 = keep
        inside = np.zeros_like(loc)
        inside[i0:i1, j0:j1] = True
        loc &= inside
    idx = np.argwhere(loc)
    order = np.argsort(-C[loc])[:k]
    return [tuple(idx[o]) for o in order]


def place_area(name, half=5000):
    r = requests.get("https://nominatim.openstreetmap.org/search",
                     params={"q": name, "format": "json", "limit": 1},
                     headers={"User-Agent": osm.USER_AGENT}, timeout=30)
    r.raise_for_status()
    hit = r.json()[0]
    e, n = Transformer.from_crs("EPSG:4326", CRS, always_xy=True).transform(float(hit["lon"]), float(hit["lat"]))
    return (e - half, n - half, e + half, n + half)


def osm_for(area):
    """OSM layers for the search area; reuses the Rosenheim cache when it covers it."""
    a0 = osm.AREA
    if area[0] >= a0[0] and area[1] >= a0[1] and area[2] <= a0[2] and area[3] <= a0[3]:
        return osm.load_layers("rosenheim")
    name = "area_" + "_".join(str(int(v)) for v in area)
    gpkg = OUT / "osm" / f"{name}.gpkg"
    if not gpkg.exists():
        osm.build_all(name, area)
    return osm.load_layers(name)


MASKS = ("road", "rail", "forest", "building")


def load_detections(det_path):
    """Detection JSON and {road, rail, forest, building} masks (None when detect.py wrote none)."""
    det = json.loads(Path(det_path).read_text())
    read = lambda p: np.asarray(Image.open(p), np.float32) / 255 if p.exists() else None
    stem = Path(det_path).stem
    return det, {k: read(Path(det_path).with_name(f"{stem}_{k}mask.png")) for k in MASKS}


def to_grid(mask, mres):
    """Average a mask at mres m/px down to the RES search grid."""
    f = int(round(RES / mres))
    th, tw = mask.shape[0] // f, mask.shape[1] // f
    return mask[:th * f, :tw * f].reshape(th, f, tw, f).mean((1, 3))


def mask_ncc(R, T):
    """NCC map of template T over search raster R (zeros when T is nearly empty) and T's coverage."""
    signal = float(T.mean())
    if signal <= MIN_SIGNAL:
        return np.zeros((R.shape[0] - T.shape[0] + 1, R.shape[1] - T.shape[1] + 1), np.float32), signal
    return ncc_map(R, T), signal


def locate(det, masks, area, layers):
    roads, inter, fac, rails, forest = (layers[k] for k in ("roads", "intersections", "facilities",
                                                             "railways", "forest"))
    gsd, mres = det["gsd"], det["roadmask_gsd"]
    ext_e, ext_n = det["width"] * gsd, det["height"] * gsd
    blur = lambda m: gaussian_filter(m, BLUR_M / RES)

    # templates on the coarse grid: road, railway, forest-edge band, building masks
    T = {"road": blur(to_grid(masks["road"], mres))}
    if masks.get("rail") is not None:
        T["rail"] = blur(to_grid(masks["rail"], mres))
    if masks.get("forest") is not None:
        T["forest"] = blur(edge_band(to_grid(masks["forest"], mres) > 0.5))
    if masks.get("building") is not None:
        T["building"] = blur(to_grid(masks["building"], mres))
    signal = {k: float(t.mean()) for k, t in T.items()}

    # offsets in metres from the image's upper-left corner (x east, y south)
    det_int = np.array([p["center_px"] for p in det["intersections"]]).reshape(-1, 2) * gsd
    osm_int = np.array([[g.x, g.y] for g in inter.geometry]).reshape(-1, 2)
    objs = [o for o in det["objects"] if o["use_for_locate"] and o["class"] in osm.DOTA_WEIGHT]
    det_fac = np.array([o["center_px"] for o in objs]).reshape(-1, 2) * gsd
    osm_fac = np.array([[c.x, c.y] for c in fac.geometry.centroid]).reshape(-1, 2)

    def block_maps(blk):
        """(a) mask NCC and (b) vote maps for the upper-left corners blk covers."""
        R = search_raster("roads", blk, RES, lambda a: rasterize_roads(roads, a, RES))
        M = {"road": mask_ncc(R, T["road"])[0]}
        zeros = np.zeros_like(M["road"])
        M["rail"] = (mask_ncc(search_raster("rails", blk, RES, lambda a: rasterize_roads(rails[rails.active], a, RES)),
                              T["rail"])[0] if "rail" in T else zeros)
        M["forest"] = (mask_ncc(search_raster("forest", blk, RES, lambda a: edge_band(rasterize_areas(forest, a, RES) > 0.5)),
                                T["forest"])[0] if "forest" in T else zeros)
        M["building"] = (mask_ncc(search_raster("buildings", blk, RES, lambda a: rasterize_areas(layers["buildings"], a, RES)),
                                  T["building"])[0] if "building" in T else zeros)
        p, big = padded(blk, RES)  # vote blur must not see the block border either
        shape = (M["road"].shape[0] + 2 * p, M["road"].shape[1] + 2 * p)
        M["inter"] = vote_map(shape, big, RES, det_int, [1.0] * len(det_int), osm_int)[p:-p, p:-p]
        M["fac"] = vote_map(shape, big, RES, det_fac, [osm.DOTA_WEIGHT[o["class"]] for o in objs],
                            osm_fac, fac.dota_class.to_numpy(), [o["class"] for o in objs])[p:-p, p:-p]
        M["C"] = (M["road"] + W_RAIL * M["rail"] + W_FOREST * M["forest"] + W_BUILDING * M["building"]
                  + W_INTER * M["inter"] + W_FAC * M["fac"])
        return M

    # upper-left corners on the global RES grid, scanned in blocks; each block reaches
    # PEAK_SEPARATION_M into its neighbours so the local-maximum test is exact
    th, tw = T["road"].shape
    rows = int(round((area[3] - area[1]) / RES)) - th + 1
    cols = int(round((area[2] - area[0]) / RES)) - tw + 1
    step, sep = int(BLOCK_M / RES), int(PEAK_SEPARATION_M / RES)
    cands = []  # (score, row, col, {map: value})
    for r0 in range(0, rows, step):
        for c0 in range(0, cols, step):
            r1, c1 = min(r0 + step, rows), min(c0 + step, cols)
            er0, er1, ec0, ec1 = max(r0 - sep, 0), min(r1 + sep, rows), max(c0 - sep, 0), min(c1 + sep, cols)
            blk = (area[0] + ec0 * RES, area[3] - (er1 + th - 1) * RES,
                   area[0] + (ec1 + tw - 1) * RES, area[3] - er0 * RES)
            M = block_maps(blk)
            for i, j in peaks(M["C"], RES, keep=(r0 - er0, r1 - er0, c0 - ec0, c1 - ec0)):
                cands.append((float(M["C"][i, j]), er0 + i, ec0 + j, {k: float(m[i, j]) for k, m in M.items()}))
    if not cands:
        return {"status": "not localizable", "reason": "no peak"}
    cands.sort(key=lambda c: -c[0])
    _, bi, bj, at = cands[0]
    second = cands[1][0] if len(cands) > 1 else 0.0
    e = area[0] + bj * RES
    n = area[3] - bi * RES
    runner_up_dist = (float(np.hypot((cands[1][2] - bj) * RES, (cands[1][1] - bi) * RES))
                      if len(cands) > 1 else None)

    # (c) local refinement at mask resolution, then least squares on intersections
    e, n = refine_roads(roads, masks["road"], e, n, mres)
    e, n, n_match, rms = refine_intersections(det_int, osm_int, e, n)
    n_fac = count_facility_matches(objs, det_fac, fac, e, n)

    score, margin = at["C"], at["C"] - second
    status = "ok" if (margin >= MIN_MARGIN and score >= MIN_SCORE) else "not localizable"
    to_wgs = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    lon, lat = to_wgs.transform(e + ext_e / 2, n - ext_n / 2)
    return {"status": status, "ul_e": round(e, 2), "ul_n": round(n, 2),
            "center_lat": round(lat, 6), "center_lon": round(lon, 6),
            "score": round(score, 3), "runner_up": round(float(second), 3), "margin": round(margin, 3),
            "runner_up_dist_m": None if runner_up_dist is None else round(runner_up_dist),
            "road_ncc": round(at["road"], 3), "rail_ncc": round(at["rail"], 3),
            "forest_ncc": round(at["forest"], 3), "building_ncc": round(at["building"], 3),
            "inter_votes": round(at["inter"], 3), "fac_votes": round(at["fac"], 3),
            **{f"{k}_signal": round(signal.get(k, 0.0), 4) for k in ("road", "rail", "forest", "building")},
            "n_det_intersections": len(det_int), "matched_intersections": n_match,
            "intersection_rms_m": rms, "matched_facilities": n_fac,
            "error_estimate_m": round(rms / np.sqrt(n_match), 2) if n_match >= 2 else None}


def refine_roads(roads, mask, e, n, mres):
    r = REFINE_RADIUS_M
    h, w = mask.shape
    area = (e - r, n - h * mres - r, e + w * mres + r, n + r)
    R = gaussian_filter(rasterize_roads(roads, area, mres), 1.0)
    T = gaussian_filter(mask, 1.0)
    if T.mean() < 0.002:
        return e, n
    M = ncc_map(R, T)
    i, j = np.unravel_index(np.argmax(M), M.shape)
    di = dj = 0.0
    if 0 < i < M.shape[0] - 1 and 0 < j < M.shape[1] - 1:  # parabolic sub-pixel peak
        a, b, c = M[i - 1, j], M[i, j], M[i + 1, j]
        di = 0.5 * (a - c) / (a - 2 * b + c + 1e-9)
        a, b, c = M[i, j - 1], M[i, j], M[i, j + 1]
        dj = 0.5 * (a - c) / (a - 2 * b + c + 1e-9)
    return area[0] + (j + dj) * mres, area[3] - (i + di) * mres


def refine_intersections(det_int, osm_int, e, n, iters=3):
    if len(det_int) == 0 or len(osm_int) == 0:
        return e, n, 0, None
    tree = cKDTree(osm_int)
    for _ in range(iters):
        world = np.c_[e + det_int[:, 0], n - det_int[:, 1]]
        d, idx = tree.query(world, distance_upper_bound=MATCH_RADIUS_M)
        ok = np.isfinite(d)
        if ok.sum() < 3:
            break
        shift = np.median(osm_int[idx[ok]] - world[ok], axis=0)
        e, n = e + shift[0], n + shift[1]
    world = np.c_[e + det_int[:, 0], n - det_int[:, 1]]
    d, _ = tree.query(world, distance_upper_bound=MATCH_RADIUS_M)
    ok = np.isfinite(d)
    rms = round(float(np.sqrt((d[ok] ** 2).mean())), 2) if ok.any() else None
    return e, n, int(ok.sum()), rms


def count_facility_matches(objs, det_fac, fac, e, n, radius=15):
    hits = 0
    for o, (dx, dy) in zip(objs, det_fac):
        cand = fac[fac.dota_class == o["class"]]
        if len(cand) and cand.distance(box(e + dx - 0.1, n - dy - 0.1, e + dx + 0.1, n - dy + 0.1)).min() < radius:
            hits += 1
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("det", type=Path, help="detect.py JSON (roadmask PNG next to it)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--search", help="minE,minN,maxE,maxN in EPSG:25832")
    g.add_argument("--place", help="place name, searched within +-5 km (Nominatim)")
    a = ap.parse_args()
    area = (tuple(float(v) for v in a.search.split(",")) if a.search
            else place_area(a.place) if a.place else osm.AREA)
    det, masks = load_detections(a.det)
    print(json.dumps({"image": det["image"], **locate(det, masks, area, osm_for(area))}, indent=1))


if __name__ == "__main__":
    main()
