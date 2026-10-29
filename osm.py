"""Fetch OSM features for the search area via Overpass and turn them into
matching layers in EPSG:25832:

  roads          buffered-width street centrelines (LineString, highway, width_m)
  intersections  junction points where >= 3 street segments meet (degree)
  facilities     static objects mapped to DOTA class names (dota_class)
  railways       track centrelines (LineString, railway, width_m, active); fetched
                 with a separate query and cache so the street data stays unchanged

Usage:
  uv run osm.py                      # fetch (cached) + build outputs/osm/rosenheim.gpkg
  uv run osm.py --overlay 32732_5306 # Gate A: draw OSM labels over a tile
"""
import argparse
import json
import time
from collections import defaultdict

import geopandas as gpd
import numpy as np
import requests
from pyproj import Transformer
from shapely.geometry import LineString, MultiPoint, Point, Polygon
from shapely.ops import polygonize, unary_union

from tiles import CRS, OUT, load_rgb, tile_origin, tile_path

# Search area: the 54-tile extent plus 1 km, in EPSG:25832.
AREA = (727000, 5298000, 736000, 5310000)  # minE, minN, maxE, maxN
OVERPASS_URLS = ("https://overpass-api.de/api/interpreter",
                 "https://overpass.private.coffee/api/interpreter",
                 "https://maps.mail.ru/osm/tools/overpass/api/interpreter")
USER_AGENT = "YOLO26-detection-poc/0.1 (aerial geolocation research)"

ROAD_WIDTH = {
    "motorway": 12, "trunk": 10, "primary": 8, "secondary": 7, "tertiary": 6,
    "motorway_link": 6, "trunk_link": 6, "primary_link": 6, "secondary_link": 5,
    "tertiary_link": 5, "residential": 5, "unclassified": 5, "living_street": 4.5,
    "service": 3.5, "track": 3,
}
# Small or hidden service ways make noisy labels and spurious junctions.
SKIP_SERVICE = {"parking_aisle", "driveway", "drive-through", "emergency_access"}
JUNCTION_MERGE_M = 8
# Each track is its own OSM way; width covers sleepers + ballast of one track.
RAIL_WIDTH = {"rail": 4, "light_rail": 3.5, "narrow_gauge": 3}
RAIL_INACTIVE = {"abandoned", "disused"}  # may or may not still be visible: ignored in training

# DOTA class -> weight for voting (rarity x OSM completeness). Vehicles are never used.
DOTA_WEIGHT = {
    "ground track field": 5.0, "soccer ball field": 2.0, "tennis court": 1.5,
    "basketball court": 1.5, "baseball diamond": 3.0, "roundabout": 4.0,
    "bridge": 2.0, "storage tank": 2.0, "swimming pool": 0.5, "harbor": 2.0,
    "helicopter": 3.0, "plane": 3.0,
}


def to_wgs84_bbox(area):
    """Overpass wants (south, west, north, east) in WGS84."""
    t = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    e0, n0, e1, n1 = area
    lons, lats = t.transform([e0, e0, e1, e1], [n0, n1, n0, n1])
    return min(lats), min(lons), max(lats), max(lons)


def query(area):
    s, w, n, e = to_wgs84_bbox(area)
    bb = f"{s:.6f},{w:.6f},{n:.6f},{e:.6f}"
    hw = "|".join(ROAD_WIDTH)
    return f"""[out:json][timeout:300];
(
  way["highway"~"^({hw})$"]({bb});
  nwr["leisure"~"^(pitch|track|swimming_pool|marina)$"]({bb});
  nwr["sport"="athletics"]({bb});
  nwr["man_made"="storage_tank"]({bb});
  way["bridge"="yes"]({bb});
  nwr["aeroway"~"^(helipad|aerodrome)$"]({bb});
);
out body geom;"""


def rail_query(area):
    s, w, n, e = to_wgs84_bbox(area)
    kinds = "|".join([*RAIL_WIDTH, *RAIL_INACTIVE])
    return f"""[out:json][timeout:300];
way["railway"~"^({kinds})$"]({s:.6f},{w:.6f},{n:.6f},{e:.6f});
out body geom;"""


def fetch(name="rosenheim", area=AREA, refresh=False, kind="raw", make_query=query):
    cache = OUT / "osm" / f"{name}_{kind}.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())
    cache.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        for url in OVERPASS_URLS:
            try:
                r = requests.post(url, data={"data": make_query(area)},
                                  headers={"User-Agent": USER_AGENT}, timeout=360)
                r.raise_for_status()
                data = r.json()
            except (requests.RequestException, ValueError) as e:
                print(f"overpass {url}: {e}")
                continue
            cache.write_text(r.text)
            return data
        time.sleep(30 * (attempt + 1))
    raise RuntimeError("all Overpass endpoints failed")


def _geom(el, proj):
    """Shapely geometry in EPSG:25832 for a way/node/relation from `out geom`."""
    if el["type"] == "node":
        return Point(proj(el["lon"], el["lat"]))
    if el["type"] == "way":
        pts = [proj(p["lon"], p["lat"]) for p in el.get("geometry", [])]
        if len(pts) < 2:
            return None
        if len(pts) >= 4 and el["nodes"][0] == el["nodes"][-1]:
            return Polygon(pts)
        return LineString(pts)
    lines = [LineString([proj(p["lon"], p["lat"]) for p in m["geometry"]])
             for m in el.get("members", []) if m.get("role") in ("outer", "")
             and len(m.get("geometry", [])) >= 2]
    polys = list(polygonize(lines))
    return unary_union(polys) if polys else None


def dota_class(tags):
    sport = tags.get("sport", "")
    leisure = tags.get("leisure")
    if leisure == "pitch":
        for key, cls in (("tennis", "tennis court"), ("soccer", "soccer ball field"),
                         ("basketball", "basketball court"), ("baseball", "baseball diamond")):
            if key in sport:
                return cls
    if leisure == "track" or (sport == "athletics" and leisure in ("track", "pitch", None)):
        return "ground track field"
    if leisure == "swimming_pool":
        return "swimming pool"
    if leisure == "marina":
        return "harbor"
    if tags.get("man_made") == "storage_tank":
        return "storage tank"
    if tags.get("aeroway") == "helipad":
        return "helicopter"
    if tags.get("aeroway") == "aerodrome":
        return "plane"
    return None


def is_street(tags):
    return (tags.get("highway") in ROAD_WIDTH
            and tags.get("service") not in SKIP_SERVICE
            and tags.get("tunnel") not in ("yes", "building_passage")
            and tags.get("area") != "yes")


def build(raw):
    proj = Transformer.from_crs("EPSG:4326", CRS, always_xy=True).transform
    roads, facilities = [], []
    neighbours = defaultdict(set)   # node id -> adjacent node ids along streets
    node_xy = {}
    roundabout_nodes = []

    for el in raw["elements"]:
        tags = el.get("tags", {})
        if el["type"] == "way" and is_street(tags):
            pts = [proj(p["lon"], p["lat"]) for p in el["geometry"]]
            roads.append({"geometry": LineString(pts), "osm_id": el["id"],
                          "highway": tags["highway"],
                          "width_m": ROAD_WIDTH[tags["highway"]]})
            ids = el["nodes"]
            for nid, xy in zip(ids, pts):
                node_xy[nid] = xy
            for a, b in zip(ids, ids[1:]):
                neighbours[a].add(b)
                neighbours[b].add(a)
            if tags.get("junction") in ("roundabout", "circular"):
                roundabout_nodes.append(set(ids))
            if tags.get("bridge") == "yes":
                facilities.append({"geometry": LineString(pts).buffer(ROAD_WIDTH[tags["highway"]] / 2),
                                   "osm_id": el["id"], "dota_class": "bridge"})
            continue
        cls = dota_class(tags)
        if cls is None and tags.get("bridge") == "yes":
            cls = "bridge"
        if cls is None:
            continue
        g = _geom(el, proj)
        if g is None or g.is_empty:
            continue
        if g.geom_type == "LineString":
            g = g.buffer(2.0)
        facilities.append({"geometry": g, "osm_id": el["id"], "dota_class": cls})

    facilities += _roundabouts(roundabout_nodes, node_xy)
    inter = _intersections(neighbours, node_xy)
    to_gdf = lambda rows: gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS)
    return to_gdf(roads), to_gdf(inter), to_gdf(facilities)


def build_rails(raw):
    proj = Transformer.from_crs("EPSG:4326", CRS, always_xy=True).transform
    rows = []
    for el in raw["elements"]:
        tags = el.get("tags", {})
        kind = tags.get("railway")
        if el["type"] != "way" or kind not in RAIL_WIDTH.keys() | RAIL_INACTIVE:
            continue
        if tags.get("tunnel", "no") != "no" or len(el.get("geometry", [])) < 2:
            continue
        rows.append({"geometry": LineString([proj(p["lon"], p["lat"]) for p in el["geometry"]]),
                     "osm_id": el["id"], "railway": kind,
                     "width_m": RAIL_WIDTH.get(kind, 4), "active": kind in RAIL_WIDTH})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS)


def build_all(name="rosenheim", area=AREA, refresh=False):
    """Fetch (cached) and write all four layers to outputs/osm/<name>.gpkg."""
    roads, inter, fac = build(fetch(name, area, refresh))
    rails = build_rails(fetch(name, area, refresh, "rail", rail_query))
    gpkg = OUT / "osm" / f"{name}.gpkg"
    gpkg.unlink(missing_ok=True)
    for layer, gdf in zip(LAYERS, (roads, inter, fac, rails)):
        gdf.to_file(gpkg, layer=layer)
    return roads, inter, fac, rails


def _roundabouts(node_sets, node_xy):
    """Roundabouts are often split into several ways; merge ways sharing nodes."""
    groups = []
    for s in node_sets:
        merged = [g for g in groups if g & s]
        for g in merged:
            groups.remove(g)
            s = s | g
        groups.append(s)
    out = []
    for g in groups:
        hull = MultiPoint([node_xy[n] for n in g]).convex_hull
        if hull.geom_type == "Polygon" and hull.area > 20:
            out.append({"geometry": hull, "osm_id": -1, "dota_class": "roundabout"})
    return out


def _intersections(neighbours, node_xy):
    """Nodes where >= 3 street segments meet, merged within JUNCTION_MERGE_M."""
    cand = [(n, len(nb)) for n, nb in neighbours.items() if len(nb) >= 3]
    if not cand:
        return []
    xy = np.array([node_xy[n] for n, _ in cand])
    deg = np.array([d for _, d in cand])
    from scipy.cluster.hierarchy import fcluster, linkage
    labels = (fcluster(linkage(xy, "single"), JUNCTION_MERGE_M, "distance")
              if len(xy) > 1 else np.array([1]))
    out = []
    for lab in np.unique(labels):
        m = labels == lab
        out.append({"geometry": Point(xy[m].mean(0)), "degree": int(deg[m].max()),
                    "n_nodes": int(m.sum())})
    return out


LAYERS = ("roads", "intersections", "facilities", "railways")


def load_layers(name="rosenheim"):
    gpkg = OUT / "osm" / f"{name}.gpkg"
    return tuple(gpd.read_file(gpkg, layer=l) for l in LAYERS)


def overlay(tile, crops=((1000, 2200), (2600, 3600), (300, 300)), size=1500):
    """Gate A: draw OSM road polygons + intersection disks over full-res crops."""
    from PIL import Image, ImageDraw
    roads, inter, fac, rails = load_layers()
    e0, n0, gsd = tile_origin(tile_path(tile))
    img = load_rgb(tile_path(tile))
    to_px = lambda coords: [((x - e0) / gsd, (n0 - y) / gsd) for x, y in coords]
    base = Image.fromarray(img)
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for _, r in roads.iterrows():
        poly = r.geometry.buffer(r.width_m / 2, cap_style="flat")
        for p in getattr(poly, "geoms", [poly]):
            d.polygon(to_px(p.exterior.coords), fill=(255, 0, 0, 70), outline=(255, 0, 0, 200))
        d.line(to_px(r.geometry.coords), fill=(255, 255, 0, 255), width=2)
    for _, r in rails.iterrows():
        poly = r.geometry.buffer(r.width_m / 2, cap_style="flat")
        color = (0, 120, 255) if r.active else (255, 0, 255)
        for p in getattr(poly, "geoms", [poly]):
            d.polygon(to_px(p.exterior.coords), fill=(*color, 70), outline=(*color, 220))
    for _, r in inter.iterrows():
        x, y = to_px(r.geometry.coords)[0]
        rad = 8 / gsd
        d.ellipse((x - rad, y - rad, x + rad, y + rad), outline=(0, 255, 255, 255), width=4)
    for _, r in fac.iterrows():
        for p in getattr(r.geometry, "geoms", [r.geometry]):
            if p.geom_type == "Polygon":
                d.polygon(to_px(p.exterior.coords), outline=(0, 255, 0, 255), width=4)
    comp = Image.alpha_composite(base.convert("RGBA"), layer).convert("RGB")
    out_dir = OUT / "gateA"
    out_dir.mkdir(parents=True, exist_ok=True)
    comp.resize((1500, 1500)).save(out_dir / f"{tile}_overview.jpg", quality=85)
    for i, (r0, c0) in enumerate(crops):
        comp.crop((c0, r0, c0 + size, r0 + size)).resize((1000, 1000)).save(
            out_dir / f"{tile}_crop{i}.jpg", quality=88)
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-query Overpass")
    ap.add_argument("--overlay", metavar="TILE")
    a = ap.parse_args()
    if a.overlay:
        print(overlay(a.overlay))
        return
    roads, inter, fac, rails = build_all(refresh=a.refresh)
    print(f"bbox S,W,N,E = {tuple(round(v, 5) for v in to_wgs84_bbox(AREA))}")
    print(f"roads {len(roads)}  ({roads.length.sum() / 1000:.0f} km)")
    print(f"intersections {len(inter)}  degree: {inter.degree.value_counts().sort_index().to_dict()}")
    print(f"facilities: {fac.dota_class.value_counts().to_dict()}")
    print(f"railways: {rails.length.groupby(rails.railway).sum().div(1000).round(1).to_dict()} km")


if __name__ == "__main__":
    main()
