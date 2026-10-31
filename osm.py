"""Read OSM features for the search area from a local PostGIS database (loaded from
the Geofabrik extract with osm_flex.lua) or straight from the .osm.pbf (pyosmium;
Overpass is archived in overpass.py) and turn them into matching layers in EPSG:25832:

  roads          buffered-width street centrelines (LineString, highway, width_m)
  intersections  junction points where >= 3 street segments meet (degree)
  facilities     static objects mapped to DOTA class names (dota_class)
  railways       track centrelines (LineString, railway, width_m, active)
  forest         landuse=forest / natural=wood polygons (ways and multipolygon
                 relations, clearings as holes), dissolved
  buildings      building=* footprints (ways and multipolygon relations, courtyards
                 as holes; underground ones dropped), dissolved

Both sources give the same elements per kind (SELECT); PBF reads are cached in
outputs/osm/<name>_<kind>.json, PostGIS answers in well under a second.

Usage:
  uv run osm.py                      # read PostGIS + build outputs/osm/rosenheim.gpkg
  uv run osm.py --source pbf         # same from the .osm.pbf (cached), no database needed
  uv run osm.py --overlay 32732_5306 # Gate A: draw OSM labels over a tile
"""
import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

import geopandas as gpd
import numpy as np
from pyproj import Transformer
import shapely
from shapely import from_wkb
from shapely.geometry import LineString, MultiPoint, Point, Polygon, box
from shapely.ops import polygonize, unary_union

from tiles import CRS, OUT, load_rgb, tile_origin, tile_path

# Search area: the 54-tile extent plus 1 km, in EPSG:25832.
AREA = (727000, 5298000, 736000, 5310000)  # minE, minN, maxE, maxN
# Geofabrik extract, read with pyosmium; snapshot date is in its header (pbf_info).
PBF = OUT / "osm" / "pbf" / "bayern-latest.osm.pbf"
# Local PostGIS with that extract imported (osm2pgsql -O flex -S osm_flex.lua); see README.
DSN = os.environ.get("OSM_DSN", "dbname=osm")
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
BUILDING_MIN_M2 = 10  # smaller footprints (bins, kiosks) are too small to segment reliably at 0.2 m

# DOTA class -> weight for voting (rarity x OSM completeness). Vehicles are never used.
DOTA_WEIGHT = {
    "ground track field": 5.0, "soccer ball field": 2.0, "tennis court": 1.5,
    "basketball court": 1.5, "baseball diamond": 3.0, "roundabout": 4.0,
    "bridge": 2.0, "storage tank": 2.0, "swimming pool": 0.5, "harbor": 2.0,
    "helicopter": 3.0, "plane": 3.0,
}


def to_wgs84_bbox(area):
    """(south, west, north, east) in WGS84, the order Overpass wants."""
    t = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    e0, n0, e1, n1 = area
    lons, lats = t.transform([e0, e0, e1, e1], [n0, n1, n0, n1])
    return min(lats), min(lons), max(lats), max(lons)


# What each cache kind holds; same selection as the archived Overpass queries (overpass.py).
# t is the osmium type char ("n", "w", "r"), tags an osmium TagList or dict.
SELECT = {
    "raw": lambda t, tags: (t == "w" and tags.get("highway") in ROAD_WIDTH
                            or tags.get("leisure") in ("pitch", "track", "swimming_pool", "marina")
                            or tags.get("sport") == "athletics"
                            or tags.get("man_made") == "storage_tank"
                            or t == "w" and tags.get("bridge") == "yes"
                            or tags.get("aeroway") in ("helipad", "aerodrome")),
    "rail": lambda t, tags: t == "w" and tags.get("railway") in RAIL_WIDTH.keys() | RAIL_INACTIVE,
    "forest": lambda t, tags: t != "n" and (tags.get("landuse") == "forest" or tags.get("natural") == "wood"),
    "building": lambda t, tags: t != "n" and "building" in tags,
}
SELECT_KEYS = ("highway", "leisure", "sport", "man_made", "bridge", "aeroway",
               "railway", "landuse", "natural", "building")
FAR_DEG = 0.5  # ways whose first node is this far outside the area are skipped (none is ~40 km long)
MEMBER_TYPE = {"n": "node", "w": "way", "r": "relation"}


def pbf_info(pbf=PBF):
    """Snapshot timestamp, coverage (minlon, minlat, maxlon, maxlat) and md5 of the extract."""
    import hashlib
    import osmium
    if not pbf.exists():
        raise FileNotFoundError(f"{pbf} missing; download it with\n  curl -L --create-dirs -o {pbf} "
                                f"https://download.geofabrik.de/europe/germany/{pbf.name}")
    r = osmium.io.Reader(str(pbf), osmium.osm.osm_entity_bits.NOTHING)
    h = r.header()
    b = h.box()
    r.close()
    md5 = hashlib.md5()
    with open(pbf, "rb") as f:
        while chunk := f.read(1 << 24):
            md5.update(chunk)
    return {"file": pbf.name, "timestamp": h.get("osmosis_replication_timestamp"), "md5": md5.hexdigest(),
            "box": (b.bottom_left.lon, b.bottom_left.lat, b.top_right.lon, b.top_right.lat)}


def read_pbf(area, pbf=PBF):
    """{kind: elements} for each SELECT kind, in Overpass `out body geom` form so build*
    work unchanged: nodes inside the area, ways with a segment in it and relations with
    a member way in it, all with full geometry, sorted by type and id like Overpass."""
    import osmium
    from osmium.filter import EntityFilter, IdFilter, KeyFilter
    info = pbf_info(pbf)
    s, w, n, e = to_wgs84_bbox(area)
    x0, y0, x1, y1 = info["box"]
    if not (x0 <= w and y0 <= s and e <= x1 and n <= y1):
        raise ValueError(f"search area {area} is outside {pbf.name}; use another Geofabrik "
                         "extract (--pbf) or the archived Overpass source (overpass.py)")
    bbox = box(w, s, e, n)
    pt = lambda loc: {"lat": round(loc.lat, 7), "lon": round(loc.lon, 7)}

    def touches(geom):
        lons, lats = [p["lon"] for p in geom], [p["lat"] for p in geom]
        if max(lons) < w or min(lons) > e or max(lats) < s or min(lats) > n:
            return False
        return len(geom) == 1 or LineString([(p["lon"], p["lat"]) for p in geom]).intersects(bbox)

    # 1) relations (no locations needed); their member ways are read in pass 3
    rels, member_ids = [], set()
    for r in osmium.FileProcessor(str(pbf), osmium.osm.RELATION).with_filter(KeyFilter(*SELECT_KEYS)):
        kinds = [k for k, f in SELECT.items() if f("r", r.tags)]
        if kinds:
            rels.append((kinds, r.id, dict(r.tags), [(m.type, m.ref, m.role) for m in r.members]))
            member_ids.update(m.ref for m in r.members if m.type == "w")

    # 2) tagged nodes and ways
    out = {k: [] for k in SELECT}
    fp = osmium.FileProcessor(str(pbf), osmium.osm.NODE | osmium.osm.WAY).with_locations()
    for o in fp.with_filter(KeyFilter(*SELECT_KEYS)):
        if o.is_node():
            if not (w <= o.location.lon <= e and s <= o.location.lat <= n):
                continue
            el = {"type": "node", "id": o.id, **pt(o.location)}
        else:
            if not o.nodes or not o.nodes[0].location.valid():
                continue
            lon, lat = o.nodes[0].lon, o.nodes[0].lat
            if not (w - FAR_DEG < lon < e + FAR_DEG and s - FAR_DEG < lat < n + FAR_DEG):
                continue
            el = {"type": "way", "id": o.id, "nodes": [nd.ref for nd in o.nodes],
                  "geometry": [pt(nd.location) for nd in o.nodes if nd.location.valid()]}
            if not touches(el["geometry"]):
                continue
        kinds = [k for k, f in SELECT.items() if f(o.type_str(), o.tags)]
        if kinds:
            el["tags"] = dict(o.tags)
            for k in kinds:
                out[k].append(el)

    # 3) relation member ways, tagged or not
    geoms = {}
    fp = osmium.FileProcessor(str(pbf), osmium.osm.NODE | osmium.osm.WAY).with_locations()
    for o in fp.with_filter(EntityFilter(osmium.osm.WAY)).with_filter(IdFilter(member_ids)):
        geoms[o.id] = [pt(nd.location) for nd in o.nodes if nd.location.valid()]
    for kinds, rid, tags, members in rels:
        if not any(t == "w" and ref in geoms and touches(geoms[ref]) for t, ref, _ in members):
            continue
        el = {"type": "relation", "id": rid,
              "members": [{"type": MEMBER_TYPE[t], "ref": ref, "role": role,
                           **({"geometry": geoms[ref]} if t == "w" and ref in geoms else {})}
                          for t, ref, role in members], "tags": tags}
        for k in kinds:
            out[k].append(el)

    order = {"node": 0, "way": 1, "relation": 2}
    meta = {"generator": "osm.read_pbf", "source": {k: v for k, v in info.items() if k != "box"}}
    return {k: {**meta, "elements": sorted(els, key=lambda el: (order[el["type"]], el["id"]))}
            for k, els in out.items()}


def read_postgis(area, dsn=DSN):
    """{kind: elements} like read_pbf, from the osm_flex.lua tables. Relations carry
    their assembled multipolygon (`polygon_wkb`, WGS84) instead of member geometries,
    since untagged member ways aren't in the database; relations osm2pgsql couldn't
    assemble (no polygon) are left out."""
    import psycopg
    s, w, n, e = to_wgs84_bbox(area)
    env = "ST_MakeEnvelope(%(w)s, %(s)s, %(e)s, %(n)s, 4326)"
    bb = {"w": w, "s": s, "e": e, "n": n}
    pt = lambda x, y: {"lat": round(y, 7), "lon": round(x, 7)}
    out = {k: [] for k in SELECT}
    with psycopg.connect(dsn) as con:
        props = dict(con.execute("select property, value from osm2pgsql_properties"))
        q = lambda sql: con.execute(sql.replace("{env}", env), bb)
        for nid, kinds, tags, wkb in q("select node_id, kinds, tags, ST_AsBinary(geom) from osm_nodes "
                                       "where ST_Intersects(geom, {env}) order by 1"):
            g = from_wkb(wkb)
            for k in kinds:
                out[k].append({"type": "node", "id": nid, **pt(g.x, g.y), "tags": tags})
        for wid, kinds, tags, nodes, wkb in q("select way_id, kinds, tags, nodes, ST_AsBinary(geom) from osm_ways "
                                              "where ST_Intersects(geom, {env}) order by 1"):
            el = {"type": "way", "id": wid, "nodes": nodes,
                  "geometry": [pt(x, y) for x, y in from_wkb(wkb).coords], "tags": tags}
            for k in kinds:
                out[k].append(el)
        for rid, kinds, tags, members, wkb in q(  # a member way touches the area = the rings do
                "select relation_id, kinds, tags, members, ST_AsBinary(geom) from osm_relations "
                "where geom is not null and ST_Intersects(ST_Boundary(geom), {env}) order by 1"):
            el = {"type": "relation", "id": rid, "tags": tags, "polygon_wkb": wkb.hex(),
                  "members": [{"type": MEMBER_TYPE[m["type"]], "ref": m["ref"], "role": m["role"]}
                              for m in members]}
            for k in kinds:
                out[k].append(el)
    meta = {"generator": "osm.read_postgis",
            "source": {"database": dsn, "timestamp": props.get("replication_timestamp"),
                       "style": Path(props.get("style", "")).name}}
    return {k: {**meta, "elements": els} for k, els in out.items()}


def fetch(name="rosenheim", area=AREA, refresh=False, source="postgis", pbf=PBF):
    """{kind: elements} for the area from PostGIS, or from the PBF cached per kind
    in outputs/osm/<name>_<kind>.json."""
    if source == "postgis":
        import psycopg
        try:
            return read_postgis(area)
        except psycopg.OperationalError as e:
            print(f"PostGIS not reachable ({str(e).strip().splitlines()[0]}); reading the PBF instead")
    cache = lambda kind: OUT / "osm" / f"{name}_{kind}.json"
    if refresh or not all(cache(k).exists() for k in SELECT):
        cache("raw").parent.mkdir(parents=True, exist_ok=True)
        for kind, data in read_pbf(area, pbf).items():
            cache(kind).write_text(json.dumps(data))
    return {k: json.loads(cache(k).read_text()) for k in SELECT}


def _project_wkb(hexwkb, proj):
    return shapely.transform(from_wkb(bytes.fromhex(hexwkb)), lambda xy: np.column_stack(proj(xy[:, 0], xy[:, 1])))


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
    if "polygon_wkb" in el:  # read_postgis; outer rings only, as below
        g = _project_wkb(el["polygon_wkb"], proj)
        return unary_union([Polygon(p.exterior) for p in getattr(g, "geoms", [g])])
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


def build_areas(raw, min_area, keep=lambda tags: True):
    """Dissolved polygons > min_area m2 from closed ways and multipolygon
    relations (outer rings minus inner rings), for elements whose tags pass `keep`."""
    proj = Transformer.from_crs("EPSG:4326", CRS, always_xy=True).transform
    ring = lambda geom: LineString([proj(p["lon"], p["lat"]) for p in geom])
    polys = []
    for el in raw["elements"]:
        if not keep(el.get("tags", {})):
            continue
        if el["type"] == "way":
            pts = [proj(p["lon"], p["lat"]) for p in el.get("geometry", [])]
            if len(pts) >= 4 and el["nodes"][0] == el["nodes"][-1]:
                polys.append(Polygon(pts).buffer(0))
        elif "polygon_wkb" in el:  # read_postgis: already assembled with its holes
            polys.append(_project_wkb(el["polygon_wkb"], proj).buffer(0))
        elif el["type"] == "relation":
            rings = {"outer": [], "inner": []}
            for m in el.get("members", []):
                if m["type"] == "way" and len(m.get("geometry", [])) >= 2:
                    rings["inner" if m.get("role") == "inner" else "outer"].append(ring(m["geometry"]))
            outer = unary_union(list(polygonize(rings["outer"])))
            if not outer.is_empty:
                polys.append(outer.difference(unary_union(list(polygonize(rings["inner"])))).buffer(0))
    merged = unary_union([p for p in polys if not p.is_empty])
    parts = [p for p in getattr(merged, "geoms", [merged]) if p.geom_type == "Polygon" and p.area > min_area]
    return gpd.GeoDataFrame({"geometry": parts}, geometry="geometry", crs=CRS)


def build_forest(raw):
    """Dissolved forest polygons, clearings as holes."""
    return build_areas(raw, 100)


def is_visible_building(tags):
    """Drop buildings that can't be seen from above."""
    return (tags.get("building") not in ("no", "underground")
            and tags.get("location") not in ("underground", "indoor")
            and not tags.get("layer", "0").lstrip().startswith("-"))


def build_buildings(raw):
    """Dissolved building footprints (adjoining houses merge into one block), courtyards as holes."""
    return build_areas(raw, BUILDING_MIN_M2, is_visible_building)


def build_all(name="rosenheim", area=AREA, refresh=False, source="postgis", pbf=PBF):
    """Read and write all layers to outputs/osm/<name>.gpkg."""
    raw = fetch(name, area, refresh, source, pbf)
    src = raw["raw"]["source"]
    print(f"OSM snapshot {src['timestamp']} from {src.get('database') or src['file']}")
    roads, inter, fac = build(raw["raw"])
    layers = {"roads": roads, "intersections": inter, "facilities": fac,
              "railways": build_rails(raw["rail"]),
              "forest": build_forest(raw["forest"]),
              "buildings": build_buildings(raw["building"])}
    gpkg = OUT / "osm" / f"{name}.gpkg"
    gpkg.unlink(missing_ok=True)
    for layer, gdf in layers.items():
        gdf.to_file(gpkg, layer=layer)
    return layers


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


LAYERS = ("roads", "intersections", "facilities", "railways", "forest", "buildings")


def load_layers(name="rosenheim"):
    gpkg = OUT / "osm" / f"{name}.gpkg"
    return {l: gpd.read_file(gpkg, layer=l) for l in LAYERS}


def overlay(tile, crops=((1000, 2200), (2600, 3600), (300, 300)), size=1500):
    """Gate A: draw OSM road polygons + intersection disks over full-res crops."""
    from PIL import Image, ImageDraw
    L = load_layers()
    roads, inter, fac, rails = L["roads"], L["intersections"], L["facilities"], L["railways"]
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
    for g in L["forest"].geometry:
        for r in [g.exterior, *g.interiors]:
            d.line(to_px(r.coords), fill=(0, 255, 0, 255), width=5)
    for g in L["buildings"].geometry:
        for r in [g.exterior, *g.interiors]:
            d.line(to_px(r.coords), fill=(255, 0, 255, 255), width=3)
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
    ap.add_argument("--source", choices=("postgis", "pbf"), default="postgis")
    ap.add_argument("--refresh", action="store_true", help="re-read the PBF extract (--source pbf)")
    ap.add_argument("--pbf", type=Path, default=PBF, help="Geofabrik .osm.pbf extract")
    ap.add_argument("--overlay", metavar="TILE")
    a = ap.parse_args()
    if a.overlay:
        print(overlay(a.overlay))
        return
    L = build_all(refresh=a.refresh, source=a.source, pbf=a.pbf)
    roads, inter, fac, rails, forest, bld = (L[k] for k in LAYERS)
    print(f"bbox S,W,N,E = {tuple(round(v, 5) for v in to_wgs84_bbox(AREA))}")
    print(f"roads {len(roads)}  ({roads.length.sum() / 1000:.0f} km)")
    print(f"intersections {len(inter)}  degree: {inter.degree.value_counts().sort_index().to_dict()}")
    print(f"facilities: {fac.dota_class.value_counts().to_dict()}")
    print(f"railways: {rails.length.groupby(rails.railway).sum().div(1000).round(1).to_dict()} km")
    print(f"forest: {len(forest)} polygons, {forest.area.sum() / 1e6:.1f} km2, edge {forest.boundary.length.sum() / 1000:.0f} km")
    print(f"buildings: {len(bld)} footprints, {bld.area.sum() / 1e6:.2f} km2")


if __name__ == "__main__":
    main()
