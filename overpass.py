"""Archived: the Overpass API source that osm.py used before it switched to the local
Geofabrik .osm.pbf extract. Overpass is rate-limited, often fails (502s, timeouts,
empty responses), and its mirrors can serve stale data (the archived rail cache came
from a mirror at 2026-05-31). Kept for areas outside the extract.

The queries match osm.SELECT; fetch() returns the same element dicts as osm.read_pbf(),
so osm.build* accept either. Caches (and the old ones, see MANIFEST.json) live in
outputs/osm/overpass/.

  import osm, overpass
  raw = overpass.fetch("myarea", area)
  roads, inter, fac = osm.build(raw["raw"])
"""
import json
import time

import requests

from osm import RAIL_INACTIVE, RAIL_WIDTH, ROAD_WIDTH, USER_AGENT, to_wgs84_bbox
from tiles import OUT

OVERPASS_URLS = ("https://overpass-api.de/api/interpreter",
                 "https://overpass.private.coffee/api/interpreter",
                 "https://maps.mail.ru/osm/tools/overpass/api/interpreter")
CACHE = OUT / "osm" / "overpass"


def _bb(area):
    s, w, n, e = to_wgs84_bbox(area)
    return f"{s:.6f},{w:.6f},{n:.6f},{e:.6f}"


def query(area):
    bb, hw = _bb(area), "|".join(ROAD_WIDTH)
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
    kinds = "|".join([*RAIL_WIDTH, *RAIL_INACTIVE])
    return f"""[out:json][timeout:300];
way["railway"~"^({kinds})$"]({_bb(area)});
out body geom;"""


def forest_query(area):
    bb = _bb(area)
    return f"""[out:json][timeout:300];
(
  way["landuse"="forest"]({bb});
  way["natural"="wood"]({bb});
  relation["landuse"="forest"]({bb});
  relation["natural"="wood"]({bb});
);
out body geom;"""


def building_query(area):
    bb = _bb(area)
    return f"""[out:json][timeout:300];
(
  way["building"]({bb});
  relation["building"]({bb});
);
out body geom;"""


QUERIES = {"raw": query, "rail": rail_query, "forest": forest_query, "building": building_query}


def fetch_kind(name, area, kind, refresh=False):
    cache = CACHE / f"{name}_{kind}.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())
    cache.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        for url in OVERPASS_URLS:
            try:
                r = requests.post(url, data={"data": QUERIES[kind](area)},
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


def fetch(name, area, refresh=False):
    """{kind: Overpass JSON} for all kinds in osm.SELECT."""
    return {k: fetch_kind(name, area, k, refresh) for k in QUERIES}
