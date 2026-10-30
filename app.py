"""Streamlit front end: upload an aerial image (or pick a dataset tile), detect, locate.

  uv run streamlit run app.py

The pipeline sees only the pixels, the pixel size and the search area. When the
file carries GeoTIFF tags, they are read *after* locating, only to show the
true position and the error.
"""
import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from PIL import Image
from pyproj import Transformer

import detect
import locate as loc
import osm
from tiles import CRS, OUT, load_rgb, splits, tile_origin, tile_path

SPLIT_NOTE = {
    "train": "Training tile: the models saw it, so the result is optimistic.",
    "val": "Validation tile: it chose the model checkpoints, so the result is slightly optimistic.",
    "test": "Test tile: held out from training and model selection.",
}
OSM_COLORS = {"roads": (255, 220, 0), "railways": (255, 0, 200), "forest": (120, 255, 120)}

st.set_page_config(page_title="Aerial geolocation", layout="wide")


@st.cache_resource(show_spinner="Loading models ...")
def models():
    return detect.load_models()


@st.cache_resource(show_spinner="Loading OSM layers ...")
def osm_layers(area):
    return loc.osm_for(area)


def split_of(name):
    return next((s for s, tiles in splits().items() if name in tiles), None)


def ground_truth(src, name):
    """True upper-left corner from GeoTIFF tags, or None. Never passed to the locator."""
    try:
        e0, n0, gsd = tile_origin(src)
    except (KeyError, AttributeError, OSError):
        return None
    if split_of(name) is None:  # renamed file: derive the dataset name from the position
        name = f"32{int(e0 // 1000)}_{int(n0 // 1000) - 1}"
    return {"e": e0, "n": n0, "gsd": gsd, "tile": name, "split": split_of(name)}


def to_model_gsd(img, gsd):
    """Resample to the 0.2 m pixel size the models were trained on."""
    if abs(gsd - detect.GSD) < 1e-6:
        return img
    f = gsd / detect.GSD
    size = (int(round(img.shape[1] * f)), int(round(img.shape[0] * f)))
    return np.asarray(Image.fromarray(img).resize(size, Image.LANCZOS))


def run_pipeline(src, name, gsd, area, reuse):
    img = to_model_gsd(load_rgb(src), gsd)
    saved = OUT / "det" / f"{name}.json"
    if reuse and saved.exists():
        det, masks = loc.load_detections(saved)
        masks_u8 = {k: None if m is None else (m * 255).astype(np.uint8) for k, m in masks.items()}
    else:
        det, masks_u8 = detect.detect(img, name, *models())
        masks = {k: None if m is None else m.astype(np.float32) / 255 for k, m in masks_u8.items()}
    res = loc.locate(det, masks, area, osm_layers(area))
    return {"preview": detect.preview(img, det, masks_u8), "det": det, "res": res,
            "masks": masks_u8, "reused": reuse and saved.exists()}


def osm_overlay(preview, res, det, layers):
    """OSM streets, railways and forest edges drawn at the located position."""
    roads, _, _, rails, forest = layers
    e, n = res["ul_e"], res["ul_n"]
    area = (e, n - det["height"] * det["gsd"], e + det["width"] * det["gsd"], n)
    rasters = {"roads": loc.rasterize_roads(roads, area, 1.0),
               "railways": loc.rasterize_roads(rails[rails.active], area, 1.0),
               "forest": loc.edge_band(loc.rasterize_areas(forest, area, 1.0) > 0.5, 1.0)}
    im = preview.convert("RGB")
    for k, r in rasters.items():
        a = Image.fromarray((r * 200).astype(np.uint8)).resize(im.size)
        im = Image.composite(Image.new("RGB", im.size, OSM_COLORS[k]), im, a)
    return im


def footprint(e, n, w_m, h_m, to_wgs):
    return [list(to_wgs.transform(x, y)) for x, y in ((e, n), (e + w_m, n), (e + w_m, n - h_m), (e, n - h_m))]


def map_view(res, det, truth, area):
    to_wgs = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    w_m, h_m = det["width"] * det["gsd"], det["height"] * det["gsd"]
    rows = [{"name": "search area", "poly": footprint(area[0], area[3], area[2] - area[0], area[3] - area[1], to_wgs),
             "fill": [0, 0, 0, 0], "line": [120, 120, 120]}]
    if truth:
        rows.append({"name": f"true position ({truth['tile']})", "poly": footprint(truth["e"], truth["n"], w_m, h_m, to_wgs),
                     "fill": [0, 200, 0, 40], "line": [0, 160, 0]})
    if "ul_e" in res:
        rows.append({"name": f"located ({res['status']})", "poly": footprint(res["ul_e"], res["ul_n"], w_m, h_m, to_wgs),
                     "fill": [30, 100, 255, 60], "line": [30, 100, 255]})
    layer = pdk.Layer("PolygonLayer", pd.DataFrame(rows), get_polygon="poly", get_fill_color="fill",
                      get_line_color="line", line_width_min_pixels=2, pickable=True)
    lon, lat = (res["center_lon"], res["center_lat"]) if "center_lon" in res else \
        to_wgs.transform((area[0] + area[2]) / 2, (area[1] + area[3]) / 2)
    st.pydeck_chart(pdk.Deck(layers=[layer], tooltip={"text": "{name}"},
                             initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=12)))


def show(out, truth, area):
    res, det = out["res"], out["det"]
    ok = res["status"] == "ok"
    c = st.columns(4)
    c[0].metric("Status", res["status"])
    c[1].metric("Margin", res.get("margin", "-"), help=f"best minus runner-up; below {loc.MIN_MARGIN} -> not localizable")
    c[2].metric("Score", res.get("score", "-"))
    if truth and "ul_e" in res:
        err = float(np.hypot(res["ul_e"] - truth["e"], res["ul_n"] - truth["n"]))
        c[3].metric("Error vs. GeoTIFF tags", f"{err:.2f} m")
        if ok and err >= 10:
            st.error("Wrong but confident: the locator placed the tile more than 10 m off.")
    if truth and truth["split"]:
        st.info(f"Dataset tile `{truth['tile']}` ({truth['split']}). {SPLIT_NOTE[truth['split']]}")
    elif truth:
        st.info("GeoTIFF tags found, but this tile is not in the dataset splits.")
    if out["reused"]:
        st.caption("Detections reused from outputs/det.")
    if "ul_e" in res:
        st.write(f"Upper-left corner **E {res['ul_e']:.1f}, N {res['ul_n']:.1f}** (EPSG:25832) · "
                 f"centre **{res['center_lat']:.6f}, {res['center_lon']:.6f}** (WGS84)")

    tabs = st.tabs(["Detections", "OSM at located position", "Map", "Details"])
    with tabs[0]:
        st.image(out["preview"], caption="red streets · blue railways · green forest · cyan intersections · "
                 "green boxes facilities (grey: vehicles, not used)", width="stretch")
    with tabs[1]:
        if "ul_e" in res:
            st.image(osm_overlay(out["preview"], res, det, osm_layers(area)), width="stretch",
                     caption="OSM at the located position: yellow streets · magenta railways · light green forest edges")
        else:
            st.write("No position found.")
    with tabs[2]:
        map_view(res, det, truth, area)
    with tabs[3]:
        cover = {k: f"{m.mean() / 2.55:.2f} %" for k, m in out["masks"].items() if m is not None}
        counts = pd.Series([o["class"] for o in det["objects"]], dtype=str).value_counts()
        st.write("Mask coverage:", cover, f"Intersections: {len(det['intersections'])}")
        st.dataframe(counts.rename("count"))
        st.json(res)


def main():
    st.title("Where was this aerial image taken?")
    st.caption("Streets, railways, forest edges, intersections and facilities are detected in the image "
               "and matched against OpenStreetMap. Assumes a north-up orthophoto with known pixel size.")
    with st.sidebar:
        source = st.radio("Image", ["Upload", "From dataset"], horizontal=True)
        if source == "Upload":
            up = st.file_uploader("Aerial image (GeoTIFF, PNG, JPG)", type=["tif", "tiff", "png", "jpg", "jpeg"])
            if up is None:
                st.stop()
            data = up.getvalue()
            name, key = Path(up.name).stem, hashlib.md5(data).hexdigest()
            src = lambda: io.BytesIO(data)
        else:
            split = st.selectbox("Split", ["test", "val", "train"])
            name = st.selectbox("Tile", splits()[split])
            path = tile_path(name)
            key, src = name, lambda: path
        gsd = st.number_input("Pixel size (m)", 0.05, 2.0, detect.GSD, 0.05, format="%.2f")
        bbox = st.text_input("Search area (minE,minN,maxE,maxN, EPSG:25832)", ",".join(map(str, osm.AREA)),
                             help="Outside the cached Rosenheim area, OSM is fetched from Overpass first.")
        area = tuple(float(v) for v in bbox.split(","))
        reuse = st.checkbox("Reuse saved detections (outputs/det) if present", value=True,
                            help="Much faster for the test and val tiles; uncheck to run the models.")
        go = st.button("Locate", type="primary", width="stretch")

    st.image(Image.open(src()).convert("RGB").reduce(8), caption=name, width=400)
    run_key = (key, gsd, area, reuse)
    if go:
        with st.status("Running pipeline ...", expanded=False) as status:
            st.session_state["out"] = run_pipeline(src(), name, gsd, area, reuse)
            st.session_state["run_key"] = run_key
            status.update(label="Done", state="complete")
    if st.session_state.get("run_key") == run_key:
        # ground truth is read only here, after locating
        show(st.session_state["out"], ground_truth(src(), name), area)


main()
