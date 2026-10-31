"""Streamlit front end: upload an aerial image (or pick a dataset tile), watch the
models detect, then locate it against OpenStreetMap.

  uv run streamlit run app.py

The pipeline sees only the pixels, the pixel size and the search area. When the
file carries GeoTIFF tags, they are read *after* locating, only to show the
true position, the error, and OSM labels at the true position.
"""
import hashlib
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from PIL import Image, ImageDraw
from pyproj import Transformer
from shapely.geometry import box

import detect
import locate as loc
import make_dataset as ds
import osm
import roadseg
from tiles import CRS, OUT, load_rgb, splits, tile_origin, tile_path, windows

SPLIT_NOTE = {
    "train": "Training tile: the models saw it, so the result is optimistic.",
    "val": "Validation tile: it chose the model checkpoints, so the result is slightly optimistic.",
    "test": "Test tile: held out from training and model selection.",
}
PREVIEW = 0.25            # display scale of the 0.2 m image (0.8 m per display pixel)
FRAME_S = 0.2             # minimum time between animation frames
STAGE_HOLD_S = 0.4        # how long a finished stage stays on screen before the next one starts
MASK_LAYERS = {"streets": "road", "railways": "rail", "forest": "forest", "buildings": "building"}
DET_LAYERS = [*MASK_LAYERS, "intersections", "facilities"]
OSM_LAYERS = ["streets", "railways", "forest", "buildings", "intersections", "facilities"]
STAGES = {"intersections": "YOLO26-seg · intersections", "road": "U-Net · streets", "rail": "U-Net · railways",
          "forest": "U-Net · forest", "building": "U-Net · buildings", "facilities": "YOLO26-obb · facilities"}
OSM_STYLE = {"streets": (255, 220, 0), "railways": (255, 0, 200), "forest": (170, 255, 170),
             "buildings": (255, 255, 255), "intersections": (255, 220, 0), "facilities": (255, 220, 0)}
LEGEND = ("**Detections** (filled): red streets · blue railways · green forest · orange buildings · "
          "cyan circles intersections · green boxes facilities (grey: vehicles, not used)  \n"
          "**OSM labels** (lines): yellow street centrelines · magenta railways · light green forest edges · "
          "white building outlines · yellow squares intersections · yellow facility outlines")

st.set_page_config(page_title="Aerial geolocation", layout="wide")


@st.cache_resource(show_spinner="Loading models ...")
def models():
    return detect.load_models()


@st.cache_resource(show_spinner="Loading OSM layers ...")
def osm_layers(area):
    return loc.osm_for(area)


@st.cache_data(max_entries=4, show_spinner="Reading image ...")
def thumbnail(key, _src):
    im = Image.open(_src()).convert("RGB")
    return im.resize((max(1, int(im.width * PREVIEW)), max(1, int(im.height * PREVIEW))), Image.BOX)


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


def expected_steps(h, w, unets):
    """Number of on_step calls detect.detect makes for an h x w image (for the progress bar)."""
    n = lambda size, win, ov: len(windows(size, win, ov))
    sh, sw = int(h * ds.SCALE), int(w * ds.SCALE)
    seg = n(sh, ds.IMGSZ, int(detect.SEG_OVERLAP * ds.SCALE)) * n(sw, ds.IMGSZ, int(detect.SEG_OVERLAP * ds.SCALE))
    unet = n(sh, ds.IMGSZ, roadseg.PRED_OVERLAP) * n(sw, ds.IMGSZ, roadseg.PRED_OVERLAP)
    obb = sum(n(int(h * s), detect.OBB_WIN, detect.OBB_OVERLAP) * n(int(w * s), detect.OBB_WIN, detect.OBB_OVERLAP)
              for s in detect.OBB_SCALES)
    n_unets = sum(u is not None for u in unets.values())
    return seg + unet * n_unets + obb + 2 + n_unets  # + one "stage finished" call per stage


# ---------- drawing ----------

def compose(base, masks=None, points=(), boxes=(), window=None, show=DET_LAYERS):
    """Detections over the display image. Masks may be bool (live, 0.4 m) or uint8 0-255 (final, 1 m);
    points and boxes are in full-resolution pixels."""
    im = base.copy()
    for layer, k in MASK_LAYERS.items():
        m = (masks or {}).get(k)
        if m is None or layer not in show:
            continue
        on = m if m.dtype == bool else m > 127
        a = Image.fromarray(on.astype(np.uint8) * 115).resize(im.size, Image.NEAREST)
        im = Image.composite(Image.new("RGB", im.size, detect.PREVIEW_COLORS[k]), im, a)
    d = ImageDraw.Draw(im)
    if "facilities" in show:
        for poly, cls in boxes:
            color = (150, 150, 150) if cls in detect.VEHICLES else (0, 255, 0)
            d.polygon([(x * PREVIEW, y * PREVIEW) for x, y in poly], outline=color, width=2)
    if "intersections" in show:
        for x, y in points:
            x, y = x * PREVIEW, y * PREVIEW
            d.ellipse((x - 7, y - 7, x + 7, y + 7), outline=(0, 255, 255), width=3)
    if window is not None:
        d.rectangle([v * PREVIEW for v in window], outline=(255, 255, 255), width=3)
    return im


def draw_osm(im, layers, e, n, which):
    """OSM labels drawn as lines over the display image, upper-left corner at (e, n)."""
    ppm = PREVIEW / detect.GSD  # display pixels per metre
    area = box(e, n - im.height / ppm, e + im.width / ppm, n)
    px = lambda coords: [((x - e) * ppm, (n - y) * ppm) for x, y in coords]
    sub = lambda gdf: gdf.iloc[gdf.sindex.query(area)]
    d = ImageDraw.Draw(im)
    rings = lambda g: [r for p in getattr(g, "geoms", [g]) if p.geom_type == "Polygon" for r in (p.exterior, *p.interiors)]
    if "forest" in which:
        for g in sub(layers["forest"]).geometry:
            for r in rings(g):
                d.line(px(r.coords), fill=OSM_STYLE["forest"], width=2)
    if "buildings" in which:
        for g in sub(layers["buildings"]).geometry:
            for r in rings(g):
                d.line(px(r.coords), fill=OSM_STYLE["buildings"], width=1)
    if "streets" in which:
        for g in sub(layers["roads"]).geometry:
            d.line(px(g.coords), fill=OSM_STYLE["streets"], width=2)
    if "railways" in which:
        rails = sub(layers["railways"])
        for g in rails[rails.active].geometry:
            d.line(px(g.coords), fill=OSM_STYLE["railways"], width=3)
    if "facilities" in which:
        for g in sub(layers["facilities"]).geometry:
            for r in rings(g):
                d.line(px(r.coords), fill=OSM_STYLE["facilities"], width=2)
    if "intersections" in which:
        for g in sub(layers["intersections"]).geometry:
            (x, y), = px(g.coords)
            d.rectangle((x - 4, y - 4, x + 4, y + 4), outline=OSM_STYLE["intersections"], width=2)
    return im


# ---------- pipeline with live animation ----------

def animate_detection(img, name, frame, bar, show):
    """Run the models and redraw the frame as each window is processed."""
    base = Image.fromarray(img).resize((int(img.shape[1] * PREVIEW), int(img.shape[0] * PREVIEW)), Image.BOX)
    obb, seg, unets = models()
    total = expected_steps(*img.shape[:2], unets)
    live = {"masks": {}, "points": [], "boxes": [], "window": None, "n": 0, "t": 0.0}

    def on_step(stage, window, part):
        live["n"] += 1
        live["window"] = window
        if "mask" in part:
            live["masks"][stage] = part["mask"]
        live["points"] = part.get("points", live["points"])
        live["boxes"] = part.get("boxes", live["boxes"])
        if window is None or time.time() - live["t"] >= FRAME_S:  # always draw a finished stage
            frame.image(compose(base, live["masks"], live["points"], live["boxes"], window, show), width="stretch")
            step = "done" if window is None else f"window {live['n']}/{total}"
            bar.progress(min(live["n"] / total, 1.0), text=f"{STAGES[stage]} · {step}")
            live["t"] = time.time()
        if window is None:
            time.sleep(STAGE_HOLD_S)

    det, masks = detect.detect(img, name, obb, seg, unets, on_step=on_step)
    return base, det, masks


def replay_detection(img, det, masks, frame, bar, show, steps=8):
    """Saved detections: reveal them stage by stage (a replay, not a model run)."""
    base = Image.fromarray(img).resize((int(img.shape[1] * PREVIEW), int(img.shape[0] * PREVIEW)), Image.BOX)
    shown, h = {}, img.shape[0]
    stages = [k for k in MASK_LAYERS.values() if masks.get(k) is not None]
    for i, k in enumerate(stages):
        m = masks[k] > 127
        for j in range(1, steps + 1):
            rows = int(m.shape[0] * j / steps)
            shown[k] = np.vstack([m[:rows], np.zeros_like(m[rows:])])
            frame.image(compose(base, shown, window=(0, 0, img.shape[1], h * j / steps), show=show), width="stretch")
            bar.progress((i + j / steps) / (len(stages) + 1), text=f"Replay · {STAGES[k]}")
            time.sleep(0.05)
    pts = [p["center_px"] for p in det["intersections"]]
    boxes = [(o["poly_px"], o["class"]) for o in det["objects"]]
    frame.image(compose(base, shown, pts, boxes, show=show), width="stretch")
    bar.progress(1.0, text="Replay · intersections and facilities")
    return base


def run_detection(img, name, bar):
    """Models without animation: one progress line per stage."""
    obb, seg, unets = models()
    done = []

    def on_step(stage, window, part):
        if window is None:
            done.append(stage)
            bar.progress(len(done) / (2 + sum(u is not None for u in unets.values())),
                         text=f"{STAGES[stage]} · done")

    bar.progress(0.0, text="Running the models ...")
    det, masks = detect.detect(img, name, obb, seg, unets, on_step=on_step)
    base = Image.fromarray(img).resize((int(img.shape[1] * PREVIEW), int(img.shape[0] * PREVIEW)), Image.BOX)
    return base, det, masks


def run_pipeline(src, name, gsd, area, reuse, frame, bar, show, animate):
    img = to_model_gsd(load_rgb(src), gsd)
    saved = OUT / "det" / f"{name}.json"
    reused = reuse and saved.exists()
    if reused:
        det, masks = loc.load_detections(saved)
        masks_u8 = {k: None if m is None else (m * 255).astype(np.uint8) for k, m in masks.items()}
        base = replay_detection(img, det, masks_u8, frame, bar, show) if animate else \
            Image.fromarray(img).resize((int(img.shape[1] * PREVIEW), int(img.shape[0] * PREVIEW)), Image.BOX)
    else:
        base, det, masks_u8 = animate_detection(img, name, frame, bar, show) if animate else \
            run_detection(img, name, bar)
        masks = {k: None if m is None else m.astype(np.float32) / 255 for k, m in masks_u8.items()}
    bar.progress(1.0, text="Locating against OSM ...")
    res = loc.locate(det, masks, area, osm_layers(area))
    return {"base": base, "det": det, "res": res, "masks": masks_u8, "reused": reused,
            "truth": ground_truth(src, name)}  # truth read only now, after locating


# ---------- result display ----------

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


def summary(out):
    res, truth = out["res"], out["truth"]
    c = st.columns(4)
    c[0].metric("Status", res["status"])
    c[1].metric("Margin", res.get("margin", "-"), help=f"best minus runner-up; below {loc.MIN_MARGIN} -> not localizable")
    c[2].metric("Score", res.get("score", "-"))
    if truth and "ul_e" in res:
        err = float(np.hypot(res["ul_e"] - truth["e"], res["ul_n"] - truth["n"]))
        c[3].metric("Error vs. GeoTIFF tags", f"{err:.2f} m")
        if res["status"] == "ok" and err >= 10:
            st.error("Wrong but confident: the locator placed the tile more than 10 m off.")
    if truth and truth["split"]:
        st.info(f"Dataset tile `{truth['tile']}` ({truth['split']}). {SPLIT_NOTE[truth['split']]}")
    elif truth:
        st.info("GeoTIFF tags found, but this tile is not in the dataset splits.")
    if "ul_e" in res:
        st.write(f"Upper-left corner **E {res['ul_e']:.1f}, N {res['ul_n']:.1f}** (EPSG:25832) · "
                 f"centre **{res['center_lat']:.6f}, {res['center_lon']:.6f}** (WGS84)")


def result_image(out, area, show, osm_which, osm_at):
    det = out["det"]
    pts = [p["center_px"] for p in det["intersections"]]
    boxes = [(o["poly_px"], o["class"]) for o in det["objects"]]
    im = compose(out["base"], out["masks"], pts, boxes, show=show)
    if osm_which:
        e, n = (out["truth"]["e"], out["truth"]["n"]) if osm_at.startswith("true") else \
            (out["res"]["ul_e"], out["res"]["ul_n"])
        im = draw_osm(im, osm_layers(area), e, n, osm_which)
    return im


def main():
    st.title("Where was this aerial image taken?")
    st.caption("Streets, railways, forest, buildings, intersections and facilities are detected in the image "
               "and matched against OpenStreetMap. Assumes a north-up orthophoto with known pixel size.")
    with st.sidebar:
        source = st.radio("Image", ["Upload", "From dataset"], horizontal=True)
        if source == "Upload":
            up = st.file_uploader("Aerial image (GeoTIFF, PNG, JPG)", type=["tif", "tiff", "png", "jpg", "jpeg"])
            if up is None:
                st.info("Upload an image, or pick one from the dataset.")
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
                             help="Outside the cached Rosenheim area, OSM is first read from the local PostGIS database (a few seconds).")
        area = tuple(float(v) for v in bbox.split(","))
        reuse = st.checkbox("Replay saved detections (outputs/det) instead of running the models",
                            help="Test and val tiles only; skips the ~40 s model run.")
        animate = st.toggle("Live animation", value=True,
                            help="Show each model's detections window by window. Off: just a progress bar (a few seconds faster).")
        go = st.button("Detect and locate", type="primary", width="stretch")

    run_key = (key, gsd, area, reuse)
    out = st.session_state.get("out") if st.session_state.get("run_key") == run_key else None
    head = st.container()
    c1, c2, c3 = st.columns([3, 3, 2])
    show = c1.multiselect("Detections", DET_LAYERS, default=DET_LAYERS, key="show")
    osm_which = c2.multiselect("OSM labels", OSM_LAYERS, default=[],
                               help="Drawn at the located position, or at the true one when the file has GeoTIFF tags.",
                               disabled=out is None, key="osm_which")
    positions = ["located position"] if out is None or "ul_e" in out["res"] else []
    if out is not None and out["truth"]:
        positions.append("true position (GeoTIFF)")
    osm_at = c3.radio("OSM labels at", positions or ["located position"], disabled=not positions, key="osm_at")
    frame, bar = st.empty(), st.empty()

    if go:
        st.session_state["out"] = run_pipeline(src(), name, gsd, area, reuse, frame, bar, show, animate)
        st.session_state["run_key"] = run_key
        st.rerun()
    if out is None:
        frame.image(thumbnail(key, src), caption=f"{name} · press *Detect and locate*", width="stretch")
        return

    with head:
        summary(out)
    if out["reused"]:
        bar.caption("Detections replayed from outputs/det.")
    frame.image(result_image(out, area, show, osm_which if positions else [], osm_at), width="stretch")
    st.markdown(LEGEND)
    tab_map, tab_details = st.tabs(["Map", "Details"])
    with tab_map:
        map_view(out["res"], out["det"], out["truth"], area)
    with tab_details:
        det = out["det"]
        cover = {k: f"{m.mean() / 2.55:.2f} %" for k, m in out["masks"].items() if m is not None}
        counts = pd.Series([o["class"] for o in det["objects"]], dtype=str).value_counts()
        st.write("Mask coverage:", cover, f"Intersections: {len(det['intersections'])}")
        st.dataframe(counts.rename("count"))
        st.json(out["res"])


main()
