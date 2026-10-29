"""Tile I/O shared by the pipeline scripts.

Georeference helpers (`tile_origin`) are only meant for label generation and
evaluation. `detect.py` and `locate.py` must never call them.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parent
TILE_DIR = ROOT / "rosenheim_area"
OUT = ROOT / "outputs"
CRS = "EPSG:25832"


def load_rgb(path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def tile_origin(path):
    """Upper-left corner (E, N) and pixel size from GeoTIFF tags (PixelIsArea)."""
    tags = Image.open(path).tag_v2
    tie, scale = tags[33922], tags[33550]
    return float(tie[3]), float(tie[4]), float(scale[0])


def splits():
    return json.loads((ROOT / "splits.json").read_text())


def tile_path(name) -> Path:
    return TILE_DIR / f"{name}.tif"


def windows(size, win, overlap):
    """Start offsets along one axis so windows of `win` px cover `size` px."""
    step = win - overlap
    starts = list(range(0, max(size - win, 0) + 1, step))
    if starts[-1] + win < size:
        starts.append(size - win)
    return starts
