# YOLO26-detection
Proof of Concept: locate aerial images from their content alone, by matching YOLO26 detections against OpenStreetMap.

## Idea
1. **Detect** in the image (pixel space, no georeference):
   - streets + intersections: `yolo26s-seg` fine-tuned on labels generated automatically from OSM
   - sports facilities, roundabouts, bridges, tanks: pretrained `yolo26s-obb` (DOTA-v1); vehicles are detected but never used for locating
2. **Fetch** the same features from OSM via the Overpass API for a coarse search area.
3. **Locate**: correlate the predicted road mask with the rasterized OSM street network (FFT), add votes from intersection and facility pairs, refine at 1 m, then least-squares on matched intersections. Ambiguous results are reported as *not localizable* instead of guessed.

## Assumptions for an image to locate
- nadir orthophoto, **north-up**, **known pixel size** (default 0.2 m, Bavarian DOP20)
- a **coarse search area** (bbox in EPSG:25832 or a place name)

## Data
`rosenheim_area/`: 54 DOP20 tiles (5000×5000 px, 0.2 m, EPSG:25832, flown 22.05.2026), not in git. `splits.json` holds a spatial split: 36 train / 6 val / 12 test tiles (test block E731–733 × N5305–5308, includes `32732_5306`).

## Setup
```
uv sync
mkdir -p weights && cd weights && uv run python -c "from ultralytics import YOLO; YOLO('yolo26s-obb.pt'); YOLO('yolo26s-seg.pt')"
```

## Pipeline
```
uv run osm.py                          # Overpass fetch (cached) -> outputs/osm/rosenheim.gpkg
uv run osm.py --overlay 32732_5306     # check OSM vs. image alignment
uv run make_dataset.py                 # OSM-labelled YOLO-seg dataset (train/val tiles only)
uv run train.py --smoke                # 1-epoch pipeline check
uv run train.py                        # full fine-tuning on MPS (hours)
uv run detect.py rosenheim_area/       # both models -> outputs/det/<tile>.json + _roadmask.png
uv run locate.py outputs/det/32732_5306.json [--search minE,minN,maxE,maxN | --place Rosenheim]
uv run evaluate.py --oracle [--degrade]  # locator check with OSM-made (degraded) detections
uv run evaluate.py                     # real detections, test split
```

`locate.py` reads only detector output and OSM; true positions (GeoTIFF tags) are used only in `make_dataset.py` and `evaluate.py`.

## Results so far
| Run | Test tiles within 10 m | Median error |
|---|---|---|
| Oracle (perfect detections) | 12/12 | 0.0 m |
| Degraded oracle (40 % streets/intersections dropped, 2 m jitter, 30 % false intersections) | 12/12 | 0.6 m |
| **Real detections, test (12 tiles)** | **11/12**, 1 not localizable, 0 wrong | **0.74 m** (max 1.26 m) |
| Real detections, val (6 tiles) | 4/6, 2 not localizable, 0 wrong | 0.53 m |

Search area for all runs: the 54-tile extent + 1 km (9 × 12 km, about 108 km²). The tiles that couldn't be localized are rural, with < 1 % road pixels and few intersections. In one of them (32729_5301) the best peak was 3.9 km off, and the margin rule rejected it.

Detector quality on test tile 32732_5306, measured against OSM:
- road mask: 90 % precision, 59 % recall (3 m tolerance)
- intersections: about 50 % precision, about 40 % recall (6–10 m)
- the DOTA model contributes little; streets carry the signal.
  - It finds the running track (needs the x0.5 scale), but it can't be matched: OSM tags that field only as `leisure=pitch` + `sport=multi` (way 103058268), with no track tag.
  - Courts and pitches are hit-and-miss, and it gives false "planes" and "storage tanks".
  - The facility vote uses a fixed 3 m sigma. For large objects this is probably too tight (not yet measured).

Caveats:
- the confidence threshold (`MIN_MARGIN = 0.05`) was set a priori. Margins of correct results range from 0.09 to 0.76, and those of rejected results from 0.002 to 0.04, so the gap is narrow on 18 tiles.
- larger search areas will lower the margins; they have not been tested yet.
