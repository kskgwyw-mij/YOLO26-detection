# YOLO26-detection
Proof of Concept: locate aerial images from their content alone, by matching YOLO26 detections against OpenStreetMap.

## Idea
1. **Detect** in the image (pixel space, no georeference):
   - streets: U-Net (ResNet34 encoder, ImageNet-pretrained) trained on road masks rasterized from OSM (`roadseg.py`). It replaced the YOLO26-seg road instances, which broke up long roads and missed dual carriageways.
   - intersections: `yolo26s-seg` fine-tuned on labels generated automatically from OSM
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
uv run roadseg.py masks                # OSM road masks for the same windows
uv run roadseg.py train                # U-Net street segmenter, 30 epochs on MPS (~1.7 h)
uv run detect.py rosenheim_area/       # all models -> outputs/det/<tile>.json + _roadmask.png  [--roads yolo]
uv run locate.py outputs/det/32732_5306.json [--search minE,minN,maxE,maxN | --place Rosenheim]
uv run evaluate.py --oracle [--degrade]  # locator check with OSM-made (degraded) detections
uv run evaluate.py                     # real detections, test split
uv run evaluate.py --quality           # street-mask precision/recall vs OSM per street type
```

`locate.py` reads only detector output and OSM; true positions (GeoTIFF tags) are used only in `make_dataset.py` and `evaluate.py`.

## Results so far
| Run | Test tiles within 10 m | Median error |
|---|---|---|
| Oracle (perfect detections) | 12/12 | 0.0 m |
| Degraded oracle (40 % streets/intersections dropped, 2 m jitter, 30 % false intersections) | 12/12 | 0.6 m |
| Real, YOLO26-seg streets, test (12 tiles) | 11/12, 1 not localizable, 0 wrong | 0.74 m (max 1.26 m) |
| Real, YOLO26-seg streets, val (6 tiles) | 4/6, 2 not localizable, 0 wrong | 0.53 m |
| **Real, U-Net streets, test (12 tiles)** | **12/12**, 0 wrong | **0.65 m** (max 1.34 m) |
| **Real, U-Net streets, val (6 tiles)** | **6/6**, 0 wrong | **0.34 m** (max 1.49 m) |

Search area for all runs: the 54-tile extent + 1 km (9 × 12 km, about 108 km²). With YOLO streets, the tiles that couldn't be localized were rural, with < 1 % road pixels and few intersections. In one of them (32729_5301) the best peak was 3.9 km off, and the margin rule rejected it. With U-Net streets all three are located correctly. The val tiles also chose the U-Net checkpoint (best val IoU), so the val numbers are slightly optimistic; test is clean.

Street mask vs OSM on the 12 test tiles (3 m tolerance):

| Streets | Share of street px | Recall, YOLO26-seg | Recall, U-Net |
|---|---|---|---|
| major (motorway…tertiary) | 17 % | 0.46 | 0.98 |
| residential / unclassified | 58 % | 0.75 | 0.96 |
| service | 7 % | 0.25 | 0.46 |
| track | 17 % | 0.20 | 0.47 |
| **all** (precision) | | 0.57 (0.90) | **0.84 (0.93)** |

Detector quality on test tile 32732_5306, measured against OSM:
- road mask (YOLO26-seg): 90 % precision, 59 % recall (3 m tolerance)
- intersections: about 50 % precision, about 40 % recall (6–10 m)
- the DOTA model contributes little; streets carry the signal.
  - It finds the running track (needs the x0.5 scale), but it can't be matched: OSM tags that field only as `leisure=pitch` + `sport=multi` (way 103058268), with no track tag.
  - Courts and pitches are hit-and-miss, and it gives false "planes" and "storage tanks".
  - The facility vote uses a fixed 3 m sigma. For large objects this is probably too tight (not yet measured).

Caveats:
- the confidence threshold (`MIN_MARGIN = 0.05`) was set a priori. With YOLO streets, margins of correct results ranged from 0.09 to 0.76, and those of rejected results from 0.002 to 0.04. With U-Net streets, correct margins are 0.28–0.97 on test and 0.08–1.04 on val; the weakest (32729_5301, 0.08) is close to the threshold.
- larger search areas will lower the margins; they have not been tested yet.
