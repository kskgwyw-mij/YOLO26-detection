# YOLO26-detection
Proof of Concept: locate aerial images from their content alone, by matching YOLO26 detections against OpenStreetMap.

## Idea
1. **Detect** in the image (pixel space, no georeference):
   - streets: U-Net (ResNet34 encoder, ImageNet-pretrained) trained on road masks rasterized from OSM (`roadseg.py`). It replaced the YOLO26-seg road instances, which broke up long roads and missed dual carriageways.
   - railways: a second U-Net with two output channels (street, railway), trained on OSM railways (`railway=rail|light_rail|narrow_gauge`, no tunnels; abandoned/disused lines ignored in the loss). Only its railway channel is used (see results).
   - forest: a third U-Net with one output channel, trained on OSM forest areas (`landuse=forest`, `natural=wood`, relations with holes). Only the **forest edge** is matched, not the area: a band of ±5 m (`FOREST_EDGE_M`) around the boundary, after an opening that drops single trees and hedges (`FOREST_MIN_M = 4`). Raster borders don't count as edges.
   - buildings: a fourth U-Net with one output channel, trained on OSM building footprints (`building=*`, multipolygons with courtyards, underground ones dropped, footprints < 10 m² ignored). Here the **area** mask is matched: buildings are small, so the footprint pattern itself is the landmark.
   - intersections: `yolo26s-seg` fine-tuned on labels generated automatically from OSM
   - sports facilities, roundabouts, bridges, tanks: pretrained `yolo26s-obb` (DOTA-v1); vehicles are detected but never used for locating
2. **Read** the same features for a coarse search area from a local OSM extract (Geofabrik `bayern-latest.osm.pbf`, read with pyosmium). The extract is a fixed snapshot: no rate limits or outages, and runs are reproducible. The Overpass API source used before is archived in `overpass.py` (with its caches in `outputs/osm/overpass/`), for areas outside the extract.
3. **Locate**: correlate the predicted road mask with the rasterized OSM street network (FFT), plus the railway mask with OSM railways the forest-edge band with OSM forest edges and the building mask with OSM footprints (weights `W_RAIL = W_FOREST = W_BUILDING = 0.5`, set a priori) when the image has any, add votes from intersection and facility pairs, refine at 1 m, then least-squares on matched intersections. Ambiguous results are reported as *not localizable* instead of guessed.

## Assumptions for an image to locate
- nadir orthophoto, **north-up**, **known pixel size** (default 0.2 m, Bavarian DOP20)
- a **coarse search area** (bbox in EPSG:25832 or a place name)

## Data
`rosenheim_area/`: 54 DOP20 tiles (5000×5000 px, 0.2 m, EPSG:25832, flown 22.05.2026), not in git. `splits.json` holds a spatial split: 36 train / 6 val / 12 test tiles (test block E731–733 × N5305–5308, includes `32732_5306`).

## Setup
```
uv sync
mkdir -p weights && cd weights && uv run python -c "from ultralytics import YOLO; YOLO('yolo26s-obb.pt'); YOLO('yolo26s-seg.pt')"
curl -L --create-dirs -o outputs/osm/pbf/bayern-latest.osm.pbf https://download.geofabrik.de/europe/germany/bayern-latest.osm.pbf   # 860 MB
```
The results below use the snapshot of 2026-10-29T21:21:26Z (md5 `2982bfa6058e35d97bce0211ec1b250b`). Geofabrik replaces `-latest` daily, so later downloads may differ slightly; `osm.py` prints the snapshot it read, and the caches `outputs/osm/<name>_<kind>.json` record it. Reading the search area takes about 2.5 min (three passes over the file, 2.2 GB RAM); a search area outside the cached one is read once and cached as well.

## Pipeline
```
uv run osm.py                          # read the PBF (cached) -> outputs/osm/rosenheim.gpkg  [--refresh] [--pbf other.osm.pbf]
uv run osm.py --overlay 32732_5306     # check OSM vs. image alignment
uv run make_dataset.py                 # OSM-labelled YOLO-seg dataset (train/val tiles only)
uv run train.py --smoke                # 1-epoch pipeline check
uv run train.py                        # full fine-tuning on MPS (hours)
uv run roadseg.py masks                # OSM street + railway + forest + building masks for the same windows
uv run roadseg.py train                # two-class U-Net, 30 epochs on MPS (~1.7 h) -> runs/roadseg/best.pt
uv run roadseg.py train --classes forest --out runs/forestseg/best.pt   # forest U-Net (~1.7 h)
uv run roadseg.py train --classes building --out runs/buildingseg/best.pt   # building U-Net (~1.7 h)
uv run detect.py rosenheim_area/       # all models -> outputs/det/<tile>.json + _road/_rail/_forest/_buildingmask.png
uv run locate.py outputs/det/32732_5306.json [--search minE,minN,maxE,maxN | --place Rosenheim]
uv run evaluate.py --oracle [--degrade]  # locator check with OSM-made (degraded) detections
uv run evaluate.py                     # real detections, test split
uv run evaluate.py --quality           # street-mask precision/recall vs OSM per street type
```

The streets-only U-Net (`runs/roadseg_streets/best.pt`) is trained from commit `f66b7e2`: there, run `uv run roadseg.py masks` and `uv run roadseg.py train`, and move `runs/roadseg/best.pt` to `runs/roadseg_streets/best.pt`. Back on the current commit, rerun `uv run roadseg.py masks`, because the mask format changed. The current `roadseg.py` trains the two-class model: it holds out `32734_5304` as railway validation (no val tile has a railway) and uses batch-level Dice. `detect.py --rail-unet none` / `--forest-unet none` / `--building-unet none` skip railways / forest / buildings; `--roads yolo` takes streets from YOLO26-seg.

### Web app
```
uv run streamlit run app.py            # http://localhost:8501
```
Upload an aerial image (GeoTIFF, PNG, JPG), or pick a train/val/test tile, set the pixel size (resampled to 0.2 m for the models) and the search area, then press *Detect and locate*. The detection runs as a live animation: a white box marks the window each model is working on, and streets, railways, forest, buildings, intersections and facility boxes appear as they are found. Each finished stage, the buildings included, stays on screen briefly before the next one starts. *Live animation* in the sidebar turns this off and shows a progress bar instead (about 35 s instead of 40 s per tile on the M3). The result view has toggles for each detection layer and, as an extra option, the OSM labels: street centrelines, railways, forest edges, building outlines, intersections and facilities. They are drawn at the located position, or at the true one when the file has GeoTIFF tags, which shows what the models missed or added. The app also shows margin, score, a map with the located footprint and per-signal details. GeoTIFF tags are read only after locating, to show the true footprint, the error and the split. Results on train tiles are optimistic, and on val tiles slightly so. For test and val tiles, *Replay saved detections* reveals the stored `outputs/det` results instead of running the models.

`locate.py` reads only detector output and OSM; true positions (GeoTIFF tags) are used only in `make_dataset.py` and `evaluate.py`.

## Results so far
| Run | Test tiles within 10 m | Median error |
|---|---|---|
| Oracle (perfect detections) | 12/12 | 0.0 m |
| Degraded oracle (40 % streets/railways/intersections/forest and building polygons dropped, 2 m jitter, 30 % false intersections) | 12/12 | 0.52 m (0.6 m before railways; the noise draws changed, so not directly comparable. Forest and buildings are drawn last, so adding them left the earlier draws unchanged) |
| Real, YOLO26-seg streets, test (12 tiles) | 11/12, 1 not localizable, 0 wrong | 0.74 m (max 1.26 m) |
| Real, YOLO26-seg streets, val (6 tiles) | 4/6, 2 not localizable, 0 wrong | 0.53 m |
| Real, U-Net streets, test (12 tiles) | 12/12, 0 wrong | 0.65 m (max 1.34 m) |
| Real, U-Net streets, val (6 tiles) | 6/6, 0 wrong | 0.34 m (max 1.49 m) |
| Real, two-class U-Net (streets + railways), test | 12/12, 0 wrong | 0.64 m |
| Real, two-class U-Net (streets + railways), val | 5/6, 1 not localizable, 0 wrong | 0.14 m |
| Real, streets U-Net + railways from two-class U-Net, test | 12/12, 0 wrong | 0.65 m (max 1.34 m) |
| Real, streets U-Net + railways from two-class U-Net, val | 6/6, 0 wrong | 0.34 m (max 1.49 m) |
| Real, + forest edges, test | 12/12, 0 wrong | 0.65 m (max 1.34 m) |
| Real, + forest edges, val | 6/6, 0 wrong | 0.34 m (max 1.49 m) |
| **Real, + buildings, test** | **12/12**, 0 wrong | **0.65 m** (max 1.34 m) |
| **Real, + buildings, val** | **6/6**, 0 wrong | **0.34 m** (max 1.49 m) |

Search area for all runs: the 54-tile extent + 1 km (9 × 12 km, about 108 km²). With YOLO streets, the tiles that couldn't be localized were rural, with < 1 % road pixels and few intersections. In one of them (32729_5301) the best peak was 3.9 km off, and the margin rule rejected it. With U-Net streets all three are located correctly. The val tiles also chose the U-Net checkpoint (best val IoU), so the val numbers are slightly optimistic; test is clean.

Railways (rows 7–8): the error is unchanged, because streets already fix the position. Railways raise the confidence margin on the 6 test tiles where track is detected, and change nothing elsewhere:

| Test tile | Margin, streets only | Margin, + railways | Railway recall |
|---|---|---|---|
| 32731_5306 | 0.49 | 0.68 | 1.00 |
| 32731_5307 | 0.66 | 0.96 | 0.85 |
| 32731_5308 | 0.28 | 0.61 | 0.84 |
| 32732_5305 | 0.97 | 1.06 | 0.97 |
| 32732_5306 | 0.74 | 0.89 | 0.76 |
| 32732_5307 | 0.89 | 1.11 | 0.71 |
| 32732_5308 | 0.71 | 0.71 | 0.20 |

- Railway mask on the test tiles: precision 0.99, recall 0.81 (3 m tolerance). No railway pixels at all on the val tiles, which have no track.
- 32732_5308 has a single overgrown branch-line track next to a road. The training data is mostly the double-track main line and the station, so this line is barely detected and contributes nothing.
- A straight track barely fixes the position along itself, so railways add evidence on top of streets; they are not a standalone locator.
- Why streets still come from the streets-only U-Net: in this one training run, the street channel of the two-class model was slightly weaker on the val tiles (IoU 0.617 vs 0.633 on the 6 original val tiles; track recall 0.52 vs 0.60). With it, val tile 32729_5301 is placed correctly (0.21 m) but its margin drops from 0.082 to 0.043, below the threshold. A 0.016 IoU gap is within plausible run-to-run noise; this is not evidence that training railways hurts streets. The choice was made on val; the two-class test numbers had already been seen.

Forest edges (rows 9–10): again the error is identical on every tile and only the margin changes. The rule was fixed before looking: val margins must rise (especially the fragile rural 32729_5301), forested test tiles are the clean evidence, and urban test tiles must not drop.

| Tile | Split | Forest (OSM) | Forest IoU | Margin before | Margin + forest |
|---|---|---|---|---|---|
| 32728_5300 | val | 41 % | 0.77 | 0.54 | 0.79 |
| 32728_5301 | val | 70 % | 0.91 | 0.36 | 0.49 |
| 32729_5300 | val | 51 % | 0.75 | 0.34 | 0.44 |
| 32729_5301 | val | 22 % | 0.77 | 0.08 | 0.25 |
| 32730_5300 | val | 6 % | 0.65 | 0.38 | 0.63 |
| 32730_5301 | val | 8 % | 0.51 | 1.04 | 1.26 |
| 32731_5305 | test | 40 % | 0.86 | 0.80 | 0.96 |
| 32731_5306 | test | 66 % | 0.92 | 0.68 | 0.90 |
| 32731_5307 | test | 62 % | 0.89 | 0.96 | 1.17 |
| 32731_5308 | test | 60 % | 0.89 | 0.61 | 0.81 |
| 32732_5306 | test | 0 % | 0.00 | 0.89 | **0.87** |
| other 7 test tiles | test | 0–6 % | | 0.71–1.11 | +0.00 to +0.16 |

- Forest mask vs OSM: IoU 0.86 on test (precision 0.91, recall 0.94), 0.80 on val.
- The one drop: on urban tile 32732_5306 about 2 % of the image is predicted as forest where OSM maps none (probably parks or tree groups). Its forest NCC at the true position is −0.01, and the margin falls by 0.02. This misses the a-priori rule slightly; nothing was retuned for it.
- The forest checkpoint (epoch 13 of 30) was chosen by val forest IoU, so the val numbers are slightly optimistic; test is clean.
- Tree rows and hedges were not added: in OSM they are too sparse here (31 km in the whole area, 0 m on most rural tiles).

Buildings (default setup, last two rows). OSM has 18,615 dissolved footprints in the search area (5.8 km²). A visual check on three train tiles (urban, village, rural) found houses almost completely mapped and well aligned; small sheds and garages are sometimes missing, and the roofs of tall buildings lean a few metres off their footprints (DOP20 is not a true orthophoto). The location error is again identical on every tile, so only the margins change. Every tile with buildings gains; the one val tile without any (32728_5301) is unchanged:

| Tiles | Building IoU | Margin before | Margin + buildings |
|---|---|---|---|
| 12 test tiles | 0.50–0.78 (all: 0.69) | 0.77–1.21 | 1.15–1.67 (each +0.34 to +0.47) |
| 5 val tiles with buildings | 0.59–0.81 (all: 0.71) | 0.25–1.26 | 0.61–1.71 (each +0.35 to +0.49) |
| val 32728_5301 (forest, no buildings) | – | 0.49 | 0.49 |

- Building NCC at the true position is 0.65–0.92, even on rural tiles with 1 % building cover (val 32729_5301: 0.78, margin 0.25 → 0.61). The scattered farms in rural areas form a distinctive pattern.
- Building mask vs OSM: test IoU 0.69 (precision 0.82, recall 0.81). The checkpoint (epoch 22 of 30) was chosen by val building IoU, so val is slightly optimistic; test is clean.

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
- the confidence threshold (`MIN_MARGIN = 0.05`) was set a priori. With YOLO streets, margins of correct results ranged from 0.09 to 0.76, and those of rejected results from 0.002 to 0.04. With the default setup, correct margins are 1.15–1.67 on test and 0.49–1.71 on val (before buildings: 0.77–1.21 and 0.25–1.26; before forest edges: 0.61–1.11 and 0.08–1.04). The weakest, val tile 32729_5301 (rural, 0/1 intersections matched), was fragile at 0.082: one U-Net retrain moved it to 0.043, below the threshold, with its position still correct. Forest edges lift it to 0.25, buildings to 0.61.
- larger search areas will lower the margins; they have not been tested yet.
