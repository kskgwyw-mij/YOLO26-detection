"""Fine-tune yolo26s-seg on the OSM-derived road/intersection dataset.

Usage:
  uv run train.py --smoke        # 1 epoch on 5 % of the data, checks the pipeline
  uv run train.py                # full run (several hours on the M3)
"""
import argparse

from ultralytics import YOLO

from make_dataset import IMGSZ
from tiles import ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--model", default=str(ROOT / "weights" / "yolo26s-seg.pt"))
    a = ap.parse_args()
    model = YOLO(a.model)
    model.train(
        data=str(ROOT / "datasets" / "roads" / "roads.yaml"),
        imgsz=IMGSZ,
        epochs=1 if a.smoke else a.epochs,
        fraction=0.05 if a.smoke else 1.0,
        batch=a.batch,
        device="mps",
        workers=4,
        project=str(ROOT / "runs"),
        name="smoke" if a.smoke else "roads",
        exist_ok=a.smoke,
        patience=15,
        # Orthophotos: geometry stays valid under flips; keep GSD roughly fixed.
        flipud=0.5, fliplr=0.5, scale=0.2, degrees=0.0,
        plots=True,
    )


if __name__ == "__main__":
    main()
