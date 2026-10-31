"""Semantic street / railway / forest / building segmentation (U-Net, ResNet34 encoder).

YOLO26-seg instance masks broke up long roads and missed dual carriageways,
so streets come from this model; intersections still come from YOLO26-seg.
Training images are the make_dataset.py windows (0.4 m/px, 640 px); masks are
rasterized from the same OSM layers that locate.py correlates against.

A model is trained for a chosen subset of CLASSES (--classes); its output
channels are independent sigmoids in that order, and the checkpoint stores
the class names. Abandoned
and disused lines are ignored in the railway loss. No val tile has a railway,
so the windows of RAIL_VAL_TILES move from train to val here (U-Net only;
splits.json and the YOLO models are unchanged).

detect.py takes streets from STREET_WEIGHTS, a streets-only U-Net trained with
this file at commit f66b7e2, railways from WEIGHTS (road + rail), forest from
FOREST_WEIGHTS (forest only) and buildings from BUILDING_WEIGHTS (buildings only). In our one training run the street channel of the
two-class model was slightly weaker on the val tiles.

Usage:
  uv run roadseg.py masks            # write datasets/roads/masks/{train,val}
  uv run roadseg.py train [--epochs 30]                                   # road + rail
  uv run roadseg.py train --classes forest --out runs/forestseg/best.pt   # forest
  uv run roadseg.py train --classes building --out runs/buildingseg/best.pt
"""
import argparse
import time
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image

import make_dataset as ds
import osm
from locate import rasterize_areas, rasterize_roads
from tiles import ROOT, tile_origin, tile_path, windows

DATA = ROOT / "datasets" / "roads"
WEIGHTS = ROOT / "runs" / "roadseg" / "best.pt"
STREET_WEIGHTS = ROOT / "runs" / "roadseg_streets" / "best.pt"
FOREST_WEIGHTS = ROOT / "runs" / "forestseg" / "best.pt"
BUILDING_WEIGHTS = ROOT / "runs" / "buildingseg" / "best.pt"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
PRED_OVERLAP = 160
THRESHOLD = 0.5
CLASSES = ("road", "rail", "forest", "building")
ROAD, RAIL, RAIL_IGNORE, FOREST, BUILDING = 1, 2, 4, 8, 16   # bits of the saved mask PNG
RAIL_VAL_TILES = ("32734_5304",)


def build_model(pretrained=True, n_classes=2):
    m = smp.Unet("resnet34", encoder_weights=None, classes=n_classes)
    if pretrained:  # ImageNet weights via torchvision (the HF hub is unreachable here)
        sd = torchvision.models.resnet34(weights="IMAGENET1K_V1").state_dict()
        m.encoder.load_state_dict({k: v for k, v in sd.items() if not k.startswith("fc.")})
    return m


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def make_masks():
    L = osm.load_layers()
    roads, rails, forest, buildings = L["roads"], L["railways"], L["forest"], L["buildings"]
    active, inactive = rails[rails.active], rails[~rails.active]
    res = ds.WIN * 0.2 / ds.IMGSZ
    origins = {}
    for split in ("train", "val"):
        out = DATA / "masks" / split
        out.mkdir(parents=True, exist_ok=True)
        for img in sorted((DATA / "images" / split).glob("*.jpg")):
            e, n, r0, c0 = img.stem.split("_")
            tile = f"{e}_{n}"
            if tile not in origins:
                origins[tile] = tile_origin(tile_path(tile))
            e0, n0, gsd = origins[tile]
            we, wn = e0 + int(c0) * gsd, n0 - int(r0) * gsd
            span = ds.WIN * gsd
            area = (we, wn - span, we + span, wn)
            m = ((rasterize_roads(roads, area, res) > 0.5) * ROAD
                 | (rasterize_roads(active, area, res) > 0.5) * RAIL
                 | (rasterize_roads(inactive, area, res) > 0.5) * RAIL_IGNORE
                 | (rasterize_areas(forest, area, res) > 0.5) * FOREST
                 | (rasterize_areas(buildings, area, res) > 0.5) * BUILDING)
            Image.fromarray(m.astype(np.uint8)).save(out / f"{img.stem}.png")
        print(split, len(list(out.glob("*.png"))))


def image_list(split, hold_rail_val=True):
    is_rail_val = lambda p: hold_rail_val and p.stem.rsplit("_", 2)[0] in RAIL_VAL_TILES
    train = sorted((DATA / "images" / "train").glob("*.jpg"))
    if split == "train":
        return [p for p in train if not is_rail_val(p)]
    return sorted((DATA / "images" / "val").glob("*.jpg")) + [p for p in train if is_rail_val(p)]


def targets(m, classes):
    """Target and validity channels for `classes` from a bit-encoded mask (rail ignored on inactive lines)."""
    bits = {"road": ROAD, "rail": RAIL, "forest": FOREST, "building": BUILDING}
    y = [m & bits[c] > 0 for c in classes]
    valid = [(m & RAIL_IGNORE == 0) | (m & RAIL > 0) if c == "rail" else torch.ones_like(m, dtype=torch.bool)
             for c in classes]
    return torch.stack(y + valid).float()


class RoadData(torch.utils.data.Dataset):
    def __init__(self, split, augment, classes=("road", "rail")):
        self.imgs = image_list(split, hold_rail_val="rail" in classes)
        self.augment, self.classes = augment, classes

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        p = self.imgs[i]
        x = torch.from_numpy(np.array(Image.open(p).convert("RGB"))).permute(2, 0, 1).float() / 255
        m = torch.from_numpy(np.array(Image.open(DATA / "masks" / p.parent.name / f"{p.stem}.png")))
        y = targets(m, self.classes)
        if self.augment:  # orthophotos: flips and 90-degree rotations keep geometry valid
            k = np.random.randint(4)
            x, y = torch.rot90(x, k, (1, 2)), torch.rot90(y, k, (1, 2))
            if np.random.rand() < 0.5:
                x, y = x.flip(2), y.flip(2)
            x = x * (0.8 + 0.4 * np.random.rand()) + 0.1 * (np.random.rand() - 0.5)
        return (x.clamp(0, 1) - MEAN) / STD, y


def dice_loss(logits, y, valid, eps=1.0):
    """Per-channel Dice over the whole batch (railways are absent from most windows)."""
    p = torch.sigmoid(logits) * valid
    inter = (p * y).sum((0, 2, 3))
    return 1 - ((2 * inter + eps) / (p.sum((0, 2, 3)) + y.sum((0, 2, 3)) + eps)).mean()


def train(epochs, batch, lr, classes, out):
    dev = device()
    n = len(classes)
    model = build_model(n_classes=n).to(dev)
    tr = torch.utils.data.DataLoader(RoadData("train", True, classes), batch_size=batch, shuffle=True,
                                     num_workers=4, persistent_workers=True)
    va = torch.utils.data.DataLoader(RoadData("val", False, classes), batch_size=batch, num_workers=2)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * len(tr), pct_start=0.1)
    out.parent.mkdir(parents=True, exist_ok=True)
    best, log = -1.0, out.parent / "log.csv"
    cols = [f"val_{c}_{m}" for c in classes for m in ("iou", "precision", "recall")]
    log.write_text(",".join(["epoch", "train_loss", *cols, "seconds"]) + "\n")
    for ep in range(1, epochs + 1):
        t0, tot = time.time(), 0.0
        model.train()
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            y, valid = y[:, :n], y[:, n:]
            logits = model(x)
            loss = (F.binary_cross_entropy_with_logits(logits, y, weight=valid, reduction="sum") / valid.sum()
                    + dice_loss(logits, y, valid))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        model.eval()
        tp, fp, fn = (torch.zeros(n) for _ in range(3))
        with torch.no_grad():
            for x, y in va:
                p = torch.sigmoid(model(x.to(dev))).cpu() > THRESHOLD
                y, valid = y[:, :n] > 0.5, y[:, n:] > 0.5
                tp += (p & y & valid).sum((0, 2, 3))
                fp += (p & ~y & valid).sum((0, 2, 3))
                fn += (~p & y & valid).sum((0, 2, 3))
        iou, prec, rec = tp / (tp + fp + fn).clamp(min=1), tp / (tp + fp).clamp(min=1), tp / (tp + fn).clamp(min=1)
        stats = [f"{v:.4f}" for c in range(n) for v in (iou[c], prec[c], rec[c])]
        row = ",".join([str(ep), f"{tot / len(tr):.4f}", *stats, f"{time.time() - t0:.0f}"])
        with log.open("a") as fh:
            fh.write(row + "\n")
        print(row, flush=True)
        # checkpoint rule, fixed before training: sum of the class IoUs
        if iou.sum() > best:
            best = float(iou.sum())
            torch.save({"classes": list(classes), "state_dict": model.state_dict()}, out)
    print(f"best val IoU sum {best:.4f} -> {out}")


def load(weights=WEIGHTS):
    """Model in eval mode with `.classes`. Older checkpoints are bare state dicts
    whose classes are CLASSES[:n] (1 = streets only, 2 = streets + railways)."""
    ck = torch.load(weights, map_location="cpu")
    sd = ck.get("state_dict", ck)
    n = sd["segmentation_head.0.weight"].shape[0]
    m = build_model(pretrained=False, n_classes=n)
    m.load_state_dict(sd)
    m.classes = tuple(ck.get("classes", CLASSES[:n]))
    return m.to(device()).eval()


def channel(model, name):
    assert name in model.classes, f"{name} not among model classes {model.classes}"
    return model.classes.index(name)


@torch.no_grad()
def predict(model, small, on_window=None):
    """Per-class probabilities (len(model.classes), h, w) for an image at the training scale (0.4 m/px).

    on_window(r0, c0, win, prob_so_far) is called before each window, for live display."""
    h, w = small.shape[:2]
    win = ds.IMGSZ
    acc = torch.zeros((len(model.classes), h, w))
    cnt = torch.zeros((h, w))
    # taper window edges so overlapping predictions blend without seams
    ramp = torch.minimum(torch.arange(win) + 1, win - torch.arange(win)).float().clamp(max=PRED_OVERLAP)
    wgt = torch.outer(ramp, ramp)
    x_all = torch.from_numpy(np.ascontiguousarray(small)).permute(2, 0, 1).float() / 255
    for r0 in windows(h, win, PRED_OVERLAP):
        for c0 in windows(w, win, PRED_OVERLAP):
            if on_window is not None:
                on_window(r0, c0, win, (acc / cnt.clamp(min=1e-6)).numpy())
            x = ((x_all[:, r0:r0 + win, c0:c0 + win] - MEAN) / STD)[None].to(device())
            p = torch.sigmoid(model(x))[0].cpu()
            acc[:, r0:r0 + win, c0:c0 + win] += p * wgt
            cnt[r0:r0 + win, c0:c0 + win] += wgt
    return (acc / cnt).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["masks", "train"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--classes", default="road,rail", help=f"comma-separated subset of {CLASSES}")
    ap.add_argument("--out", type=Path, default=WEIGHTS)
    a = ap.parse_args()
    classes = tuple(a.classes.split(","))
    assert set(classes) <= set(CLASSES), classes
    make_masks() if a.cmd == "masks" else train(a.epochs, a.batch, a.lr, classes, a.out)


if __name__ == "__main__":
    main()
