"""Semantic street + railway segmentation (U-Net, ResNet34 encoder).

YOLO26-seg instance masks broke up long roads and missed dual carriageways,
so streets come from this model; intersections still come from YOLO26-seg.
Training images are the make_dataset.py windows (0.4 m/px, 640 px); masks are
rasterized from the same OSM streets and railways that locate.py correlates against.

Output channels are independent (sigmoid): 0 = street, 1 = railway. Abandoned
and disused lines are ignored in the railway loss. No val tile has a railway,
so the windows of RAIL_VAL_TILES move from train to val here (U-Net only;
splits.json and the YOLO models are unchanged).

detect.py takes streets from STREET_WEIGHTS, a streets-only U-Net trained with
this file at commit f66b7e2, and railways from WEIGHTS. In our one training run
the street channel of the two-class model was slightly weaker on the val tiles.

Usage:
  uv run roadseg.py masks            # write datasets/roads/masks/{train,val}
  uv run roadseg.py train [--epochs 30]
"""
import argparse
import time

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image

import make_dataset as ds
import osm
from locate import rasterize_roads
from tiles import ROOT, tile_origin, tile_path, windows

DATA = ROOT / "datasets" / "roads"
WEIGHTS = ROOT / "runs" / "roadseg" / "best.pt"
STREET_WEIGHTS = ROOT / "runs" / "roadseg_streets" / "best.pt"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
PRED_OVERLAP = 160
THRESHOLD = 0.5
CLASSES = ("road", "rail")
ROAD, RAIL, RAIL_IGNORE = 1, 2, 4   # bits of the saved mask PNG
RAIL_VAL_TILES = ("32734_5304",)


def build_model(pretrained=True, n_classes=len(CLASSES)):
    m = smp.Unet("resnet34", encoder_weights=None, classes=n_classes)
    if pretrained:  # ImageNet weights via torchvision (the HF hub is unreachable here)
        sd = torchvision.models.resnet34(weights="IMAGENET1K_V1").state_dict()
        m.encoder.load_state_dict({k: v for k, v in sd.items() if not k.startswith("fc.")})
    return m


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def make_masks():
    roads, _, _, rails = osm.load_layers()
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
                 | (rasterize_roads(inactive, area, res) > 0.5) * RAIL_IGNORE)
            Image.fromarray(m.astype(np.uint8)).save(out / f"{img.stem}.png")
        print(split, len(list(out.glob("*.png"))))


def image_list(split):
    is_rail_val = lambda p: p.stem.rsplit("_", 2)[0] in RAIL_VAL_TILES
    train = sorted((DATA / "images" / "train").glob("*.jpg"))
    if split == "train":
        return [p for p in train if not is_rail_val(p)]
    return sorted((DATA / "images" / "val").glob("*.jpg")) + [p for p in train if is_rail_val(p)]


class RoadData(torch.utils.data.Dataset):
    def __init__(self, split, augment):
        self.imgs = image_list(split)
        self.augment = augment

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        p = self.imgs[i]
        x = torch.from_numpy(np.array(Image.open(p).convert("RGB"))).permute(2, 0, 1).float() / 255
        m = torch.from_numpy(np.array(Image.open(DATA / "masks" / p.parent.name / f"{p.stem}.png")))
        # channels: road, rail targets, then their validity (rail ignored on inactive lines)
        y = torch.stack([m & ROAD > 0, m & RAIL > 0, torch.ones_like(m, dtype=torch.bool),
                         (m & RAIL_IGNORE == 0) | (m & RAIL > 0)]).float()
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


def train(epochs, batch, lr):
    dev = device()
    model = build_model().to(dev)
    tr = torch.utils.data.DataLoader(RoadData("train", True), batch_size=batch, shuffle=True,
                                     num_workers=4, persistent_workers=True)
    va = torch.utils.data.DataLoader(RoadData("val", False), batch_size=batch, num_workers=2)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * len(tr), pct_start=0.1)
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    best, log = -1.0, WEIGHTS.parent / "log.csv"
    cols = [f"val_{c}_{m}" for c in CLASSES for m in ("iou", "precision", "recall")]
    log.write_text(",".join(["epoch", "train_loss", *cols, "seconds"]) + "\n")
    for ep in range(1, epochs + 1):
        t0, tot = time.time(), 0.0
        model.train()
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            y, valid = y[:, :2], y[:, 2:]
            out = model(x)
            loss = (F.binary_cross_entropy_with_logits(out, y, weight=valid, reduction="sum") / valid.sum()
                    + dice_loss(out, y, valid))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        model.eval()
        tp, fp, fn = (torch.zeros(len(CLASSES)) for _ in range(3))
        with torch.no_grad():
            for x, y in va:
                p = torch.sigmoid(model(x.to(dev))).cpu() > THRESHOLD
                y, valid = y[:, :2] > 0.5, y[:, 2:] > 0.5
                tp += (p & y & valid).sum((0, 2, 3))
                fp += (p & ~y & valid).sum((0, 2, 3))
                fn += (~p & y & valid).sum((0, 2, 3))
        iou, prec, rec = tp / (tp + fp + fn).clamp(min=1), tp / (tp + fp).clamp(min=1), tp / (tp + fn).clamp(min=1)
        stats = [f"{v:.4f}" for c in range(len(CLASSES)) for v in (iou[c], prec[c], rec[c])]
        row = ",".join([str(ep), f"{tot / len(tr):.4f}", *stats, f"{time.time() - t0:.0f}"])
        with log.open("a") as fh:
            fh.write(row + "\n")
        print(row, flush=True)
        # checkpoint rule, fixed before training: street IoU + railway IoU
        if iou.sum() > best:
            best = float(iou.sum())
            torch.save(model.state_dict(), WEIGHTS)
    print(f"best val IoU sum {best:.4f} -> {WEIGHTS}")


def load(weights=WEIGHTS):
    """Model in eval mode; its classes are CLASSES[:n] (a 1-class checkpoint is streets only)."""
    sd = torch.load(weights, map_location="cpu")
    m = build_model(pretrained=False, n_classes=sd["segmentation_head.0.weight"].shape[0])
    m.load_state_dict(sd)
    return m.to(device()).eval()


@torch.no_grad()
def predict(model, small):
    """Per-class probabilities (len(CLASSES), h, w) for an image at the training scale (0.4 m/px)."""
    h, w = small.shape[:2]
    win = ds.IMGSZ
    acc = torch.zeros((len(CLASSES), h, w))
    cnt = torch.zeros((h, w))
    # taper window edges so overlapping predictions blend without seams
    ramp = torch.minimum(torch.arange(win) + 1, win - torch.arange(win)).float().clamp(max=PRED_OVERLAP)
    wgt = torch.outer(ramp, ramp)
    x_all = torch.from_numpy(np.ascontiguousarray(small)).permute(2, 0, 1).float() / 255
    for r0 in windows(h, win, PRED_OVERLAP):
        for c0 in windows(w, win, PRED_OVERLAP):
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
    a = ap.parse_args()
    make_masks() if a.cmd == "masks" else train(a.epochs, a.batch, a.lr)


if __name__ == "__main__":
    main()
