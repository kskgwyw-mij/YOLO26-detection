"""Semantic road segmentation (U-Net, ResNet34 encoder) as the street detector.

YOLO26-seg instance masks broke up long roads and missed dual carriageways,
so streets come from this model; intersections still come from YOLO26-seg.
Training images are the make_dataset.py windows (0.4 m/px, 640 px); masks are
rasterized from the same OSM streets that locate.py correlates against.

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
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
PRED_OVERLAP = 160
THRESHOLD = 0.5


def build_model(pretrained=True):
    m = smp.Unet("resnet34", encoder_weights=None, classes=1)
    if pretrained:  # ImageNet weights via torchvision (the HF hub is unreachable here)
        sd = torchvision.models.resnet34(weights="IMAGENET1K_V1").state_dict()
        m.encoder.load_state_dict({k: v for k, v in sd.items() if not k.startswith("fc.")})
    return m


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def make_masks():
    roads, _, _ = osm.load_layers()
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
            m = rasterize_roads(roads, (we, wn - span, we + span, wn), res) > 0.5
            Image.fromarray((m * 255).astype(np.uint8)).save(out / f"{img.stem}.png")
        print(split, len(list(out.glob("*.png"))))


class RoadData(torch.utils.data.Dataset):
    def __init__(self, split, augment):
        self.imgs = sorted((DATA / "images" / split).glob("*.jpg"))
        self.split, self.augment = split, augment

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        p = self.imgs[i]
        x = torch.from_numpy(np.asarray(Image.open(p).convert("RGB"))).permute(2, 0, 1).float() / 255
        y = torch.from_numpy(np.asarray(Image.open(DATA / "masks" / self.split / f"{p.stem}.png")) > 127)
        y = y.float()[None]
        if self.augment:  # orthophotos: flips and 90-degree rotations keep geometry valid
            k = np.random.randint(4)
            x, y = torch.rot90(x, k, (1, 2)), torch.rot90(y, k, (1, 2))
            if np.random.rand() < 0.5:
                x, y = x.flip(2), y.flip(2)
            x = x * (0.8 + 0.4 * np.random.rand()) + 0.1 * (np.random.rand() - 0.5)
        return (x.clamp(0, 1) - MEAN) / STD, y


def dice_loss(logits, y, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * y).sum((1, 2, 3))
    return 1 - ((2 * inter + eps) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + eps)).mean()


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
    log.write_text("epoch,train_loss,val_iou,val_precision,val_recall,seconds\n")
    for ep in range(1, epochs + 1):
        t0, tot = time.time(), 0.0
        model.train()
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            out = model(x)
            loss = F.binary_cross_entropy_with_logits(out, y) + dice_loss(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        model.eval()
        tp = fp = fn = 0
        with torch.no_grad():
            for x, y in va:
                p = torch.sigmoid(model(x.to(dev))).cpu() > THRESHOLD
                y = y > 0.5
                tp += (p & y).sum().item(); fp += (p & ~y).sum().item(); fn += (~p & y).sum().item()
        iou, prec, rec = tp / max(tp + fp + fn, 1), tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        row = f"{ep},{tot / len(tr):.4f},{iou:.4f},{prec:.4f},{rec:.4f},{time.time() - t0:.0f}"
        with log.open("a") as fh:
            fh.write(row + "\n")
        print(row, flush=True)
        if iou > best:
            best = iou
            torch.save(model.state_dict(), WEIGHTS)
    print(f"best val IoU {best:.4f} -> {WEIGHTS}")


def load(weights=WEIGHTS):
    m = build_model(pretrained=False)
    m.load_state_dict(torch.load(weights, map_location="cpu"))
    return m.to(device()).eval()


@torch.no_grad()
def predict(model, small):
    """Road probability for an image already at the training scale (0.4 m/px)."""
    h, w = small.shape[:2]
    win = ds.IMGSZ
    acc = torch.zeros((h, w))
    cnt = torch.zeros((h, w))
    # taper window edges so overlapping predictions blend without seams
    ramp = torch.minimum(torch.arange(win) + 1, win - torch.arange(win)).float().clamp(max=PRED_OVERLAP)
    wgt = torch.outer(ramp, ramp)
    x_all = torch.from_numpy(np.ascontiguousarray(small)).permute(2, 0, 1).float() / 255
    for r0 in windows(h, win, PRED_OVERLAP):
        for c0 in windows(w, win, PRED_OVERLAP):
            x = ((x_all[:, r0:r0 + win, c0:c0 + win] - MEAN) / STD)[None].to(device())
            p = torch.sigmoid(model(x))[0, 0].cpu()
            acc[r0:r0 + win, c0:c0 + win] += p * wgt
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
