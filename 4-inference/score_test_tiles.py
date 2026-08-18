"""Score every test tile under all three replica checkpoints.

Produces the per-tile IoU distribution needed to choose figure examples on
evidence rather than by random draw, and to state in the caption how the
chosen tiles sit relative to the rest of the test set.

Note: evaluation random-crops (the reference protocol), so a tile's IoU is
partly a function of the crop drawn. The seed is fixed here and in
make_prediction_figures.py so the two agree.
"""
import json
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, "/workspace/0-reproduction")
from dataset import load_dataset, esegui_trasformazioni, ArcheoDataset
from model import _EpochBufferedArcheoModel, SegformerArcheoModel

ARCH_CONFIG = {
    "manet":     {"arch": "MAnet",         "encoder": "efficientnet-b3", "ckpt": "manet_full/lightning_logs/version_0/checkpoints/epoch=19-step=2940.ckpt"},
    "deeplab":   {"arch": "DeepLabV3Plus", "encoder": "resnet50",        "ckpt": "deeplab_full/lightning_logs/version_0/checkpoints/epoch=19-step=2940.ckpt"},
    "segformer": {"arch": None,            "encoder": "b0",              "ckpt": "segformer_full/lightning_logs/version_0/checkpoints/epoch=19-step=2940.ckpt"},
}
DATASET_PATH = "/workspace/bing_1k"
RUNS_ROOT = "/workspace/0-reproduction/runs"
OUT = "/workspace/runs/figs/test_tile_scores.json"
RANDOM_SEED = 1234
THRESHOLD = 0.5
device = "cuda"

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

n_files = len(os.listdir(os.path.join(DATASET_PATH, "train", "originals", "sites"))) \
    + len(os.listdir(os.path.join(DATASET_PATH, "train", "negs", "sites")))
indices = np.arange(0, n_files)
np.random.shuffle(indices)
path_lookup, train_files, val_files, test_files = load_dataset(DATASET_PATH, RANDOM_SEED, indices)
_, val_transform = esegui_trasformazioni()

models = {}
for key, cfg in ARCH_CONFIG.items():
    ckpt = os.path.join(RUNS_ROOT, cfg["ckpt"])
    config = {"arch": cfg["arch"], "encoder": cfg["encoder"], "weights": "imagenet",
              "in_channels": 3, "batch_size": 32, "lr": 1e-4, "loss": "focal"}
    if key == "segformer":
        m = SegformerArcheoModel(encoder_name=cfg["encoder"], in_channels=3, out_classes=1, config=config)
        m.load_state_dict(torch.load(ckpt, map_location="cpu")["state_dict"])
    else:
        m = _EpochBufferedArcheoModel.load_from_checkpoint(
            ckpt, arch=cfg["arch"], encoder_name=cfg["encoder"],
            encoder_weights="imagenet", in_channels=3, out_classes=1, config=config)
    models[key] = m.to(device).eval()

ds = ArcheoDataset(np.array(sorted(test_files)), path_lookup, transform=val_transform)
rows = []
with torch.no_grad():
    for i in range(len(ds)):
        image, mask, fname = ds[i]
        mask_np = mask.squeeze(0).numpy()
        x = image.unsqueeze(0).to(device).float()
        rec = {"file": fname, "positive": bool(mask_np.any()),
               "site_frac": float((mask_np >= 0.5).mean())}
        for key, m in models.items():
            logits = m(x)
            if logits.shape[-2:] != mask_np.shape:
                logits = torch.nn.functional.interpolate(
                    logits, size=mask_np.shape, mode="bilinear", align_corners=False)
            pred = (torch.sigmoid(logits).squeeze().cpu().numpy() >= THRESHOLD)
            gt = mask_np >= 0.5
            if gt.any():
                union = np.logical_or(pred, gt).sum()
                rec[key] = float(np.logical_and(pred, gt).sum() / union) if union else 0.0
            else:
                # negative tile: record whether anything was predicted at all
                rec[key] = None
                rec[f"{key}_fp_px"] = int(pred.sum())
        rows.append(rec)
        if (i + 1) % 100 == 0:
            print(f"  scored {i+1}/{len(ds)}", flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as fh:
    json.dump(rows, fh, indent=1)

pos = [r for r in rows if r["positive"]]
neg = [r for r in rows if not r["positive"]]
print(f"\ntest tiles: {len(rows)}  ({len(pos)} with a site, {len(neg)} negative)")
for key in ARCH_CONFIG:
    v = np.array([r[key] for r in pos])
    print(f"{key:>10}: median {np.median(v):.3f}  mean {v.mean():.3f}  "
          f"p90 {np.percentile(v,90):.3f}  frac<0.1 {(v<0.1).mean():.1%}  frac>0.5 {(v>0.5).mean():.1%}")
    fp = np.array([r[f"{key}_fp_px"] for r in neg])
    print(f"{'':>10}  negatives with any spurious pixel: {(fp>0).mean():.1%}")

# Tiles where all three architectures agree and score well -- the defensible
# "this works" examples, as opposed to one architecture getting lucky.
allgood = sorted(pos, key=lambda r: min(r[k] for k in ARCH_CONFIG), reverse=True)
print("\ntop 12 tiles by WORST-of-three IoU (all three architectures succeed):")
for r in allgood[:12]:
    print(f"  {r['file']:<16} site {r['site_frac']*100:5.1f}%  "
          + "  ".join(f"{k} {r[k]:.2f}" for k in ARCH_CONFIG))
cleanneg = [r for r in neg if all(r[f"{k}_fp_px"] == 0 for k in ARCH_CONFIG)]
print(f"\nnegatives cleanly rejected by all three: {len(cleanneg)}/{len(neg)}")
for r in cleanneg[:5]:
    print("  ", r["file"])
