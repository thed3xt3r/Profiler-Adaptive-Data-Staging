import os
import sys
import random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/workspace/0-reproduction")
from dataset import load_dataset, esegui_trasformazioni, ArcheoDataset
from model import _EpochBufferedArcheoModel, SegformerArcheoModel

ARCH_CONFIG = {
    "manet":     {"arch": "MAnet",         "encoder": "efficientnet-b3", "ckpt": "manet_full/lightning_logs/version_0/checkpoints/epoch=19-step=2940.ckpt"},
    "deeplab":   {"arch": "DeepLabV3Plus",  "encoder": "resnet50",        "ckpt": "deeplab_full/lightning_logs/version_0/checkpoints/epoch=19-step=2940.ckpt"},
    "segformer": {"arch": None,             "encoder": "b0",              "ckpt": "segformer_full/lightning_logs/version_0/checkpoints/epoch=19-step=2940.ckpt"},
}
DATASET_PATH = "/workspace/bing_1k"
RUNS_ROOT = "/workspace/0-reproduction/runs"
OUT_DIR = "/workspace/thesis/figures"
RANDOM_SEED = 1234
N_EXAMPLES = 4
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

# Pick a few site-bearing test tiles (skip the "neg" empty-mask ones for the figure)
site_files = [f for f in test_files if not f.startswith("neg")]
rng = random.Random(RANDOM_SEED)
chosen = rng.sample(list(site_files), N_EXAMPLES)

test_dataset = ArcheoDataset(np.array(chosen), path_lookup, transform=val_transform)

models = {}
for arch_key, cfg in ARCH_CONFIG.items():
    ckpt_path = os.path.join(RUNS_ROOT, cfg["ckpt"])
    config = {"arch": cfg["arch"], "encoder": cfg["encoder"], "weights": "imagenet",
              "in_channels": 3, "batch_size": 32, "lr": 1e-4, "loss": "focal"}
    if arch_key == "segformer":
        model = SegformerArcheoModel(encoder_name=cfg["encoder"], in_channels=3, out_classes=1, config=config)
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state["state_dict"])
    else:
        model = _EpochBufferedArcheoModel.load_from_checkpoint(
            ckpt_path, arch=cfg["arch"], encoder_name=cfg["encoder"],
            encoder_weights="imagenet", in_channels=3, out_classes=1, config=config,
        )
    models[arch_key] = model.to(device).eval()

# The dataset transform does NOT normalize -- each model normalizes internally
# in forward() via its own registered mean/std buffers (raw pixel scale varies
# by encoder's pretrained convention, e.g. 0-255 "advprop" vs 0-1 ImageNet).
# Percentile-clip the raw tensor for display rather than assuming a fixed
# [0,1]/[0,255] range.
def denorm(img_tensor):
    x = img_tensor.numpy().transpose(1, 2, 0)  # raw pixel values, whatever scale the dataset produced
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    x = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
    return x

n_cols = 2 + len(ARCH_CONFIG)
fig, axes = plt.subplots(N_EXAMPLES, n_cols, figsize=(3.1 * n_cols, 3.1 * N_EXAMPLES))
col_titles = ["Input", "Ground truth"] + [f"{k} prediction" for k in ARCH_CONFIG]

for row in range(N_EXAMPLES):
    image, mask, fname = test_dataset[row]
    img_np = denorm(image)
    mask_np = mask.squeeze(0).numpy()

    axes[row, 0].imshow(img_np)
    axes[row, 0].set_ylabel(fname.replace(".jpg", ""), fontsize=8)

    axes[row, 1].imshow(img_np)
    axes[row, 1].imshow(np.ma.masked_where(mask_np == 0, mask_np), cmap="spring_r", alpha=0.65, vmin=0, vmax=1)
    axes[row, 1].contour(mask_np, levels=[0.5], colors="black", linewidths=1.2)

    with torch.no_grad():
        x = image.unsqueeze(0).to(device).float()
        for col, (arch_key, model) in enumerate(models.items(), start=2):
            prob = torch.sigmoid(model(x)).squeeze().cpu().numpy()
            axes[row, col].imshow(img_np)
            axes[row, col].imshow(prob, cmap="inferno", alpha=0.55, vmin=0, vmax=1)

    for col in range(n_cols):
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
        if row == 0:
            axes[row, col].set_title(col_titles[col], fontsize=10)

fig.suptitle("Example predictions, faithful-replica checkpoints (test set, unseen during training)", fontsize=11, y=1.0)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/prediction-examples.png", dpi=150, bbox_inches="tight")
print(f"Written {OUT_DIR}/prediction-examples.png using files: {chosen}")
