import os
import sys
import random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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

# Mostly site-bearing tiles, plus one "neg" empty-mask tile: a third of the
# corpus is negative, and a figure of only positives would not show whether the
# model correctly produces nothing when there is nothing to find.
site_files = [f for f in test_files if not f.startswith("neg")]
neg_files = [f for f in test_files if f.startswith("neg")]
rng = random.Random(RANDOM_SEED)
chosen = rng.sample(list(site_files), N_EXAMPLES)
if neg_files:
    # Prefer a *hard* negative. Negatives are drawn from urban areas, intensive
    # agriculture, water and rocky terrain; a uniform field is trivially
    # rejected and demonstrates nothing, whereas visually complex terrain is
    # where a false positive would actually occur. Pick the highest-variance
    # candidate from a random sample as a cheap proxy for visual complexity.
    probe = ArcheoDataset(np.array(rng.sample(list(neg_files), min(12, len(neg_files)))),
                          path_lookup, transform=val_transform)
    variances = [(float(probe[i][0].float().std()), probe.images_filenames[i])
                 for i in range(len(probe))]
    chosen = chosen + [max(variances)[1]]
N_EXAMPLES = len(chosen)

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
    x = img_tensor.numpy().transpose(1, 2, 0).astype(np.float32)
    # Detect the scale rather than contrast-stretching per tile. A per-image
    # percentile stretch renders low-variance tiles (uniform fields, which is
    # exactly what the negative examples are) as a saturated flat wash, because
    # the 1st and 99th percentiles nearly coincide and the tiny spread gets
    # amplified to the full display range. Scaling by a fixed constant keeps
    # colour faithful and comparable across rows.
    if x.max() > 1.5:
        x = x / 255.0
    return np.clip(x, 0, 1)

THRESHOLD = 0.5
GT_COLOR = "#00d0ff"     # ground-truth contour
PRED_COLOR = "#ffd400"   # predicted contour / fill

n_cols = 2 + len(ARCH_CONFIG)
fig, axes = plt.subplots(N_EXAMPLES, n_cols, figsize=(3.1 * n_cols, 3.25 * N_EXAMPLES))
col_titles = ["Input", "Ground truth"] + [f"{k} prediction" for k in ARCH_CONFIG]


def outline(ax, m, color, lw, ls="solid"):
    """Draw a mask boundary, but only if the mask is non-empty -- contour() on an
    all-zero array emits a warning and nothing useful."""
    if m.any():
        ax.contour(m, levels=[0.5], colors=color, linewidths=lw, linestyles=ls)


for row in range(N_EXAMPLES):
    image, mask, fname = test_dataset[row]
    img_np = denorm(image)
    mask_np = mask.squeeze(0).numpy()

    axes[row, 0].imshow(img_np)
    axes[row, 0].set_ylabel(fname.replace(".jpg", ""), fontsize=8)

    # Ground truth: translucent fill plus a hard boundary, so the annotated
    # extent is readable against busy imagery.
    axes[row, 1].imshow(img_np)
    axes[row, 1].imshow(np.ma.masked_where(mask_np < 0.5, mask_np),
                        cmap="cool", alpha=0.45, vmin=0, vmax=1)
    outline(axes[row, 1], mask_np, GT_COLOR, 1.6)

    with torch.no_grad():
        x = image.unsqueeze(0).to(device).float()
        for col, (arch_key, model) in enumerate(models.items(), start=2):
            logits = model(x)
            # SegFormer decodes at reduced resolution; bring every architecture
            # back to mask resolution so contours and IoU are comparable.
            if logits.shape[-2:] != mask_np.shape:
                logits = torch.nn.functional.interpolate(
                    logits, size=mask_np.shape, mode="bilinear", align_corners=False)
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()
            pred = (prob >= THRESHOLD).astype(np.float32)

            ax = axes[row, col]
            ax.imshow(img_np)
            # Thresholded prediction, drawn the same way as the ground truth
            # rather than as a probability wash: the question this figure has to
            # answer is *where* the model says the site is, not how confident it
            # is at every background pixel.
            ax.imshow(np.ma.masked_where(pred < 0.5, pred),
                      cmap="autumn", alpha=0.45, vmin=0, vmax=1)
            outline(ax, pred, PRED_COLOR, 1.6)
            # Ground-truth boundary repeated on the prediction panel, so
            # agreement and disagreement are visible without eye-tracking
            # between columns.
            outline(ax, mask_np, GT_COLOR, 1.2, ls="dashed")

            inter = np.logical_and(pred >= 0.5, mask_np >= 0.5).sum()
            union = np.logical_or(pred >= 0.5, mask_np >= 0.5).sum()
            if not mask_np.any():
                # Negative tile. IoU is either 0/0 (undefined) or 0/|pred|, and
                # reporting "0.00" would read as a missed site rather than as
                # what it is -- a spurious detection where nothing is annotated.
                label = "correct reject" if not pred.any() else "false positive"
            else:
                label = f"IoU {inter / union:.2f}"
            ax.text(0.03, 0.03, label, transform=ax.transAxes, fontsize=8,
                    color="white", va="bottom", ha="left",
                    bbox=dict(facecolor="black", alpha=0.6, pad=2, edgecolor="none"))

    for col in range(n_cols):
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
        if row == 0:
            axes[row, col].set_title(col_titles[col], fontsize=10)

handles = [
    Line2D([0], [0], color=GT_COLOR, lw=2, label="Ground-truth boundary"),
    Line2D([0], [0], color=GT_COLOR, lw=1.5, ls="--", label="Ground truth (on prediction panels)"),
    Line2D([0], [0], color=PRED_COLOR, lw=2, label=f"Predicted boundary (p $\\geq$ {THRESHOLD})"),
]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
           fontsize=9, bbox_to_anchor=(0.5, -0.012))
fig.suptitle("Example predictions, faithful-replica checkpoints (test set, unseen during training)",
             fontsize=11, y=1.0)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/prediction-examples.png", dpi=150, bbox_inches="tight")
print(f"Written {OUT_DIR}/prediction-examples.png using files: {chosen}")
