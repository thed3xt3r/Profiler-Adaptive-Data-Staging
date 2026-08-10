import os
import sys
import time
import json
import random
import numpy as np
import torch

sys.path.insert(0, "/workspace/casini_reference")
from dataset import load_dataset, esegui_trasformazioni, ArcheoDataset
from model import _EpochBufferedArcheoModel, SegformerArcheoModel

ARCH_CONFIG = {
    "manet":     {"arch": "MAnet",         "encoder": "efficientnet-b3", "ckpt": "manet_full/lightning_logs/version_0/checkpoints/epoch=19-step=2940.ckpt"},
    "deeplab":   {"arch": "DeepLabV3Plus",  "encoder": "resnet50",        "ckpt": "deeplab_full/lightning_logs/version_0/checkpoints/epoch=19-step=2940.ckpt"},
    "segformer": {"arch": None,             "encoder": "b0",              "ckpt": "segformer_full/lightning_logs/version_0/checkpoints/epoch=19-step=2940.ckpt"},
}

DATASET_PATH = "/workspace/bing_1k"
RUNS_ROOT = "/workspace/casini_reference/runs"
BATCH_SIZE = 32
RANDOM_SEED = 1234
N_WARMUP_BATCHES = 5
N_TIMED_BATCHES = 30

device = "cuda"
results = {}

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

n_files = len(os.listdir(os.path.join(DATASET_PATH, "train", "originals", "sites"))) \
    + len(os.listdir(os.path.join(DATASET_PATH, "train", "negs", "sites")))
indices = np.arange(0, n_files)
np.random.shuffle(indices)

path_lookup, train_files, val_files, test_files = load_dataset(DATASET_PATH, RANDOM_SEED, indices)
_, val_transform = esegui_trasformazioni()
test_dataset = ArcheoDataset(test_files, path_lookup, transform=val_transform)
loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=True)

for arch_key, cfg in ARCH_CONFIG.items():
    ckpt_path = os.path.join(RUNS_ROOT, cfg["ckpt"])
    config = {
        "arch": cfg["arch"], "encoder": cfg["encoder"], "weights": "imagenet",
        "in_channels": 3, "batch_size": BATCH_SIZE, "lr": 1e-4, "loss": "focal",
    }
    if arch_key == "segformer":
        model = SegformerArcheoModel(encoder_name=cfg["encoder"], in_channels=3, out_classes=1, config=config)
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state["state_dict"])
    else:
        model = _EpochBufferedArcheoModel.load_from_checkpoint(
            ckpt_path, arch=cfg["arch"], encoder_name=cfg["encoder"],
            encoder_weights="imagenet", in_channels=3, out_classes=1, config=config,
        )
    model = model.to(device).eval()

    it = iter(loader)
    with torch.no_grad():
        for _ in range(N_WARMUP_BATCHES):
            try:
                images, masks, fnames = next(it)
            except StopIteration:
                it = iter(loader)
                images, masks, fnames = next(it)
            images = images.to(device)
            _ = model(images)
        torch.cuda.synchronize()

        n_images = 0
        start = time.perf_counter()
        for _ in range(N_TIMED_BATCHES):
            try:
                images, masks, fnames = next(it)
            except StopIteration:
                it = iter(loader)
                images, masks, fnames = next(it)
            images = images.to(device)
            _ = model(images)
            n_images += images.shape[0]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    images_per_sec = n_images / elapsed
    results[arch_key] = {
        "batch_size": BATCH_SIZE, "n_images": n_images,
        "elapsed_s": elapsed, "images_per_sec": images_per_sec,
    }
    print(f"[{arch_key}] {n_images} images in {elapsed:.3f}s -> {images_per_sec:.2f} images/sec")
    del model
    torch.cuda.empty_cache()

out_path = "/workspace/runs/inference_bench_result.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nWritten to {out_path}")
