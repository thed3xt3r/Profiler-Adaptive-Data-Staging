# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "torch",
#     "lightning",
#     "transformers",
#     "torchmetrics",
#     "albumentations",
#     "optuna",
#     "numpy",
#     "matplotlib",
#     "pillow",
#     "gdown",
# ]
# ///

import marimo

__generated_with = "0.9.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # PADS — SegFormer (MiT-B0, HuggingFace) on bing_1k

        Self-contained marimo port of `PADS/segformer`: policy-aware data loading
        (`loose` / `shard` / `stage`), Optuna hyperparameter tuning, and PyTorch
        profiler integration for the archaeological site segmentation task.

        Fill in the configuration below, optionally pull the dataset from Google
        Drive, then click **Start PADS pipeline** to run.
        """
    )
    return


@app.cell
def _():
    import os
    import json
    import random
    import re
    import time
    import tarfile
    import shutil
    import tempfile
    import urllib.error
    import urllib.parse
    import urllib.request
    from io import BytesIO
    from datetime import datetime

    import numpy as np
    import optuna
    import torch
    import torch.nn.functional as F
    from torch.nn import BCEWithLogitsLoss
    import lightning.pytorch as pl
    from lightning.pytorch import loggers as pl_loggers
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
    from torch.utils.data import Dataset, DataLoader
    from torch.profiler import profile, record_function, ProfilerActivity

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import albumentations as A
    from PIL import Image

    from transformers import SegformerForSemanticSegmentation
    from torchmetrics import Accuracy, JaccardIndex

    return (
        A,
        Accuracy,
        BCEWithLogitsLoss,
        BytesIO,
        Dataset,
        DataLoader,
        EarlyStopping,
        F,
        Image,
        JaccardIndex,
        ModelCheckpoint,
        ProfilerActivity,
        SegformerForSemanticSegmentation,
        datetime,
        json,
        np,
        optuna,
        os,
        pl,
        pl_loggers,
        plt,
        profile,
        random,
        re,
        record_function,
        shutil,
        tarfile,
        tempfile,
        time,
        torch,
        urllib,
    )


@app.cell
def _(torch):
    # GPU compatibility workaround: bypass Lightning's Ampere-or-later check,
    # which can misfire on older/newer GPUs depending on driver/torch build.
    torch.set_float32_matmul_precision("medium")

    try:
        import lightning.fabric.accelerators.cuda

        def _dummy_is_ampere_or_later(device):
            return False

        lightning.fabric.accelerators.cuda._is_ampere_or_later = _dummy_is_ampere_or_later
    except (ImportError, AttributeError):
        pass
    return


@app.cell
def _(pl, plt):
    class TrainingPlotCallback(pl.Callback):
        """
        Collects loss and IoU each epoch and saves a PNG at the end of training.
        Output: <ckpt_dir>/training_curves.png
        """

        def __init__(self, save_path: str = "training_curves.png"):
            self.save_path = save_path
            self.train_loss, self.val_loss = [], []
            self.train_iou, self.val_iou = [], []

        def on_train_epoch_end(self, trainer, pl_module):
            metrics = trainer.callback_metrics
            self._append(self.train_loss, metrics.get("train/loss_epoch"))
            self._append(self.train_iou, metrics.get("train/iou"))

        def on_validation_epoch_end(self, trainer, pl_module):
            if trainer.sanity_checking:
                return
            metrics = trainer.callback_metrics
            self._append(self.val_loss, metrics.get("valid/loss"))
            self._append(self.val_iou, metrics.get("valid/iou"))

        def on_train_end(self, trainer, pl_module):
            if not self.train_loss:
                return

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            train_epochs = range(1, len(self.train_loss) + 1)
            val_epochs = range(1, len(self.val_loss) + 1)

            ax1.plot(train_epochs, self.train_loss, label="Train Loss")
            if self.val_loss:
                ax1.plot(val_epochs, self.val_loss, label="Val Loss")
            ax1.set_title("Loss")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.legend()
            ax1.grid(True)

            if self.train_iou:
                ax2.plot(train_epochs, self.train_iou, label="Train IoU")
            if self.val_iou:
                ax2.plot(val_epochs, self.val_iou, label="Val IoU")
            ax2.set_title("IoU (Jaccard Index)")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("IoU")
            ax2.legend()
            ax2.grid(True)

            fig.suptitle("Training Curves — SegFormer B0", fontsize=14)
            plt.tight_layout()
            plt.savefig(self.save_path, dpi=150)
            plt.close(fig)
            print(f"\nTraining curves saved to: {self.save_path}")

        @staticmethod
        def _append(lst, value):
            if value is not None:
                lst.append(float(value))

    return (TrainingPlotCallback,)


@app.cell
def _(json, os, re, urllib):
    # ---------------------------------------------------------------------
    # Google Drive API helpers (streaming data source)
    # ---------------------------------------------------------------------
    # Lets training read bing_1k directly from a shared Drive folder,
    # fetching + caching each image/mask the first time it's actually used
    # instead of bulk-downloading ~11,700 files upfront. Uses the Drive API
    # v3 REST endpoints directly (stdlib urllib only, no extra dependency)
    # with an API key -- works against "Anyone with the link" folders
    # without an OAuth login flow.

    _DRIVE_ID_RE = re.compile(r"[-\w]{25,}")

    def extract_drive_id(url_or_id):
        """Pull a Drive file/folder ID out of a share URL, or pass an ID through."""
        matches = _DRIVE_ID_RE.findall(url_or_id or "")
        return max(matches, key=len) if matches else (url_or_id or "").strip()

    def _drive_api_get(path, api_key, **params):
        params["key"] = api_key
        url = f"https://www.googleapis.com/drive/v3/{path}?{urllib.parse.urlencode(params)}"
        try:
            with urllib.request.urlopen(url) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Drive API request failed ({exc.code}): {body}") from exc

    def list_drive_folder(folder_id, api_key):
        """List a Drive folder's immediate children as {name: (id, mimeType)}."""
        entries = {}
        page_token = None
        while True:
            params = {
                "q": f"'{folder_id}' in parents and trashed = false",
                "fields": "nextPageToken, files(id, name, mimeType)",
                "pageSize": 1000,
            }
            if page_token:
                params["pageToken"] = page_token
            data = _drive_api_get("files", api_key, **params)
            for f in data.get("files", []):
                entries[f["name"]] = (f["id"], f["mimeType"])
            page_token = data.get("nextPageToken")
            if not page_token:
                break
        return entries

    def find_drive_subfolder(parent_id, name, api_key):
        entries = list_drive_folder(parent_id, api_key)
        if name not in entries:
            raise FileNotFoundError(f"Subfolder '{name}' not found under Drive folder {parent_id}")
        file_id, mime = entries[name]
        if mime != "application/vnd.google-apps.folder":
            raise ValueError(f"'{name}' under Drive folder {parent_id} is not a folder")
        return file_id

    def resolve_bing1k_drive_folders(root_id, api_key):
        """Walk train/{originals,negs}/{sites,masks} under the shared root
        folder, returning {"originals_sites": {filename: file_id}, ...}."""
        train_id = find_drive_subfolder(root_id, "train", api_key)
        result = {}
        for source in ("originals", "negs"):
            source_id = find_drive_subfolder(train_id, source, api_key)
            for kind in ("sites", "masks"):
                kind_id = find_drive_subfolder(source_id, kind, api_key)
                entries = list_drive_folder(kind_id, api_key)
                result[f"{source}_{kind}"] = {name: file_id for name, (file_id, _mime) in entries.items()}
        return result

    def download_drive_file(file_id, api_key, dest_path):
        """Fetch a single file's bytes by ID and cache it at dest_path (no-op if already cached)."""
        if os.path.exists(dest_path):
            return dest_path
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&key={api_key}"
        with urllib.request.urlopen(url) as resp:
            data = resp.read()
        tmp_path = dest_path + ".part"
        with open(tmp_path, "wb") as fh:
            fh.write(data)
        os.replace(tmp_path, dest_path)
        return dest_path

    return (
        download_drive_file,
        extract_drive_id,
        list_drive_folder,
        resolve_bing1k_drive_folders,
    )


@app.cell
def _(A, BytesIO, Dataset, Image, np, os, shutil, tarfile, tempfile, time):
    def load_dataset(PATH, SEED, indices):
        """
        Load and split dataset into train (80%), validation (10%), and test (10%).

        Combines both 'originals' and 'negs' datasets from:
        - PATH/train/originals/sites/ and PATH/train/originals/masks/
        - PATH/train/negs/sites/ and PATH/train/negs/masks/
        """
        rng = np.random.RandomState(SEED)
        rng.shuffle(indices)

        root_directory = os.path.join(PATH)

        originals_images_dir = os.path.join(root_directory, "train/originals/sites")
        originals_masks_dir = os.path.join(root_directory, "train/originals/masks")
        originals_images = sorted(os.listdir(originals_images_dir)) if os.path.exists(originals_images_dir) else []

        negs_images_dir = os.path.join(root_directory, "train/negs/sites")
        negs_masks_dir = os.path.join(root_directory, "train/negs/masks")
        negs_images = sorted(os.listdir(negs_images_dir)) if os.path.exists(negs_images_dir) else []

        combined_data = []
        for fname in originals_images:
            combined_data.append({"filename": fname, "source": "originals"})
        for fname in negs_images:
            combined_data.append({"filename": fname, "source": "negs"})

        print(f"Loaded {len(originals_images)} originals images")
        print(f"Loaded {len(negs_images)} negs images")
        print(f"Total combined images: {len(combined_data)}")

        valid_split = -int(len(combined_data) * 0.2)
        test_split = valid_split // 2

        train_indices = indices[:valid_split]
        valid_indices = indices[valid_split:test_split]
        test_indices = indices[test_split:]

        train_data = [combined_data[i] for i in train_indices]
        val_data = [combined_data[i] for i in valid_indices]
        test_data = [combined_data[i] for i in test_indices]

        print(f"Train images: {len(train_data)}")
        print(f"Validation images: {len(val_data)}")
        print(f"Test images: {len(test_data)}")

        return (originals_images_dir, originals_masks_dir, negs_images_dir, negs_masks_dir,
                train_data, val_data, test_data)

    def get_transforms():
        """Augmentation transforms for bing_1k (512x512 images resized to 256x256)."""
        train_transform = A.Compose([
            A.RandomCrop(512, 512, p=1.0),
            A.HorizontalFlip(p=0.25),
            A.VerticalFlip(p=0.25),
            A.RandomRotate90(p=0.25),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.25),
            A.Resize(256, 256),
        ])

        val_transform = A.Compose([
            A.RandomCrop(512, 512, p=1.0),
            A.Resize(256, 256),
        ])

        return train_transform, val_transform

    def _finalize(image, mask, transform):
        """Apply the augmentation pipeline and convert to the model's expected
        layout: float32 image normalised to [0, 1] and transposed to CHW,
        mask as (1, H, W)."""
        if transform:
            transformed = transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))

        mask = np.expand_dims(mask.astype(np.float32), 0)

        return image, mask

    class ArcheoDataset(Dataset):
        """Dataset class for archaeological image segmentation."""

        def __init__(self, data_items, originals_images_dir, originals_masks_dir,
                     negs_images_dir, negs_masks_dir, transform=None):
            self.data_items = data_items
            self.originals_images_dir = originals_images_dir
            self.originals_masks_dir = originals_masks_dir
            self.negs_images_dir = negs_images_dir
            self.negs_masks_dir = negs_masks_dir
            self.transform = transform
            self.profile_enabled = False
            self.profile_stats = []

        def __len__(self):
            return len(self.data_items)

        def reset_profile_stats(self):
            self.profile_stats = []

        def __getitem__(self, idx):
            data_item = self.data_items[idx]
            image_filename = data_item["filename"]
            source = data_item["source"]

            if source == "originals":
                image_dir = self.originals_images_dir
                mask_dir = self.originals_masks_dir
            else:
                image_dir = self.negs_images_dir
                mask_dir = self.negs_masks_dir

            start_time = time.perf_counter()
            image_path = os.path.join(image_dir, image_filename)
            mask_filename = image_filename.replace(".jpg", ".png")
            mask_path = os.path.join(mask_dir, mask_filename)

            image = np.asarray(Image.open(image_path).convert("RGB"))

            if os.path.exists(mask_path):
                mask = ~np.array(Image.open(mask_path).convert("L"))
            else:
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

            mask = mask.astype(np.float32)
            mask[mask > 0.0] = 1.0

            image, mask = _finalize(image, mask, self.transform)

            if self.profile_enabled:
                self.profile_stats.append(time.perf_counter() - start_time)

            return image, mask, image_filename

    class TarArcheoDataset(Dataset):
        """Dataset that reads from tar-shard archives instead of loose files."""

        def __init__(self, data_items, originals_tar_path, negs_tar_path, transform=None):
            self.data_items = data_items
            self.originals_tar_path = originals_tar_path
            self.negs_tar_path = negs_tar_path
            self.transform = transform
            self.profile_enabled = False
            self.profile_stats = []
            self._tar_cache = {}

        def __len__(self):
            return len(self.data_items)

        def reset_profile_stats(self):
            self.profile_stats = []

        def _get_tar(self, tar_path):
            if tar_path not in self._tar_cache:
                self._tar_cache[tar_path] = tarfile.open(tar_path, "r:gz")
            return self._tar_cache[tar_path]

        def __getitem__(self, idx):
            data_item = self.data_items[idx]
            image_filename = data_item["filename"]
            source = data_item["source"]

            start_time = time.perf_counter()

            if source == "originals":
                tar_path = self.originals_tar_path
            else:
                tar_path = self.negs_tar_path
            image_member = f"images/{image_filename}"
            mask_member = f"masks/{image_filename.replace('.jpg', '.png')}"

            try:
                tar = self._get_tar(tar_path)
                image_f = tar.extractfile(image_member)
                image = np.asarray(Image.open(BytesIO(image_f.read())).convert("RGB"))
                try:
                    mask_f = tar.extractfile(mask_member)
                    mask = ~np.array(Image.open(BytesIO(mask_f.read())).convert("L"))
                except KeyError:
                    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            except (KeyError, tarfile.TarError):
                image = np.zeros((256, 256, 3), dtype=np.uint8)
                mask = np.zeros((256, 256), dtype=np.uint8)

            mask = mask.astype(np.float32)
            mask[mask > 0.0] = 1.0

            image, mask = _finalize(image, mask, self.transform)

            if self.profile_enabled:
                self.profile_stats.append(time.perf_counter() - start_time)

            return image, mask, image_filename

        def __del__(self):
            for tar in self._tar_cache.values():
                try:
                    tar.close()
                except Exception:
                    pass

    class StagingArcheoDataset(Dataset):
        """Dataset that stages data to node-local scratch before loading."""

        def __init__(self, data_items, originals_images_dir, originals_masks_dir,
                     negs_images_dir, negs_masks_dir, transform=None, scratch_dir=None):
            self.data_items = data_items
            self.originals_images_dir = originals_images_dir
            self.originals_masks_dir = originals_masks_dir
            self.negs_images_dir = negs_images_dir
            self.negs_masks_dir = negs_masks_dir
            self.transform = transform
            self.profile_enabled = False
            self.profile_stats = []
            self.scratch_dir = scratch_dir or tempfile.gettempdir()
            self._staged = set()
            self._stage_batch(0, min(64, len(data_items)))

        def _stage_batch(self, start_idx, end_idx):
            for idx in range(start_idx, end_idx):
                if idx >= len(self.data_items):
                    break
                data_item = self.data_items[idx]
                image_filename = data_item["filename"]
                source = data_item["source"]

                if source == "originals":
                    src_img_dir = self.originals_images_dir
                    src_mask_dir = self.originals_masks_dir
                else:
                    src_img_dir = self.negs_images_dir
                    src_mask_dir = self.negs_masks_dir

                src_img = os.path.join(src_img_dir, image_filename)
                src_mask = os.path.join(src_mask_dir, image_filename.replace(".jpg", ".png"))

                dst_img = os.path.join(self.scratch_dir, image_filename)
                dst_mask = os.path.join(self.scratch_dir, image_filename.replace(".jpg", ".png"))

                try:
                    if not os.path.exists(dst_img) and os.path.exists(src_img):
                        shutil.copy2(src_img, dst_img)
                    if not os.path.exists(dst_mask) and os.path.exists(src_mask):
                        shutil.copy2(src_mask, dst_mask)
                    self._staged.add(idx)
                except (IOError, OSError):
                    pass

        def __len__(self):
            return len(self.data_items)

        def reset_profile_stats(self):
            self.profile_stats = []

        def __getitem__(self, idx):
            if idx % 16 == 0:
                self._stage_batch(idx + 64, idx + 128)

            data_item = self.data_items[idx]
            image_filename = data_item["filename"]

            start_time = time.perf_counter()

            scratch_img = os.path.join(self.scratch_dir, image_filename)
            scratch_mask = os.path.join(self.scratch_dir, image_filename.replace(".jpg", ".png"))

            if os.path.exists(scratch_img):
                image = np.asarray(Image.open(scratch_img).convert("RGB"))
                if os.path.exists(scratch_mask):
                    mask = ~np.array(Image.open(scratch_mask).convert("L"))
                else:
                    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            else:
                source = data_item["source"]
                if source == "originals":
                    image_dir = self.originals_images_dir
                    mask_dir = self.originals_masks_dir
                else:
                    image_dir = self.negs_images_dir
                    mask_dir = self.negs_masks_dir

                image_path = os.path.join(image_dir, image_filename)
                mask_path = os.path.join(mask_dir, image_filename.replace(".jpg", ".png"))

                image = np.asarray(Image.open(image_path).convert("RGB"))
                if os.path.exists(mask_path):
                    mask = ~np.array(Image.open(mask_path).convert("L"))
                else:
                    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

            mask = mask.astype(np.float32)
            mask[mask > 0.0] = 1.0

            image, mask = _finalize(image, mask, self.transform)

            if self.profile_enabled:
                self.profile_stats.append(time.perf_counter() - start_time)

            return image, mask, image_filename

    def create_dataset(data_items, originals_images_dir, originals_masks_dir,
                        negs_images_dir, negs_masks_dir, transform, policy="loose",
                        originals_tar_path=None, negs_tar_path=None, scratch_dir=None):
        """Factory function to create the appropriate dataset based on policy."""
        if policy == "shard":
            if originals_tar_path and negs_tar_path:
                return TarArcheoDataset(data_items, originals_tar_path, negs_tar_path, transform=transform)
            print("[Warning] shard policy requested but tar files not found, falling back to loose")
            return ArcheoDataset(data_items, originals_images_dir, originals_masks_dir,
                                  negs_images_dir, negs_masks_dir, transform=transform)
        elif policy == "stage":
            return StagingArcheoDataset(data_items, originals_images_dir, originals_masks_dir,
                                         negs_images_dir, negs_masks_dir, transform=transform,
                                         scratch_dir=scratch_dir)
        else:  # loose
            return ArcheoDataset(data_items, originals_images_dir, originals_masks_dir,
                                  negs_images_dir, negs_masks_dir, transform=transform)

    return ArcheoDataset, StagingArcheoDataset, TarArcheoDataset, _finalize, create_dataset, get_transforms, load_dataset


@app.cell
def _(Dataset, Image, _finalize, download_drive_file, np, os, resolve_bing1k_drive_folders, time):
    def load_dataset_drive(root_id, api_key, seed):
        """Drive-backed equivalent of load_dataset(): lists the shared Drive
        folder instead of local directories, but produces the same
        combined/shuffled/split structure."""
        folders = resolve_bing1k_drive_folders(root_id, api_key)

        originals_images = sorted(folders["originals_sites"].keys())
        negs_images = sorted(folders["negs_sites"].keys())

        combined_data = []
        for fname in originals_images:
            combined_data.append({"filename": fname, "source": "originals"})
        for fname in negs_images:
            combined_data.append({"filename": fname, "source": "negs"})

        print(f"[Drive] Found {len(originals_images)} originals images")
        print(f"[Drive] Found {len(negs_images)} negs images")
        print(f"[Drive] Total combined images: {len(combined_data)}")

        indices = np.arange(0, len(combined_data))
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)

        valid_split = -int(len(combined_data) * 0.2)
        test_split = valid_split // 2

        train_indices = indices[:valid_split]
        valid_indices = indices[valid_split:test_split]
        test_indices = indices[test_split:]

        train_data = [combined_data[i] for i in train_indices]
        val_data = [combined_data[i] for i in valid_indices]
        test_data = [combined_data[i] for i in test_indices]

        print(f"Train images: {len(train_data)}")
        print(f"Validation images: {len(val_data)}")
        print(f"Test images: {len(test_data)}")

        return folders, train_data, val_data, test_data

    class DriveArcheoDataset(Dataset):
        """Dataset that fetches each image/mask from Google Drive on first
        access and caches it to local disk, so training can start without a
        bulk download step. Later epochs read from the local cache."""

        def __init__(self, data_items, drive_folders, api_key, cache_dir, transform=None):
            self.data_items = data_items
            self.drive_folders = drive_folders  # {"originals_sites": {filename: file_id}, ...}
            self.api_key = api_key
            self.cache_dir = cache_dir
            self.transform = transform
            self.profile_enabled = False
            self.profile_stats = []
            os.makedirs(cache_dir, exist_ok=True)

        def __len__(self):
            return len(self.data_items)

        def reset_profile_stats(self):
            self.profile_stats = []

        def _fetch(self, source, kind, filename):
            ids_map = self.drive_folders[f"{source}_{kind}"]
            cache_path = os.path.join(self.cache_dir, source, kind, filename)
            if not os.path.exists(cache_path):
                file_id = ids_map.get(filename)
                if file_id is None:
                    return None
                download_drive_file(file_id, self.api_key, cache_path)
            return cache_path

        def __getitem__(self, idx):
            data_item = self.data_items[idx]
            image_filename = data_item["filename"]
            source = data_item["source"]
            mask_filename = image_filename.replace(".jpg", ".png")

            start_time = time.perf_counter()

            image_path = self._fetch(source, "sites", image_filename)
            image = np.asarray(Image.open(image_path).convert("RGB"))

            mask_path = self._fetch(source, "masks", mask_filename)
            if mask_path is not None:
                mask = ~np.array(Image.open(mask_path).convert("L"))
            else:
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

            mask = mask.astype(np.float32)
            mask[mask > 0.0] = 1.0

            image, mask = _finalize(image, mask, self.transform)

            if self.profile_enabled:
                self.profile_stats.append(time.perf_counter() - start_time)

            return image, mask, image_filename

    return DriveArcheoDataset, load_dataset_drive


@app.cell
def _(Accuracy, BCEWithLogitsLoss, F, JaccardIndex, SegformerForSemanticSegmentation, pl, torch):
    class ArcheoModel(pl.LightningModule):
        """Lightning module for archaeological site segmentation using NVlabs'
        SegFormer via HuggingFace Transformers.

        Loads a pretrained SegFormer variant (b0-b5) from the
        'nvidia/segformer-{variant}-finetuned-ade-512-512' checkpoint and
        replaces the classification head for binary segmentation.

        NOTE: the SegFormer decode head outputs logits at 1/4 of the input
        spatial resolution; they are upsampled back to full size with
        bilinear interpolation before computing loss/metrics.
        """

        _VARIANT_MAP = {
            "mit_b0": "b0", "mit_b1": "b1", "mit_b2": "b2",
            "mit_b3": "b3", "mit_b4": "b4", "mit_b5": "b5",
            "b0": "b0", "b1": "b1", "b2": "b2",
            "b3": "b3", "b4": "b4", "b5": "b5",
        }

        def __init__(self, encoder_name, in_channels, out_classes, config):
            super().__init__()
            self.save_hyperparameters()
            self.config = config

            variant = self._VARIANT_MAP.get(encoder_name, "b0")
            pretrained_name = f"nvidia/segformer-{variant}-finetuned-ade-512-512"

            self.model = SegformerForSemanticSegmentation.from_pretrained(
                pretrained_name,
                num_labels=out_classes,
                ignore_mismatched_sizes=True,
            )

            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

            self.loss_fn = BCEWithLogitsLoss()

            metric_kwargs = dict(task="binary", threshold=0.5)
            self.train_iou = JaccardIndex(**metric_kwargs)
            self.train_acc = Accuracy(**metric_kwargs)
            self.val_iou = JaccardIndex(**metric_kwargs)
            self.val_acc = Accuracy(**metric_kwargs)
            self.test_iou = JaccardIndex(**metric_kwargs)
            self.test_acc = Accuracy(**metric_kwargs)

        def forward(self, image):
            image = (image - self.mean) / self.std
            outputs = self.model(pixel_values=image)
            logits = outputs.logits  # (B, out_classes, H/4, W/4)
            logits = F.interpolate(
                logits, size=image.shape[2:], mode="bilinear", align_corners=False,
            )
            return logits

        def shared_step(self, batch, stage):
            image = batch[0]
            if not torch.is_tensor(image):
                image = torch.tensor(image)
            image = image.float()

            mask = batch[1]
            if not torch.is_tensor(mask):
                mask = torch.tensor(mask)
            mask = mask.float()

            logits_mask = self.forward(image)
            loss = self.loss_fn(logits_mask, mask)

            prob_mask = logits_mask.sigmoid()
            pred_mask = (prob_mask > 0.5).long()

            pred_flat = pred_mask.squeeze(1)
            mask_flat = mask.squeeze(1).long()

            if stage == "train":
                self.train_iou(pred_flat, mask_flat)
                self.train_acc(pred_flat, mask_flat)
                self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True,
                         batch_size=self.config["batch_size"])
                self.log("train/iou", self.train_iou, on_step=False, on_epoch=True, prog_bar=True,
                         batch_size=self.config["batch_size"])
                self.log("train/acc", self.train_acc, on_step=False, on_epoch=True,
                         batch_size=self.config["batch_size"])
            elif stage == "valid":
                self.val_iou(pred_flat, mask_flat)
                self.val_acc(pred_flat, mask_flat)
                self.log("valid/loss", loss, on_step=False, on_epoch=True, prog_bar=True,
                         batch_size=self.config["batch_size"])
                self.log("valid/iou", self.val_iou, on_step=False, on_epoch=True, prog_bar=True,
                         batch_size=self.config["batch_size"])
                self.log("valid/acc", self.val_acc, on_step=False, on_epoch=True,
                         batch_size=self.config["batch_size"])
            elif stage == "test":
                self.test_iou(pred_flat, mask_flat)
                self.test_acc(pred_flat, mask_flat)
                self.log("test/loss", loss, on_step=False, on_epoch=True,
                         batch_size=self.config["batch_size"])
                self.log("test/iou", self.test_iou, on_step=False, on_epoch=True,
                         batch_size=self.config["batch_size"])
                self.log("test/acc", self.test_acc, on_step=False, on_epoch=True,
                         batch_size=self.config["batch_size"])

            return {"loss": loss}

        def training_step(self, batch, batch_idx):
            return self.shared_step(batch, "train")

        def validation_step(self, batch, batch_idx):
            return self.shared_step(batch, "valid")

        def test_step(self, batch, batch_idx):
            return self.shared_step(batch, "test")

        def configure_optimizers(self):
            return torch.optim.AdamW(self.parameters(), lr=self.config["learning_rate"])

    return (ArcheoModel,)


@app.cell
def _(DataLoader, np, os, json, time, torch):
    def classify_bottleneck(metrics, scratch_available=False):
        """Classify the dominant bottleneck from profiled timings."""
        t_data = float(metrics.get("t_data", 0.0))
        t_decode = float(metrics.get("t_decode", 0.0))
        t_h2d = float(metrics.get("t_h2d", 0.0))
        t_gpu = float(metrics.get("t_gpu", 0.0))

        if scratch_available and t_h2d > max(t_decode, t_gpu, 0.05) * 1.25:
            return "stage"
        if t_decode > max(t_h2d, t_gpu) * 1.5:
            return "shard"
        if t_data > max(t_decode, t_gpu) * 1.5:
            return "shard"
        return "loose"

    def select_policy(metrics, scratch_available=False):
        """Select a PADS policy from collected profiler metrics."""
        if scratch_available and float(metrics.get("t_h2d", 0.0)) > 0.08:
            return "stage"
        return classify_bottleneck(metrics, scratch_available=scratch_available)

    def build_dataloader(dataset, batch_size, shuffle, drop_last, policy="loose", num_workers=4):
        """Create a DataLoader with policy-based worker settings."""
        if policy == "drive":
            # Network-bound: more parallel workers hides per-file HTTP
            # latency; deeper prefetch keeps the GPU fed across fetches.
            workers = min(max(num_workers, 4), 8)
            prefetch_factor = 4
        elif policy == "stage":
            workers = min(max(num_workers, 4), 8)
            prefetch_factor = 2
        elif policy == "shard":
            workers = min(max(num_workers, 2), 8)
            prefetch_factor = 2
        else:
            workers = max(1, num_workers - 1)
            prefetch_factor = 1

        loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "drop_last": drop_last,
            "pin_memory": torch.cuda.is_available(),
            "num_workers": workers,
        }
        if workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = prefetch_factor

        return DataLoader(dataset, **loader_kwargs)

    def profile_data_pipeline(model, dataset, batch_size=32, device="cuda",
                               num_batches=12, scratch_available=True,
                               profile_dir="profiler_logs", num_workers=0):
        """Profile batch acquisition, decode and H2D timings and select a PADS policy."""
        os.makedirs(profile_dir, exist_ok=True)
        dataset.reset_profile_stats()
        dataset.profile_enabled = True

        model = model.to(device)
        model.eval()

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=(device == "cuda"),
        )

        iterator = iter(dataloader)
        metrics = {"t_data": 0.0, "t_decode": 0.0, "t_h2d": 0.0, "t_gpu": 0.0}
        batches_seen = 0

        for _ in range(num_batches):
            try:
                batch_start = time.perf_counter()
                batch = next(iterator)
                t_data = time.perf_counter() - batch_start
            except StopIteration:
                break

            sample_count = batch[0].shape[0] if isinstance(batch[0], torch.Tensor) else len(batch[0])
            decode_values = dataset.profile_stats[-sample_count:] if dataset.profile_stats else []
            t_decode = float(np.mean(decode_values)) if decode_values else 0.0
            dataset.reset_profile_stats()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            h2d_start = time.perf_counter()
            images = batch[0].to(device, non_blocking=True).float()
            t_h2d = time.perf_counter() - h2d_start

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            gpu_start = time.perf_counter()
            with torch.no_grad():
                _ = model(images)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_gpu = time.perf_counter() - gpu_start

            metrics["t_data"] += t_data
            metrics["t_decode"] += t_decode
            metrics["t_h2d"] += t_h2d
            metrics["t_gpu"] += t_gpu
            batches_seen += 1

        if batches_seen == 0:
            metrics = {"t_data": 0.0, "t_decode": 0.0, "t_h2d": 0.0, "t_gpu": 0.0}
            policy = "loose"
        else:
            for key in metrics:
                metrics[key] /= batches_seen
            policy = select_policy(metrics, scratch_available=scratch_available)

        dataset.profile_enabled = False

        summary = {
            "policy": policy,
            "scratch_available": scratch_available,
            "metrics": metrics,
            "batches_profiled": batches_seen,
        }
        summary_path = os.path.join(profile_dir, "data_policy_summary.json")
        with open(summary_path, "w") as handle:
            json.dump(summary, handle, indent=2)

        print("\n[PADS Profiler] Warm-up summary")
        print(f"  Policy: {policy}")
        print(f"  t_data: {metrics['t_data']:.4f}s")
        print(f"  t_decode: {metrics['t_decode']:.4f}s")
        print(f"  t_h2d: {metrics['t_h2d']:.4f}s")
        print(f"  t_gpu: {metrics['t_gpu']:.4f}s")
        print(f"  Summary saved to: {summary_path}")

        return summary

    return build_dataloader, profile_data_pipeline


@app.cell
def _(ProfilerActivity, os, profile, record_function, torch):
    def profile_model(model, dataloader, num_batches=5, device="cuda",
                       profile_dir="profiler_logs", wait=1, warmup=1, active=3):
        """Profile model inference using PyTorch Profiler. Saves Chrome trace + text report."""
        os.makedirs(profile_dir, exist_ok=True)
        model = model.to(device)
        model.eval()

        print(f"\n{'=' * 60}")
        print(f"PROFILING MODEL ON {num_batches} BATCHES")
        print(f"{'=' * 60}")
        print(f"wait={wait}, warmup={warmup}, active={active}")

        activities = [ProfilerActivity.CPU]
        if device == "cuda":
            activities.append(ProfilerActivity.CUDA)

        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
            on_trace_ready=lambda p: p.export_chrome_trace(os.path.join(profile_dir, "trace.json")),
        ) as prof:
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    if batch_idx >= num_batches:
                        break
                    images = batch[0].to(device).float()
                    with record_function("model_forward"):
                        _ = model(images)
                    if (batch_idx + 1) % max(1, num_batches // 3) == 0:
                        print(f"  Profiled {batch_idx + 1}/{num_batches} batches...")
                    prof.step()

        print("\nProfiler Summary (Top 15 operations by CPU time):")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))
        if device == "cuda":
            print("\nProfiler Summary (Top 15 operations by CUDA time):")
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

        report_path = os.path.join(profile_dir, "profiler_report.txt")
        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("PROFILER REPORT — SegFormer B0\n")
            f.write("=" * 80 + "\n\n")
            f.write("Top 50 operations by CPU time:\n")
            f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=50))
            if device == "cuda":
                f.write("\n\nTop 50 operations by CUDA time:\n")
                f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))
                f.write("\n\nTop 50 operations by CUDA memory usage:\n")
                f.write(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=50))

        print(f"\nDetailed report saved to: {report_path}")
        print(f"Chrome trace saved to: {os.path.join(profile_dir, 'trace.json')}")
        print(f"{'=' * 60}\n")

    return (profile_model,)


@app.cell
def _(
    ArcheoModel,
    DataLoader,
    EarlyStopping,
    ModelCheckpoint,
    TrainingPlotCallback,
    optuna,
    os,
    pl,
    pl_loggers,
    profile_model,
    torch,
):
    def _build_pruning_callback(trial):
        if trial is None:
            return None
        try:
            from optuna.integration import PyTorchLightningPruningCallback
            return PyTorchLightningPruningCallback(trial, monitor="valid/iou")
        except Exception as exc:
            print(f"[Optuna] Warning: pruning callback unavailable ({exc}); continuing without it.")
            return None

    def run_training(config, train_loader, val_loader, val_dataset, trial=None):
        model = ArcheoModel(
            encoder_name=config["encoder"],
            in_channels=config["in_channels"],
            out_classes=1,
            config=config,
        )

        callbacks_list = []
        checkpoint_callback = ModelCheckpoint(
            dirpath=config["checkpoint_path"],
            filename=f"segformer-b0-trial{trial.number if trial else '0'}-{{epoch:02d}}-{{valid/iou:.4f}}",
            monitor="valid/iou",
            mode="max",
            save_top_k=1,
            save_last=True,
            verbose=True,
        )
        callbacks_list.append(checkpoint_callback)

        early_stopping = EarlyStopping(
            monitor="valid/iou",
            mode="max",
            patience=config["patience"],
            verbose=True,
        )
        callbacks_list.append(early_stopping)

        if trial is None:
            plot_callback = TrainingPlotCallback(
                save_path=os.path.join(config["checkpoint_path"], "training_curves.png")
            )
            callbacks_list.append(plot_callback)
        else:
            pruning_callback = _build_pruning_callback(trial)
            if pruning_callback is not None:
                callbacks_list.append(pruning_callback)

        trainer = pl.Trainer(
            max_epochs=config["epochs"],
            precision=config["precision"],
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            logger=pl_loggers.TensorBoardLogger(config["checkpoint_path"]),
            log_every_n_steps=1,
            enable_progress_bar=True,
            callbacks=callbacks_list,
            # "warn" (not True): SegFormer's decode head does a bilinear upsample,
            # whose CUDA backward kernel has no deterministic implementation —
            # deterministic=True hard-crashes on GPU.
            deterministic="warn",
        )

        cfg_text = "\n".join([f"{key}: {config[key]}" for key in config])
        if trial is None:
            print("\nTraining Configuration:")
            print(cfg_text)
            trainer.logger.experiment.add_text(tag="config", text_string=cfg_text)

        # Resume from this run's own last checkpoint if one already exists
        # (e.g. the marimo session was interrupted mid-training) instead of
        # silently restarting from epoch 0 and losing that progress.
        resume_ckpt_path = os.path.join(config["checkpoint_path"], "last.ckpt")
        resume_ckpt_path = resume_ckpt_path if os.path.exists(resume_ckpt_path) else None
        if resume_ckpt_path:
            print(f"\n[Resume] Found existing checkpoint at {resume_ckpt_path}; resuming training from there.")

        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=resume_ckpt_path,
        )

        best_iou = trainer.callback_metrics.get("valid/iou")
        best_iou_val = float(best_iou) if best_iou is not None else 0.0

        if config["profile"] and trial is None:
            print("\n[Profile] Profiling best model checkpoint...")
            try:
                best_ckpt = checkpoint_callback.best_model_path
                if best_ckpt:
                    profiled_model = ArcheoModel.load_from_checkpoint(
                        best_ckpt,
                        encoder_name=config["encoder"],
                        in_channels=config["in_channels"],
                        out_classes=1,
                        config=config,
                    )
                else:
                    print("[Profile] No best checkpoint found, using current model...")
                    profiled_model = model

                profile_dir = os.path.join(config["checkpoint_path"], "profiler_results")
                profile_loader = DataLoader(
                    val_dataset, batch_size=config["batch_size"],
                    shuffle=False, drop_last=False, num_workers=4, pin_memory=True
                )

                profile_model(
                    profiled_model, profile_loader,
                    num_batches=config["profile_wait"] + config["profile_warmup"] + config["profile_active"] + 2,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    profile_dir=profile_dir,
                    wait=config["profile_wait"],
                    warmup=config["profile_warmup"],
                    active=config["profile_active"],
                )
            except Exception as e:
                print(f"[Profile] Warning: Could not profile model: {e}")
                import traceback
                traceback.print_exc()

        return best_iou_val, checkpoint_callback.best_model_path

    return (run_training,)


@app.cell
def _(mo):
    mo.md(r"""## Configuration""")
    return


@app.cell
def _(mo):
    ui_dataset_path = mo.ui.text(value="./bing_1k", label="Dataset root (bing_1k layout)", full_width=True)
    ui_checkpoint_dir = mo.ui.text(value="./checkpoints_segformer", label="Checkpoint output directory", full_width=True)
    ui_encoder = mo.ui.dropdown(options=["b0", "b1", "b2", "b3", "b4", "b5"], value="b0", label="SegFormer (MiT) variant")
    ui_mode = mo.ui.dropdown(options=["tune_and_profile", "standard"], value="tune_and_profile", label="Mode")
    ui_epochs = mo.ui.number(start=1, stop=1000, step=1, value=100, label="Epochs")
    ui_batch_size = mo.ui.number(start=1, stop=256, step=1, value=32, label="Batch size")
    ui_tune_trials = mo.ui.number(start=1, stop=200, step=1, value=10, label="Optuna trials")
    ui_learning_rate = mo.ui.number(start=0.000001, stop=0.1, step=0.000001, value=0.0001, label="Learning rate (standard mode)")
    ui_profile = mo.ui.checkbox(value=True, label="Profile best model with PyTorch profiler")
    ui_profile_policy = mo.ui.checkbox(value=True, label="Profile data pipeline to pick a PADS policy")

    ui_gdrive_url = mo.ui.text(value="", label="Google Drive folder URL (bing_1k directory structure)", full_width=True)
    ui_download_button = mo.ui.run_button(label="Download dataset from Google Drive")

    ui_data_source = mo.ui.dropdown(
        options=["local", "google_drive"],
        value="local",
        label="Data source",
    )
    ui_drive_root_url = mo.ui.text(
        value="", label="Google Drive bing_1k folder URL (streaming mode)", full_width=True
    )
    ui_drive_api_key = mo.ui.text(
        value="", label="Google Drive API key (streaming mode)", kind="password", full_width=True
    )

    ui_start = mo.ui.run_button(label="Start PADS pipeline")

    mo.vstack([
        ui_dataset_path,
        ui_checkpoint_dir,
        ui_encoder,
        ui_mode,
        ui_epochs,
        ui_batch_size,
        ui_tune_trials,
        ui_learning_rate,
        ui_profile,
        ui_profile_policy,
        mo.md("---"),
        mo.md(
            "**Dataset access** — either download once via the button below and use `local`, "
            "or pick `google_drive` to stream images on demand without downloading everything first."
        ),
        ui_gdrive_url,
        ui_download_button,
        ui_data_source,
        ui_drive_root_url,
        ui_drive_api_key,
        mo.md("---"),
        ui_start,
    ])
    return (
        ui_batch_size,
        ui_checkpoint_dir,
        ui_data_source,
        ui_dataset_path,
        ui_download_button,
        ui_drive_api_key,
        ui_drive_root_url,
        ui_encoder,
        ui_epochs,
        ui_gdrive_url,
        ui_learning_rate,
        ui_mode,
        ui_profile,
        ui_profile_policy,
        ui_start,
        ui_tune_trials,
    )


@app.cell
def _(mo, os, ui_dataset_path, ui_download_button, ui_gdrive_url):
    if ui_download_button.value:
        _dest = ui_dataset_path.value
        os.makedirs(_dest, exist_ok=True)
        try:
            import gdown
        except ImportError:
            download_status = mo.md(
                "**`gdown` is not installed.** Run `pip install gdown` in this environment, then click the button again."
            )
        else:
            _url = ui_gdrive_url.value.strip()
            if not _url:
                download_status = mo.md("Paste a Google Drive folder link above, then click the button again.")
            else:
                # Drive folder downloads are scraped page-by-page and can be
                # flaky/rate-limited on large trees (bing_1k has ~11,700+
                # files across 4 subfolders) -- retry a few times, and
                # remaining_ok=True stops gdown hard-erroring on the
                # "folder has more than 50 files" warning.
                _last_error = None
                for _attempt in range(1, 4):
                    try:
                        gdown.download_folder(
                            url=_url, output=_dest, quiet=False,
                            use_cookies=False, remaining_ok=True,
                        )
                        _last_error = None
                        break
                    except Exception as _exc:
                        _last_error = _exc
                        print(f"[Download] Attempt {_attempt}/3 failed: {_exc}")

                if _last_error is not None:
                    download_status = mo.md(
                        f"Download failed after 3 attempts: `{_last_error}`. Try clicking the button again."
                    )
                else:
                    # Drive folder scraping can silently drop files past a
                    # few hundred per folder -- report counts so you can
                    # eyeball completeness instead of training on partial data.
                    def _count(*parts):
                        _p = os.path.join(_dest, *parts)
                        return len(os.listdir(_p)) if os.path.isdir(_p) else 0

                    _counts = {
                        "train/originals/sites": _count("train", "originals", "sites"),
                        "train/originals/masks": _count("train", "originals", "masks"),
                        "train/negs/sites": _count("train", "negs", "sites"),
                        "train/negs/masks": _count("train", "negs", "masks"),
                    }
                    _summary = "\n".join(f"- `{k}`: {v} files" for k, v in _counts.items())
                    download_status = mo.md(
                        f"Downloaded into `{_dest}`.\n\n**File counts (verify these look right):**\n\n{_summary}"
                    )
    else:
        download_status = mo.md(
            "Dataset not downloaded yet this session — click **Download dataset from Google Drive** above, "
            "or skip this if the data already exists at the path below."
        )
    download_status
    return


@app.cell
def _(
    datetime,
    os,
    ui_batch_size,
    ui_checkpoint_dir,
    ui_data_source,
    ui_dataset_path,
    ui_drive_api_key,
    ui_drive_root_url,
    ui_encoder,
    ui_epochs,
    ui_learning_rate,
    ui_mode,
    ui_profile,
    ui_profile_policy,
    ui_tune_trials,
):
    base_config = {
        "timestamp": datetime.now().strftime("%d-%m-%Y_%H%M%S"),
        "dataset_path": ui_dataset_path.value,
        "checkpoint_path": ui_checkpoint_dir.value,
        "random_seed": 1234,
        "encoder": ui_encoder.value,
        "loss": "focal",
        "learning_rate": ui_learning_rate.value,
        "precision": 32,
        "epochs": int(ui_epochs.value),
        "batch_size": int(ui_batch_size.value),
        "in_channels": 3,
        "patience": 15,
        "mode": ui_mode.value,
        "tune_trials": int(ui_tune_trials.value),
        "profile": ui_profile.value,
        "profile_policy": ui_profile_policy.value,
        "profile_batches": 8,
        "profile_wait": 1,
        "profile_warmup": 1,
        "profile_active": 5,
        "data_source": ui_data_source.value,
        "drive_root_url": ui_drive_root_url.value,
        "drive_api_key": ui_drive_api_key.value,
    }
    os.makedirs(base_config["checkpoint_path"], exist_ok=True)
    base_config
    return (base_config,)


@app.cell
def _(
    base_config,
    extract_drive_id,
    load_dataset,
    load_dataset_drive,
    mo,
    np,
    os,
    random,
    torch,
    ui_start,
):
    mo.stop(
        not ui_start.value,
        mo.md("Click **Start PADS pipeline** above to load the dataset and begin training."),
    )

    random.seed(base_config["random_seed"])
    np.random.seed(base_config["random_seed"])
    torch.manual_seed(base_config["random_seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base_config["random_seed"])

    if base_config["data_source"] == "google_drive":
        _root_id = extract_drive_id(base_config["drive_root_url"])
        drive_folders, train_data, val_data, test_data = load_dataset_drive(
            _root_id, base_config["drive_api_key"], base_config["random_seed"],
        )
        originals_images_dir = None
        originals_masks_dir = None
        negs_images_dir = None
        negs_masks_dir = None
    else:
        _originals_count = len(sorted(os.listdir(
            os.path.join(base_config["dataset_path"], "train/originals/sites"))))
        _negs_count = len(sorted(os.listdir(
            os.path.join(base_config["dataset_path"], "train/negs/sites"))))
        _total_count = _originals_count + _negs_count
        _indices = np.arange(0, _total_count)

        (originals_images_dir, originals_masks_dir, negs_images_dir, negs_masks_dir,
         train_data, val_data, test_data) = load_dataset(
            base_config["dataset_path"], base_config["random_seed"], _indices,
        )
        drive_folders = None

    return (
        drive_folders,
        negs_images_dir,
        negs_masks_dir,
        originals_images_dir,
        originals_masks_dir,
        test_data,
        train_data,
        val_data,
    )


@app.cell
def _(
    DriveArcheoDataset,
    base_config,
    create_dataset,
    drive_folders,
    get_transforms,
    negs_images_dir,
    negs_masks_dir,
    originals_images_dir,
    originals_masks_dir,
    os,
    test_data,
    train_data,
    val_data,
):
    train_transform, val_transform = get_transforms()

    if base_config["data_source"] == "google_drive":
        _cache_dir = os.path.join(base_config["dataset_path"], "_drive_cache")
        train_dataset = DriveArcheoDataset(
            train_data, drive_folders, base_config["drive_api_key"], _cache_dir, transform=train_transform,
        )
        val_dataset = DriveArcheoDataset(
            val_data, drive_folders, base_config["drive_api_key"], _cache_dir, transform=val_transform,
        )
        test_dataset = DriveArcheoDataset(
            test_data, drive_folders, base_config["drive_api_key"], _cache_dir, transform=val_transform,
        )
    else:
        _scratch_dir = os.path.join(base_config["checkpoint_path"], "scratch")
        train_dataset = create_dataset(
            train_data, originals_images_dir, originals_masks_dir,
            negs_images_dir, negs_masks_dir, transform=train_transform,
            policy=base_config.get("data_policy", "loose"), scratch_dir=_scratch_dir,
        )
        val_dataset = create_dataset(
            val_data, originals_images_dir, originals_masks_dir,
            negs_images_dir, negs_masks_dir, transform=val_transform,
            policy=base_config.get("data_policy", "loose"), scratch_dir=_scratch_dir,
        )
        test_dataset = create_dataset(
            test_data, originals_images_dir, originals_masks_dir,
            negs_images_dir, negs_masks_dir, transform=val_transform,
            policy=base_config.get("data_policy", "loose"), scratch_dir=_scratch_dir,
        )
    return test_dataset, train_dataset, val_dataset


@app.cell
def _(ArcheoModel, base_config, mo, profile_data_pipeline, torch, train_dataset):
    if base_config["data_source"] == "google_drive":
        data_policy = "drive"
    elif base_config["profile_policy"]:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _probe_model = ArcheoModel(
            encoder_name=base_config["encoder"],
            in_channels=base_config["in_channels"],
            out_classes=1,
            config=base_config,
        )
        _policy_summary = profile_data_pipeline(
            _probe_model,
            train_dataset,
            batch_size=base_config["batch_size"],
            device=_device,
            num_batches=base_config["profile_batches"],
            scratch_available=True,
            profile_dir=base_config["checkpoint_path"],
        )
        data_policy = _policy_summary["policy"]
    else:
        data_policy = "loose"
    mo.md(f"**Selected PADS data-loading policy:** `{data_policy}`")
    return (data_policy,)


@app.cell
def _(base_config, build_dataloader, data_policy, train_dataset, val_dataset):
    train_loader = build_dataloader(
        train_dataset,
        batch_size=base_config["batch_size"],
        shuffle=True,
        drop_last=True,
        policy=data_policy,
        num_workers=4,
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=base_config["batch_size"],
        shuffle=False,
        drop_last=False,
        policy=data_policy,
        num_workers=4,
    )
    return train_loader, val_loader


@app.cell
def _(base_config, optuna, os, run_training, train_loader, val_dataset, val_loader):
    if base_config["mode"] == "standard":
        _best_iou_val, _best_ckpt_path = run_training(
            base_config, train_loader, val_loader, val_dataset, trial=None
        )
        results = {
            "mode": "standard",
            "best_iou": _best_iou_val,
            "best_checkpoint": _best_ckpt_path,
        }
    else:
        def _objective(trial):
            config = base_config.copy()
            config["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
            config["checkpoint_path"] = os.path.join(base_config["checkpoint_path"], f"trial_{trial.number}")
            os.makedirs(config["checkpoint_path"], exist_ok=True)
            iou, _ = run_training(config, train_loader, val_loader, val_dataset, trial=trial)
            return iou

        # Persist the study to disk so an interrupted/restarted marimo session
        # resumes instead of losing every previously-completed trial and
        # restarting trial numbering (and therefore checkpoint directories)
        # from zero.
        _optuna_storage = f"sqlite:///{os.path.join(base_config['checkpoint_path'], 'optuna_study.db')}"
        _study = optuna.create_study(
            direction="maximize",
            study_name="segformer_tuning",
            storage=_optuna_storage,
            load_if_exists=True,
        )
        _completed_trials = sum(
            1 for t in _study.trials if t.state == optuna.trial.TrialState.COMPLETE
        )
        _remaining_trials = max(0, base_config["tune_trials"] - _completed_trials)
        if _remaining_trials > 0:
            print(f"\n[Optuna] {_completed_trials} trial(s) already completed; running {_remaining_trials} more "
                  f"toward the target of {base_config['tune_trials']}.")
            _study.optimize(_objective, n_trials=_remaining_trials)
        else:
            print(f"\n[Optuna] Target of {base_config['tune_trials']} completed trials already reached; skipping optimize().")

        results = {
            "mode": "tune_and_profile",
            "best_trial": _study.best_trial.number,
            "best_iou": _study.best_value,
            "best_params": _study.best_params,
        }

        if base_config["profile"]:
            _best_config = base_config.copy()
            _best_config.update(_study.best_params)
            _best_config["checkpoint_path"] = os.path.join(base_config["checkpoint_path"], "best_trial_profiling")
            os.makedirs(_best_config["checkpoint_path"], exist_ok=True)
            run_training(_best_config, train_loader, val_loader, val_dataset, trial=None)

    return (results,)


@app.cell
def _(mo, results):
    mo.md(
        f"""
        ## Results

        ```
        {results}
        ```
        """
    )
    return


if __name__ == "__main__":
    app.run()
