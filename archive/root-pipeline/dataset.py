"""
dataset.py
----------
Dataset and LightningDataModule for Tell site segmentation.
Handles originals (positive) and negs (negative) samples with
stratified 81/9/10 train/val/test split matching Cassini et al. 2023.
Images are tiled from 1024x1024 to 512x512 patches during loading.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold


# ---------------------------------------------------------------------------
# Albumentations pipelines
# ---------------------------------------------------------------------------

def get_train_transforms():
    return A.Compose([
        A.RandomCrop(width=512, height=512),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_val_transforms():
    return A.Compose([
        A.CenterCrop(width=512, height=512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TellSiteDataset(Dataset):
    """
    Loads pairs of (satellite image, binary mask) for Tell site segmentation.

    Mask convention (from Cassini dataset):
        0   -> Tell site (foreground)
        255 -> Background

    We remap to:
        1   -> Tell site (foreground)
        0   -> Background
    """

    def __init__(self, samples: list, transform=None):
        """
        Args:
            samples: list of (image_path, mask_path) tuples
            transform: albumentations transform pipeline
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = np.array(Image.open(img_path).convert("RGB"))

        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path).convert("L"))
        else:
            # Negative sample with no mask file — treat as all background
            mask = np.full((image.shape[0], image.shape[1]), 255, dtype=np.uint8)

        # Remap: 0 (Tell site) -> 1, 255 (background) -> 0
        mask = (mask == 0).astype(np.uint8)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]          # CxHxW float tensor
            mask = augmented["mask"].long()     # HxW long tensor

        return image, mask


# ---------------------------------------------------------------------------
# Sample collection helpers
# ---------------------------------------------------------------------------

def collect_samples(root_dir: str) -> tuple:
    """
    Walk root_dir and collect (image_path, mask_path) pairs for both
    originals (positive) and negs (negative) subsets.

    Expected structure:
        root_dir/
            originals/
                sites/   <- satellite images
                masks/   <- binary masks
            negs/
                sites/
                masks/

    Returns:
        positives: list of (img_path, mask_path)
        negatives: list of (img_path, mask_path)
    """
    positives = []
    negatives = []

    for subset, target_list in [("originals", positives), ("negs", negatives)]:
        sites_dir = os.path.join(root_dir, subset, "sites")
        masks_dir = os.path.join(root_dir, subset, "masks")

        if not os.path.isdir(sites_dir):
            print(f"Warning: {sites_dir} not found, skipping.")
            continue

        for fname in sorted(os.listdir(sites_dir)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            img_path = os.path.join(sites_dir, fname)
            
            stem = os.path.splitext(fname)[0]
            mask_path = None
            for ext in [os.path.splitext(fname)[1], ".png", ".jpg", ".jpeg"]:
                candidate = os.path.join(masks_dir, stem + ext)
                if os.path.exists(candidate):
                    mask_path = candidate
                    break
            if mask_path is None:
                mask_path = os.path.join(masks_dir, fname)  # fallback, will be caught as missing

            target_list.append((img_path, mask_path))
    return positives, negatives


def stratified_split(positives: list, negatives: list, seed: int = 42):
    """
    Stratified split matching Cassini et al. 2023:
        - 90% train, 10% holdout test  (stratified on pos/neg)
        - 10% of training -> validation (stratified on pos/neg)
    Effective split: ~81% train / ~9% val / ~10% test
    """
    def split_stratum(samples, test_size, seed):
        train, test = train_test_split(samples, test_size=test_size,
                                       random_state=seed, shuffle=True)
        return train, test

    # First split: 90/10
    pos_train, pos_test = split_stratum(positives, test_size=0.10, seed=seed)
    neg_train, neg_test = split_stratum(negatives, test_size=0.10, seed=seed)

    # Second split: 10% of training becomes validation
    pos_train, pos_val = split_stratum(pos_train, test_size=0.10, seed=seed)
    neg_train, neg_val = split_stratum(neg_train, test_size=0.10, seed=seed)

    train = pos_train + neg_train
    val   = pos_val   + neg_val
    test  = pos_test  + neg_test

    random.seed(seed)
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


def kfold_split(positives: list, negatives: list, n_splits: int = 5, seed: int = 42):
    """
    K-Fold cross-validation split with stratification on positive/negative balance.
    
    Returns:
        List of tuples (train_samples, val_samples) for each fold.
        Validation set for each fold is 1/n_splits of total data.
    """
    # Combine with labels for stratification
    all_samples = positives + negatives
    labels = [1] * len(positives) + [0] * len(negatives)  # 1=positive, 0=negative
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    
    for train_idx, val_idx in skf.split(all_samples, labels):
        train_samples = [all_samples[i] for i in train_idx]
        val_samples = [all_samples[i] for i in val_idx]
        folds.append((train_samples, val_samples))
    
    return folds


# ---------------------------------------------------------------------------
# LightningDataModule
# ---------------------------------------------------------------------------

class TellSiteDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        batch_size: int = 8,
        num_workers: int = 4,
        seed: int = 42,
    ):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

    def setup(self, stage=None):
        positives, negatives = collect_samples(self.data_root)

        print(f"Collected {len(positives)} positive samples (originals)")
        print(f"Collected {len(negatives)} negative samples (negs)")

        train_samples, val_samples, test_samples = stratified_split(
            positives, negatives, seed=self.seed
        )

        print(f"Split -> Train: {len(train_samples)} | "
              f"Val: {len(val_samples)} | Test: {len(test_samples)}")

        self.train_dataset = TellSiteDataset(train_samples, transform=get_train_transforms())
        self.val_dataset   = TellSiteDataset(val_samples,   transform=get_val_transforms())
        self.test_dataset  = TellSiteDataset(test_samples,  transform=get_val_transforms())

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class TellSiteKFoldDataModule(pl.LightningDataModule):
    """K-Fold cross-validation datamodule for Tell site segmentation."""
    
    def __init__(
        self,
        data_root: str,
        batch_size: int = 8,
        num_workers: int = 4,
        n_splits: int = 5,
        seed: int = 42,
    ):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_splits = n_splits
        self.seed = seed
        self.folds = []
        self.current_fold = 0

    def setup(self, stage=None):
        positives, negatives = collect_samples(self.data_root)
        
        print(f"Collected {len(positives)} positive samples (originals)")
        print(f"Collected {len(negatives)} negative samples (negs)")
        
        self.folds = kfold_split(positives, negatives, n_splits=self.n_splits, seed=self.seed)
        
        print(f"Created {len(self.folds)} folds for cross-validation")
        self._setup_fold(self.current_fold)

    def _setup_fold(self, fold_idx: int):
        """Set up datasets for a specific fold."""
        if fold_idx >= len(self.folds):
            raise ValueError(f"Fold {fold_idx} out of range (max {len(self.folds)-1})")
        
        train_samples, val_samples = self.folds[fold_idx]
        
        print(f"\n--- Fold {fold_idx + 1}/{len(self.folds)} ---")
        print(f"Train: {len(train_samples)} | Val: {len(val_samples)}")
        
        self.train_dataset = TellSiteDataset(train_samples, transform=get_train_transforms())
        self.val_dataset = TellSiteDataset(val_samples, transform=get_val_transforms())

    def set_fold(self, fold_idx: int):
        """Switch to a different fold."""
        self.current_fold = fold_idx
        self._setup_fold(fold_idx)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        """For k-fold, validation set is used for testing."""
        return self.val_dataloader()


# ---------------------------------------------------------------------------
# Visualisation helper
# ---------------------------------------------------------------------------

def visualize_sample(datamodule: TellSiteDataModule, split: str = "train"):
    """
    Display one random (image, mask) pair after albumentations transforms.
    Call after datamodule.setup().
    """
    if split == "train":
        dataset = datamodule.train_dataset
    elif split == "val":
        dataset = datamodule.val_dataset
    else:
        dataset = datamodule.test_dataset

    idx = random.randint(0, len(dataset) - 1)
    image, mask = dataset[idx]

    # Denormalise image for display
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img_np = image.permute(1, 2, 0).numpy()
    img_np = (img_np * std + mean).clip(0, 1)

    mask_np = mask.numpy()

#     fig, axes = plt.subplots(1, 2, figsize=(12, 6))
#     axes[0].imshow(img_np)
#     axes[0].set_title(f"Satellite Image (sample {idx})")
#     axes[0].axis("off")
# 
#     axes[1].imshow(mask_np, cmap="gray", vmin=0, vmax=1)
#     axes[1].set_title(f"Mask (white=Tell site)")
#     axes[1].axis("off")
# 
#     plt.suptitle(f"Post-albumentations sample from {split} split", fontsize=14)
#     plt.tight_layout()
#     plt.savefig("sample_visualization.png", dpi=150)
#     plt.show()
#     print("Saved to sample_visualization.png")

# ---------------------------------------------------------------------------
# Class distribution helper
# ---------------------------------------------------------------------------

def compute_class_distribution(datamodule: TellSiteDataModule):
    """
    Compute pixel-level foreground/background distribution across
    train, val, and test splits. Prints summary and returns dict.
    Call after datamodule.setup().
    """
    results = {}

    for split_name, dataset in [
        ("train", datamodule.train_dataset),
        ("val",   datamodule.val_dataset),
        ("test",  datamodule.test_dataset),
    ]:
        total_fg = 0
        total_px = 0
        per_image_ratios = []

        # Use raw masks without transforms for accurate pixel counting
        raw_dataset = TellSiteDataset(dataset.samples, transform=None)

        for img_path, mask_path in raw_dataset.samples:
            if os.path.exists(mask_path):
                mask = np.array(Image.open(mask_path).convert("L"))
            else:
                mask = np.full((1024, 1024), 255, dtype=np.uint8)

            # Remap same as dataset: 0 -> foreground
            binary = (mask == 0).astype(np.uint8)
            fg = binary.sum()
            px = binary.size
            total_fg += fg
            total_px += px
            per_image_ratios.append(fg / px)

        global_fg_ratio = total_fg / total_px
        global_bg_ratio = 1.0 - global_fg_ratio

        results[split_name] = {
            "global_fg_ratio": global_fg_ratio,
            "global_bg_ratio": global_bg_ratio,
            "per_image_mean":  float(np.mean(per_image_ratios)),
            "per_image_min":   float(np.min(per_image_ratios)),
            "per_image_max":   float(np.max(per_image_ratios)),
            "n_samples":       len(dataset),
        }

        print(f"\n[{split_name.upper()}] {len(dataset)} samples")
        print(f"  Global foreground ratio : {global_fg_ratio:.4f} ({global_fg_ratio*100:.2f}%)")
        print(f"  Global background ratio : {global_bg_ratio:.4f} ({global_bg_ratio*100:.2f}%)")
        print(f"  Per-image fg ratio      : "
              f"mean={np.mean(per_image_ratios):.4f} | "
              f"min={np.min(per_image_ratios):.4f} | "
              f"max={np.max(per_image_ratios):.4f}")

    return results
