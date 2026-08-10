"""Bare-bones, faithful port of the dataset/transform cells from Definitivo.ipynb
(the Casini et al. reference notebook), restricted to the Bing 1k / no-CORONA
branch (config["dim_input"]=='1k', config["corona_path"]=="").

This is a literal port, not a redesign: kept identical wherever the notebook's
own code is directly reusable, and adapted only where the on-disk layout of
this dataset differs from what the notebook expects.

ONE REAL ADAPTATION (documented, not silent): the notebook's load_dataset()
lists a single combined `train/sites` / `train/masks` directory pair. On this
cluster the same corpus is split into `train/originals/{sites,masks}` and
`train/negs/{sites,masks}` (PADS's own layout choice, made after this
notebook was written). The negs/sites filenames are literally `neg<N>.jpg` --
exactly the prefix the notebook's own empty-mask check
(`i.startswith("neg")`) already expects -- so this is the same corpus under a
different directory split, not a different corpus. load_dataset() below
reconstructs the notebook's single combined, sorted filename list from the
two split directories and keeps a filename -> (images_dir, masks_dir) lookup
for __getitem__.
"""
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations import pytorch


def load_dataset(path, seed, indices):
    """Faithful port of Definitivo.ipynb cell 5, adapted only for the
    split originals/negs directory layout (see module docstring).
    """
    originals_images_dir = os.path.join(path, "train", "originals", "sites")
    originals_masks_dir = os.path.join(path, "train", "originals", "masks")
    negs_images_dir = os.path.join(path, "train", "negs", "sites")
    negs_masks_dir = os.path.join(path, "train", "negs", "masks")

    originals_files = sorted(os.listdir(originals_images_dir))
    negs_files = sorted(os.listdir(negs_images_dir))
    # Combined + re-sorted, matching what sorted(os.listdir(<single dir>))
    # would have produced on the notebook's original undivided layout.
    filenames_train = np.asarray(sorted(originals_files + negs_files))
    print("total files:", len(filenames_train))

    path_lookup = {}
    for fn in originals_files:
        path_lookup[fn] = (originals_images_dir, originals_masks_dir)
    for fn in negs_files:
        path_lookup[fn] = (negs_images_dir, negs_masks_dir)

    valid_split = -int(len(indices) * 0.2)
    test_split = valid_split // 2

    train_indices = indices[:valid_split]
    valid_indices = indices[valid_split:test_split]
    test_indices = indices[test_split:]
    train_images_filenames = filenames_train[train_indices]
    val_images_filenames = filenames_train[valid_indices]
    test_images_filenames = filenames_train[test_indices]

    print("root:", path,
          "\n---",
          "\ntrain images", len(train_images_filenames),
          "\nval images", len(val_images_filenames),
          "\ntest images", len(test_images_filenames),
          "\n---\ntotal images", len(filenames_train))

    print("empty masks percentage: %.4f %.4f %.4f" %
          (np.sum([i.startswith("neg") for i in train_images_filenames]) / len(train_images_filenames),
           np.sum([i.startswith("neg") for i in val_images_filenames]) / len(val_images_filenames),
           np.sum([i.startswith("neg") for i in test_images_filenames]) / len(test_images_filenames)))

    return path_lookup, train_images_filenames, val_images_filenames, test_images_filenames


def esegui_trasformazioni():
    """Faithful port of Definitivo.ipynb cell 6, Bing 1k / no-CORONA branch
    only (the other three branches -- 1k+CORONA, 2k, 2k+CORONA -- are out of
    scope: this thesis's runs are Bing 1k without CORONA)."""
    train_transform = A.Compose([
        A.RandomCrop(512, 512, p=1.0),
        A.Flip(p=0.25), A.RandomRotate90(p=0.25),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.25),
        A.Resize(256, 256),
        A.pytorch.ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.RandomCrop(512, 512, p=1.0),
        A.Resize(256, 256),
        A.pytorch.ToTensorV2(),
    ])
    return train_transform, val_transform


class ArcheoDataset(Dataset):
    """Faithful port of Definitivo.ipynb cell 8, no-CORONA branch only."""

    def __init__(self, images_filenames, path_lookup, transform=None):
        self.images_filenames = images_filenames
        self.path_lookup = path_lookup
        self.transform = transform

    def __len__(self):
        return len(self.images_filenames)

    def __getitem__(self, idx):
        image_filename = self.images_filenames[idx]
        images_directory, masks_directory = self.path_lookup[image_filename]
        image_path = os.path.join(images_directory, image_filename)
        mask_path = os.path.join(masks_directory, image_filename.replace(".jpg", ".png"))

        image = Image.open(image_path)

        mask = ~np.array(Image.open(mask_path).convert("L"))  # masks are flipped (qgis)
        mask = mask.astype("float")
        mask[mask > 0.0] = 1.0
        mask = np.expand_dims(mask, -1)

        transformed = self.transform(image=np.asarray(image), mask=np.asarray(mask))
        image = transformed["image"]
        mask = transformed["mask"].permute(2, 0, 1)

        return image, mask, image_filename
