import os
import time
import tarfile
import shutil
import tempfile
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations import pytorch
from PIL import Image
from io import BytesIO


def load_dataset(PATH, SEED, indices):
    """
    Load and split dataset into train (80%), validation (10%), and test (10%)
    
    This function combines both 'originals' and 'negs' datasets from:
    - PATH/train/originals/sites/ and PATH/train/originals/masks/
    - PATH/train/negs/sites/ and PATH/train/negs/masks/
    """
    # Shuffle indices with seed for reproducible random split
    rng = np.random.RandomState(SEED)
    rng.shuffle(indices)

    root_directory = os.path.join(PATH)
    
    # Load originals
    originals_images_dir = os.path.join(root_directory, "train/originals/sites")
    originals_masks_dir = os.path.join(root_directory, "train/originals/masks")
    originals_images = sorted(os.listdir(originals_images_dir)) if os.path.exists(originals_images_dir) else []
    
    # Load negs
    negs_images_dir = os.path.join(root_directory, "train/negs/sites")
    negs_masks_dir = os.path.join(root_directory, "train/negs/masks")
    negs_images = sorted(os.listdir(negs_images_dir)) if os.path.exists(negs_images_dir) else []
    
    # Combine datasets: mark originals with (source='originals') and negs with (source='negs')
    combined_data = []
    for fname in originals_images:
        combined_data.append({"filename": fname, "source": "originals"})
    for fname in negs_images:
        combined_data.append({"filename": fname, "source": "negs"})
    
    print(f"Loaded {len(originals_images)} originals images")
    print(f"Loaded {len(negs_images)} negs images")
    print(f"Total combined images: {len(combined_data)}")

    # Create balanced indices for train/val/test split
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
    """Get augmentation transforms for bing_1k (512x512 images resized to 256x256)"""
    train_transform = A.Compose([
        A.RandomCrop(512, 512, p=1.0),
        A.HorizontalFlip(p=0.25),
        A.VerticalFlip(p=0.25),
        A.RandomRotate90(p=0.25),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.25),
        A.Resize(256, 256),
        A.pytorch.ToTensorV2(),
    ])
    
    val_transform = A.Compose([
        A.RandomCrop(512, 512, p=1.0),
        A.Resize(256, 256),
        A.pytorch.ToTensorV2()
    ])
    
    return train_transform, val_transform


class ArcheoDataset(Dataset):
    """Dataset class for archaeological image segmentation"""
    def __init__(self, data_items, originals_images_dir, originals_masks_dir, 
                 negs_images_dir, negs_masks_dir, transform=None):
        """
        Args:
            data_items: List of dicts with 'filename' and 'source' keys
            originals_images_dir: Path to originals/sites
            originals_masks_dir: Path to originals/masks
            negs_images_dir: Path to negs/sites
            negs_masks_dir: Path to negs/masks
            transform: albumentations transform pipeline
        """
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
        
        # Select correct directory based on source
        if source == "originals":
            image_dir = self.originals_images_dir
            mask_dir = self.originals_masks_dir
        else:  # negs
            image_dir = self.negs_images_dir
            mask_dir = self.negs_masks_dir
        
        start_time = time.perf_counter()
        image_path = os.path.join(image_dir, image_filename)
        mask_filename = image_filename.replace(".jpg", ".png")
        mask_path = os.path.join(mask_dir, mask_filename)
        
        image = Image.open(image_path).convert("RGB")
        
        # Load mask if it exists, otherwise create blank mask (for negs)
        if os.path.exists(mask_path):
            mask = ~np.array(Image.open(mask_path).convert("L"))  # masks are flipped
        else:
            mask = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)
        
        mask = mask.astype(np.float32)
        mask[mask > 0.0] = 1.0
        mask = np.expand_dims(mask, -1)

        if self.transform:
            transformed = self.transform(image=np.asarray(image), mask=np.asarray(mask))
            image = transformed["image"]
            mask = transformed["mask"].permute(2, 0, 1)

        if self.profile_enabled:
            self.profile_stats.append(time.perf_counter() - start_time)
        
        return image, mask, image_filename


class TarArcheoDataset(Dataset):
    """
    Dataset that reads from tar-shard archives instead of loose files.
    Reduces many-small-file overhead.
    """
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
        """Get or open tar archive."""
        if tar_path not in self._tar_cache:
            self._tar_cache[tar_path] = tarfile.open(tar_path, 'r:gz')
        return self._tar_cache[tar_path]

    def __getitem__(self, idx):
        data_item = self.data_items[idx]
        image_filename = data_item["filename"]
        source = data_item["source"]

        start_time = time.perf_counter()

        if source == "originals":
            tar_path = self.originals_tar_path
            image_member = f"images/{image_filename}"
            mask_member = f"masks/{image_filename.replace('.jpg', '.png')}"
        else:
            tar_path = self.negs_tar_path
            image_member = f"images/{image_filename}"
            mask_member = f"masks/{image_filename.replace('.jpg', '.png')}"

        try:
            tar = self._get_tar(tar_path)
            
            # Read image
            image_f = tar.extractfile(image_member)
            image = Image.open(BytesIO(image_f.read())).convert("RGB")
            
            # Read mask
            try:
                mask_f = tar.extractfile(mask_member)
                mask = ~np.array(Image.open(BytesIO(mask_f.read())).convert("L"))
            except KeyError:
                mask = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)
        except (KeyError, tarfile.TarError):
            # Fallback to placeholder if member not found
            image = Image.new("RGB", (256, 256), color=(0, 0, 0))
            mask = np.zeros((256, 256), dtype=np.uint8)

        mask = mask.astype(np.float32)
        mask[mask > 0.0] = 1.0
        mask = np.expand_dims(mask, -1)

        if self.transform:
            transformed = self.transform(image=np.asarray(image), mask=np.asarray(mask))
            image = transformed["image"]
            mask = transformed["mask"].permute(2, 0, 1)

        if self.profile_enabled:
            self.profile_stats.append(time.perf_counter() - start_time)

        return image, mask, image_filename

    def __del__(self):
        """Close tar files on cleanup."""
        for tar in self._tar_cache.values():
            try:
                tar.close()
            except:
                pass


class StagingArcheoDataset(Dataset):
    """
    Dataset that stages data to node-local scratch before loading.
    Reduces shared filesystem latency.
    """
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
        
        # Use /tmp or provided scratch directory
        self.scratch_dir = scratch_dir or tempfile.gettempdir()
        self._staged = set()
        self._stage_batch(0, min(64, len(data_items)))

    def _stage_batch(self, start_idx, end_idx):
        """Pre-copy a batch of files to scratch."""
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
        # Stage next batch ahead of time
        if idx % 16 == 0:
            self._stage_batch(idx + 64, idx + 128)

        data_item = self.data_items[idx]
        image_filename = data_item["filename"]

        start_time = time.perf_counter()

        # Try to load from scratch first
        scratch_img = os.path.join(self.scratch_dir, image_filename)
        scratch_mask = os.path.join(self.scratch_dir, image_filename.replace(".jpg", ".png"))

        if os.path.exists(scratch_img):
            image = Image.open(scratch_img).convert("RGB")
            if os.path.exists(scratch_mask):
                mask = ~np.array(Image.open(scratch_mask).convert("L"))
            else:
                mask = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)
        else:
            # Fallback to original location
            source = data_item["source"]
            if source == "originals":
                image_dir = self.originals_images_dir
                mask_dir = self.originals_masks_dir
            else:
                image_dir = self.negs_images_dir
                mask_dir = self.negs_masks_dir

            image_path = os.path.join(image_dir, image_filename)
            mask_path = os.path.join(mask_dir, image_filename.replace(".jpg", ".png"))

            image = Image.open(image_path).convert("RGB")
            if os.path.exists(mask_path):
                mask = ~np.array(Image.open(mask_path).convert("L"))
            else:
                mask = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)

        mask = mask.astype(np.float32)
        mask[mask > 0.0] = 1.0
        mask = np.expand_dims(mask, -1)

        if self.transform:
            transformed = self.transform(image=np.asarray(image), mask=np.asarray(mask))
            image = transformed["image"]
            mask = transformed["mask"].permute(2, 0, 1)

        if self.profile_enabled:
            self.profile_stats.append(time.perf_counter() - start_time)

        return image, mask, image_filename

