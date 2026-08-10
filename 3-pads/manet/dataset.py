import os
import time
import tarfile
import shutil
import tempfile
import math
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


def get_transforms(val_crop="random"):
    """Augmentation transforms for bing_1k, matching Casini et al. (2023).

    The paper's pipeline (Methods, "Setting the input images"): images are saved
    at ~1 px/m, so L=1000 m gives 1024x1024; a random L/2 = 512 crop is taken as
    the input; then "the inputs were then scaled down to half of that to ease
    computational requirements", i.e. to 256x256. The augmentation is "a random
    rotation and mirroring, as well as a slight shift in brightness and
    contrast", which is what is configured below.

    NOTE: the Resize(256, 256) was briefly removed here on the assumption that
    it was discarding resolution. That was wrong -- 256 is the paper's actual
    input size, and training at 512 is a deviation from it, not a fix.
    """
    train_transform = A.Compose([
        A.RandomCrop(512, 512, p=1.0),
        A.HorizontalFlip(p=0.25),
        A.VerticalFlip(p=0.25),
        A.RandomRotate90(p=0.25),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.25),
        A.Resize(256, 256),
        A.pytorch.ToTensorV2(),
    ])

    # The reference RandomCrops at evaluation and averages ten runs; that is the
    # default here so reported numbers follow its protocol. --val_crop center
    # substitutes a deterministic crop, which removes crop noise from per-epoch
    # checkpoint selection and early stopping but is a deviation.
    crop = (A.CenterCrop(512, 512, p=1.0) if val_crop == "center"
            else A.RandomCrop(512, 512, p=1.0))
    val_transform = A.Compose([
        crop,
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
        """Get or open tar archive.

        'r:*' autodetects instead of forcing 'r:gz'. Uncompressed archives are
        strongly preferred here: the payload is already-compressed JPEG/PNG, so
        gzip saves almost nothing on size while forcing extractfile() to
        decompress from the stream start on every random access -- which is
        exactly the access pattern a shuffled DataLoader produces.
        """
        if tar_path not in self._tar_cache:
            self._tar_cache[tar_path] = tarfile.open(tar_path, 'r:*')
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

    full_prestage=True copies the entire dataset to scratch in __init__
    (evaluation method 4, the "full pre-stage ceiling" -- see
    Section~\\ref{sec:evaluation-plan}), rather than the rolling
    stage_depth-item lookahead window that predictive staging uses. It is a
    static, non-adaptive baseline: everything is staged up front regardless
    of whether it fits or whether adaptivity would have helped.

    stage_depth is d in Section~\\ref{ch:methodology}'s Stage 4 (predictive
    staging): the lookahead window is re-staged every stage_depth//4 items
    consumed (matching the original fixed 64-item window's 64/16=4 ratio of
    window size to re-trigger interval), copying the next stage_depth items
    to scratch each time. RQ4's fixed-depth sweep sets this directly
    (1/2/4/8); the "auto" arm derives it from measured timings via
    compute_adaptive_stage_depth() below and passes the result in here --
    train.py, not this class, owns which one happens. Was hardcoded to 64
    with no way to override it or trace it back to a measurement; that made
    "auto-tuned depth" aspirational (Section~\\ref{ch:methodology} describes
    d = min(ceil(t_stage/t_consume) + 1, d_max), but nothing computed it).

    scratch_capacity_pct, if given (0.0-1.0), caps the number of items ever
    staged at that fraction of the corpus -- a synthetic "scratch is smaller
    than the dataset" regime for Chapter 6's scratch-capacity ablation
    (Table~\\ref{tab:ablation-scratch}). Once the cap is hit, later items
    permanently fall back to the loose read path in __getitem__ rather than
    being staged; this is the regime the design is actually aimed at
    (Section~\\ref{sec:ablation-capacity} -- real scratch on this cluster
    comfortably fits the Bing 1k corpus, so the constraint has to be
    synthetic to be exercised at all). None = unconstrained (real capacity).
    """
    def __init__(self, data_items, originals_images_dir, originals_masks_dir,
                 negs_images_dir, negs_masks_dir, transform=None, scratch_dir=None,
                 full_prestage=False, scratch_capacity_pct=None, stage_depth=64):
        self.data_items = data_items
        self.originals_images_dir = originals_images_dir
        self.originals_masks_dir = originals_masks_dir
        self.negs_images_dir = negs_images_dir
        self.negs_masks_dir = negs_masks_dir
        self.transform = transform
        self.profile_enabled = False
        self.profile_stats = []
        self.full_prestage = full_prestage
        self.stage_depth = max(1, int(stage_depth))
        self._restage_interval = max(1, self.stage_depth // 4)
        self.scratch_capacity_items = (
            max(0, int(len(data_items) * scratch_capacity_pct))
            if scratch_capacity_pct is not None else None
        )
        self._staging_hits = 0
        self._staging_misses = 0

        # Use /tmp or provided scratch directory
        self.scratch_dir = scratch_dir or tempfile.gettempdir()
        self._staged = set()
        if full_prestage:
            stage_start = time.perf_counter()
            self._stage_batch(0, len(data_items))
            elapsed = time.perf_counter() - stage_start
            print(f"[StagingArcheoDataset] full_prestage: staged "
                  f"{len(self._staged)}/{len(data_items)} items to "
                  f"{self.scratch_dir} in {elapsed:.1f}s")
        else:
            self._stage_batch(0, min(self.stage_depth, len(data_items)))

    def _stage_batch(self, start_idx, end_idx):
        """Pre-copy a batch of files to scratch."""
        for idx in range(start_idx, end_idx):
            if idx >= len(self.data_items):
                break
            if (self.scratch_capacity_items is not None
                    and len(self._staged) >= self.scratch_capacity_items):
                # Synthetic scratch is full: everything from here on falls
                # back to the loose read path in __getitem__ permanently,
                # not just until space frees up (there is no eviction here --
                # this ablation is about whether PADS degrades gracefully
                # when it never fits, not about steady-state churn).
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
        # Stage next batch ahead of time. Skipped under full_prestage: everything
        # is already on scratch from __init__, so there is nothing left to stage
        # and no adaptive lookahead logic runs during the timed portion.
        if not self.full_prestage and idx % self._restage_interval == 0:
            self._stage_batch(idx + self.stage_depth, idx + 2 * self.stage_depth)

        data_item = self.data_items[idx]
        image_filename = data_item["filename"]

        start_time = time.perf_counter()

        # Try to load from scratch first
        scratch_img = os.path.join(self.scratch_dir, image_filename)
        scratch_mask = os.path.join(self.scratch_dir, image_filename.replace(".jpg", ".png"))

        if os.path.exists(scratch_img):
            self._staging_hits += 1
            image = Image.open(scratch_img).convert("RGB")
            if os.path.exists(scratch_mask):
                mask = ~np.array(Image.open(scratch_mask).convert("L"))
            else:
                mask = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)
        else:
            self._staging_misses += 1
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

    def staging_hit_rate(self):
        """Fraction of __getitem__ calls so far served from scratch rather
        than falling back to the loose path -- the "Staging hit rate" column
        of Table~\\ref{tab:ablation-scratch}. None if __getitem__ hasn't been
        called yet."""
        total = self._staging_hits + self._staging_misses
        return (self._staging_hits / total) if total else None


def measure_staging_timings(data_items, originals_images_dir, originals_masks_dir,
                             negs_images_dir, negs_masks_dir, scratch_dir, sample_n=16):
    """Measure t_stage and t_consume for the adaptive prefetch-depth formula
    (Section~\\ref{ch:methodology}, Stage 4): t_stage is the time to copy one
    item to scratch, t_consume the time to read+decode it back from scratch.
    Also returns the mean bytes copied per item, needed for d_max.

    The sampled items are left on scratch afterwards (copies are idempotent,
    same skip-if-exists check as _stage_batch), so this measurement pass
    doubles as real staging rather than being throwaway work.
    """
    sample_n = min(sample_n, len(data_items))
    stage_times, consume_times, byte_counts = [], [], []

    for idx in range(sample_n):
        data_item = data_items[idx]
        image_filename = data_item["filename"]
        source = data_item["source"]
        if source == "originals":
            src_img_dir, src_mask_dir = originals_images_dir, originals_masks_dir
        else:
            src_img_dir, src_mask_dir = negs_images_dir, negs_masks_dir

        src_img = os.path.join(src_img_dir, image_filename)
        src_mask = os.path.join(src_mask_dir, image_filename.replace(".jpg", ".png"))
        dst_img = os.path.join(scratch_dir, image_filename)
        dst_mask = os.path.join(scratch_dir, image_filename.replace(".jpg", ".png"))

        stage_start = time.perf_counter()
        copied_bytes = 0
        try:
            if not os.path.exists(dst_img) and os.path.exists(src_img):
                shutil.copy2(src_img, dst_img)
            if not os.path.exists(dst_mask) and os.path.exists(src_mask):
                shutil.copy2(src_mask, dst_mask)
            if os.path.exists(dst_img):
                copied_bytes += os.path.getsize(dst_img)
            if os.path.exists(dst_mask):
                copied_bytes += os.path.getsize(dst_mask)
        except (IOError, OSError):
            continue
        stage_times.append(time.perf_counter() - stage_start)
        byte_counts.append(copied_bytes)

        consume_start = time.perf_counter()
        try:
            _ = np.asarray(Image.open(dst_img).convert("RGB"))
            if os.path.exists(dst_mask):
                _ = np.array(Image.open(dst_mask).convert("L"))
        except (IOError, OSError):
            pass
        consume_times.append(time.perf_counter() - consume_start)

    t_stage = float(np.mean(stage_times)) if stage_times else 0.0
    t_consume = float(np.mean(consume_times)) if consume_times else 0.0
    s_item = float(np.mean(byte_counts)) if byte_counts else 0.0
    return t_stage, t_consume, s_item


def compute_adaptive_stage_depth(data_items, originals_images_dir, originals_masks_dir,
                                  negs_images_dir, negs_masks_dir, scratch_dir, sample_n=16):
    """d = min(ceil(t_stage / t_consume) + 1, d_max) -- the auto-tuned
    prefetch depth from Section~\\ref{ch:methodology} (Stage 4: predictive
    staging). d_max is how many items fit in the scratch dir's free capacity;
    the +1 is slack against jitter in t_stage, same as the thesis formula.
    """
    t_stage, t_consume, s_item = measure_staging_timings(
        data_items, originals_images_dir, originals_masks_dir,
        negs_images_dir, negs_masks_dir, scratch_dir, sample_n=sample_n,
    )
    free_bytes = shutil.disk_usage(scratch_dir).free
    d_max = (len(data_items) if s_item <= 0
              else max(1, min(len(data_items), int(free_bytes // s_item))))
    t_consume_safe = max(t_consume, 1e-6)
    depth = max(1, min(math.ceil(t_stage / t_consume_safe) + 1, d_max))
    return {
        "depth": depth,
        "t_stage": t_stage,
        "t_consume": t_consume,
        "s_item_bytes": s_item,
        "d_max": d_max,
        "scratch_free_bytes": free_bytes,
        "sample_n": sample_n,
    }

