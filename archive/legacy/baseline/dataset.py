import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations import pytorch
from PIL import Image


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

    def __len__(self):
        return len(self.data_items)

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
        
        return image, mask, image_filename
