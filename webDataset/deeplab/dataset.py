import os
import webdataset as wds
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
from PIL import Image

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

def process_sample(sample, transform):
    # Decode jpg and png from the sample
    image_pil = sample["jpg"]
    mask_pil = sample["png"]
    
    # Preprocess image and mask
    image = image_pil.convert("RGB")
    mask = mask_pil.convert("L")
    
    # Preprocess matching baseline logic
    # masks are flipped
    mask_np = ~np.array(mask)
    mask_np = mask_np.astype(np.float32)
    mask_np[mask_np > 0.0] = 1.0
    mask_np = np.expand_dims(mask_np, -1)
    
    image_np = np.asarray(image)
    
    if transform:
        transformed = transform(image=image_np, mask=mask_np)
        image_tensor = transformed["image"]
        mask_tensor = transformed["mask"].permute(2, 0, 1)
    else:
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask_np).permute(2, 0, 1).float()
        
    return image_tensor, mask_tensor

def get_webdataset(shard_urls, transform, is_training=True, shuffle_buffer=1000):
    shardshuffle = 1 if is_training else False
    dataset = wds.WebDataset(shard_urls, shardshuffle=shardshuffle)
    if is_training:
        dataset = dataset.shuffle(shuffle_buffer)
    
    dataset = dataset.decode("pil")
    dataset = dataset.map(lambda sample: process_sample(sample, transform))
    return dataset
