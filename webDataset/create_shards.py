import os
import random
import numpy as np
import webdataset as wds
from PIL import Image
import io

def create_shards(dataset_path, output_dir, seed=1234, max_count_per_shard=100):
    # Setup directories
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)

    # Load originals
    originals_images_dir = os.path.join(dataset_path, "train/originals/sites")
    originals_masks_dir = os.path.join(dataset_path, "train/originals/masks")
    originals_images = sorted(os.listdir(originals_images_dir)) if os.path.exists(originals_images_dir) else []

    # Load negs
    negs_images_dir = os.path.join(dataset_path, "train/negs/sites")
    negs_masks_dir = os.path.join(dataset_path, "train/negs/masks")
    negs_images = sorted(os.listdir(negs_images_dir)) if os.path.exists(negs_images_dir) else []

    # Combine datasets
    combined_data = []
    for fname in originals_images:
        combined_data.append({
            "filename": fname,
            "source": "originals",
            "image_path": os.path.join(originals_images_dir, fname),
            "mask_path": os.path.join(originals_masks_dir, fname.replace(".jpg", ".png"))
        })
    for fname in negs_images:
        combined_data.append({
            "filename": fname,
            "source": "negs",
            "image_path": os.path.join(negs_images_dir, fname),
            "mask_path": os.path.join(negs_masks_dir, fname.replace(".jpg", ".png"))
        })

    print(f"Total combined images: {len(combined_data)}")

    # Shuffle combined data with reproducible seed
    rng = np.random.RandomState(seed)
    indices = np.arange(len(combined_data))
    rng.shuffle(indices)
    
    # Same splits as baseline
    valid_split = -int(len(combined_data) * 0.2)
    test_split = valid_split // 2

    train_indices = indices[:valid_split]
    val_indices = indices[valid_split:test_split]
    test_indices = indices[test_split:]

    train_data = [combined_data[i] for i in train_indices]
    val_data = [combined_data[i] for i in val_indices]
    test_data = [combined_data[i] for i in test_indices]

    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")

    def write_dataset_to_shards(data_list, split_name):
        pattern = os.path.join(output_dir, split_name, f"shard-%06d.tar")
        print(f"Writing {split_name} shards to {pattern}...")
        
        # We use wds.ShardWriter to automatically manage shard rotation
        with wds.ShardWriter(pattern, maxcount=max_count_per_shard) as sink:
            for idx, item in enumerate(data_list):
                # Read image file
                with open(item["image_path"], "rb") as f:
                    img_bytes = f.read()

                # Read or create mask file
                if os.path.exists(item["mask_path"]):
                    with open(item["mask_path"], "rb") as f:
                        mask_bytes = f.read()
                else:
                    # Create empty (all-black) mask matching the image dimensions
                    img = Image.open(item["image_path"])
                    mask = Image.new("L", img.size, 0)
                    mask_io = io.BytesIO()
                    mask.save(mask_io, format="PNG")
                    mask_bytes = mask_io.getvalue()

                # Write to shard
                key = f"{split_name}_{idx:06d}"
                sink.write({
                    "__key__": key,
                    "jpg": img_bytes,
                    "png": mask_bytes,
                    "filename.txt": item["filename"].encode("utf-8"),
                    "source.txt": item["source"].encode("utf-8")
                })
        print(f"Finished writing {split_name} shards.")

    write_dataset_to_shards(train_data, "train")
    write_dataset_to_shards(val_data, "val")
    write_dataset_to_shards(test_data, "test")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="/home/users/nahsan/Thesis/bing_1k")
    parser.add_argument("--output_dir", type=str, default="/home/users/nahsan/Thesis/webDataset/shards")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max_count", type=int, default=100)
    args = parser.parse_args()

    create_shards(args.dataset_path, args.output_dir, args.seed, args.max_count)
