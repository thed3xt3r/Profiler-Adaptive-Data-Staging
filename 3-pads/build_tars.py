r"""Build the tar archives that the PADS 'shard' policy reads.

Why this exists
---------------
`TarArcheoDataset` expects two archives whose members are named
`images/<file>.jpg` and `masks/<file>.png`, addressed by the *original*
filename. Nothing in the repo ever produced them: `2-webdataset/create_shards.py`
writes WebDataset shards (`shard-000000.tar`, members keyed `train_000123.jpg`),
which that dataset cannot open or address. So every time PADS selected `shard`
it hit the "tar files not found" fallback and silently ran `loose` instead.

This script closes that gap, making the `shard` arm of the policy actually
executable and therefore measurable.

Uncompressed by default
-----------------------
The payload is already-compressed JPEG/PNG, so gzip saves almost nothing on
size, while making `extractfile()` decompress from the stream start on every
random access -- crippling for a shuffled DataLoader. Pass --gzip if you
specifically want to measure the compressed case.

Usage
-----
    python build_tars.py                      # -> <data_root>/bing_1k_tars/
    python build_tars.py --gzip
    python build_tars.py --out D:\shards
"""

import argparse
import json
import os
import sys
import tarfile
import time


def build(dataset_path, out_dir, use_gzip=False):
    os.makedirs(out_dir, exist_ok=True)
    ext = ".tar.gz" if use_gzip else ".tar"
    mode = "w:gz" if use_gzip else "w"
    build_started = time.perf_counter()

    sources = {
        "originals": (
            os.path.join(dataset_path, "train/originals/sites"),
            os.path.join(dataset_path, "train/originals/masks"),
        ),
        "negs": (
            os.path.join(dataset_path, "train/negs/sites"),
            os.path.join(dataset_path, "train/negs/masks"),
        ),
    }

    # Persisted for Table~\ref{tab:overhead} (profiling overhead): archive
    # construction is a one-off, offline cost, but only if it's saved
    # somewhere -- it used to only ever reach stdout.
    per_archive = {}

    for name, (img_dir, mask_dir) in sources.items():
        if not os.path.isdir(img_dir):
            raise SystemExit(f"Image directory not found: {img_dir}")

        out_path = os.path.join(out_dir, f"{name}{ext}")
        images = sorted(os.listdir(img_dir))
        started = time.perf_counter()
        n_img = n_mask = 0

        # Member layout must match TarArcheoDataset exactly:
        #   images/<filename>.jpg   masks/<filename>.png
        with tarfile.open(out_path, mode) as tar:
            for fname in images:
                src_img = os.path.join(img_dir, fname)
                if os.path.isfile(src_img):
                    tar.add(src_img, arcname=f"images/{fname}")
                    n_img += 1

                mask_name = fname.replace(".jpg", ".png")
                src_mask = os.path.join(mask_dir, mask_name)
                # Masks are stored verbatim; TarArcheoDataset applies the same
                # ~invert as the loose path, so pre-processing here would
                # double-invert and flip the labels.
                if os.path.isfile(src_mask):
                    tar.add(src_mask, arcname=f"masks/{mask_name}")
                    n_mask += 1

        size_mb = os.path.getsize(out_path) / 1024 ** 2
        elapsed_s = time.perf_counter() - started
        print(f"{name:10s} -> {out_path}")
        print(f"           {n_img} images, {n_mask} masks, {size_mb:.1f} MB, "
              f"{elapsed_s:.1f}s")
        per_archive[name] = {
            "path": out_path, "n_images": n_img, "n_masks": n_mask,
            "size_mb": size_mb, "elapsed_s": elapsed_s,
        }

    total_elapsed_s = time.perf_counter() - build_started
    summary = {
        "out_dir": out_dir,
        "gzip": use_gzip,
        "total_elapsed_s": total_elapsed_s,
        "archives": per_archive,
    }
    summary_path = os.path.join(out_dir, "archive_build_summary.json")
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nTotal archive-construction time: {total_elapsed_s:.1f}s")
    print(f"Summary saved to: {summary_path}")

    return out_dir


def verify(out_dir, dataset_path, use_gzip=False):
    """Read back through TarArcheoDataset and compare against the loose path."""
    import numpy as np
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "manet"))
    from dataset import load_dataset, get_transforms, ArcheoDataset, TarArcheoDataset

    ext = ".tar.gz" if use_gzip else ".tar"
    o_tar = os.path.join(out_dir, f"originals{ext}")
    n_tar = os.path.join(out_dir, f"negs{ext}")

    n_o = len(os.listdir(os.path.join(dataset_path, "train/originals/sites")))
    n_n = len(os.listdir(os.path.join(dataset_path, "train/negs/sites")))
    oi, om, ni, nm, train_data, _, _ = load_dataset(dataset_path, 1234, np.arange(n_o + n_n))
    _, val_tf = get_transforms()

    loose = ArcheoDataset(train_data, oi, om, ni, nm, transform=val_tf)
    shard = TarArcheoDataset(train_data, o_tar, n_tar, transform=val_tf)

    print("\nverifying shard output matches loose output...")
    mismatches = 0
    for i in (0, 1, 7, 50, 123, len(train_data) - 1):
        li, lm, lname = loose[i]
        si, sm, sname = shard[i]
        same_img = bool((np.asarray(li) == np.asarray(si)).all())
        same_mask = bool((np.asarray(lm) == np.asarray(sm)).all())
        ok = same_img and same_mask and lname == sname
        if not ok:
            mismatches += 1
        print(f"  idx {i:<6} {lname:<22} image_match={same_img} mask_match={same_mask}")
    print("MISMATCHES:", mismatches)
    return mismatches == 0


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    default_data = os.path.join(os.path.dirname(os.path.dirname(repo)), "Thesis", "source")

    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default=os.environ.get("PADS_DATA_ROOT") or default_data,
                   help="Directory containing the bing_1k folder")
    p.add_argument("--out", default=None, help="Output directory (default <data_root>/bing_1k_tars)")
    p.add_argument("--gzip", action="store_true", help="Compress (slow random access; little size win)")
    p.add_argument("--no_verify", action="store_true")
    args = p.parse_args()

    dataset_path = os.path.join(args.data_root, "bing_1k")
    if not os.path.isdir(os.path.join(dataset_path, "train")):
        raise SystemExit(f"Dataset not found: {os.path.join(dataset_path, 'train')}")

    out_dir = args.out or os.path.join(args.data_root, "bing_1k_tars")
    print(f"dataset : {dataset_path}")
    print(f"output  : {out_dir}")
    print(f"format  : {'tar.gz (compressed)' if args.gzip else 'tar (uncompressed, seekable)'}\n")

    build(dataset_path, out_dir, use_gzip=args.gzip)

    if not args.no_verify:
        ok = verify(out_dir, dataset_path, use_gzip=args.gzip)
        print("\nRESULT:", "shard output matches loose" if ok else "MISMATCH - do not use")
        if not ok:
            raise SystemExit(1)

    print(f"\nSet PADS_SHARD_DIR={out_dir} (or leave it - this is the default location).")


if __name__ == "__main__":
    main()
