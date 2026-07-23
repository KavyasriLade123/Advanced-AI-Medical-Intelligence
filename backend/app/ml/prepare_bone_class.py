"""
Add a BONE_FRACTURE class from a public bone X-ray dataset so non-chest
images are not forced into PNEUMONIA.

Uses existing chest folders under data/chest_xray and writes BONE_FRACTURE
into train/val/test.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def clear_class(root: Path, class_name: str) -> None:
    for split in ("train", "val", "test"):
        path = root / split / class_name
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def save_images(images: list, dest: Path, prefix: str) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    for i, image in enumerate(tqdm(images, desc=f"Writing {prefix}")):
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")
        image.save(dest / f"{prefix}_{i:05d}.jpg", format="JPEG", quality=90)
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data/chest_xray"))
    parser.add_argument("--max-images", type=int, default=900)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not (args.out_dir / "train" / "NORMAL").exists():
        raise SystemExit("Chest dataset missing. Run prepare_dataset.py first.")

    print("Downloading Hemg/bone-fracture-detection ...")
    ds = load_dataset("Hemg/bone-fracture-detection")

    # Flatten all available splits and keep fractured-looking samples when label exists.
    rows = []
    for split_name in ds.keys():
        for row in ds[split_name]:
            rows.append(row)

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    images = []
    for row in rows:
        image = row.get("image")
        if image is None:
            continue
        # Prefer fractured if a label/field exists; otherwise take all bone X-rays.
        label = row.get("label")
        if label is not None:
            label_name = str(label).lower()
            if label_name in {"0", "not fractured", "nofracture", "normal", "negative"}:
                continue
        images.append(image)
        if len(images) >= args.max_images:
            break

    if len(images) < 100:
        # Fallback: take any images from the dataset
        images = []
        for row in rows:
            if row.get("image") is not None:
                images.append(row["image"])
            if len(images) >= args.max_images:
                break

    rng.shuffle(images)
    n = len(images)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    train_imgs = images[:n_train]
    val_imgs = images[n_train : n_train + n_val]
    test_imgs = images[n_train + n_val :]

    clear_class(args.out_dir, "BONE_FRACTURE")
    print("train", save_images(train_imgs, args.out_dir / "train" / "BONE_FRACTURE", "train_bone"))
    print("val", save_images(val_imgs, args.out_dir / "val" / "BONE_FRACTURE", "val_bone"))
    print("test", save_images(test_imgs, args.out_dir / "test" / "BONE_FRACTURE", "test_bone"))
    print(f"Done. Added BONE_FRACTURE under {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
