"""
Export multi-body-part medical classes + UNSUPPORTED (non-medical) images.

Body coverage (top -> bottom style):
  BRAIN_* (existing), BREAST_*, CHEST (NORMAL/PNEUMONIA existing),
  ABDOMEN, LOWER_LIMB, BONE_FRACTURE (existing), EYE_RETINA, SKIN,
  UNSUPPORTED
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def clear_class(root: Path, class_name: str) -> None:
    for split in ("train", "val", "test"):
        path = root / split / class_name
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def split_images(images: list[Image.Image], seed: int):
    rng = random.Random(seed)
    items = list(images)
    rng.shuffle(items)
    n = len(items)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    return items[:n_train], items[n_train : n_train + n_val], items[n_train + n_val :]


def save_split(images: list[Image.Image], dest: Path, prefix: str) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    for i, image in enumerate(tqdm(images, desc=prefix)):
        image.convert("RGB").resize((224, 224)).save(dest / f"{prefix}_{i:05d}.jpg", quality=90)
    return len(images)


def write_class(root: Path, class_name: str, images: list[Image.Image], seed: int) -> None:
    clear_class(root, class_name)
    train_i, val_i, test_i = split_images(images, seed)
    print(class_name, "train", save_split(train_i, root / "train" / class_name, f"train_{class_name.lower()}"))
    print(class_name, "val", save_split(val_i, root / "val" / class_name, f"val_{class_name.lower()}"))
    print(class_name, "test", save_split(test_i, root / "test" / class_name, f"test_{class_name.lower()}"))


def from_medmnist(name: str, label_map: dict[int, str], max_per_class: int, seed: int) -> dict[str, list[Image.Image]]:
    import medmnist

    info = medmnist.INFO[name]
    DataClass = getattr(medmnist, info["python_class"])
    dataset = DataClass(split="train", download=True, size=28)
    buckets: dict[str, list[Image.Image]] = {v: [] for v in set(label_map.values())}
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    for idx in indices:
        img, label = dataset[idx]
        label_id = int(np.asarray(label).reshape(-1)[0])
        if label_id not in label_map:
            continue
        class_name = label_map[label_id]
        if len(buckets[class_name]) >= max_per_class:
            if all(len(buckets[c]) >= max_per_class for c in buckets):
                break
            continue
        if not isinstance(img, Image.Image):
            arr = np.asarray(img)
            if arr.ndim == 2:
                img = Image.fromarray(arr).convert("RGB")
            else:
                img = Image.fromarray(arr).convert("RGB")
        else:
            img = img.convert("RGB")
        buckets[class_name].append(img)
    return buckets


def from_cifar_unsupported(max_images: int, seed: int) -> list[Image.Image]:
    """Create non-medical distractors without external download (SSL-safe)."""
    rng = random.Random(seed)
    images: list[Image.Image] = []
    for i in range(max_images):
        arr = np.zeros((224, 224, 3), dtype=np.uint8)
        mode = i % 4
        if mode == 0:
            # colorful noise
            arr = np.random.default_rng(seed + i).integers(0, 255, size=(224, 224, 3), dtype=np.uint8)
        elif mode == 1:
            # solid + shapes (photo-like junk)
            arr[:, :] = (rng.randint(20, 220), rng.randint(20, 220), rng.randint(20, 220))
            for _ in range(12):
                x0, y0 = rng.randint(0, 180), rng.randint(0, 180)
                x1, y1 = x0 + rng.randint(20, 40), y0 + rng.randint(20, 40)
                arr[y0:y1, x0:x1] = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        elif mode == 2:
            # gradient
            for y in range(224):
                arr[y, :, 0] = y
                arr[y, :, 1] = 255 - y
                arr[y, :, 2] = (y * 2) % 255
        else:
            # checkerboard
            for y in range(0, 224, 16):
                for x in range(0, 224, 16):
                    if ((x // 16) + (y // 16)) % 2 == 0:
                        arr[y : y + 16, x : x + 16] = (240, 240, 40)
                    else:
                        arr[y : y + 16, x : x + 16] = (40, 40, 200)
        images.append(Image.fromarray(arr, mode="RGB"))
    return images


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data/chest_xray"))
    parser.add_argument("--max-per-class", type=int, default=350)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not (args.out_dir / "train" / "NORMAL").exists():
        raise SystemExit("Base chest/brain/bone dataset missing.")

    # Abdomen + lower limb from OrganAMNIST
    organ_map = {
        0: "ABDOMEN",
        4: "ABDOMEN",
        5: "ABDOMEN",
        6: "ABDOMEN",
        9: "ABDOMEN",
        10: "ABDOMEN",
        1: "LOWER_LIMB",
        2: "LOWER_LIMB",
    }
    print("Preparing OrganAMNIST body regions...")
    organ_buckets = from_medmnist("organamnist", organ_map, args.max_per_class, args.seed)
    for class_name, images in organ_buckets.items():
        if images:
            write_class(args.out_dir, class_name, images, args.seed)

    # Breast
    print("Preparing BreastMNIST...")
    breast_map = {0: "BREAST_MALIGNANT", 1: "BREAST_NORMAL"}
    breast_buckets = from_medmnist("breastmnist", breast_map, args.max_per_class, args.seed)
    for class_name, images in breast_buckets.items():
        if images:
            write_class(args.out_dir, class_name, images, args.seed)

    # Skin (dermatology)
    print("Preparing DermaMNIST as SKIN...")
    derma_map = {i: "SKIN" for i in range(7)}
    derma_buckets = from_medmnist("dermamnist", derma_map, args.max_per_class, args.seed)
    for class_name, images in derma_buckets.items():
        if images:
            write_class(args.out_dir, class_name, images, args.seed)

    # Eye / retina
    print("Preparing RetinaMNIST as EYE_RETINA...")
    retina_map = {i: "EYE_RETINA" for i in range(5)}
    retina_buckets = from_medmnist("retinamnist", retina_map, args.max_per_class, args.seed)
    for class_name, images in retina_buckets.items():
        if images:
            write_class(args.out_dir, class_name, images, args.seed)

    # Unsupported / unrelated images
    print("Preparing UNSUPPORTED from CIFAR-10...")
    unsupported = from_cifar_unsupported(args.max_per_class, args.seed)
    write_class(args.out_dir, "UNSUPPORTED", unsupported, args.seed)

    print("Done. Current train classes:")
    for p in sorted((args.out_dir / "train").glob("*")):
        if p.is_dir():
            n = len(list(p.glob("*.jpg"))) + len(list(p.glob("*.png")))
            print(f"  {p.name}={n}")


if __name__ == "__main__":
    main()
