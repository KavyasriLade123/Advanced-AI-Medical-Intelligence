"""
Download the Kermany chest X-ray pneumonia dataset from Hugging Face
and write ImageFolder directories for training.

Source: hf-vision/chest-xray-pneumonia
Labels: NORMAL (0), PNEUMONIA (1)
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def save_split(rows, out_root: Path, split_name: str, label_names: dict[int, str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for i, row in enumerate(tqdm(rows, desc=f"Writing {split_name}")):
        label_id = int(row["label"])
        label = label_names[label_id]
        dest_dir = out_root / split_name / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        image = row["image"]
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")
        path = dest_dir / f"{split_name}_{label.lower()}_{i:05d}.jpg"
        image.save(path, format="JPEG", quality=92)
        counts[label] += 1
    return dict(counts)


def stratified_holdout(rows, val_ratio: float, seed: int):
    by_label: dict[int, list] = defaultdict(list)
    for row in rows:
        by_label[int(row["label"])].append(row)

    train_rows: list = []
    val_rows: list = []
    rng = random.Random(seed)
    for label, items in by_label.items():
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_ratio))
        val_rows.extend(items[:n_val])
        train_rows.extend(items[n_val:])
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data/chest_xray"))
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-class", type=int, default=0, help="0 = use all training images")
    args = parser.parse_args()

    print("Downloading hf-vision/chest-xray-pneumonia ...")
    ds = load_dataset("hf-vision/chest-xray-pneumonia")
    label_names = {0: "NORMAL", 1: "PNEUMONIA"}

    train_all = list(ds["train"])
    if args.max_per_class > 0:
        by_label: dict[int, list] = defaultdict(list)
        for row in train_all:
            by_label[int(row["label"])].append(row)
        limited: list = []
        for items in by_label.values():
            limited.extend(items[: args.max_per_class])
        train_all = limited

    train_rows, val_rows = stratified_holdout(train_all, args.val_ratio, args.seed)
    # Official test split kept for final evaluation
    test_rows = list(ds["test"])

    if args.out_dir.exists():
        print(f"Cleaning existing folder: {args.out_dir}")
        for path in sorted(args.out_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()

    print(save_split(train_rows, args.out_dir, "train", label_names))
    print(save_split(val_rows, args.out_dir, "val", label_names))
    print(save_split(test_rows, args.out_dir, "test", label_names))
    print(f"Done. Dataset ready at {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
