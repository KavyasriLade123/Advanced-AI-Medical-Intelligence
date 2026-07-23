"""Add BRAIN_NORMAL / BRAIN_TUMOR by downloading a subset of brain CT images."""

from __future__ import annotations

import argparse
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image
from tqdm import tqdm

REPO_ID = "Reddysridhar1/CT-Scan-Brain-Tumor"


def clear_class(root: Path, class_name: str) -> None:
    for split in ("train", "val", "test"):
        path = root / split / class_name
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def split_paths(paths: list[str], seed: int) -> tuple[list[str], list[str], list[str]]:
    rng = random.Random(seed)
    paths = list(paths)
    rng.shuffle(paths)
    n = len(paths)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    return paths[:n_train], paths[n_train : n_train + n_val], paths[n_train + n_val :]


def download_one(rel_path: str) -> Path:
    local = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=rel_path)
    return Path(local)


def copy_split(rel_paths: list[str], dest: Path, prefix: str, workers: int = 8) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_one, p): p for p in rel_paths}
        for i, fut in enumerate(tqdm(as_completed(futures), total=len(futures), desc=f"Writing {prefix}")):
            src = fut.result()
            image = Image.open(src).convert("RGB")
            image.save(dest / f"{prefix}_{i:05d}.jpg", format="JPEG", quality=90)
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data/chest_xray"))
    parser.add_argument("--max-per-class", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not (args.out_dir / "train" / "NORMAL").exists():
        raise SystemExit("Base dataset missing. Prepare chest/bone data first.")

    print("Listing files from", REPO_ID)
    files = list_repo_files(REPO_ID, repo_type="dataset")
    normal_files = [
        f for f in files if f.startswith("NORMAL/") and f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    tumor_files = [
        f
        for f in files
        if (f.startswith("TUMOR/") or f.startswith("Tumor/") or f.startswith("tumor/"))
        and f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    # Prefer unique basenames without duplicate png/jpg pairs: keep jpg preferred
    def prefer_jpg(paths: list[str]) -> list[str]:
        by_stem: dict[str, str] = {}
        for p in paths:
            stem = Path(p).stem.lower().replace(" ", "_")
            if stem not in by_stem or p.lower().endswith(".jpg"):
                by_stem[stem] = p
        return list(by_stem.values())

    normal_files = prefer_jpg(normal_files)[: args.max_per_class]
    tumor_files = prefer_jpg(tumor_files)[: args.max_per_class]
    print(f"Selected NORMAL={len(normal_files)} TUMOR={len(tumor_files)}")

    mapping = {
        "BRAIN_NORMAL": normal_files,
        "BRAIN_TUMOR": tumor_files,
    }

    for class_name, paths in mapping.items():
        if not paths:
            raise SystemExit(f"No files found for {class_name}")
        clear_class(args.out_dir, class_name)
        train_p, val_p, test_p = split_paths(paths, args.seed)
        print(class_name, "train", copy_split(train_p, args.out_dir / "train" / class_name, f"train_{class_name.lower()}"))
        print(class_name, "val", copy_split(val_p, args.out_dir / "val" / class_name, f"val_{class_name.lower()}"))
        print(class_name, "test", copy_split(test_p, args.out_dir / "test" / class_name, f"test_{class_name.lower()}"))

    print(f"Done. Brain classes added under {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
