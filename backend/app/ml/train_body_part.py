"""Train optional body-part classifier from dataset/{body_part}/... folders.

Usage (from backend/ with venv):
  python -m app.ml.train_body_part --data-dir ../dataset --epochs 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Training entrypoint placeholder — extend with torchvision ImageFolder loaders
# pointing at dataset/<body_part>/<disease>/ for per-part disease models, or
# dataset_body_part/<part>/*.jpg for Stage-2 body-part classification.


def main() -> None:
    parser = argparse.ArgumentParser(description="Train body-part / disease models")
    parser.add_argument("--data-dir", type=Path, default=Path("../dataset"))
    parser.add_argument("--task", choices=["body_part", "disease"], default="body_part")
    parser.add_argument("--body-part", type=str, default="chest")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    print(
        f"Data dir: {args.data_dir.resolve()}\n"
        f"Task: {args.task}\n"
        "Populate dataset/ then extend this script with ImageFolder + ResNet training.\n"
        "Checkpoints go to backend/models/body_part_resnet18.pth or "
        "backend/models/disease/{body_part}.pth"
    )


if __name__ == "__main__":
    main()
