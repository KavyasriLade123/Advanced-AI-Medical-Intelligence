"""
Train dedicated Stage-2 body-part ResNet18.

  python -m app.ml.train_body_part --data-dir ../data/body_part --epochs 10

Writes: backend/models/body_part_resnet18.pth
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms

from app.config import MODEL_DIR, ensure_directories
from app.ml.pipeline.catalog import BODY_PART_MODEL_CLASSES

BODY_PART_WEIGHTS = MODEL_DIR / "body_part_resnet18.pth"
LABELS_META = MODEL_DIR / "body_part_labels.json"


def build_loaders(data_dir: Path, image_size: int, batch_size: int):
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(12),
            transforms.ColorJitter(brightness=0.2, contrast=0.25),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=eval_tf)

    # class-balanced sampling
    counts = torch.bincount(torch.tensor(train_ds.targets), minlength=len(train_ds.classes)).float()
    weights = 1.0 / counts.clamp(min=1.0)
    sample_w = weights[torch.tensor(train_ds.targets)]
    sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, train_ds.classes, train_ds.targets


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


def train(data_dir: Path, epochs: int, batch_size: int, lr: float, image_size: int) -> Path:
    ensure_directories()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, classes, targets = build_loaders(data_dir, image_size, batch_size)
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Device={device} train_batches={len(train_loader)} val_batches={len(val_loader)}")

    # Prefer catalog order when possible; otherwise use ImageFolder order.
    catalog = [c for c in BODY_PART_MODEL_CLASSES if c in classes]
    extra = [c for c in classes if c not in catalog]
    ordered = catalog + extra
    # Remap ImageFolder indices -> ordered label indices via class name
    folder_to_name = {i: n for i, n in enumerate(classes)}
    name_to_ordered = {n: i for i, n in enumerate(ordered)}

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(ordered))

    # Warm-start backbone from chest disease model when available
    prior = MODEL_DIR / "chest_xray_resnet18.pth"
    if prior.exists():
        state = torch.load(prior, map_location="cpu", weights_only=True)
        current = model.state_dict()
        matched = {
            k: v
            for k, v in state.items()
            if k in current and current[k].shape == v.shape and not k.startswith("fc.")
        }
        current.update(matched)
        model.load_state_dict(current)
        print(f"Warm-started backbone ({len(matched)} tensors) from {prior.name}")
    model.to(device)

    counts = torch.bincount(torch.tensor(targets), minlength=len(classes)).float()
    # map folder-class weights into ordered class weights
    ordered_counts = torch.zeros(len(ordered))
    for fi, name in folder_to_name.items():
        ordered_counts[name_to_ordered[name]] = counts[fi]
    class_w = (ordered_counts.sum() / (len(ordered) * ordered_counts.clamp(min=1))).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        full = epoch > 2
        for name, param in model.named_parameters():
            param.requires_grad = full or name.startswith("fc.")
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=lr if full else lr * 4)

        model.train()
        running = 0.0
        seen = 0
        for images, labels in train_loader:
            # remap labels to ordered index space
            mapped = torch.tensor(
                [name_to_ordered[folder_to_name[int(x)]] for x in labels.tolist()],
                dtype=torch.long,
            )
            images, mapped = images.to(device), mapped.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), mapped)
            loss.backward()
            optimizer.step()
            running += loss.item() * mapped.size(0)
            seen += mapped.size(0)

        # val with remapped labels
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                mapped = torch.tensor(
                    [name_to_ordered[folder_to_name[int(x)]] for x in labels.tolist()],
                    dtype=torch.long,
                )
                images, mapped = images.to(device), mapped.to(device)
                preds = model(images).argmax(dim=1)
                correct += (preds == mapped).sum().item()
                total += mapped.size(0)
        acc = correct / max(total, 1)
        print(f"Epoch {epoch}/{epochs} loss={running / max(seen, 1):.4f} val_acc={acc:.3f}")

        if acc >= best_acc:
            best_acc = acc
            BODY_PART_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), BODY_PART_WEIGHTS)
            LABELS_META.write_text(json.dumps({"classes": ordered}, indent=2), encoding="utf-8")
            print(f"  saved {BODY_PART_WEIGHTS.name} (best val_acc={best_acc:.3f})")

    print(f"Done. Best val_acc={best_acc:.3f} -> {BODY_PART_WEIGHTS}")
    return BODY_PART_WEIGHTS


def main() -> None:
    parser = argparse.ArgumentParser(description="Train body-part classifier")
    parser.add_argument("--data-dir", type=Path, default=Path("../data/body_part"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()
    if not (args.data_dir / "train").exists():
        raise SystemExit(
            f"Missing {args.data_dir / 'train'}. Run: python scripts/prepare_body_part_data.py"
        )
    train(args.data_dir, args.epochs, args.batch_size, args.lr, args.image_size)


if __name__ == "__main__":
    main()
