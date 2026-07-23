"""
Train binary X-ray / clinical-scan detector (xray vs not_xray).

  python -m app.ml.train_xray_gate --data-dir ../data/xray_gate --epochs 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from app.config import MODEL_DIR, ensure_directories

XRAY_GATE_WEIGHTS = MODEL_DIR / "xray_gate_resnet18.pth"
CLASS_NAMES = ["not_xray", "xray"]  # ImageFolder sorts alphabetically


def build_loaders(data_dir: Path, image_size: int, batch_size: int):
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(8),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
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
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, train_ds.classes


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
    train_loader, val_loader, classes = build_loaders(data_dir, image_size, batch_size)
    print(f"Classes: {classes} | device={device}")
    assert set(classes) == set(CLASS_NAMES), f"Expected {CLASS_NAMES}, got {classes}"

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    # Warm-start backbone from disease classifier when available
    prior_path = MODEL_DIR / "chest_xray_resnet18.pth"
    if prior_path.exists():
        prior = torch.load(prior_path, map_location="cpu", weights_only=True)
        current = model.state_dict()
        matched = {
            k: v
            for k, v in prior.items()
            if k in current and current[k].shape == v.shape and not k.startswith("fc.")
        }
        current.update(matched)
        model.load_state_dict(current)
        print(f"Warm-started backbone ({len(matched)} tensors) from {prior_path.name}")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    # Head-only then full
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        full = epoch > 2
        for name, param in model.named_parameters():
            param.requires_grad = full or name.startswith("fc.")
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=lr if full else lr * 5)

        model.train()
        running = 0.0
        seen = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * labels.size(0)
            seen += labels.size(0)

        val_acc = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch}/{epochs} loss={running / max(seen, 1):.4f} "
            f"val_acc={val_acc:.4f} ({'full' if full else 'head'})"
        )
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), XRAY_GATE_WEIGHTS)
            print(f"Saved {XRAY_GATE_WEIGHTS} (val_acc={best_acc:.4f})")

    print(f"Done. Best val_acc={best_acc:.4f}")
    return XRAY_GATE_WEIGHTS


def main() -> None:
    parser = argparse.ArgumentParser(description="Train X-ray gate detector")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()
    train(args.data_dir, args.epochs, args.batch_size, args.lr, args.image_size)


if __name__ == "__main__":
    main()
