"""
Fine-tune ResNet18 on chest X-ray NORMAL vs PNEUMONIA.

Example:
  python -m app.ml.train --data-dir ./data/chest_xray --epochs 6 --batch-size 16
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from app.config import MODEL_DIR, WEIGHTS_PATH, ensure_directories


def build_loaders(data_dir: Path, image_size: int, batch_size: int):
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
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

    test_loader = None
    test_dir = data_dir / "test"
    if test_dir.exists():
        test_ds = datasets.ImageFolder(test_dir, transform=eval_tf)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, train_ds.classes, train_ds.targets


def class_weights(targets: list[int], num_classes: int, device: torch.device) -> torch.Tensor:
    counts = torch.bincount(torch.tensor(targets), minlength=num_classes).float()
    weights = counts.sum() / (num_classes * counts.clamp(min=1))
    return weights.to(device)


def evaluate(model, loader, device, criterion=None) -> tuple[float, float]:
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            if criterion is not None:
                loss_sum += criterion(logits, labels).item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / max(total, 1)
    avg_loss = loss_sum / max(total, 1)
    return acc, avg_loss


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for name, param in model.named_parameters():
        if name.startswith("fc."):
            param.requires_grad = True
        else:
            param.requires_grad = trainable


def train(
    data_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    image_size: int,
    freeze_epochs: int,
) -> None:
    ensure_directories()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, classes, targets = build_loaders(
        data_dir, image_size, batch_size
    )
    print(f"Classes: {classes} | Device: {device}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    # Warm-start from previous checkpoint when shapes match (backbone / old head).
    if WEIGHTS_PATH.exists():
        prior = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
        current = model.state_dict()
        matched = {
            k: v for k, v in prior.items() if k in current and current[k].shape == v.shape
        }
        current.update(matched)
        model.load_state_dict(current)
        print(f"Warm-started {len(matched)}/{len(current)} tensors from {WEIGHTS_PATH.name}")
    model.to(device)

    weights = class_weights(targets, len(classes), device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    print(f"Class weights: { {c: float(weights[i]) for i, c in enumerate(classes)} }")

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        trainable_backbone = epoch > freeze_epochs
        set_backbone_trainable(model, trainable_backbone)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=lr if trainable_backbone else lr * 5)

        model.train()
        running = 0.0
        seen = 0
        for step, (images, labels) in enumerate(train_loader, start=1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * labels.size(0)
            seen += labels.size(0)
            if step % 20 == 0 or step == len(train_loader):
                print(
                    f"  epoch {epoch} step {step}/{len(train_loader)} "
                    f"loss={running / max(seen, 1):.4f} "
                    f"({'full' if trainable_backbone else 'head-only'})"
                )

        val_acc, val_loss = evaluate(model, val_loader, device, criterion)
        print(
            f"Epoch {epoch}/{epochs} — train_loss={running / max(seen, 1):.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if val_acc >= best_acc:
            best_acc = val_acc
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), WEIGHTS_PATH)
            print(f"Saved best weights to {WEIGHTS_PATH} (val_acc={best_acc:.4f})")

    if test_loader is not None:
        # Reload best checkpoint for honest test score
        state = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
        model.load_state_dict(state)
        test_acc, test_loss = evaluate(model, test_loader, device, criterion)
        print(f"Test — loss={test_loss:.4f} acc={test_acc:.4f}")

    print(f"Done. Best val accuracy: {best_acc:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MedIntel chest X-ray classifier")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--freeze-epochs",
        type=int,
        default=2,
        help="Train classifier head only for N epochs, then fine-tune full network",
    )
    args = parser.parse_args()
    train(
        args.data_dir,
        args.epochs,
        args.batch_size,
        args.lr,
        args.image_size,
        args.freeze_epochs,
    )


if __name__ == "__main__":
    main()
