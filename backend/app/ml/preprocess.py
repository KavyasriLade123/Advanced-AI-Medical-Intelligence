from io import BytesIO
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from app.config import get_settings


def get_transform() -> transforms.Compose:
    size = get_settings().image_size
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_image(path: Path | str | bytes) -> Image.Image:
    if isinstance(path, bytes):
        return Image.open(BytesIO(path)).convert("RGB")
    return Image.open(path).convert("RGB")


def preprocess_image(image: Image.Image) -> torch.Tensor:
    tensor = get_transform()(image)
    return tensor.unsqueeze(0)
