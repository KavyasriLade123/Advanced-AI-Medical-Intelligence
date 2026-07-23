from io import BytesIO
from pathlib import Path

import torch
from PIL import Image, ImageOps
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
        img = Image.open(BytesIO(path))
    else:
        img = Image.open(path)
    # Honor camera/scanner orientation tags
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Normalize brightness/contrast for varied X-ray exposures, then ImageNet normalize."""
    img = ImageOps.exif_transpose(image.convert("RGB"))
    # Mild autocontrast helps dark/bright films without destroying clinical LUTs
    img = ImageOps.autocontrast(img, cutoff=1)
    tensor = get_transform()(img)
    return tensor.unsqueeze(0)
