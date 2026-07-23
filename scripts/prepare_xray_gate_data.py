"""
Prepare binary dataset for X-ray / clinical-scan detection.

Layout:
  data/xray_gate/{train,val}/{xray,not_xray}/

Usage:
  python scripts/prepare_xray_gate_data.py
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "xray_gate"
SAMPLES_XRAY = ROOT / "samples" / "xray"
SAMPLES_TEXT = ROOT / "samples" / "text"

# Public research chest X-rays (ieee8023 covid-chestxray-dataset)
REMOTE_XRAYS = [
    "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/ryct.2020200034.fig2.jpeg",
    "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/ryct.2020200028.fig1a.jpeg",
    "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/1-s2.0-S0929664620300449-gr2_lrg-a.jpg",
    "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/auntminnie-a-2020_01_28_23_51_6665_2020_01_28_Vietnam_coronavirus.jpeg",
    "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/auntminnie-b-2020_01_28_23_51_6665_2020_01_28_Vietnam_coronavirus.jpeg",
    "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/auntminnie-c-2020_01_28_23_51_6665_2020_01_28_Vietnam_coronavirus.jpeg",
    "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/lancet-case2a.jpg",
    "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/lancet-case2b.jpg",
    "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/nejmc2001573_f1a.jpeg",
    "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/nejmc2001573_f1b.jpeg",
    "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/radiopaedia.org.20200316.covid.001.PA.jpg",
    "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/radiopaedia.org.20200316.covid.002.PA.jpg",
]


def _download(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 8_000:
        return True
    try:
        req = Request(url, headers={"User-Agent": "MedIntelXrayGate/1.0"})
        data = urlopen(req, timeout=60).read()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        Image.open(dest).convert("RGB").save(dest.with_suffix(".jpg"), quality=90)
        if dest.suffix.lower() != ".jpg":
            dest.unlink(missing_ok=True)
            dest = dest.with_suffix(".jpg")
        print("ok", dest.name, len(data))
        return True
    except Exception as exc:
        print("skip", url, exc)
        return False


def _synth_bone_xray(seed: int, size: int = 256) -> Image.Image:
    rng = np.random.default_rng(seed)
    base = rng.integers(20, 60, (size, size), dtype=np.uint8)
    yy, xx = np.ogrid[:size, :size]
    # soft elliptical "lung / skull" tissue
    for _ in range(3):
        cy, cx = rng.integers(60, size - 60, 2)
        ry, rx = rng.integers(40, 90, 2)
        mask = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 < 1
        base[mask] = np.clip(base[mask].astype(np.int16) + rng.integers(40, 120), 0, 255).astype(
            np.uint8
        )
    noise = rng.integers(-15, 15, (size, size), dtype=np.int16)
    gray = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(gray, mode="L").convert("RGB")
    return img.filter(ImageFilter.GaussianBlur(radius=0.8))


def _synth_blue_mri(seed: int, size: int = 256) -> Image.Image:
    gray = np.array(_synth_bone_xray(seed, size).convert("L"), dtype=np.float32)
    r = np.clip(gray * 0.45, 0, 255)
    g = np.clip(gray * 0.65, 0, 255)
    b = np.clip(gray * 1.25 + 20, 0, 255)
    arr = np.stack([r, g, b], axis=-1).astype(np.uint8)
    return Image.fromarray(arr)


def _synth_person(seed: int, size: int = 256) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[:, :, 0] = rng.integers(50, 180, (size, size))
    arr[:, :, 1] = rng.integers(40, 150, (size, size))
    arr[:, :, 2] = rng.integers(30, 120, (size, size))
    yy, xx = np.ogrid[:size, :size]
    for _ in range(rng.integers(1, 4)):
        cy, cx = rng.integers(40, size - 40, 2)
        mask = ((yy - cy) / 70) ** 2 + ((xx - cx) / 55) ** 2 < 1
        arr[mask] = [200, 150, 120]
    return Image.fromarray(arr)


def _synth_ui(seed: int, size: int = 512) -> Image.Image:
    rng = random.Random(seed)
    im = Image.new("RGB", (size, size), (30, 30, 36))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, size // 4, size), fill=(37, 37, 38))
    d.rectangle((0, 0, size, 36), fill=(45, 45, 48))
    for i in range(18):
        d.text((10, 50 + i * 22), f"File{i}.tsx  Project Manager", fill=(180, 180, 180))
        col = (86, 156, 214) if i % 3 == 0 else (200, 200, 200)
        d.text((size // 4 + 12, 50 + i * 24), f"const x = predict(img); // {i}", fill=col)
    if rng.random() > 0.4:
        d.rectangle((size - 180, 50, size - 10, size - 10), fill=(40, 40, 44))
    return im


def _synth_document(seed: int, size: int = 512) -> Image.Image:
    im = Image.new("RGB", (size, size), (250, 250, 250))
    d = ImageDraw.Draw(im)
    for i in range(24):
        d.text((24, 20 + i * 20), f"Assignment notes chapter {i} fees WhatsApp", fill=(20, 20, 20))
    return im


def _copy_or_save(img: Image.Image, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(dest, quality=92)


def _gather_local_xrays() -> list[Image.Image]:
    images: list[Image.Image] = []
    if SAMPLES_XRAY.exists():
        for path in SAMPLES_XRAY.iterdir():
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                continue
            try:
                images.append(Image.open(path).convert("RGB"))
            except Exception:
                continue
    return images


def _gather_local_negatives() -> list[Image.Image]:
    images: list[Image.Image] = []
    if SAMPLES_TEXT.exists():
        for path in SAMPLES_TEXT.iterdir():
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                continue
            try:
                images.append(Image.open(path).convert("RGB"))
            except Exception:
                continue
    return images


def build() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    rng = random.Random(42)
    positives: list[Image.Image] = _gather_local_xrays()
    negatives: list[Image.Image] = _gather_local_negatives()

    cache = ROOT / "data" / "xray_gate_downloads"
    cache.mkdir(parents=True, exist_ok=True)
    for i, url in enumerate(REMOTE_XRAYS):
        dest = cache / f"remote_{i}.jpg"
        if _download(url, dest):
            try:
                positives.append(Image.open(dest.with_suffix(".jpg") if dest.with_suffix(".jpg").exists() else dest).convert("RGB"))
            except Exception:
                try:
                    positives.append(Image.open(dest).convert("RGB"))
                except Exception:
                    pass

    for i in range(40):
        positives.append(_synth_bone_xray(1000 + i))
    for i in range(25):
        positives.append(_synth_blue_mri(2000 + i))

    for i in range(35):
        negatives.append(_synth_person(3000 + i))
    for i in range(35):
        negatives.append(_synth_ui(4000 + i))
    for i in range(25):
        negatives.append(_synth_document(5000 + i))

    # light augment by flips
    def expand(imgs: list[Image.Image], limit: int) -> list[Image.Image]:
        out = list(imgs)
        while len(out) < limit:
            src = rng.choice(imgs)
            im = src.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if rng.random() > 0.5 else src.copy()
            if rng.random() > 0.5:
                im = im.rotate(rng.choice([-8, -4, 4, 8]), expand=False, fillcolor=(0, 0, 0))
            out.append(im)
        return out

    positives = expand(positives, 120)
    negatives = expand(negatives, 120)
    rng.shuffle(positives)
    rng.shuffle(negatives)

    def split_save(label: str, images: list[Image.Image]) -> None:
        n_val = max(8, len(images) // 5)
        val, train = images[:n_val], images[n_val:]
        for split, items in (("train", train), ("val", val)):
            for i, img in enumerate(items):
                _copy_or_save(img, OUT / split / label / f"{label}_{i:04d}.jpg")

    split_save("xray", positives)
    split_save("not_xray", negatives)
    print(
        "Prepared",
        OUT,
        "train",
        len(list((OUT / "train" / "xray").glob("*"))),
        len(list((OUT / "train" / "not_xray").glob("*"))),
        "val",
        len(list((OUT / "val" / "xray").glob("*"))),
        len(list((OUT / "val" / "not_xray").glob("*"))),
    )


if __name__ == "__main__":
    build()
