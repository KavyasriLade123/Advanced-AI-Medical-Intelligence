"""
Build Stage-2 body-part X-ray dataset:

  data/body_part/{train,val}/<part>/*.jpg

Sources:
  - Wikimedia Commons radiology categories (hand, knee, chest, …)
  - HuggingFace chest / brain samples when available
  - Local samples/xray/*

Usage:
  python scripts/prepare_body_part_data.py --max-per-class 100
"""

from __future__ import annotations

import argparse
import json
import random
import ssl
import time
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from PIL import Image, ImageEnhance, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "body_part"
SAMPLES = ROOT / "samples" / "xray"
UA = "MedIntelBodyPartTrainer/1.0 (educational research; local training)"
SSL_CTX = ssl._create_unverified_context()

# Catalog class -> Wikimedia Commons categories (best-effort)
PART_CATEGORIES: dict[str, list[str]] = {
    "hand": [
        "Category:X-rays of fractures of the human wrist and hand",
        "Category:X-rays of phalangeal fractures of the hand",
        "Category:X-rays of injuries to the wrist and hand",
        "Category:X-rays of diseases and disorders of the hands",
    ],
    "wrist": [
        "Category:X-rays of fractures of the human wrist and hand",
        "Category:X-rays of injuries to the wrist and hand",
        "Category:X-rays of distal radius fractures",
    ],
    "elbow": [
        "Category:X-rays of olecranon fracture",
        "Category:X-rays of elbow luxation",
        "Category:Fractures of the human forearm",
    ],
    "shoulder": [
        "Category:X-rays of shoulder luxation",
        "Category:X-rays of shoulder impingement syndrome",
        "Category:X-rays of proximal humerus fractures",
    ],
    "arm": [
        "Category:X-rays of humerus fractures",
        "Category:X-rays of proximal humerus fractures",
        "Category:Fractures of the human forearm",
    ],
    "knee": [
        "Category:X-rays of knee osteoarthritis",
        "Category:X-rays of knee effusion",
        "Category:X-rays of fractures of the human knee",
        "Category:X-rays of the knee",
    ],
    "foot": [
        "Category:X-rays of the foot",
        "Category:X-rays of the right foot",
        "Category:X-rays of metatarsal fractures",
    ],
    "ankle": [
        "Category:X-rays of injuries to the ankle and foot",
        "Category:X-rays of ankle fractures",
        "Category:X-rays of the ankle",
    ],
    "hip": [
        "Category:X-rays of hip fractures",
        "Category:X-rays of hip dysplasia",
        "Category:X-rays of the hip",
    ],
    "pelvis": [
        "Category:X-rays of the female pelvis",
        "Category:Fractures of human pelvis",
        "Category:X-rays of the pelvis",
    ],
    "spine": [
        "Category:X-rays of the spine",
        "Category:X-rays of diseases and disorders of the vertebral column",
    ],
    "cervical_spine": [
        "Category:X-rays of the cervical spine",
        "Category:X-rays of the neck",
    ],
    "lumbar_spine": [
        "Category:X-rays of the lumbar vertebrae",
        "Category:X-rays of the spine",
    ],
    "chest": [
        "Category:X-rays of the chest",
        "Category:X-rays of pneumonia",
        "Category:COVID-19 pneumonia",
    ],
    "skull": [
        "Category:X-rays of fractures of the human skull",
        "Category:X-rays of the skull",
    ],
    "abdomen": [
        "Category:X-rays of the abdomen",
        "Category:Abdominal X-ray",
    ],
    "leg": [
        "Category:X-rays of tibia fractures",
        "Category:X-rays of fibula fractures",
        "Category:X-rays of the lower leg",
    ],
    "femur": [
        "Category:X-rays of femur fractures",
        "Category:X-rays of the femur",
    ],
    "brain": [
        "Category:MRI of the brain",
        "Category:Computed tomography of the brain",
    ],
    "bone": [
        "Category:X-rays of bone fractures",
    ],
}

LOCAL_HINTS: dict[str, tuple[str, ...]] = {
    "chest": ("covid", "chest", "lung", "pneumonia", "color_chest"),
    "bone": ("bone", "humerus", "fracture"),
    "brain": ("abc", "brain", "mri"),
    "hand": ("hand",),
    "knee": ("knee",),
}


def _get_json(params: dict, retries: int = 4) -> dict:
    url = "https://commons.wikimedia.org/w/api.php?" + urlencode(params)
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": UA})
            with urlopen(req, timeout=45, context=SSL_CTX) as resp:
                return json.loads(resp.read())
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Wikimedia API failed: {last_err}")


def _download_bytes(url: str) -> bytes | None:
    try:
        req = Request(url, headers={"User-Agent": UA})
        with urlopen(req, timeout=60, context=SSL_CTX) as resp:
            return resp.read()
    except Exception:
        return None


def list_category_urls(category: str, limit: int) -> list[str]:
    """Prefer Wikimedia thumbnails (higher rate limits than full originals)."""
    urls: list[str] = []
    cont: dict[str, str] = {}
    while len(urls) < limit:
        params = {
            "action": "query",
            "generator": "categorymembers",
            "gcmtitle": category,
            "gcmtype": "file",
            "gcmlimit": "40",
            "prop": "imageinfo",
            "iiprop": "url|size",
            "iiurlwidth": "512",
            "format": "json",
            **cont,
        }
        data = _get_json(params)
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            info = (page.get("imageinfo") or [{}])[0]
            u = info.get("thumburl") or info.get("url")
            if u and any(u.lower().split("?")[0].endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff")):
                urls.append(u)
                if len(urls) >= limit:
                    break
        cont_token = data.get("continue")
        if not cont_token:
            break
        cont = cont_token
        time.sleep(1.0)
    return urls


def load_image_bytes(data: bytes) -> Image.Image | None:
    try:
        img = Image.open(BytesIO(data))
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")
    except Exception:
        return None


def augment(img: Image.Image, rng: random.Random) -> Image.Image:
    out = img.copy()
    if rng.random() < 0.5:
        out = ImageOps.mirror(out)
    if rng.random() < 0.4:
        out = out.rotate(rng.uniform(-18, 18), expand=True, fillcolor=(0, 0, 0))
    if rng.random() < 0.5:
        out = ImageEnhance.Contrast(out).enhance(rng.uniform(0.75, 1.35))
    if rng.random() < 0.5:
        out = ImageEnhance.Brightness(out).enhance(rng.uniform(0.8, 1.2))
    # mild random crop
    w, h = out.size
    if min(w, h) > 80 and rng.random() < 0.45:
        scale = rng.uniform(0.82, 0.98)
        nw, nh = int(w * scale), int(h * scale)
        x0 = rng.randint(0, max(w - nw, 0))
        y0 = rng.randint(0, max(h - nh, 0))
        out = out.crop((x0, y0, x0 + nw, y0 + nh))
    return out.resize((224, 224))


def save_split(images: list[Image.Image], dest: Path, prefix: str) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        img.convert("RGB").resize((224, 224)).save(dest / f"{prefix}_{i:04d}.jpg", quality=90)
    return len(images)


def collect_local(part: str) -> list[Image.Image]:
    if not SAMPLES.exists():
        return []
    hints = LOCAL_HINTS.get(part, (part,))
    out: list[Image.Image] = []
    for path in SAMPLES.iterdir():
        if not path.is_file():
            continue
        name = path.name.lower()
        if any(h in name for h in hints):
            try:
                out.append(Image.open(path).convert("RGB"))
            except Exception:
                pass
    return out


def collect_hf_chest(max_n: int) -> list[Image.Image]:
    images: list[Image.Image] = []
    try:
        from datasets import load_dataset

        ds = load_dataset("keremberke/chest-xray-classification", name="full", split="train", streaming=True)
        for i, row in enumerate(ds):
            img = row.get("image")
            if img is None:
                continue
            images.append(img.convert("RGB"))
            if len(images) >= max_n:
                break
    except Exception as exc:  # noqa: BLE001
        print("HF chest skip:", exc)
    return images


def collect_hf_brain(max_n: int) -> list[Image.Image]:
    images: list[Image.Image] = []
    try:
        from huggingface_hub import hf_hub_download, list_repo_files

        files = list_repo_files("Reddysridhar1/CT-Scan-Brain-Tumor", repo_type="dataset")
        files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        random.shuffle(files)
        for rel in files[:max_n]:
            local = Path(hf_hub_download(repo_id="Reddysridhar1/CT-Scan-Brain-Tumor", repo_type="dataset", filename=rel))
            images.append(Image.open(local).convert("RGB"))
    except Exception as exc:  # noqa: BLE001
        print("HF brain skip:", exc)
    return images


def _synth_part(part: str, seed: int, size: int = 224) -> Image.Image:
    """Distinctive grayscale anatomic sketches so rare classes still train."""
    import numpy as np

    rng = np.random.default_rng(seed)
    canvas = np.full((size, size), rng.integers(8, 25), dtype=np.uint8)
    yy, xx = np.ogrid[:size, :size]

    def ellipse(cy, cx, ry, rx, val):
        mask = ((yy - cy) / max(ry, 1)) ** 2 + ((xx - cx) / max(rx, 1)) ** 2 <= 1.0
        canvas[mask] = np.clip(canvas[mask].astype(np.int16) + val, 0, 255).astype(np.uint8)

    def bone(x0, y0, x1, y1, thickness, val=160):
        # crude line bone
        length = max(int(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5), 1)
        for t in range(length):
            x = int(x0 + (x1 - x0) * t / length)
            y = int(y0 + (y1 - y0) * t / length)
            ellipse(y, x, thickness, thickness, val - canvas[min(max(y, 0), size - 1), min(max(x, 0), size - 1)])

    if part in {"hand", "wrist"}:
        # palm + 5 fingers
        ellipse(150, 112, 55, 48, 90)
        tips = [(30, 40), (20, 70), (18, 112), (28, 150), (55, 175)]
        bases = [(110, 80), (115, 95), (118, 112), (115, 130), (120, 150)]
        for (y1, x1), (y0, x0) in zip(tips, bases):
            bone(x0, y0, x1, y1, 5, 170)
            bone(x1, y1, x1 - 8, max(y1 - 18, 8), 4, 155)
    elif part == "knee":
        ellipse(110, 112, 70, 60, 70)
        ellipse(95, 112, 22, 28, 110)  # patella
        bone(112, 20, 112, 95, 14, 150)
        bone(112, 130, 112, 210, 14, 150)
    elif part == "elbow":
        bone(112, 20, 112, 110, 12, 150)
        bone(90, 120, 60, 200, 10, 145)
        bone(130, 120, 155, 200, 10, 145)
        ellipse(115, 112, 20, 24, 100)
    elif part == "shoulder":
        ellipse(90, 80, 35, 40, 85)
        bone(100, 95, 170, 200, 14, 150)
        ellipse(85, 70, 18, 22, 120)
    elif part == "foot":
        ellipse(140, 120, 35, 70, 80)
        for i, x in enumerate([50, 75, 100, 125, 150]):
            bone(x, 140, x - 10 + i, 40, 4, 160)
    elif part == "ankle":
        bone(112, 20, 112, 130, 14, 150)
        ellipse(150, 112, 28, 35, 95)
        bone(90, 160, 50, 200, 8, 140)
        bone(130, 160, 165, 200, 8, 140)
    elif part in {"spine", "cervical_spine", "lumbar_spine"}:
        for i, y in enumerate(range(30, 200, 18)):
            ellipse(y, 112, 8, 22, 100 + (i % 3) * 10)
    elif part in {"leg", "femur", "arm"}:
        bone(112, 15, 112, 210, 16, 155)
        ellipse(40, 112, 18, 20, 110)
        ellipse(190, 112, 18, 20, 110)
    elif part == "abdomen":
        ellipse(120, 112, 80, 70, 60)
        ellipse(100, 90, 25, 20, 40)
        ellipse(110, 140, 30, 22, 45)
    else:
        ellipse(112, 112, 60, 50, 80)
        bone(112, 40, 112, 180, 12, 140)

    noise = rng.integers(-12, 12, (size, size), dtype=np.int16)
    gray = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(gray, mode="L").convert("RGB")


def collect_wikimedia(part: str, max_n: int) -> list[Image.Image]:
    cats = PART_CATEGORIES.get(part, [])
    urls: list[str] = []
    per_cat = max(15, max_n // max(len(cats), 1) + 8)
    for cat in cats:
        try:
            found = list_category_urls(cat, per_cat)
            print(f"  {part}/{cat}: {len(found)} urls", flush=True)
            urls.extend(found)
            time.sleep(1.2)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {cat}: {exc}", flush=True)
    seen: set[str] = set()
    uniq = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    random.shuffle(uniq)
    images: list[Image.Image] = []
    for u in uniq:
        if len(images) >= max_n:
            break
        raw = _download_bytes(u)
        if not raw:
            time.sleep(2.0)
            raw = _download_bytes(u)
        if not raw:
            continue
        img = load_image_bytes(raw)
        if img is None or min(img.size) < 64:
            continue
        images.append(img)
        time.sleep(1.6)  # stay under Wikimedia request limits
    return images


def balance_with_aug(images: list[Image.Image], target: int, seed: int) -> list[Image.Image]:
    if not images:
        return []
    rng = random.Random(seed)
    out = [im.resize((224, 224)) for im in images]
    i = 0
    while len(out) < target:
        out.append(augment(images[i % len(images)], rng))
        i += 1
    return out[:target]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--max-per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--parts",
        type=str,
        default=",".join(PART_CATEGORIES.keys()),
        help="Comma-separated body-part ids",
    )
    args = parser.parse_args()
    random.seed(args.seed)

    parts = [p.strip() for p in args.parts.split(",") if p.strip()]
    buckets: dict[str, list[Image.Image]] = defaultdict(list)

    for part in parts:
        print(f"\n=== Collecting {part} ===")
        imgs = collect_local(part)
        print(f"  local={len(imgs)}")
        if part == "chest":
            imgs.extend(collect_hf_chest(args.max_per_class))
            print(f"  after HF chest={len(imgs)}")
        if part == "brain":
            imgs.extend(collect_hf_brain(min(args.max_per_class, 120)))
            print(f"  after HF brain={len(imgs)}")
        wiki = collect_wikimedia(part, args.max_per_class)
        print(f"  wikimedia={len(wiki)}", flush=True)
        imgs.extend(wiki)
        # If still sparse, add distinctive anatomic sketches (better than empty class)
        if len(imgs) < max(12, args.max_per_class // 3):
            need = max(12, args.max_per_class // 2) - len(imgs)
            print(f"  synth fill={need}", flush=True)
            for i in range(max(need, 0)):
                imgs.append(_synth_part(part, args.seed + i * 17 + hash(part) % 10000))
        uniq: list[Image.Image] = []
        sigs: set[tuple[int, int, int]] = set()
        for im in imgs:
            small = im.resize((8, 8)).convert("L")
            sig = (im.size[0], im.size[1], sum(list(small.getdata())) % 9973)
            if sig in sigs:
                continue
            sigs.add(sig)
            uniq.append(im)
        buckets[part] = balance_with_aug(uniq, args.max_per_class, args.seed + hash(part) % 1000)
        print(f"  final={len(buckets[part])}", flush=True)

    # write train/val 80/20
    for part, images in buckets.items():
        if len(images) < 8:
            print(f"WARNING: {part} has only {len(images)} images — skipping")
            continue
        rng = random.Random(args.seed)
        items = list(images)
        rng.shuffle(items)
        n_val = max(2, int(len(items) * 0.2))
        val_i, train_i = items[:n_val], items[n_val:]
        save_split(train_i, args.out_dir / "train" / part, f"train_{part}")
        save_split(val_i, args.out_dir / "val" / part, f"val_{part}")
        print(f"Wrote {part}: train={len(train_i)} val={len(val_i)}")

    print("\nDone. Train classes:")
    train_root = args.out_dir / "train"
    if train_root.exists():
        for p in sorted(train_root.iterdir()):
            if p.is_dir():
                n = len(list(p.glob("*.jpg")))
                print(f"  {p.name}={n}")


if __name__ == "__main__":
    main()
