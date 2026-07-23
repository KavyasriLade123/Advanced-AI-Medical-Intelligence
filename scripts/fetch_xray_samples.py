"""Download open chest X-ray samples and verify accept/reject behavior."""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.ml import get_classifier, preprocess_image  # noqa: E402
from app.ml.image_gate import is_medical_prediction, looks_like_photo_or_text  # noqa: E402
from app.ml.preprocess import load_image  # noqa: E402

XRAY_DIR = ROOT / "samples" / "xray"
TEXT_DIR = ROOT / "samples" / "text"

# Open GitHub covid-chestxray-dataset images (public research samples)
SAMPLES = {
    "covid1.jpg": "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/ryct.2020200034.fig2.jpeg",
    "covid2.jpg": "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/ryct.2020200028.fig1a.jpeg",
    "normal2.jpg": "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/1-s2.0-S0929664620300449-gr2_lrg-a.jpg",
}


def download() -> None:
    XRAY_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in SAMPLES.items():
        dest = XRAY_DIR / name
        if dest.exists() and dest.stat().st_size > 10_000:
            continue
        req = Request(url, headers={"User-Agent": "MedIntelSampleBot/1.0"})
        data = urlopen(req, timeout=90).read()
        dest.write_bytes(data)
        print("downloaded", dest, len(data))

    doc = TEXT_DIR / "document.png"
    im = Image.new("RGB", (640, 480), (255, 255, 255))
    d = ImageDraw.Draw(im)
    for i in range(22):
        d.text((20, 12 + i * 20), f"This is NOT an X-ray. Document line {i}", fill=(0, 0, 0))
    im.save(doc)

    dark = TEXT_DIR / "dark_ui.png"
    im2 = Image.new("RGB", (640, 480), (18, 18, 18))
    d2 = ImageDraw.Draw(im2)
    for i in range(22):
        d2.text((16, 10 + i * 20), f"Dark theme screenshot text {i}", fill=(220, 220, 220))
    im2.save(dark)


def evaluate() -> None:
    clf = get_classifier()
    paths = sorted(XRAY_DIR.glob("*")) + sorted(TEXT_DIR.glob("*"))
    for path in paths:
        try:
            raw = path.read_bytes()
            img = load_image(raw)
            img.thumbnail((1024, 1024))
        except Exception as exc:
            print(path.name, "skip", exc)
            continue
        gate = looks_like_photo_or_text(img)
        pred, conf, probs = clf.predict(preprocess_image(img))
        ok = (not gate) and is_medical_prediction(pred, conf, probs, 0.40, img)
        kind = "XRAY" if "xray" in str(path.parent).lower() else "TEXT"
        print(
            f"{kind} {path.name}: gate={gate} pred={pred} conf={conf:.3f} "
            f"ACCEPT={ok} (want {'True' if kind=='XRAY' else 'False'})"
        )


if __name__ == "__main__":
    download()
    evaluate()
