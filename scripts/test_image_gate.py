from PIL import Image, ImageDraw
import sys

sys.path.insert(0, "backend")
from app.ml.image_gate import looks_like_photo_or_text, is_medical_prediction

# Dark text screenshot
img = Image.new("RGB", (400, 300), (20, 20, 20))
d = ImageDraw.Draw(img)
for i, line in enumerate(
    ["Hello world", "This is a text document", "abc 12345", "Brain tumor? no just text"] * 5
):
    d.text((8, 8 + i * 16), line, fill=(230, 230, 230))
print("dark text -> reject?", looks_like_photo_or_text(img))

# White text doc
img2 = Image.new("RGB", (400, 300), (255, 255, 255))
d2 = ImageDraw.Draw(img2)
for i in range(14):
    d2.text((10, 8 + i * 20), f"Report line {i} lorem ipsum dolor sit amet", fill=(0, 0, 0))
print("white text -> reject?", looks_like_photo_or_text(img2))

# Smooth grayscale "xray-like"
img3 = Image.new("RGB", (400, 300), (25, 25, 25))
d3 = ImageDraw.Draw(img3)
d3.ellipse((70, 30, 330, 270), fill=(150, 150, 150))
d3.ellipse((140, 90, 260, 210), fill=(90, 90, 90))
print("xray-like -> reject?", looks_like_photo_or_text(img3))

probs_ok = {"PNEUMONIA": 0.72, "NORMAL": 0.15, "UNSUPPORTED": 0.05}
print(
    "pneumonia on xray medical?",
    is_medical_prediction("PNEUMONIA", 0.72, probs_ok, 0.45, img3),
)
probs_bad = {"BRAIN_TUMOR": 0.70, "UNSUPPORTED": 0.10, "NORMAL": 0.05}
print(
    "brain on text medical?",
    is_medical_prediction("BRAIN_TUMOR", 0.70, probs_bad, 0.45, img),
)
