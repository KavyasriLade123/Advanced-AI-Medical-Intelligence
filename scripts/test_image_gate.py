from PIL import Image, ImageDraw
import sys

sys.path.insert(0, "backend")
from app.ml.image_gate import looks_like_photo_or_text, is_medical_prediction

img = Image.new("RGB", (400, 300), (20, 20, 20))
d = ImageDraw.Draw(img)
for i, line in enumerate(
    ["Hello world", "This is a text document", "abc 12345", "Brain tumor? no just text"] * 4
):
    d.text((10, 10 + i * 18), line, fill=(230, 230, 230))
print("dark text -> reject?", looks_like_photo_or_text(img))

img2 = Image.new("RGB", (400, 300), (255, 255, 255))
d2 = ImageDraw.Draw(img2)
for i in range(12):
    d2.text((12, 12 + i * 22), f"Medical report line {i} lorem ipsum dolor", fill=(0, 0, 0))
print("white text -> reject?", looks_like_photo_or_text(img2))

img3 = Image.new("RGB", (400, 300), (30, 30, 30))
d3 = ImageDraw.Draw(img3)
d3.ellipse((80, 40, 320, 260), fill=(160, 160, 160))
print("smooth gray clinical-like -> reject?", looks_like_photo_or_text(img3))

probs = {"BRAIN_TUMOR": 0.55, "UNSUPPORTED": 0.12, "NORMAL": 0.1}
print(
    "brain on text medical?",
    is_medical_prediction("BRAIN_TUMOR", 0.55, probs, 0.55, img),
)
probs2 = {"BRAIN_TUMOR": 0.81, "UNSUPPORTED": 0.05, "NORMAL": 0.04}
print(
    "strong brain on gray medical?",
    is_medical_prediction("BRAIN_TUMOR", 0.81, probs2, 0.55, img3),
)
