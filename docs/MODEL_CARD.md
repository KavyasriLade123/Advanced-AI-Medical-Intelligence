# Trained model card — MedIntel ResNet18

## File
`backend/models/chest_xray_resnet18.pth`

## Architecture
ResNet18 (ImageNet-pretrained backbone) with a custom fully-connected head for multi-class medical image triage.

## Classes
ABDOMEN, BONE_FRACTURE, BRAIN_NORMAL, BRAIN_TUMOR, BREAST_MALIGNANT, BREAST_NORMAL,
EYE_RETINA, LOWER_LIMB, NORMAL (chest), PNEUMONIA, SKIN, UNSUPPORTED

## Input
RGB image, resized to 224×224, ImageNet mean/std normalization.

## Training notes
- Optimizer: Adam
- Loss: class-weighted CrossEntropy
- Pipeline: head-only warm-up then full fine-tune
- Script: `python -m app.ml.train --data-dir ./data/chest_xray`

## Intended use
Educational decision support / academic demonstration only. Not for clinical diagnosis.
