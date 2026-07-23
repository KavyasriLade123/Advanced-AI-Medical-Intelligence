# 2. Trained Model

## What to submit
1. Weight file: `backend/models/chest_xray_resnet18.pth` (~45 MB)
2. Supporting description: `docs/MODEL_CARD.md`

## File
| Item | Value |
|------|--------|
| Path | `backend/models/chest_xray_resnet18.pth` |
| Architecture | ResNet18 (ImageNet backbone + custom head) |
| Task | Multi-class medical image triage |
| Framework | PyTorch |

## Classes
ABDOMEN, BONE_FRACTURE, BRAIN_NORMAL, BRAIN_TUMOR, BREAST_MALIGNANT, BREAST_NORMAL,  
EYE_RETINA, LOWER_LIMB, NORMAL, PNEUMONIA, SKIN, UNSUPPORTED

## Related docs
- Model card: `docs/MODEL_CARD.md`
- Training script: `backend/app/ml/train.py` (or `python -m app.ml.train`)

## Disclaimer
Educational / academic demonstration only. **Not for clinical diagnosis.**

## Status
**Ready** — checkpoint is present in the repo and tracked for submission.
