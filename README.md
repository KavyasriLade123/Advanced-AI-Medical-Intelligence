# Advanced AI Medical Intelligence Platform (MedIntel)

## Project title
**Advanced AI Medical Intelligence Platform**

## Project objective
End-to-end AI application for medical image analysis with deep learning prediction, Grad-CAM explainability, LLM-assisted reporting, REST APIs, SQLite history, and a React web UI.

## Submission checklist

| Requirement | Location / status |
|-------------|-------------------|
| Complete source code | This repository (`backend/`, `frontend/`) |
| Trained model | [`backend/models/chest_xray_resnet18.pth`](backend/models/chest_xray_resnet18.pth) |
| GitHub repository link | *Add after push* — see [GitHub setup](#github-setup) |
| README documentation | This file |
| PDF project report | [`docs/Project_Report.pdf`](docs/Project_Report.pdf) |
| `requirements.txt` | [`requirements.txt`](requirements.txt) and [`backend/requirements.txt`](backend/requirements.txt) |
| Dockerfile | [`backend/Dockerfile`](backend/Dockerfile), [`frontend/Dockerfile`](frontend/Dockerfile) |
| Live deployment link | *Optional* — local run / Docker; cloud URL if you deploy |

## Objective coverage

| Capability | Implementation |
|------------|----------------|
| Analyzing medical images | Upload + RGB preprocess (224×224, ImageNet normalize) |
| Predicting diseases (Deep Learning) | Fine-tuned ResNet18 multi-class classifier |
| Explainable AI | Grad-CAM heatmaps on ResNet `layer4` |
| AI-assisted medical reports (LLM) | OpenAI-compatible chat API + template fallback |
| REST APIs | FastAPI: `/api/predict`, `/api/history`, `/api/reports`, `/api/health` |
| Prediction history database | SQLite + SQLAlchemy (`predictions` table) |
| User-friendly interface / deploy | React + Vite UI; Docker Compose |

## Model classes (trained)

`ABDOMEN`, `BONE_FRACTURE`, `BRAIN_NORMAL`, `BRAIN_TUMOR`, `BREAST_MALIGNANT`, `BREAST_NORMAL`, `EYE_RETINA`, `LOWER_LIMB`, `NORMAL` (chest), `PNEUMONIA`, `SKIN`, `UNSUPPORTED` (rejected uploads)

Unrelated / low-confidence images return an error asking the user to upload a correct medical image.

## Project structure

```
advanced-ai-medical-intelligence-platform/
├── backend/
│   ├── app/
│   │   ├── api/routes/     # REST endpoints
│   │   ├── ml/             # classifier, Grad-CAM, train, dataset prep
│   │   ├── models/         # SQLAlchemy ORM
│   │   ├── schemas/        # Pydantic schemas
│   │   ├── services/       # LLM reports + disease info
│   │   ├── config.py
│   │   ├── database.py
│   │   └── main.py
│   ├── models/chest_xray_resnet18.pth   # trained weights
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
├── frontend/               # React + TypeScript + Vite
│   └── Dockerfile
├── docs/Project_Report.pdf
├── scripts/                # Windows run helpers
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.11+ (3.13 supported)
- Node.js 20+
- Optional: Docker Desktop
- Optional: `OPENAI_API_KEY` for live LLM reports

## Quick start (local)

### Backend

```bat
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

- API docs: http://127.0.0.1:8000/docs  
- Health: http://127.0.0.1:8000/api/health  

### Frontend

```bat
cd frontend
npm install
npm run dev
```

Or from repo root:

```bat
npm run install:frontend
npm run dev
```

UI: http://localhost:5173

### Helper scripts (Windows)

```bat
scripts\run-backend.bat
scripts\run-frontend.bat
```

## Docker deployment

```bat
docker compose up --build
```

- Frontend: http://localhost:5173  
- Backend: http://localhost:8000  

Ensure `backend/models/chest_xray_resnet18.pth` exists before building/running.

## REST API summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health, model mode, classes |
| POST | `/api/predict` | Image → prediction + Grad-CAM + disease info + report |
| GET | `/api/history` | Paginated prediction history |
| GET | `/api/history/{id}` | Single prediction |
| POST | `/api/reports` | Regenerate report for a prediction |
| DELETE | `/api/history/{id}` | Delete history row |

```bat
curl -X POST http://127.0.0.1:8000/api/predict -F "file=@sample.png"
```

## Environment variables

Copy `backend/.env.example` → `backend/.env`:

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | LLM reports (optional) |
| `OPENAI_BASE_URL` | OpenAI-compatible base URL |
| `OPENAI_MODEL` | Default `gpt-4o-mini` |
| `CORS_ORIGINS` | Frontend origins |
| `DATABASE_URL` | SQLite URL |

Without an API key, reports use the built-in clinical template.

## Training / retraining

```bat
cd backend
.venv\Scripts\activate
python -m app.ml.train --data-dir ./data/chest_xray --epochs 3 --batch-size 16
```

Weights are written to `backend/models/chest_xray_resnet18.pth`. Restart the API after training.

## Evaluation criteria mapping

| Criterion | How this project addresses it |
|-----------|-------------------------------|
| DL model performance | Fine-tuned ResNet18; validation metrics logged during training |
| Code quality & structure | Modular FastAPI + React TypeScript layout |
| Explainable AI | Grad-CAM overlays returned by `/api/predict` |
| LLM integration | OpenAI-compatible client + template fallback |
| API development | Versioned REST routes + OpenAPI (`/docs`) |
| Database design | SQLAlchemy `predictions` table with probs/report paths |
| Web application | React UI: upload, heatmap, finding, disease card, history |
| Documentation | README + PDF report |
| Deployment | Dockerfiles + `docker-compose.yml` |
| System design | Separation of ML / API / DB / UI; reject unsupported images |

## GitHub setup

`gh` is not logged in on this machine. To publish:

```bat
gh auth login
git add .
git commit -m "Initial submission: Advanced AI Medical Intelligence Platform"
gh repo create advanced-ai-medical-intelligence-platform --public --source=. --remote=origin --push
```

Then paste the repo URL into this README under **GitHub repository link**.

## Live deployment

Not deployed to a public cloud by default. For submission you may:

1. Run locally / Docker and record a demo video, or  
2. Deploy backend (e.g. Render/Railway) + frontend (e.g. Vercel/Netlify) and add the URL here:

**Live deployment link:** _Not deployed yet_

## Disclaimer

Educational decision-support prototype only. **Not a medical diagnosis.** All outputs require review by qualified clinicians.
