# Advanced AI Medical Intelligence Platform (MedIntel)

## Project title
**Advanced AI Medical Intelligence Platform**

## Project objective
Build a complete end-to-end AI application capable of:

- Analyzing medical images
- Predicting diseases using Deep Learning
- Explaining predictions using Explainable AI (Grad-CAM)
- Generating AI-assisted medical reports using an LLM
- Providing REST APIs
- Storing prediction history in a database
- Deploying the application with a user-friendly interface

## Submission checklist

| Requirement | Location / status |
|-------------|-------------------|
| Complete source code | This repository (`backend/`, `frontend/`) |
| Trained model | [`backend/models/chest_xray_resnet18.pth`](backend/models/chest_xray_resnet18.pth) |
| GitHub repository link | https://github.com/KavyasriLade123/Advanced-AI-Medical-Intelligence |
| README documentation | This file |
| PDF project report | [`docs/Project_Report.pdf`](docs/Project_Report.pdf) |
| Model card | [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) |
| `requirements.txt` | [`requirements.txt`](requirements.txt) and [`backend/requirements.txt`](backend/requirements.txt) |
| Dockerfile | [`backend/Dockerfile`](backend/Dockerfile), [`frontend/Dockerfile`](frontend/Dockerfile) |
| Live deployment link | Optional — local/Docker demo, or add cloud URL below |
| Per-item submission notes | [`docs/submission/`](docs/submission/00_SUBMISSION_INDEX.md) |

**Live deployment link:** _Add Vercel URL after deploy_ (see [Live cloud deploy](#live-cloud-deploy-vercel--render))  
**GitHub repository link:** https://github.com/KavyasriLade123/Advanced-AI-Medical-Intelligence

## Objective coverage

| Capability | Implementation |
|------------|----------------|
| Analyzing medical images | Image upload + RGB preprocess (224×224, ImageNet normalize) |
| Predicting diseases (Deep Learning) | Fine-tuned ResNet18 multi-class classifier |
| Explainable AI | Grad-CAM heatmaps on ResNet `layer4` |
| AI-assisted medical reports (LLM) | OpenAI-compatible chat API + template fallback |
| REST APIs | FastAPI: `/api/predict`, `/api/history`, `/api/reports`, `/api/health` |
| Prediction history database | SQLite + SQLAlchemy (`predictions` table) |
| User-friendly interface / deploy | React + Vite UI; Docker Compose |

## Tech stack

| Area | Stack |
|------|--------|
| Frontend | React, TypeScript, Vite |
| Backend | FastAPI, Uvicorn, Pydantic, SQLAlchemy |
| Deep Learning | PyTorch, Torchvision (ResNet18) |
| XAI | Grad-CAM |
| LLM | OpenAI-compatible API (optional key) |
| Database | SQLite |
| Deploy | Docker, Docker Compose |

## Features

- Drag-and-drop medical image upload (JPG/PNG)
- Deep learning prediction with confidence score
- Grad-CAM visual explanation
- Disease information card for the predicted finding only
- AI-assisted report (LLM or template)
- Prediction history stored in SQLite
- Rejection message for unrelated / unsupported images

## Model classes (trained)

`ABDOMEN`, `BONE_FRACTURE`, `BRAIN_NORMAL`, `BRAIN_TUMOR`, `BREAST_MALIGNANT`, `BREAST_NORMAL`, `EYE_RETINA`, `LOWER_LIMB`, `NORMAL` (chest), `PNEUMONIA`, `SKIN`, `UNSUPPORTED`

Unrelated or low-confidence images are rejected with: please upload a correct medical image.

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
├── docs/
│   ├── Project_Report.pdf
│   └── MODEL_CARD.md
├── scripts/                # Windows run helpers + PDF generator
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

### 1. Backend

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

### 2. Frontend

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

### Windows helper scripts

```bat
scripts\run-backend.bat
scripts\run-frontend.bat
```

## How to use the app

1. Start backend (port `8000`) and frontend (port `5173`)
2. Open http://localhost:5173
3. Upload a medical image (brain, chest, bone, etc.)
4. Click **Predict + explain**
5. Review finding, Grad-CAM heatmap, disease info, and AI report
6. Open **Prediction history** to revisit past results

## Docker deployment

```bat
docker compose up --build
```

- Frontend: http://localhost:5173  
- Backend: http://localhost:8000  

Ensure `backend/models/chest_xray_resnet18.pth` exists before building/running.

## Live cloud deploy (Vercel + Render)

Do **not** deploy the whole monorepo as a Vercel Python app (that caused the “No python entrypoint” error). Use:

| Piece | Host | Notes |
|-------|------|--------|
| Frontend | **Vercel** | Root Directory = `frontend`, or use root `vercel.json` |
| Backend | **Render** | Docker, context = `backend/` |

### Render (API first)

1. New **Web Service** → Docker → Root Directory `backend`
2. Set `CORS_ORIGINS` to your Vercel URL (after frontend exists; you can update later)
3. Optional: `OPENAI_API_KEY`
4. Copy API URL, e.g. `https://medintel-api.onrender.com`
5. Check `https://…onrender.com/api/health`

**Free** plan works for demos (CPU Torch image). Expect cold starts after idle sleep; if it OOMs, upgrade later or use Hugging Face Spaces.

### Vercel (UI)

1. Import the same GitHub repo
2. **Root Directory:** `frontend` (recommended) — avoids Python detection
3. Env: `VITE_API_BASE_URL` = your Render origin (no trailing slash)
4. Deploy → submit the `*.vercel.app` URL as the live link

Full checklist: [`docs/submission/08_Live_Deployment_Link.md`](docs/submission/08_Live_Deployment_Link.md)

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
| DL model performance | Fine-tuned ResNet18; validation checkpoint saved |
| Code quality & structure | Modular FastAPI + React TypeScript layout |
| Explainable AI | Grad-CAM overlays returned by `/api/predict` |
| LLM integration | OpenAI-compatible client + template fallback |
| API development | REST routes + OpenAPI docs at `/docs` |
| Database design | SQLAlchemy `predictions` table with probs/report paths |
| Web application | React UI: upload, heatmap, finding, disease card, history |
| Documentation | README + PDF report + model card |
| Deployment | Dockerfiles + `docker-compose.yml` |
| System design | Layered ML / API / DB / UI; reject unsupported images |

## GitHub setup

The project is already committed locally. To publish:

```bat
gh auth login
gh repo create advanced-ai-medical-intelligence-platform --public --source=. --remote=origin --push
```

Or with an existing empty repo:

```bat
git remote add origin https://github.com/<your-username>/advanced-ai-medical-intelligence-platform.git
git push -u origin master
```

Then paste the repo URL into **GitHub repository link** at the top of this README.

## Disclaimer

Educational decision-support prototype only. **Not a medical diagnosis.** All outputs require review by qualified clinicians.
