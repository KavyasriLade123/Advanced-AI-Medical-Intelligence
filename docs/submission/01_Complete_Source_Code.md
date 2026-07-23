# 1. Complete Source Code

## What to submit
The full application source (backend + frontend + scripts + configs), as a ZIP of the repository **or** via the GitHub link.

## Primary locations
| Part | Path |
|------|------|
| Backend (FastAPI, ML, APIs) | `backend/app/` |
| Trained weights folder | `backend/models/` |
| Frontend (React + Vite) | `frontend/src/` |
| Run scripts | `scripts/` |
| Docker Compose | `docker-compose.yml` |
| Docs / report | `docs/` |

## Key modules
- `backend/app/main.py` — API entry
- `backend/app/ml/` — model, Grad-CAM, train
- `backend/app/routers/` — predict, history, reports, health
- `backend/app/services/` — LLM report / disease info
- `frontend/src/pages/` — Home, About, Analyze
- `frontend/src/components/` — layout / UI pieces

## How to package (ZIP)
From the project root, zip the repo **excluding** caches and secrets:

- Include: `backend/`, `frontend/`, `docs/`, `scripts/`, `README.md`, `requirements.txt`, `docker-compose.yml`, `package.json`
- Exclude: `node_modules/`, `__pycache__/`, `.venv/`, `backend/data/uploads/`, `.env` with secrets

## Status
**Ready** — complete source is in this repository.
