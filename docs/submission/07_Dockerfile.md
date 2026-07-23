# 7. Dockerfile (implemented)

## What to submit
Docker build files (and compose if allowed):

| File | Path | Purpose |
|------|------|---------|
| Backend image | `backend/Dockerfile` | FastAPI + PyTorch API on port 8000 |
| Frontend image | `frontend/Dockerfile` | Built React app served via nginx |
| Frontend nginx | `frontend/nginx.conf` | Reverse proxy / static serve |
| Compose | `docker-compose.yml` | Run full stack together |

## Backend Dockerfile (summary)
- Base: `python:3.11-slim`
- Installs OpenCV system libs
- Copies `requirements.txt`, `app/`, `models/`
- Exposes `8000`
- Runs `uvicorn app.main:app --host 0.0.0.0 --port 8000`

## Run (when Docker Desktop is installed)
```powershell
cd C:\Users\kanir\advanced-ai-medical-intelligence-platform
docker compose up --build
```

## Status
**Ready / implemented.** Note: Docker may not be installed on the development PC; files are still valid for submission and for machines with Docker.
