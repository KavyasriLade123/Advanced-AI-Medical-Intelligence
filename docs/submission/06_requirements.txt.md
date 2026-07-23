# 6. requirements.txt

## What to submit
Python dependency files used to install the backend:

| File | Path | Role |
|------|------|------|
| Root wrapper | `requirements.txt` | `pip install -r requirements.txt` from repo root |
| Backend list | `backend/requirements.txt` | Actual pinned/listed packages (FastAPI, torch, etc.) |

## Root file content
```
# Install backend dependencies from repository root:
#   pip install -r requirements.txt
-r backend/requirements.txt
```

## Frontend dependencies (separate)
Not in `requirements.txt` — use:

- `frontend/package.json`
- `npm install` inside `frontend/`

## Install command
```powershell
cd C:\Users\kanir\advanced-ai-medical-intelligence-platform
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Status
**Ready** — both root and backend requirement files are present.
