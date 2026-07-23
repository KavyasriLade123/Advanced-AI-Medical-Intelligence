# Live deployment — Vercel (frontend) + Render (backend)

## Status
Configured in repo. Publish after you push these files and create the two cloud services.

## 1) Deploy backend on Render (do this first)

1. Push this repo to GitHub (include model weights + Docker files).
2. Open [https://dashboard.render.com](https://dashboard.render.com) → **New** → **Web Service**.
3. Connect `KavyasriLade123/Advanced-AI-Medical-Intelligence`.
4. Settings:
   - **Runtime:** Docker
   - **Root Directory:** `backend`  
     (or use Blueprint `render.yaml` from repo root)
   - **Dockerfile path:** `./Dockerfile` (inside `backend`)
5. Environment variables:
   - `CORS_ORIGINS` = your Vercel URL(s), comma-separated  
     Example: `https://advanced-ai-medical-intelligence.vercel.app`
   - Optional: `OPENAI_API_KEY`
6. Deploy and copy the API URL, e.g. `https://medintel-api.onrender.com`
7. Test: `https://medintel-api.onrender.com/api/health`

**Note:** PyTorch needs RAM. Prefer **Starter** (or higher), not Free, or the service may crash/OOM.

## 2) Deploy frontend on Vercel

1. Open [https://vercel.com](https://vercel.com) → **Add New Project** → import the same GitHub repo.
2. Important (avoids the Python entrypoint error):
   - **Framework Preset:** Vite
   - **Root Directory:** `frontend`  
     OR keep repo root and use the root `vercel.json` (builds `frontend/`)
3. Environment variable:
   - `VITE_API_BASE_URL` = `https://medintel-api.onrender.com`  
     (no trailing slash; use your real Render URL)
4. Deploy. Copy the frontend URL, e.g. `https://….vercel.app`

## 3) Wire CORS

On Render, set `CORS_ORIGINS` to the exact Vercel URL (and preview URLs if needed).  
Backend also allows `*.vercel.app` via regex, but setting `CORS_ORIGINS` is safest.

Redeploy Render after changing env vars.

## 4) What to submit as Live Deployment Link

Use the **Vercel frontend URL** (the site users open).

| Service | Example |
|---------|---------|
| Live app (submit this) | `https://YOUR-APP.vercel.app` |
| API (supporting) | `https://YOUR-API.onrender.com` |
| API docs | `https://YOUR-API.onrender.com/docs` |

## Why not all-on-Vercel?

Vercel failed looking for a Python entrypoint at repo root. Even with an entrypoint, PyTorch + uploads is a poor fit for Vercel serverless. Frontend on Vercel + API on Render is the intended live setup.
