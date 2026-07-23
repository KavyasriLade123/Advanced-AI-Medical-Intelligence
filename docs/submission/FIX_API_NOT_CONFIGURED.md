# Fix “API not configured” (Vercel + Render)

Do these in order. The Vercel site alone cannot run the PyTorch model.

## Step A — Deploy API on Render Free

1. Open https://dashboard.render.com → **New +** → **Web Service**
2. Connect GitHub repo: `Advanced-AI-Medical-Intelligence`
3. Settings:
   - **Name:** `medintel-api`
   - **Root Directory:** `backend`
   - **Runtime:** Docker
   - **Instance type:** **Free**
4. Environment → Add:
   - `CORS_ORIGINS` = `https://advanced-ai-medical-intelligence-jy.vercel.app`
5. Click **Create Web Service** and wait until **Live** (first Docker build can take 15–25 min).
6. Copy your URL, e.g. `https://medintel-api.onrender.com`
7. Open: `https://YOUR-SERVICE.onrender.com/api/health` (must return JSON)

### Free plan notes
- Service **sleeps** after ~15 min idle → first Predict can take 30–90s (cold start). Wait; don’t spam refresh.
- If the service crashes with **OOM / ran out of memory**, Free can’t run Torch reliably — then use Hugging Face Spaces or a paid starter later.
- Keep **one** free web service on the account if Render limits apply.

## Step B — Point Vercel at that URL

### Option 1 (recommended): Vercel Environment Variable

1. Open https://vercel.com → project `advanced-ai-medical-intelligence`
2. **Settings** → **Environment Variables**
3. Add:
   - **Key:** `VITE_API_BASE_URL`
   - **Value:** `https://YOUR-SERVICE.onrender.com`  (no trailing slash)
   - Environments: Production, Preview, Development
4. **Deployments** → latest → **⋯** → **Redeploy**  
   (check **Use existing Build Cache** = OFF if available)

### Option 2: config.json in the repo

Edit `frontend/public/config.json`:

```json
{
  "apiBaseUrl": "https://YOUR-SERVICE.onrender.com"
}
```

Commit, push to `main`, wait for Vercel auto-deploy.

## Step C — Verify

1. Hard-refresh the site (Ctrl+F5)
2. Top pill should say **API online**
3. Red banner should disappear
4. Go to **Image Analysis** → upload JPG/PNG → **Predict + explain**

## Common mistakes

| Mistake | Result |
|---------|--------|
| Only Vercel, no Render | This exact error |
| Env set but no Redeploy | Still old build, still error |
| Trailing `/` on API URL | Usually OK, but avoid it |
| Wrong CORS_ORIGINS | Predict may fail after “API online” |
| Render still building / slept | Timeout / offline |
