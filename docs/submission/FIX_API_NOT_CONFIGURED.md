# Fix “Failed to fetch” / 502 on Vercel → Render Free

## Fastest working demo (recommended)

Use the **Render URL only** (UI + API on same host after redeploy with root `Dockerfile`):

**https://advanced-ai-medical-intelligence-4og2.onrender.com/analyze**

Same origin → no browser CORS → no `Failed to fetch`.

### Redeploy Render for that
1. Render → service → **Settings**
2. **Root Directory:** clear / leave blank (repo root)
3. **Dockerfile path:** `Dockerfile` (root file)
4. **Manual Deploy** → clear build cache if available → Deploy
5. Wait until **Live**, open `/analyze`

## If you keep using Vercel UI
1. Open health and wait for JSON (wake Free tier)
2. Click Predict and **wait up to 2 minutes** (app now polls health first)
3. Prefer a real medical JPG/PNG (not a random `abc.webp`)

## Why Vercel keeps failing
Browser calls from `*.vercel.app` to `*.onrender.com` fail while Render is sleeping (response has no CORS headers → `Failed to fetch`). Vercel proxy also returns **502** when Render is down or Predict is too slow.
