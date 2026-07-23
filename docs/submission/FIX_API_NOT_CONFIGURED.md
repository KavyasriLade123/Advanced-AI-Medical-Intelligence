# Live app (fixed)

## Use this URL (solved)
**https://advanced-ai-medical-intelligence-4og2.onrender.com/analyze**

UI + API are on the **same Render service** → no CORS → no `Failed to fetch` from Vercel.

## After this push — redeploy Render
1. Render dashboard → your web service  
2. **Manual Deploy** → **Deploy latest commit**  
3. Wait until **Live** (build can take 10–20 min)  
4. Open `/analyze` above  

Keep **Root Directory = `backend`** (Dockerfile now includes `static/`).

## Vercel
Vercel now redirects to Render. Prefer the Render link for demos and submission.
