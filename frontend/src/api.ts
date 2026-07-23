export type ProbabilityItem = {
  label: string;
  probability: number;
};

export type DiseaseInfo = {
  title: string;
  body_region: string;
  summary: string;
  related_conditions: string[];
  common_symptoms_to_correlate: string[];
  typical_xray_findings: string[];
  possible_causes_if_symptomatic: string[];
  recommended_next_steps: string[];
  urgency: string;
  disclaimer: string;
};

export type Prediction = {
  id: number;
  predicted_class: string;
  confidence: number;
  probabilities: ProbabilityItem[];
  disease_info?: DiseaseInfo | null;
  image_url: string;
  heatmap_url: string | null;
  report: string | null;
  model_mode: string;
  created_at: string;
};

export type HistoryItem = {
  id: number;
  original_filename: string;
  predicted_class: string;
  confidence: number;
  model_mode: string;
  created_at: string;
  image_url: string;
  heatmap_url: string | null;
  has_report: boolean;
};

export type HistoryList = {
  total: number;
  items: HistoryItem[];
};

export type Health = {
  status: string;
  app: string;
  model_loaded: boolean;
  model_mode: string;
  classes: string[];
};

/** Build-time default; overridden at runtime by /config.json then VITE_API_BASE_URL. */
let API_BASE = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

/** Call once before rendering (main.tsx). Loads public/config.json if present. */
export async function initApiBase(): Promise<void> {
  try {
    const res = await fetch("/config.json", { cache: "no-store" });
    if (!res.ok) return;
    const cfg = (await res.json()) as { apiBaseUrl?: string };
    // If apiBaseUrl is present (even ""), prefer it so Vercel same-origin /api proxy works
    // even when VITE_API_BASE_URL was set at build time.
    if (Object.prototype.hasOwnProperty.call(cfg, "apiBaseUrl")) {
      API_BASE = String(cfg.apiBaseUrl ?? "")
        .trim()
        .replace(/\/$/, "");
    }
  } catch {
    /* keep build-time value */
  }
}

export function getApiBase(): string {
  return API_BASE;
}

export function isApiConfigured(): boolean {
  // Local Vite and Vercel both proxy /api (vite.config / vercel.json rewrites).
  // Optional absolute API_BASE still works if set via env or config.json.
  return true;
}

async function readError(res: Response): Promise<string> {
  try {
    const data = await res.json();
    if (typeof data.detail === "string") return data.detail;
    return JSON.stringify(data.detail ?? data);
  } catch {
    return res.statusText || `Request failed (${res.status})`;
  }
}

function apiHint(cause?: string): string {
  const target = API_BASE || "the proxied /api route";
  const extra = cause ? ` (${cause})` : "";
  return (
    `Cannot reach the MedIntel API via ${target}${extra}. ` +
    "Render Free may be waking up — open https://advanced-ai-medical-intelligence-4og2.onrender.com/api/health, " +
    "wait until JSON appears, then click Predict again (can take 30–90 seconds)."
  );
}

async function apiFetch(path: string, init?: RequestInit): Promise<Response> {
  const url = `${API_BASE}${path}`;
  let lastErr: unknown;
  // Render Free cold-starts often fail the first 1–2 browser requests.
  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      const res = await fetch(url, init);
      return res;
    } catch (err) {
      lastErr = err;
      if (attempt < 3) {
        await new Promise((r) => setTimeout(r, attempt * 2500));
      }
    }
  }
  const msg = lastErr instanceof Error ? lastErr.message : "network error";
  throw new Error(apiHint(msg));
}

export async function fetchHealth(): Promise<Health> {
  const res = await apiFetch(`/api/health`);
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export async function predictImage(file: File): Promise<Prediction> {
  const form = new FormData();
  form.append("file", file);
  const res = await apiFetch(`/api/predict?generate_report=true`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export async function fetchHistory(limit = 12): Promise<HistoryList> {
  const res = await apiFetch(`/api/history?limit=${limit}`);
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export async function fetchPrediction(id: number): Promise<Prediction> {
  const res = await apiFetch(`/api/history/${id}`);
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export async function regenerateReport(id: number): Promise<{ report: string; source: string }> {
  const res = await apiFetch(`/api/reports`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prediction_id: id }),
  });
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export function mediaUrl(path: string | null | undefined): string {
  if (!path) return "";
  return path.startsWith("http") ? path : `${API_BASE}${path}`;
}
