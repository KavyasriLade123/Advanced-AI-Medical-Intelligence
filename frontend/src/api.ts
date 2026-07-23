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
    const fromFile = (cfg.apiBaseUrl ?? "").trim().replace(/\/$/, "");
    if (fromFile) API_BASE = fromFile;
  } catch {
    /* keep build-time value */
  }
}

export function getApiBase(): string {
  return API_BASE;
}

export function isApiConfigured(): boolean {
  // Local Vite uses same-origin proxy; production needs an absolute API origin.
  if (import.meta.env.DEV) return true;
  return Boolean(API_BASE);
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

function apiHint(): string {
  if (!isApiConfigured()) {
    return (
      "Backend URL is not configured. On Vercel set VITE_API_BASE_URL to your Render API " +
      "(e.g. https://medintel-api.onrender.com), or put that URL in frontend/public/config.json as apiBaseUrl, then redeploy."
    );
  }
  return (
    `Cannot reach the MedIntel API at ${API_BASE}. Ensure Render is Live and CORS_ORIGINS includes this Vercel site.`
  );
}

async function apiFetch(path: string, init?: RequestInit): Promise<Response> {
  if (!isApiConfigured()) {
    throw new Error(apiHint());
  }
  try {
    return await fetch(`${API_BASE}${path}`, init);
  } catch {
    throw new Error(apiHint());
  }
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
