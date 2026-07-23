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

/** Build-time default; overridden at runtime by /config.json. */
let API_BASE = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

/** Call once before rendering (main.tsx). Loads public/config.json if present. */
export async function initApiBase(): Promise<void> {
  try {
    const res = await fetch("/config.json", { cache: "no-store" });
    if (!res.ok) return;
    const cfg = (await res.json()) as { apiBaseUrl?: string };
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
  const target = API_BASE || "/api";
  const extra = cause ? ` (${cause})` : "";
  return (
    `Cannot reach the MedIntel API at ${target}${extra}. ` +
    "Prefer the Render app (same server as the API): https://advanced-ai-medical-intelligence-4og2.onrender.com/analyze — " +
    "or wake https://advanced-ai-medical-intelligence-4og2.onrender.com/api/health first (Free tier 30–90s)."
  );
}

/** Poll health until Render Free finishes waking (up to ~2 minutes). */
export async function wakeApi(onStatus?: (msg: string) => void): Promise<Health> {
  const started = Date.now();
  const deadline = started + 120_000;
  let attempt = 0;
  let lastErr: unknown;

  while (Date.now() < deadline) {
    attempt += 1;
    onStatus?.(`Waking API… attempt ${attempt} (${Math.round((Date.now() - started) / 1000)}s)`);
    try {
      const res = await fetch(`${API_BASE}/api/health`, { cache: "no-store" });
      if (res.ok) {
        const health = (await res.json()) as Health;
        if (health.status === "ok") return health;
      }
    } catch (err) {
      lastErr = err;
    }
    await new Promise((r) => setTimeout(r, 4000));
  }

  const msg = lastErr instanceof Error ? lastErr.message : "timeout";
  throw new Error(apiHint(msg));
}

async function apiFetch(path: string, init?: RequestInit): Promise<Response> {
  const url = `${API_BASE}${path}`;
  let lastErr: unknown;
  for (let attempt = 1; attempt <= 2; attempt++) {
    try {
      return await fetch(url, init);
    } catch (err) {
      lastErr = err;
      if (attempt < 2) await new Promise((r) => setTimeout(r, 3000));
    }
  }
  const msg = lastErr instanceof Error ? lastErr.message : "network error";
  throw new Error(apiHint(msg));
}

export async function fetchHealth(): Promise<Health> {
  return wakeApi();
}

export async function predictImage(file: File, onStatus?: (msg: string) => void): Promise<Prediction> {
  await wakeApi(onStatus);
  onStatus?.("Running model…");
  const form = new FormData();
  form.append("file", file);
  const res = await apiFetch(`/api/predict?generate_report=false`, {
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
