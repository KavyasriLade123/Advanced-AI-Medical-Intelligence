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

/** Empty in local Vite (proxy /api). Set VITE_API_BASE_URL on Vercel to the Render API origin. */
const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

async function readError(res: Response): Promise<string> {
  try {
    const data = await res.json();
    if (typeof data.detail === "string") return data.detail;
    return JSON.stringify(data.detail ?? data);
  } catch {
    return res.statusText;
  }
}

export async function fetchHealth(): Promise<Health> {
  const res = await fetch(`${API_BASE}/api/health`);
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export async function predictImage(file: File): Promise<Prediction> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/api/predict?generate_report=true`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export async function fetchHistory(limit = 12): Promise<HistoryList> {
  const res = await fetch(`${API_BASE}/api/history?limit=${limit}`);
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export async function fetchPrediction(id: number): Promise<Prediction> {
  const res = await fetch(`${API_BASE}/api/history/${id}`);
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export async function regenerateReport(id: number): Promise<{ report: string; source: string }> {
  const res = await fetch(`${API_BASE}/api/reports`, {
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
