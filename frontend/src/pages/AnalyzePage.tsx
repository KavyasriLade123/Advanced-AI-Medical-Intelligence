import { useCallback, useEffect, useRef, useState, type DragEvent } from "react";
import {
  fetchHistory,
  fetchPrediction,
  mediaUrl,
  predictImage,
  regenerateReport,
  type HistoryItem,
  type Prediction,
} from "../api";

function formatPct(n: number): string {
  return `${(n * 100).toFixed(1)}%`;
}

function formatWhen(iso: string): string {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

export default function AnalyzePage() {
  const [file, setFile] = useState<File | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<Prediction | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [historyTotal, setHistoryTotal] = useState(0);
  const workspaceRef = useRef<HTMLElement>(null);

  const refreshHistory = useCallback(async () => {
    try {
      const data = await fetchHistory(12);
      setHistory(data.items);
      setHistoryTotal(data.total);
    } catch {
      /* backend may be offline */
    }
  }, []);

  useEffect(() => {
    refreshHistory();
  }, [refreshHistory]);

  const onAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const prediction = await predictImage(file, (msg) => setError(msg));
      setError(null);
      setResult(prediction);
      await refreshHistory();
      workspaceRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      setResult(null);
      setError(
        err instanceof Error
          ? err.message
          : "Please upload a valid medical X-ray image.",
      );
    } finally {
      setLoading(false);
    }
  };

  const onPickHistory = async (id: number) => {
    setLoading(true);
    setError(null);
    try {
      const prediction = await fetchPrediction(id);
      setResult(prediction);
      setFile(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load prediction");
    } finally {
      setLoading(false);
    }
  };

  const onRegenReport = async () => {
    if (!result) return;
    setLoading(true);
    setError(null);
    try {
      const { report } = await regenerateReport(result.id);
      setResult({ ...result, report });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Report generation failed");
    } finally {
      setLoading(false);
    }
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const next = e.dataTransfer.files?.[0];
    if (next) setFile(next);
  };

  return (
    <main className="workspace" ref={workspaceRef}>
      <section className="panel">
        <h2>Image analysis</h2>
        <p className="lede">
          Upload a medical X-ray only. The system validates the image, detects the body part, then
          predicts disease. Non-X-ray uploads show:{" "}
          <strong>Please upload a valid medical X-ray image.</strong>
        </p>

        <div
          className={`dropzone ${dragOver ? "active" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
        >
          <input
            type="file"
            accept="image/png,image/jpeg,image/jpg,image/webp,image/bmp"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
          <div>
            <strong>Drop a medical X-ray here</strong>
            <span>or click to browse from your device</span>
            {file ? <div className="preview-name">{file.name}</div> : null}
          </div>
        </div>

        <div className="cta-row" style={{ marginTop: "1rem" }}>
          <button className="btn btn-primary" type="button" disabled={!file || loading} onClick={onAnalyze}>
            {loading ? "Please wait (waking API / running model)…" : "Predict + explain"}
          </button>
          {result ? (
            <button className="btn btn-ghost" type="button" disabled={loading} onClick={onRegenReport}>
              Regenerate report
            </button>
          ) : null}
        </div>

        {error ? <p className="error" style={{ marginTop: "1rem" }}>{error}</p> : null}
      </section>

      {result ? (
        <section className="panel grid-2">
          <div>
            <h2>Visual explanation</h2>
            <p className="lede">Original study beside Grad-CAM heatmap for this prediction.</p>
            <div className="image-pair">
              <figure className="image-frame">
                <img src={mediaUrl(result.image_url)} alt="Uploaded study" />
                <figcaption>Original</figcaption>
              </figure>
              <figure className="image-frame">
                {result.heatmap_url ? (
                  <img src={mediaUrl(result.heatmap_url)} alt="Grad-CAM heatmap" />
                ) : (
                  <div className="empty" style={{ padding: "2rem" }}>No heatmap</div>
                )}
                <figcaption>Grad-CAM</figcaption>
              </figure>
            </div>
          </div>

          <div className="result-meta">
            <div>
              <h2>Prediction</h2>
              <p className="lede">
                {result.is_xray !== false
                  ? `✅ ${result.body_part || "Medical"} X-ray detected.`
                  : "Result for this upload"}
              </p>
            </div>
            {result.body_part ? (
              <div className="metric">
                <span className="label">Body Part</span>
                <span className="value">{result.body_part}</span>
              </div>
            ) : null}
            <div className="metric">
              <span className="label">Prediction</span>
              <span
                className={`value ${
                  /pneumonia|fracture|tumor|covid|tuberculosis|malignant|abnormal/i.test(
                    result.disease || result.predicted_class,
                  )
                    ? "warn"
                    : ""
                }`}
              >
                {result.disease || result.predicted_class}
              </span>
            </div>
            <div className="metric">
              <span className="label">Confidence</span>
              <span className="value">{formatPct(result.confidence)}</span>
            </div>
            {result.recommendation ? (
              <div className="metric">
                <span className="label">Doctor recommendation</span>
                <span className="value" style={{ fontSize: "1rem", fontWeight: 600 }}>
                  {result.recommendation}
                </span>
              </div>
            ) : null}
            <div className="bars">
              <div className="bar-row">
                <header>
                  <span>{result.disease || result.predicted_class}</span>
                  <span>{formatPct(result.confidence)}</span>
                </header>
                <div className="track">
                  <div className="fill" style={{ width: `${Math.max(result.confidence * 100, 2)}%` }} />
                </div>
              </div>
            </div>
          </div>

          <div style={{ gridColumn: "1 / -1" }}>
            <h2>About this finding</h2>
            <p className="lede">Information for the predicted result of your upload only.</p>
            {result.disease_info ? (
              <div className="disease-card">
                <div className="disease-head">
                  <strong>{result.disease_info.title}</strong>
                  <span className="badge bone">{result.disease_info.body_region}</span>
                </div>
                <p>{result.disease_info.summary}</p>
                <p className="urgency">{result.disease_info.urgency}</p>
                <div className="disease-grid">
                  <div>
                    <h3>Related to this finding</h3>
                    <ul>
                      {result.disease_info.related_conditions.map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ul>
                  </div>
                  <div>
                    <h3>Symptoms to correlate</h3>
                    <ul>
                      {result.disease_info.common_symptoms_to_correlate.map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ul>
                  </div>
                  <div>
                    <h3>Typical imaging findings</h3>
                    <ul>
                      {result.disease_info.typical_xray_findings.map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ul>
                  </div>
                  <div>
                    <h3>Possible causes</h3>
                    <ul>
                      {result.disease_info.possible_causes_if_symptomatic.map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ul>
                  </div>
                  <div style={{ gridColumn: "1 / -1" }}>
                    <h3>Recommended next steps</h3>
                    <ul>
                      {result.disease_info.recommended_next_steps.map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ul>
                  </div>
                </div>
                <p className="disease-disclaimer">{result.disease_info.disclaimer}</p>
              </div>
            ) : (
              <p className="empty">No details available for this prediction.</p>
            )}
          </div>

          <div style={{ gridColumn: "1 / -1" }}>
            <h2>AI-assisted report</h2>
            <p className="lede">Report generated for this uploaded image.</p>
            <pre className="report-box">{result.report || "No report generated."}</pre>
          </div>
        </section>
      ) : null}

      <section className="panel" id="history-panel">
        <h2>Prediction history</h2>
        <p className="lede">
          {historyTotal} stored result{historyTotal === 1 ? "" : "s"} in SQLite.
        </p>
        {history.length === 0 ? (
          <p className="empty">No analyses yet. Upload an image to begin.</p>
        ) : (
          <div className="history-list">
            {history.map((item) => (
              <button
                key={item.id}
                type="button"
                className={`history-item ${result?.id === item.id ? "active" : ""}`}
                onClick={() => onPickHistory(item.id)}
              >
                <div>
                  <strong>{item.original_filename}</strong>
                  <small>
                    {formatWhen(item.created_at)} · {formatPct(item.confidence)}
                  </small>
                </div>
                <span
                  className={`badge ${
                    item.predicted_class.toUpperCase() === "PNEUMONIA"
                      ? "pneumonia"
                      : item.predicted_class.toUpperCase() === "BONE_FRACTURE"
                        ? "bone"
                        : item.predicted_class.toUpperCase().startsWith("BRAIN")
                          ? "brain"
                          : "normal"
                  }`}
                >
                  {item.predicted_class}
                </span>
              </button>
            ))}
          </div>
        )}
      </section>
    </main>
  );
}
