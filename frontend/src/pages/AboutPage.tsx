import { Link } from "react-router-dom";

const objectives = [
  "Analyzing medical images",
  "Predicting diseases using Deep Learning",
  "Explaining predictions using Explainable AI (Grad-CAM)",
  "Generating AI-assisted medical reports using an LLM",
  "Providing REST APIs",
  "Storing prediction history in a database",
  "Deploying with a user-friendly interface",
];

export default function AboutPage() {
  return (
    <main className="workspace about-page">
      <section className="panel">
        <h2>About MedIntel</h2>
        <p className="lede">
          MedIntel is an end-to-end educational platform for medical image intelligence. Upload a
          study, receive a deep-learning prediction, inspect Grad-CAM explanations, and read an
          AI-assisted report — with history saved for review.
        </p>
      </section>

      <section className="panel">
        <h2>Project objectives</h2>
        <p className="lede">This application was built to satisfy the full platform objective set.</p>
        <ul className="about-list">
          {objectives.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </section>

      <section className="panel grid-2 about-grid">
        <div>
          <h2>How it works</h2>
          <ol className="about-list numbered">
            <li>Upload a supported medical image (JPG/PNG).</li>
            <li>ResNet18 predicts the most likely finding.</li>
            <li>Grad-CAM highlights regions that influenced the decision.</li>
            <li>An LLM/template report and disease card are generated.</li>
            <li>Results are stored in SQLite for history.</li>
          </ol>
        </div>
        <div>
          <h2>Tech stack</h2>
          <ul className="about-list">
            <li>Frontend: React + TypeScript + Vite</li>
            <li>Backend: FastAPI + PyTorch</li>
            <li>XAI: Grad-CAM</li>
            <li>Reports: OpenAI-compatible LLM + template fallback</li>
            <li>Database: SQLite (SQLAlchemy)</li>
            <li>Deploy: Docker Compose</li>
          </ul>
        </div>
      </section>

      <section className="panel">
        <h2>Important disclaimer</h2>
        <p className="lede">
          MedIntel is for academic and decision-support demonstration only. It is <strong>not</strong> a
          medical diagnosis tool. All outputs must be reviewed by qualified clinicians.
        </p>
        <div className="cta-row" style={{ marginTop: "1rem" }}>
          <Link className="btn btn-primary" to="/analyze">
            Go to Image Analysis
          </Link>
        </div>
      </section>
    </main>
  );
}
