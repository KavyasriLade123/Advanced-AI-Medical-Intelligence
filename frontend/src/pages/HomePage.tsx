import { Link } from "react-router-dom";

export default function HomePage() {
  return (
    <section className="hero home-hero">
      <div className="hero-copy">
        <p className="hero-kicker">Advanced AI Medical Intelligence Platform</p>
        <h1>MedIntel</h1>
        <p>
          Deep learning triage for medical images — with Grad-CAM explanations and AI-assisted
          clinical reports in one workspace.
        </p>
        <div className="cta-row">
          <Link className="btn btn-primary" to="/analyze">
            Start image analysis
          </Link>
          <Link className="btn btn-ghost" to="/about">
            About the platform
          </Link>
        </div>
      </div>
    </section>
  );
}
