import { NavLink, Outlet } from "react-router-dom";
import { useEffect, useState } from "react";
import { fetchHealth, getApiBase, type Health } from "../api";

export default function Layout() {
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);

  useEffect(() => {
    fetchHealth()
      .then((h) => {
        setHealth(h);
        setHealthError(null);
      })
      .catch((err) => {
        setHealth(null);
        setHealthError(err instanceof Error ? err.message : "API offline");
      });
  }, []);

  const online = health?.status === "ok";
  const apiBase = getApiBase();

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            Med<span>Intel</span>
          </div>
          <div className="brand-sub">Advanced AI Medical Intelligence Platform</div>
        </div>

        <nav className="main-nav" aria-label="Primary">
          <NavLink to="/" end className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}>
            Home
          </NavLink>
          <NavLink to="/about" className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}>
            About
          </NavLink>
          <NavLink to="/analyze" className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}>
            Image Analysis
          </NavLink>
        </nav>

        <div className="status-pill">
          <span className={`status-dot ${online ? "" : "offline"}`} />
          {online ? `API online · ${health?.model_mode}` : "API waking / offline"}
        </div>
      </header>

      {!online ? (
        <p className="error" style={{ margin: "0 1.25rem 0.75rem" }}>
          {healthError ||
            `API is waking up or offline${apiBase ? ` (${apiBase})` : ""}. Open https://advanced-ai-medical-intelligence-4og2.onrender.com/api/health, wait for JSON, then refresh this page.`}
        </p>
      ) : null}

      <Outlet />

      <p className="footnote">
        MedIntel is an educational decision-support prototype. Outputs must be reviewed by qualified clinicians.
      </p>
    </div>
  );
}
