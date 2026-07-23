import { NavLink, Outlet } from "react-router-dom";
import { useEffect, useState } from "react";
import { fetchHealth, getApiBase, isApiConfigured, type Health } from "../api";

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
          {online
            ? `API online · ${health?.model_mode}`
            : !isApiConfigured()
              ? "API not configured"
              : "API offline — start Render backend"}
        </div>
      </header>

      {!online ? (
        <p className="error" style={{ margin: "0 1.25rem 0.75rem" }}>
          {healthError ||
            (!isApiConfigured()
              ? "Set VITE_API_BASE_URL on Vercel to your Render URL (e.g. https://….onrender.com), then Redeploy."
              : `Cannot reach API${apiBase ? ` at ${apiBase}` : ""}. Deploy/start the Render backend and allow this site in CORS_ORIGINS.`)}
        </p>
      ) : null}

      <Outlet />

      <p className="footnote">
        MedIntel is an educational decision-support prototype. Outputs must be reviewed by qualified clinicians.
      </p>
    </div>
  );
}
