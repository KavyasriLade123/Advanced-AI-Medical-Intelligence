import { NavLink, Outlet } from "react-router-dom";
import { useEffect, useState } from "react";
import { fetchHealth, type Health } from "../api";

export default function Layout() {
  const [health, setHealth] = useState<Health | null>(null);

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  const online = health?.status === "ok";

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
          {online ? `API online · ${health?.model_mode}` : "API offline — start the backend"}
        </div>
      </header>

      <Outlet />

      <p className="footnote">
        MedIntel is an educational decision-support prototype. Outputs must be reviewed by qualified clinicians.
      </p>
    </div>
  );
}
