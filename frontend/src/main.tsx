import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { initApiBase } from "./api";
import "./index.css";

async function boot() {
  await initApiBase();
  createRoot(document.getElementById("root")!).render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
}

void boot();
