"""Generate docs/Project_Report.pdf — formal project document for submission."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "Project_Report.pdf"

ACCENT = colors.HexColor("#0B3A5C")
LIGHT = colors.HexColor("#E8F1F8")
MUTED = colors.HexColor("#4A5568")


def styles():
    base = getSampleStyleSheet()
    return {
        "cover_title": ParagraphStyle(
            "cover_title",
            parent=base["Title"],
            fontSize=22,
            leading=28,
            textColor=ACCENT,
            alignment=TA_CENTER,
            spaceAfter=12,
        ),
        "cover_sub": ParagraphStyle(
            "cover_sub",
            parent=base["Normal"],
            fontSize=12,
            leading=16,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontSize=14,
            leading=18,
            textColor=ACCENT,
            spaceBefore=14,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontSize=12,
            leading=15,
            textColor=ACCENT,
            spaceBefore=10,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["Normal"],
            fontSize=10,
            leading=14,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            parent=base["Normal"],
            fontSize=10,
            leading=13,
            leftIndent=4,
        ),
        "footer": ParagraphStyle(
            "footer",
            parent=base["Normal"],
            fontSize=8,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
        "meta": ParagraphStyle(
            "meta",
            parent=base["Normal"],
            fontSize=10,
            leading=14,
            alignment=TA_CENTER,
            textColor=MUTED,
        ),
        "table_cell": ParagraphStyle(
            "table_cell",
            parent=base["Normal"],
            fontSize=9,
            leading=12,
        ),
    }


def bullets(items: list[str], s) -> ListFlowable:
    return ListFlowable(
        [ListItem(Paragraph(i, s["bullet"]), leftIndent=12, value="bullet") for i in items],
        bulletType="bullet",
        start="•",
        leftIndent=18,
        spaceBefore=2,
        spaceAfter=8,
    )


def simple_table(rows: list[list[str]], s, col_widths=None) -> Table:
    data = [[Paragraph(c, s["table_cell"]) for c in row] for row in rows]
    t = Table(data, colWidths=col_widths or [2.2 * inch, 4.3 * inch])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 1), (-1, -1), LIGHT),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#C5D6E6")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return t


def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawCentredString(
        A4[0] / 2,
        0.55 * inch,
        f"MedIntel Project Report  |  Page {doc.page}",
    )
    canvas.restoreState()


def build():
    s = styles()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.85 * inch,
        title="Advanced AI Medical Intelligence Platform — Project Report",
        author="MedIntel Team",
    )

    story = []

    # ----- Cover -----
    story.append(Spacer(1, 1.4 * inch))
    story.append(Paragraph("PROJECT DOCUMENT / TECHNICAL REPORT", s["cover_sub"]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(
        Paragraph(
            "Advanced AI Medical Intelligence Platform<br/>(MedIntel)",
            s["cover_title"],
        )
    )
    story.append(
        Paragraph(
            "End-to-end Deep Learning system for medical image analysis,<br/>"
            "explainable AI, LLM-assisted reporting, REST APIs, and web deployment",
            s["cover_sub"],
        )
    )
    story.append(Spacer(1, 0.5 * inch))
    story.append(
        Paragraph(
            "GitHub: https://github.com/KavyasriLade123/Advanced-AI-Medical-Intelligence",
            s["meta"],
        )
    )
    story.append(Paragraph("Document type: Academic / submission project report", s["meta"]))
    story.append(Paragraph("Disclaimer: Educational decision support only — not clinical diagnosis", s["meta"]))
    story.append(PageBreak())

    # ----- 1. Introduction -----
    story.append(Paragraph("1. Introduction", s["h1"]))
    story.append(
        Paragraph(
            "Medical imaging is central to modern diagnosis, but interpreting scans at scale is "
            "time-consuming and expertise-intensive. This project builds an integrated software "
            "platform that accepts a medical image, predicts a finding using a deep neural network, "
            "explains the prediction visually with Grad-CAM, generates an AI-assisted textual report, "
            "exposes results through REST APIs, stores history in a database, and presents everything "
            "in a user-friendly React web interface.",
            s["body"],
        )
    )
    story.append(
        Paragraph(
            "The system is designed as an academic demonstration of full-stack AI engineering: "
            "data preprocessing, model inference, explainability, generative reporting, API design, "
            "persistence, containerization, and frontend UX — not as a certified clinical device.",
            s["body"],
        )
    )

    # ----- 2. Objectives -----
    story.append(Paragraph("2. Project Objectives", s["h1"]))
    story.append(
        Paragraph("The platform was built to satisfy the following end-to-end capabilities:", s["body"])
    )
    story.append(
        bullets(
            [
                "Analyze uploaded medical images (preprocess and normalize for CNN input).",
                "Predict findings using Deep Learning (fine-tuned ResNet18 classifier).",
                "Explain predictions using Explainable AI (Grad-CAM heatmaps).",
                "Generate AI-assisted medical-style reports using an LLM (with template fallback).",
                "Provide REST APIs for prediction, history, reports, and health checks.",
                "Store prediction history in a relational database (SQLite).",
                "Deploy with a user-friendly interface (React) and optional Docker Compose.",
            ],
            s,
        )
    )

    # ----- 3. Scope -----
    story.append(Paragraph("3. Scope and Supported Findings", s["h1"]))
    story.append(
        Paragraph(
            "The trained classifier supports multi-region medical image triage with the following labels:",
            s["body"],
        )
    )
    story.append(
        bullets(
            [
                "Chest: NORMAL, PNEUMONIA",
                "Brain: BRAIN_NORMAL, BRAIN_TUMOR",
                "Breast: BREAST_NORMAL, BREAST_MALIGNANT",
                "Other: ABDOMEN, BONE_FRACTURE, EYE_RETINA, LOWER_LIMB, SKIN",
                "Reject class: UNSUPPORTED (plus low-confidence rejection messaging)",
            ],
            s,
        )
    )
    story.append(
        Paragraph(
            "Unrelated or low-confidence uploads are rejected with a clear message asking the user "
            "to upload a correct medical image from supported body regions.",
            s["body"],
        )
    )

    # ----- 4. Architecture -----
    story.append(Paragraph("4. System Architecture", s["h1"]))
    story.append(
        Paragraph(
            "MedIntel follows a client–server architecture. The React frontend communicates with a "
            "FastAPI backend. The backend loads a PyTorch ResNet18 checkpoint, runs inference and "
            "Grad-CAM, optionally calls an OpenAI-compatible LLM for report text, and persists records "
            "in SQLite via SQLAlchemy.",
            s["body"],
        )
    )
    story.append(Paragraph("4.1 High-level flow", s["h2"]))
    story.append(
        bullets(
            [
                "User opens the web UI (Home / About / Analyze).",
                "User uploads a JPG/PNG medical image on the Analyze page.",
                "Frontend POSTs the file to <b>/api/predict</b>.",
                "Backend preprocesses the image (RGB, 224×224, ImageNet normalize).",
                "ResNet18 predicts class probabilities; Grad-CAM produces a heatmap overlay.",
                "Disease info card and LLM/template report are assembled for the top finding.",
                "Result is stored in SQLite and returned as JSON (prediction, confidence, heatmap URL, report).",
                "UI displays only the predicted finding (not a full class bar chart).",
            ],
            s,
        )
    )
    story.append(Paragraph("4.2 Technology stack", s["h2"]))
    story.append(
        simple_table(
            [
                ["Layer", "Technologies"],
                ["Frontend", "React, TypeScript, Vite"],
                ["Backend", "FastAPI, Uvicorn, Pydantic, SQLAlchemy"],
                ["Deep Learning", "PyTorch, Torchvision (ResNet18)"],
                ["Explainability", "Grad-CAM on ResNet layer4"],
                ["LLM reports", "OpenAI-compatible chat API + template fallback"],
                ["Database", "SQLite"],
                ["Packaging", "Docker, Docker Compose, nginx (frontend image)"],
            ],
            s,
        )
    )

    story.append(PageBreak())

    # ----- 5. Deep Learning -----
    story.append(Paragraph("5. Deep Learning Model", s["h1"]))
    story.append(
        Paragraph(
            "The classifier is a ResNet18 network pretrained on ImageNet, with a replaced fully "
            "connected head for the project’s multi-class medical labels. Weights are stored at "
            "<b>backend/models/chest_xray_resnet18.pth</b>.",
            s["body"],
        )
    )
    story.append(Paragraph("5.1 Training approach", s["h2"]))
    story.append(
        bullets(
            [
                "Input: RGB images resized to 224×224 with ImageNet mean/std normalization.",
                "Loss: class-weighted CrossEntropy to mitigate class imbalance.",
                "Optimizer: Adam.",
                "Pipeline: head-only warm-up followed by full fine-tuning.",
                "Training entrypoint: <b>python -m app.ml.train</b> (see backend ML module).",
            ],
            s,
        )
    )
    story.append(Paragraph("5.2 Inference safeguards", s["h2"]))
    story.append(
        Paragraph(
            "A minimum confidence threshold (default 0.50) and an UNSUPPORTED class help reject "
            "non-medical or out-of-distribution images. This improves UX honesty and reduces "
            "misleading high-confidence errors on unrelated photos.",
            s["body"],
        )
    )

    # ----- 6. XAI -----
    story.append(Paragraph("6. Explainable AI (Grad-CAM)", s["h1"]))
    story.append(
        Paragraph(
            "Grad-CAM computes class-discriminative localization maps from gradients flowing into "
            "the final convolutional block (ResNet <b>layer4</b>). The heatmap is overlaid on the "
            "original image and returned to the client so users can see which regions most influenced "
            "the model’s decision. This supports transparency and educational review of model focus.",
            s["body"],
        )
    )

    # ----- 7. LLM -----
    story.append(Paragraph("7. AI-Assisted Medical Reports", s["h1"]))
    story.append(
        Paragraph(
            "After prediction, the backend builds a structured disease-information card for the "
            "predicted label and generates a narrative report. If an OpenAI-compatible API key is "
            "configured, the system requests a concise clinical-style summary from the LLM. If the "
            "key is missing or the call fails, a deterministic template report is returned so the "
            "demo remains fully functional offline.",
            s["body"],
        )
    )

    # ----- 8. APIs -----
    story.append(Paragraph("8. REST API Design", s["h1"]))
    story.append(
        Paragraph(
            "APIs are served under the <b>/api</b> prefix. Interactive OpenAPI docs are available at "
            "<b>/docs</b> when the backend is running.",
            s["body"],
        )
    )
    story.append(
        simple_table(
            [
                ["Endpoint", "Purpose"],
                ["/api/health", "Service health / readiness check"],
                ["/api/predict", "Upload image → prediction, Grad-CAM, report"],
                ["/api/history", "List stored prediction records"],
                ["/api/reports", "Report-related retrieval / generation helpers"],
            ],
            s,
            col_widths=[1.8 * inch, 4.7 * inch],
        )
    )

    # ----- 9. Database -----
    story.append(Paragraph("9. Database and Persistence", s["h1"]))
    story.append(
        Paragraph(
            "Prediction history is stored in SQLite using SQLAlchemy. Typical fields include "
            "timestamp, predicted class, confidence, image path, heatmap path, and report text. "
            "This enables auditability of demo sessions and a history view in the UI/API.",
            s["body"],
        )
    )

    # ----- 10. Frontend -----
    story.append(Paragraph("10. Frontend Application", s["h1"]))
    story.append(
        Paragraph(
            "The frontend is a React + TypeScript SPA built with Vite and React Router. Pages include:",
            s["body"],
        )
    )
    story.append(
        bullets(
            [
                "<b>Home (/)</b> — product introduction and call-to-action to analyze images.",
                "<b>About (/about)</b> — project purpose, capabilities, and disclaimer.",
                "<b>Analyze (/analyze)</b> — image upload, prediction result, Grad-CAM, report.",
            ],
            s,
        )
    )
    story.append(
        Paragraph(
            "Development servers can bind to <b>0.0.0.0</b> so the UI is reachable on the local "
            "network (e.g., http://192.168.x.x:5173). Vite proxies <b>/api</b> to the FastAPI backend "
            "on port 8000.",
            s["body"],
        )
    )

    story.append(PageBreak())

    # ----- 11. Docker -----
    story.append(Paragraph("11. Deployment and Docker", s["h1"]))
    story.append(
        Paragraph(
            "Containerization is implemented for reproducible deployment:",
            s["body"],
        )
    )
    story.append(
        bullets(
            [
                "<b>backend/Dockerfile</b> — Python 3.11 image running Uvicorn on port 8000.",
                "<b>frontend/Dockerfile</b> — multi-stage build serving static assets via nginx.",
                "<b>docker-compose.yml</b> — orchestrates backend and frontend together.",
            ],
            s,
        )
    )
    story.append(
        Paragraph(
            "Public cloud live deployment is optional and may be added later (Render, Railway, Azure, "
            "etc.). Local and LAN demos satisfy academic evaluation when Docker/cloud is unavailable.",
            s["body"],
        )
    )

    # ----- 12. Structure -----
    story.append(Paragraph("12. Repository Structure (Summary)", s["h1"]))
    story.append(
        bullets(
            [
                "<b>backend/app/</b> — FastAPI app, ML, routes, services, schemas, database.",
                "<b>backend/models/</b> — trained PyTorch weights (.pth).",
                "<b>frontend/src/</b> — React pages and components.",
                "<b>docs/</b> — this PDF, model card, submission notes.",
                "<b>scripts/</b> — helpers including this PDF generator.",
                "<b>requirements.txt</b> — root wrapper installing backend dependencies.",
            ],
            s,
        )
    )

    # ----- 13. Submission -----
    story.append(Paragraph("13. Submission Deliverables", s["h1"]))
    story.append(
        simple_table(
            [
                ["Deliverable", "Location / Link"],
                ["Complete source code", "GitHub repository (backend + frontend)"],
                ["Trained model", "backend/models/chest_xray_resnet18.pth"],
                ["GitHub link", "github.com/KavyasriLade123/Advanced-AI-Medical-Intelligence"],
                ["README", "README.md"],
                ["PDF project report", "docs/Project_Report.pdf (this document)"],
                ["requirements.txt", "requirements.txt + backend/requirements.txt"],
                ["Dockerfile", "backend/Dockerfile, frontend/Dockerfile, compose"],
                ["Live deployment", "Optional — local/LAN demo if not cloud-hosted"],
            ],
            s,
        )
    )

    # ----- 14. Results -----
    story.append(Paragraph("14. Results and Observations", s["h1"]))
    story.append(
        Paragraph(
            "The integrated pipeline successfully performs upload → inference → Grad-CAM → report → "
            "history in a single user flow. Multi-class expansion beyond binary chest pneumonia "
            "improves coverage across common imaging domains used in demos. Rejection of unsupported "
            "images reduces false clinical-sounding outputs on random photographs.",
            s["body"],
        )
    )
    story.append(
        Paragraph(
            "Limitations include dependence on training data quality/diversity, possible confusion "
            "between visually similar modalities, and the fact that template/LLM text is assistive "
            "narrative rather than a radiologist report. Performance metrics should be re-evaluated "
            "whenever the class set or dataset is updated.",
            s["body"],
        )
    )

    # ----- 15. Ethics -----
    story.append(Paragraph("15. Ethics, Safety, and Disclaimer", s["h1"]))
    story.append(
        Paragraph(
            "This project is intended strictly for education, research demonstration, and software "
            "engineering evaluation. It must not be used for real patient diagnosis, triage decisions, "
            "or clinical treatment planning. Any LLM-generated text may contain inaccuracies and must "
            "be reviewed by qualified professionals in any real-world medical context.",
            s["body"],
        )
    )

    # ----- 16. Conclusion -----
    story.append(Paragraph("16. Conclusion and Future Work", s["h1"]))
    story.append(
        Paragraph(
            "MedIntel demonstrates a complete Advanced AI Medical Intelligence stack: deep learning "
            "prediction, visual explainability, generative reporting, APIs, persistence, and a modern "
            "web UI with Docker packaging. Future improvements may include larger curated multi-modal "
            "datasets, stronger calibration, DICOM support, role-based auth, cloud deployment with "
            "HTTPS, and formal clinical validation studies.",
            s["body"],
        )
    )

    story.append(Spacer(1, 0.35 * inch))
    story.append(
        KeepTogether(
            [
                Paragraph("— End of Project Document —", s["meta"]),
                Paragraph(
                    "Generated for submission · Advanced AI Medical Intelligence Platform (MedIntel)",
                    s["meta"],
                ),
            ]
        )
    )

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    build()
