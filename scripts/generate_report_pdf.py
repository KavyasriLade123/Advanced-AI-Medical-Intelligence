"""Generate docs/Project_Report.pdf for submission."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
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


def build() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        fontSize=18,
        spaceAfter=12,
    )
    h1 = ParagraphStyle("H1Custom", parent=styles["Heading1"], fontSize=14, spaceBefore=14, spaceAfter=8)
    h2 = ParagraphStyle("H2Custom", parent=styles["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=6)
    body = ParagraphStyle("BodyCustom", parent=styles["BodyText"], fontSize=10, leading=14, spaceAfter=6)

    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title="Advanced AI Medical Intelligence Platform — Project Report",
    )
    story = []

    story.append(Paragraph("Advanced AI Medical Intelligence Platform", title))
    story.append(Paragraph("Project Report (Academic / Submission)", styles["Heading2"]))
    story.append(Paragraph("<b>Product name:</b> MedIntel", body))
    story.append(
        Paragraph(
            "This report documents an end-to-end AI system for medical image analysis, "
            "disease prediction with deep learning, Grad-CAM explanations, LLM-assisted "
            "reporting, REST APIs, persistent prediction history, and a React web interface.",
            body,
        )
    )

    story.append(Paragraph("1. Project objective", h1))
    story.append(
        Paragraph(
            "Build a complete application capable of: analyzing medical images; predicting "
            "diseases using deep learning; explaining predictions with Grad-CAM; generating "
            "AI-assisted reports with an LLM; exposing REST APIs; storing history in a database; "
            "and deploying a user-friendly interface.",
            body,
        )
    )

    story.append(Paragraph("2. System architecture", h1))
    story.append(
        Paragraph(
            "The system uses a layered design: (1) React frontend for upload and visualization; "
            "(2) FastAPI backend for orchestration; (3) PyTorch ResNet18 classifier; "
            "(4) Grad-CAM explainer; (5) LLM/template report service; (6) SQLite persistence.",
            body,
        )
    )
    arch = [
        ["Layer", "Technology"],
        ["Frontend", "React + TypeScript + Vite"],
        ["Backend API", "FastAPI + Uvicorn"],
        ["Deep learning", "PyTorch ResNet18 (fine-tuned)"],
        ["XAI", "Grad-CAM on ResNet layer4"],
        ["LLM reports", "OpenAI-compatible API + template fallback"],
        ["Database", "SQLite via SQLAlchemy"],
        ["Deployment", "Docker + Docker Compose"],
    ]
    t = Table(arch, colWidths=[1.6 * inch, 4.2 * inch])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123040")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.Color(0.93, 0.95, 0.96)]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(t)

    story.append(Paragraph("3. Deep learning model", h1))
    story.append(
        Paragraph(
            "A ResNet18 backbone pretrained on ImageNet is fine-tuned for multi-class medical "
            "image triage. Supported classes include brain (normal/tumor), chest (normal/pneumonia), "
            "bone fracture, abdomen, breast, eye/retina, skin, lower limb, and an UNSUPPORTED class "
            "used to reject unrelated uploads.",
            body,
        )
    )
    story.append(Paragraph("Training pipeline highlights:", h2))
    story.append(
        ListFlowable(
            [
                ListItem(Paragraph("Data preparation scripts for chest, bone, brain, MedMNIST body regions", body)),
                ListItem(Paragraph("Augmentation: flip, rotation, color jitter", body)),
                ListItem(Paragraph("Class-weighted cross-entropy for imbalance", body)),
                ListItem(Paragraph("Head-only warm-up then full fine-tuning", body)),
                ListItem(Paragraph("Best validation checkpoint saved to <font face='Courier'>backend/models/chest_xray_resnet18.pth</font>", body)),
            ],
            bulletType="bullet",
        )
    )

    story.append(Paragraph("4. Explainable AI (Grad-CAM)", h1))
    story.append(
        Paragraph(
            "Grad-CAM computes class-discriminative localization maps from gradients flowing into "
            "the final convolutional block. The heatmap is overlaid on the original image and served "
            "via the media API so clinicians/users can see which regions influenced the prediction.",
            body,
        )
    )

    story.append(Paragraph("5. LLM-assisted medical reports", h1))
    story.append(
        Paragraph(
            "After prediction, the backend generates a structured report. If <font face='Courier'>OPENAI_API_KEY</font> "
            "is configured, an OpenAI-compatible chat completion is used. Otherwise a professional "
            "template report is produced. The UI also shows a disease-information card scoped to the "
            "predicted finding only.",
            body,
        )
    )

    story.append(Paragraph("6. REST API design", h1))
    api_rows = [
        ["Method", "Endpoint", "Purpose"],
        ["GET", "/api/health", "Health + model status"],
        ["POST", "/api/predict", "Predict + Grad-CAM + report"],
        ["GET", "/api/history", "List stored predictions"],
        ["GET", "/api/history/{id}", "Fetch one prediction"],
        ["POST", "/api/reports", "Regenerate report"],
        ["DELETE", "/api/history/{id}", "Delete record"],
    ]
    api = Table(api_rows, colWidths=[0.8 * inch, 1.8 * inch, 3.2 * inch])
    api.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123040")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.Color(0.93, 0.95, 0.96)]),
            ]
        )
    )
    story.append(api)

    story.append(Paragraph("7. Database design", h1))
    story.append(
        Paragraph(
            "SQLite stores prediction history in table <font face='Courier'>predictions</font> with fields: "
            "id, filename, original_filename, predicted_class, confidence, class_probabilities (JSON text), "
            "heatmap_path, report_text, model_mode, created_at.",
            body,
        )
    )

    story.append(Paragraph("8. Web application", h1))
    story.append(
        Paragraph(
            "The React UI provides drag-and-drop upload, prediction display (finding + confidence only), "
            "Grad-CAM visualization, disease information for the predicted class, AI report text, and "
            "history browsing. Unsupported images are rejected with a clear message to upload a correct "
            "medical image.",
            body,
        )
    )

    story.append(Paragraph("9. Deployment", h1))
    story.append(
        Paragraph(
            "Dockerfiles exist for backend and frontend. <font face='Courier'>docker compose up --build</font> "
            "runs the stack. Local Windows scripts are also provided under <font face='Courier'>scripts/</font>. "
            "Public cloud deployment is optional for submission.",
            body,
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("10. Evaluation criteria alignment", h1))
    eval_rows = [
        ["Criterion", "Evidence in project"],
        ["DL model performance", "Fine-tuned ResNet18 + validation checkpoint"],
        ["Code quality / structure", "Modular backend & frontend packages"],
        ["Explainable AI", "Grad-CAM heatmaps in API/UI"],
        ["LLM integration", "OpenAI-compatible service + fallback"],
        ["API development", "FastAPI OpenAPI docs at /docs"],
        ["Database design", "SQLAlchemy prediction history schema"],
        ["Web application", "React medical triage UI"],
        ["Documentation", "README + this PDF report"],
        ["Deployment", "Dockerfile(s) + docker-compose.yml"],
        ["System design / best practices", "Layered architecture, reject unsupported inputs"],
    ]
    ev = Table(eval_rows, colWidths=[2.0 * inch, 3.8 * inch])
    ev.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123040")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.Color(0.93, 0.95, 0.96)]),
            ]
        )
    )
    story.append(ev)

    story.append(Paragraph("11. Ethical disclaimer", h1))
    story.append(
        Paragraph(
            "MedIntel is an educational decision-support prototype. Predictions and reports are <b>not</b> "
            "clinical diagnoses and must be reviewed by licensed medical professionals before any clinical action.",
            body,
        )
    )

    story.append(Paragraph("12. Submission artifacts", h1))
    story.append(
        ListFlowable(
            [
                ListItem(Paragraph("Complete source code in repository", body)),
                ListItem(Paragraph("Trained model: backend/models/chest_xray_resnet18.pth", body)),
                ListItem(Paragraph("README.md documentation", body)),
                ListItem(Paragraph("This PDF project report", body)),
                ListItem(Paragraph("requirements.txt (+ backend/requirements.txt)", body)),
                ListItem(Paragraph("Dockerfile(s) and docker-compose.yml", body)),
                ListItem(Paragraph("GitHub repository link (add after push)", body)),
                ListItem(Paragraph("Live deployment link (optional)", body)),
            ],
            bulletType="bullet",
        )
    )

    story.append(Spacer(1, 16))
    story.append(Paragraph("— End of report —", ParagraphStyle("End", parent=body, alignment=1)))

    doc.build(story)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()
