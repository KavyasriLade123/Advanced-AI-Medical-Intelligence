from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import api_router
from app.config import DATA_DIR, HEATMAP_DIR, UPLOAD_DIR, ensure_directories, get_settings
from app.database import init_db
from app.ml import get_classifier

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_directories()
    init_db()
    get_classifier()
    yield


settings = get_settings()
ensure_directories()
app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins,
    allow_origin_regex=settings.cors_origin_regex,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.api_prefix)
app.mount("/api/media/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/api/media/heatmaps", StaticFiles(directory=str(HEATMAP_DIR)), name="heatmaps")


@app.get("/api/media/file")
def get_media_file(kind: str, name: str) -> FileResponse:
    base = UPLOAD_DIR if kind == "uploads" else HEATMAP_DIR if kind == "heatmaps" else None
    if base is None:
        raise HTTPException(status_code=400, detail="Invalid media kind.")
    path = (base / Path(name).name).resolve()
    if not str(path).startswith(str(base.resolve())) or not path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(path)


def _spa_index() -> FileResponse:
    index = STATIC_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=404, detail="Frontend not built.")
    return FileResponse(index)


if (STATIC_DIR / "index.html").is_file():
    assets_dir = STATIC_DIR / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="frontend-assets")

    @app.get("/")
    def spa_root() -> FileResponse:
        return _spa_index()

    @app.get("/analyze")
    @app.get("/about")
    def spa_pages() -> FileResponse:
        return _spa_index()

    @app.get("/{full_path:path}")
    def spa_fallback(full_path: str) -> FileResponse:
        # Do not swallow API/docs
        if full_path.startswith(("api/", "docs", "openapi", "redoc")):
            raise HTTPException(status_code=404, detail="Not Found")
        candidate = (STATIC_DIR / full_path).resolve()
        if str(candidate).startswith(str(STATIC_DIR.resolve())) and candidate.is_file():
            return FileResponse(candidate)
        return _spa_index()
else:

    @app.get("/")
    def root() -> dict:
        return {
            "name": settings.app_name,
            "docs": "/docs",
            "health": "/api/health",
        }
