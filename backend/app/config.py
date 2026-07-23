from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
HEATMAP_DIR = DATA_DIR / "heatmaps"
MODEL_DIR = BASE_DIR / "models"
WEIGHTS_PATH = MODEL_DIR / "chest_xray_resnet18.pth"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "MedIntel - Advanced AI Medical Intelligence Platform"
    api_prefix: str = "/api"
    database_url: str = f"sqlite:///{(DATA_DIR / 'medintel.db').as_posix()}"
    cors_origins: str = (
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,"
        "http://192.168.0.8:5173"
    )
    cors_origin_regex: str = r"https?://(localhost|127\.0\.0\.1|192\.168\.\d+\.\d+)(:\d+)?"

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"

    class_names: str = (
        "ABDOMEN,BONE_FRACTURE,BRAIN_NORMAL,BRAIN_TUMOR,BREAST_MALIGNANT,"
        "BREAST_NORMAL,EYE_RETINA,LOWER_LIMB,NORMAL,PNEUMONIA,SKIN,UNSUPPORTED"
    )
    image_size: int = 224
    min_confidence: float = 0.50
    supported_hint: str = (
        "Upload a correct medical image from supported body regions: "
        "Brain, Eye/Retina, Breast, Chest X-ray, Abdomen CT, Skin, "
        "Bone fracture, or Lower limb. Non-medical or unrelated images are rejected."
    )

    @property
    def origins(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def labels(self) -> list[str]:
        return [c.strip() for c in self.class_names.split(",") if c.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


def ensure_directories() -> None:
    for path in (DATA_DIR, UPLOAD_DIR, HEATMAP_DIR, MODEL_DIR):
        path.mkdir(parents=True, exist_ok=True)
