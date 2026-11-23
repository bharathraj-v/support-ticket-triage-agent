"""Settings, read from environment variables (optionally a local .env file)."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


@dataclass(frozen=True)
class Settings:
    helpdesk_url: str
    helpdesk_key: str
    candidate_email: str
    gemini_api_key: str
    gemini_model: str
    confidence_threshold: float
    state_path: Path
    log_path: Path
    faq_path: Path
    llm_cache_path: Path


def load_settings(need_llm: bool = True) -> Settings:
    return Settings(
        helpdesk_url=_require("HELPDESK_URL").rstrip("/"),
        helpdesk_key=_require("HELPDESK_KEY"),
        candidate_email=_require("CANDIDATE_EMAIL"),
        gemini_api_key=_require("GEMINI_API_KEY") if need_llm else os.environ.get("GEMINI_API_KEY", ""),
        gemini_model=_require("GEMINI_MODEL") if need_llm else os.environ.get("GEMINI_MODEL", ""),
        confidence_threshold=float(os.environ.get("CONFIDENCE_THRESHOLD", "0.7")),
        state_path=ROOT / "state" / "decisions.json",
        log_path=ROOT / "logs" / "run.jsonl",
        faq_path=ROOT / "data" / "helpcentre.md",
        llm_cache_path=ROOT / "state" / "llm_cache.jsonl",
    )
