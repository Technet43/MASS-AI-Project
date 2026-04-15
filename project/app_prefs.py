from __future__ import annotations

import json
import os
from pathlib import Path


def resolve_app_prefs_dir() -> Path:
    if os.name == "nt":
        base_dir = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "MASS-AI"
    else:
        base_dir = Path.home() / ".mass-ai"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def resolve_theme_prefs_path() -> Path:
    return resolve_app_prefs_dir() / "mass_ai_prefs.json"


def load_prefs(path: str | os.PathLike[str] | None = None) -> dict:
    prefs_path = Path(path) if path else resolve_theme_prefs_path()
    if not prefs_path.exists():
        return {}
    try:
        return json.loads(prefs_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_prefs(data: dict, path: str | os.PathLike[str] | None = None) -> Path:
    prefs_path = Path(path) if path else resolve_theme_prefs_path()
    prefs_path.parent.mkdir(parents=True, exist_ok=True)
    prefs_path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")
    return prefs_path


def load_theme_preference(path: str | os.PathLike[str] | None = None) -> str | None:
    theme_name = load_prefs(path).get("theme")
    return str(theme_name).strip() if theme_name else None


def save_theme_preference(theme_name: str, path: str | os.PathLike[str] | None = None) -> Path:
    prefs = load_prefs(path)
    prefs["theme"] = str(theme_name).strip()
    return save_prefs(prefs, path)
