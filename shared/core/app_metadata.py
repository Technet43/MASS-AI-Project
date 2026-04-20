from __future__ import annotations

import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

APP_NAME = "MASS-AI Desktop"
APP_VERSION = "1.3.0"
BUILD_CHANNEL = "desktop-pilot"


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def build_timestamp() -> str:
    override = os.environ.get("MASS_AI_BUILD_TIME")
    if override:
        return override
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def metadata_dict() -> dict[str, Any]:
    return {
        "app_name": APP_NAME,
        "version": APP_VERSION,
        "build_channel": BUILD_CHANNEL,
        "build_timestamp": build_timestamp(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "workspace": str(project_root()),
    }


def version_label() -> str:
    return f"v{APP_VERSION}"


def support_bundle_name(prefix: str = "mass_ai_support_bundle") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
