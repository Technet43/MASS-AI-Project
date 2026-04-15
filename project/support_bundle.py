from __future__ import annotations

import json
import traceback
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from app_metadata import metadata_dict


def _safe_json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, default=str)


def create_support_bundle(
    bundle_path: str | Path,
    *,
    theme_name: str,
    overview: dict[str, Any] | None,
    ops_metrics: dict[str, Any] | None,
    selected_case: dict[str, Any] | None,
    selected_notes: list[dict[str, Any]] | None,
    case_history: list[dict[str, Any]] | None,
    log_lines: list[str] | None,
    current_df: pd.DataFrame | None = None,
    extra_sections: dict[str, Any] | None = None,
) -> Path:
    bundle_file = Path(bundle_path)
    bundle_file.parent.mkdir(parents=True, exist_ok=True)
    metadata = metadata_dict()
    payload = {
        "metadata": metadata,
        "theme": theme_name,
        "overview": overview or {},
        "ops_metrics": ops_metrics or {},
        "selected_case": selected_case or {},
        "selected_notes": selected_notes or [],
        "case_history": case_history or [],
        "extra_sections": extra_sections or {},
    }
    summary_lines = [
        f"{metadata['app_name']} {metadata['version']}",
        f"Build channel: {metadata['build_channel']}",
        f"Generated: {metadata['build_timestamp']}",
        f"Theme: {theme_name}",
        "",
        "Overview:",
        _safe_json(overview or {}),
        "",
        "Ops metrics:",
        _safe_json(ops_metrics or {}),
        "",
        "Selected case:",
        _safe_json(selected_case or {}),
    ]
    if selected_notes:
        summary_lines.extend(["", "Notes:", _safe_json(selected_notes)])
    if case_history:
        summary_lines.extend(["", "Case history:", _safe_json(case_history)])
    if log_lines:
        summary_lines.extend(["", "Recent log lines:", *log_lines[-80:]])

    csv_rows = current_df.head(200) if current_df is not None and not current_df.empty else pd.DataFrame()
    with zipfile.ZipFile(bundle_file, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("metadata.json", _safe_json(payload))
        archive.writestr("summary.txt", "\n".join(summary_lines))
        archive.writestr("logs.txt", "\n".join(log_lines or []))
        if not csv_rows.empty:
            archive.writestr("current_scored_sample.csv", csv_rows.to_csv(index=False))
    return bundle_file


def support_failure_message(exc: Exception) -> str:
    return f"{exc}\n\n{traceback.format_exc(limit=3)}"
