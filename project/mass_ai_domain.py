from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from html import escape
from typing import Any

import pandas as pd

from ops_store import is_overdue_value, parse_datetime, priority_for_risk_band

CURRENCY_CODE = "TRY"
RISK_LABELS = ["Low", "Moderate", "High", "Critical", "Urgent"]
CRITICAL_RISK_LABELS = {"Critical", "Urgent"}


def fmt_currency(value: Any) -> str:
    return f"{CURRENCY_CODE} {float(value or 0):,.0f}"


def fmt_percent(value: Any, decimals: int = 1) -> str:
    return f"{float(value or 0) * 100:.{decimals}f}%"


def safe_text(value: Any) -> str:
    if value is None:
        return "-"
    try:
        if pd.isna(value):
            return "-"
    except TypeError:
        pass
    return str(value)


def format_local_datetime(value: Any, fallback: str = "-") -> str:
    dt = parse_datetime(value)
    if dt is None:
        return fallback
    return dt.strftime("%Y-%m-%d %H:%M")


def normalize_follow_up_input(raw_text: Any) -> str | None:
    raw = str(raw_text or "").strip()
    if not raw:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(raw, fmt).isoformat(timespec="minutes")
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(raw).isoformat(timespec="minutes")
    except ValueError as exc:
        raise ValueError("Follow-up date must use YYYY-MM-DD or YYYY-MM-DD HH:MM.") from exc


def priority_for_risk(risk_band: Any) -> str:
    return priority_for_risk_band(risk_band)


def is_case_overdue(follow_up_at: Any, status: Any) -> bool:
    return is_overdue_value(follow_up_at, status)


def build_case_recommendation(risk_band: Any, status: Any = None, overdue: bool = False) -> str:
    risk = safe_text(risk_band)
    state = safe_text(status)
    if state == "Resolved":
        return "This case is resolved. Keep the documentation complete in case the customer returns to the queue."
    if overdue:
        return "This follow-up is overdue. Re-prioritize the case and confirm the next operational action today."
    if risk in CRITICAL_RISK_LABELS:
        return "Escalate for field inspection immediately and keep this account in the active verification queue."
    if risk == "High":
        return "Move the case into analyst review, attach supporting evidence, and schedule a secondary validation."
    if state == "Monitoring":
        return "Keep this customer under monitoring and refresh the score after the next dataset arrives."
    return "Review the score snapshot, capture notes, and decide whether the case should move into review or monitoring."


def summarize_case_notes(notes: list[dict[str, Any]] | None, limit: int = 3) -> str:
    if not notes:
        return "No analyst notes have been captured yet."
    latest = notes[-limit:]
    return " ".join(
        f"{format_local_datetime(note.get('created_at'))}: {safe_text(note.get('note_text'))}" for note in latest
    )


def summarize_risk_story(value: Any, fallback: str = "No explainability summary available.") -> str:
    text = safe_text(value)
    if text == "-":
        return fallback
    text = text.replace(" | ", ", ")
    return text


def _brief_risk_driver(row: pd.Series | dict[str, Any]) -> str:
    return summarize_risk_story(
        row.get("risk_drivers") or row.get("risk_summary") or row.get("risk_reason_1"),
        fallback="No risk driver summary available.",
    )


def filter_case_dataframe(
    df: pd.DataFrame | None,
    search: str = "",
    status: str = "All statuses",
    risk_band: str = "All risk bands",
    priority: str = "All priorities",
    overdue_only: bool = False,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=list(df.columns) if df is not None else [])

    filtered = df.copy()
    if "is_overdue" not in filtered.columns:
        filtered["is_overdue"] = filtered.apply(
            lambda row: is_case_overdue(row.get("follow_up_at"), row.get("status")),
            axis=1,
        )

    query = str(search or "").strip().lower()
    if query:
        mask = pd.Series(False, index=filtered.index)
        for column in ["customer_id", "profile", "fraud_pattern", "case_title"]:
            if column in filtered.columns:
                mask = mask | filtered[column].astype(str).str.lower().str.contains(query, na=False)
        filtered = filtered[mask]

    if status and status not in {"All statuses", "All open"}:
        filtered = filtered[filtered["status"].astype(str) == status]
    elif status == "All open":
        filtered = filtered[filtered["status"].astype(str) != "Resolved"]

    if risk_band and risk_band != "All risk bands":
        filtered = filtered[filtered["risk_band"].astype(str) == risk_band]

    if priority and priority != "All priorities":
        filtered = filtered[filtered["priority"].astype(str) == priority]

    if overdue_only:
        filtered = filtered[filtered["is_overdue"]]

    return filtered.reset_index(drop=True)


def build_executive_brief_html(
    overview: dict[str, Any],
    ops_metrics: dict[str, Any],
    top_rows: pd.DataFrame,
    selected_case: dict[str, Any] | None,
    selected_notes: list[dict[str, Any]] | None,
) -> str:
    risk_colors = {
        "Low": "#eaf8ef",
        "Moderate": "#fff4dd",
        "High": "#ffe9d6",
        "Critical": "#ffe0de",
        "Urgent": "#ffd6d3",
    }
    status_blocks = "".join(
        f"<div class='mini-card'><div class='mini-label'>{escape(status)}</div><div class='mini-value'>{count}</div></div>"
        for status, count in ops_metrics.get("open_by_status", {}).items()
    )
    priority_rows = []
    for _, row in top_rows.iterrows():
        risk = safe_text(row.get("risk_band", row.get("risk_category", "-")))
        priority_rows.append(
            "<tr>"
            f"<td>{escape(safe_text(row.get('customer_id', '-')))}</td>"
            f"<td>{escape(safe_text(row.get('profile', '-')))}</td>"
            f"<td><span class='badge' style='background:{risk_colors.get(risk, '#eef0f4')}'>{escape(risk)}</span></td>"
            f"<td>{escape(fmt_percent(row.get('fraud_probability', row.get('theft_probability', 0)), decimals=2))}</td>"
            f"<td>{escape(safe_text(row.get('priority', '-')))}</td>"
            f"<td>{escape(safe_text(row.get('status', '-')))}</td>"
            f"<td>{escape(fmt_currency(row.get('est_monthly_loss', 0)))}</td>"
            f"<td>{escape(_brief_risk_driver(row))}</td>"
            "</tr>"
        )

    selected_case = selected_case or {}
    follow_up_text = format_local_datetime(selected_case.get("follow_up_at"))
    selected_summary = summarize_case_notes(selected_notes)
    selected_recommendation = safe_text(selected_case.get("recommended_action") or "No case recommendation available.")
    selected_explanation = summarize_risk_story(selected_case.get("risk_summary") or selected_case.get("risk_drivers"))
    preset_name = safe_text(overview.get("preset_name"))
    preset_summary = safe_text(overview.get("preset_summary"))
    explainability_summary = summarize_risk_story(overview.get("explainability_summary"))

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MASS-AI Executive Brief</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:#f5f5f7; color:#1d1d1f; margin:0; padding:32px; }}
    .page {{ max-width:1180px; margin:0 auto; }}
    .hero, .panel {{ background:#fff; border:1px solid #e5e5ea; border-radius:24px; box-shadow:0 12px 32px rgba(20,20,40,0.06); }}
    .hero {{ padding:30px; margin-bottom:18px; }}
    .eyebrow {{ color:#0071e3; font-size:13px; font-weight:700; text-transform:uppercase; letter-spacing:.08em; }}
    h1, h2, h3 {{ margin:8px 0 12px; }}
    p {{ color:#6e6e73; line-height:1.5; }}
    .grid {{ display:grid; grid-template-columns:repeat(4, 1fr); gap:14px; margin-bottom:18px; }}
    .card, .mini-card {{ background:#fff; border:1px solid #e5e5ea; border-radius:18px; padding:18px; }}
    .mini-grid {{ display:grid; grid-template-columns:repeat(5, 1fr); gap:12px; }}
    .label, .mini-label {{ color:#6e6e73; font-size:13px; margin-bottom:10px; }}
    .value {{ font-size:24px; font-weight:700; }}
    .mini-value {{ font-size:20px; font-weight:700; }}
    .panel {{ padding:24px; margin-bottom:18px; }}
    .two-col {{ display:grid; grid-template-columns:1.35fr .95fr; gap:18px; }}
    table {{ width:100%; border-collapse:collapse; }}
    th, td {{ text-align:left; padding:12px 10px; border-bottom:1px solid #eef0f4; vertical-align:top; }}
    th {{ color:#6e6e73; font-size:13px; }}
    .badge {{ display:inline-block; padding:6px 10px; border-radius:999px; font-size:12px; font-weight:700; }}
    .note {{ background:#f7f8fb; border-radius:16px; padding:16px; color:#1d1d1f; }}
    .meta-row {{ display:grid; grid-template-columns:repeat(2, 1fr); gap:10px; margin-bottom:14px; }}
    .meta-item {{ background:#f7f8fb; border-radius:14px; padding:14px; }}
    @media (max-width: 960px) {{
      .grid {{ grid-template-columns:repeat(2, 1fr); }}
      .mini-grid, .two-col {{ grid-template-columns:1fr; }}
      body {{ padding:16px; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="eyebrow">MASS-AI Ops Center</div>
      <h1>Executive Brief</h1>
      <p>Generated on {escape(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))} from <strong>{escape(overview['data_source'])}</strong>. Leading model: <strong>{escape(overview['best_model'])}</strong>.</p>
      <p><strong>Synthetic preset:</strong> {escape(preset_name)}<br><strong>Preset context:</strong> {escape(preset_summary)}<br><strong>Explainability snapshot:</strong> {escape(explainability_summary)}</p>
    </section>
    <section class="grid">
      <div class="card"><div class="label">Customers scored</div><div class="value">{overview['customer_count']}</div></div>
      <div class="card"><div class="label">High-risk customers</div><div class="value">{overview['high_risk_count']}</div></div>
      <div class="card"><div class="label">Open Ops cases</div><div class="value">{ops_metrics.get('open_cases', 0)}</div></div>
      <div class="card"><div class="label">Overdue follow-ups</div><div class="value">{ops_metrics.get('overdue', 0)}</div></div>
    </section>
    <section class="panel"><h2>Open Case Status Mix</h2><div class="mini-grid">{status_blocks}</div></section>
    <section class="two-col">
      <section class="panel">
        <h2>Top Priority Open Cases</h2>
        <table>
          <thead><tr><th>Customer ID</th><th>Profile</th><th>Risk Band</th><th>Fraud Probability</th><th>Priority</th><th>Status</th><th>Monthly Exposure</th><th>Why Flagged</th></tr></thead>
          <tbody>{''.join(priority_rows)}</tbody>
        </table>
      </section>
      <section class="panel">
        <h2>Selected Case</h2>
        <h3>{escape(safe_text(selected_case.get('case_title', 'No case selected')))}</h3>
        <div class="meta-row">
          <div class="meta-item"><div class="mini-label">Status</div><div class="mini-value">{escape(safe_text(selected_case.get('status', '-')))}</div></div>
          <div class="meta-item"><div class="mini-label">Priority</div><div class="mini-value">{escape(safe_text(selected_case.get('priority', '-')))}</div></div>
          <div class="meta-item"><div class="mini-label">Risk Band</div><div class="mini-value">{escape(safe_text(selected_case.get('risk_band', '-')))}</div></div>
          <div class="meta-item"><div class="mini-label">Follow-up</div><div class="mini-value">{escape(follow_up_text)}</div></div>
        </div>
        <h3>Recommendation</h3><div class="note">{escape(selected_recommendation)}</div>
        <h3>Why This Case Was Flagged</h3><div class="note">{escape(selected_explanation)}</div>
        <h3>Notes Summary</h3><div class="note">{escape(selected_summary)}</div>
      </section>
    </section>
  </div>
</body>
</html>"""


def build_executive_brief_text(
    overview: dict[str, Any],
    ops_metrics: dict[str, Any],
    top_rows: pd.DataFrame,
    selected_case: dict[str, Any] | None,
    selected_notes: list[dict[str, Any]] | None,
) -> str:
    lines = [
        "# MASS-AI Executive Brief",
        "",
        f"- Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Data source: {overview['data_source']}",
        f"- Leading model: {overview['best_model']}",
        f"- Synthetic preset: {safe_text(overview.get('preset_name'))}",
        f"- Preset context: {safe_text(overview.get('preset_summary'))}",
        f"- Explainability snapshot: {summarize_risk_story(overview.get('explainability_summary'))}",
        f"- Customers scored: {overview['customer_count']}",
        f"- High-risk customers: {overview['high_risk_count']}",
        f"- Open Ops cases: {ops_metrics.get('open_cases', 0)}",
        f"- Overdue follow-ups: {ops_metrics.get('overdue', 0)}",
        "",
        "## Open Case Status Mix",
        "",
    ]
    for status, count in ops_metrics.get("open_by_status", {}).items():
        lines.append(f"- {status}: {count}")
    lines += [
        "",
        "## Top Priority Open Cases",
        "",
        "| Customer ID | Profile | Risk Band | Fraud Probability | Priority | Status | Monthly Exposure | Why Flagged |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in top_rows.iterrows():
        lines.append(
            f"| {safe_text(row.get('customer_id', '-'))} | {safe_text(row.get('profile', '-'))} | "
            f"{safe_text(row.get('risk_band', row.get('risk_category', '-')))} | "
            f"{fmt_percent(row.get('fraud_probability', row.get('theft_probability', 0)), decimals=2)} | "
            f"{safe_text(row.get('priority', '-'))} | {safe_text(row.get('status', '-'))} | "
            f"{fmt_currency(row.get('est_monthly_loss', 0))} | {summarize_risk_story(row.get('risk_drivers') or row.get('risk_summary') or row.get('risk_reason_1'))} |"
        )

    selected_case = selected_case or {}
    lines += [
        "",
        "## Selected Case",
        "",
        f"- Case: {safe_text(selected_case.get('case_title', 'No case selected'))}",
        f"- Status: {safe_text(selected_case.get('status', '-'))}",
        f"- Priority: {safe_text(selected_case.get('priority', '-'))}",
        f"- Risk band: {safe_text(selected_case.get('risk_band', '-'))}",
        f"- Recommendation: {safe_text(selected_case.get('recommended_action') or 'No case recommendation available.')}",
        f"- Why flagged: {summarize_risk_story(selected_case.get('risk_summary') or selected_case.get('risk_drivers'))}",
        "",
        "## Notes Summary",
        "",
        summarize_case_notes(selected_notes),
    ]
    return "\n".join(lines)


def normalize_column_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = (
        text.replace("Ä±", "i")
        .replace("Ä°", "i")
        .replace("ÅŸ", "s")
        .replace("Å", "s")
        .replace("ÄŸ", "g")
        .replace("Ä", "g")
        .replace("Ã¼", "u")
        .replace("Ãœ", "u")
        .replace("Ã¶", "o")
        .replace("Ã–", "o")
        .replace("Ã§", "c")
        .replace("Ã‡", "c")
    )
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[\s\-\/\\]+", "_", text)
    text = re.sub(r"[^\w]+", "", text)
    return text.strip("_")


COLUMN_ALIASES = {
    "customer_id": {"customer_id", "customerid", "customer", "aboneno", "abone_no", "musteri_no", "musteri_id", "subscriber_id", "client_id", "account_id"},
    "profile": {"profile", "segment", "customer_type", "type", "kategori", "profil"},
    "label": {"label", "target", "flag", "fraud_flag", "is_fraud", "theft_label", "etiket", "kacak_flag"},
    "theft_type": {"theft_type", "fraud_pattern", "pattern", "anomaly_type", "event_type", "kacak_turu"},
    "mean_consumption": {"mean_consumption", "avg_consumption", "average_consumption", "mean_usage", "avg_usage", "ortalama_tuketim", "ortalama_kullanim"},
    "std_consumption": {"std_consumption", "stdev_consumption", "std_usage", "consumption_std", "standart_sapma"},
    "zero_measurement_pct": {"zero_measurement_pct", "zero_ratio", "zero_day_pct", "zero_pct", "sifir_orani", "sifir_olcum_orani"},
    "sudden_change_ratio": {"sudden_change_ratio", "abrupt_change_ratio", "change_ratio", "spike_ratio", "ani_degisim_orani"},
    "mean_daily_total": {"mean_daily_total", "daily_total_mean", "avg_daily_total", "gunluk_ortalama", "gunluk_toplam_ortalama"},
}
