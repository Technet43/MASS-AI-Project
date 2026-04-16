"""Shared UI helpers for the MASS-AI dashboard.

Keep cross-cutting presentation concerns here so the main `app.py`
stays focused on screen composition.
"""
from __future__ import annotations

from typing import Any

import streamlit as st


def friendly_error(title: str, hint: str | None = None, detail: Any | None = None) -> None:
    """Render a one-line user-facing error with an optional technical expander.

    Kept deliberately simple so tracebacks never leak into the main flow.
    """
    st.error(title)
    if hint:
        st.caption(hint)
    if detail is not None:
        with st.expander("Technical detail", expanded=False):
            st.code(str(detail))


def action_card(label: str, value: str, subtitle: str | None = None, tone: str = "neutral") -> None:
    """Render a large action-oriented card used in the decision-first overview.

    `tone` controls the left border color: "danger" (red), "warn" (amber),
    "ok" (green), "neutral" (blue).
    """
    palette = {
        "danger": "#e74c3c",
        "warn": "#f1c40f",
        "ok": "#27ae60",
        "neutral": "#3498db",
    }
    border = palette.get(tone, palette["neutral"])
    html = f"""
    <div style="
        border-left: 6px solid {border};
        border-radius: 10px;
        padding: 14px 18px;
        background: rgba(255,255,255,0.03);
        margin-bottom: 8px;
    ">
        <div style="font-size: 0.82rem; opacity: 0.75; text-transform: uppercase; letter-spacing: 0.04em;">{label}</div>
        <div style="font-size: 1.9rem; font-weight: 700; margin-top: 4px;">{value}</div>
        {f'<div style="font-size: 0.85rem; opacity: 0.72; margin-top: 4px;">{subtitle}</div>' if subtitle else ''}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
