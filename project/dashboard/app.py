"""
MASS-AI Dashboard v2.0
======================
Akilli Sayac Verileri Icin Gercek Zamanli Anomali Izleme Paneli

Ozellikler:
- Bolgesel anomali haritasi
- Risk dagilimi grafikleri
- Musteri detay inceleme
- Alarm tablosu
- Kacak turu analizi
- [YENi] Zaman serisi karsilastirma (normal vs kacak)
- [YENi] Model performans sayfasi (ROC, confusion matrix, PR curve)
- [YENi] Canli simulasyon modu (akan veri efekti)

Yazar: Omer Burak Kocak
Calistirma: streamlit run dashboard/app.py
"""

import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ui_helpers import action_card, friendly_error  # noqa: E402

# TODO: centralised logging — mirrors mass_ai_engine setup
_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(_LOG_DIR / "mass_ai.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
_logger = logging.getLogger("dashboard")

# TODO: watchdog-based realtime ingest — monitors WATCH_DIR for new CSV/JSON drops
_WATCH_DIR = Path(__file__).resolve().parent.parent / "watch"
_WATCH_DIR.mkdir(parents=True, exist_ok=True)

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    _WATCHDOG_AVAILABLE = True
except ImportError:
    _WATCHDOG_AVAILABLE = False
    _logger.warning("watchdog not installed — realtime file monitoring disabled; pip install watchdog>=3.0")


class _IngestHandler(FileSystemEventHandler if _WATCHDOG_AVAILABLE else object):
    """Watches WATCH_DIR for new .csv/.json files and signals Streamlit to rerun."""

    def on_created(self, event):
        if event.is_directory:
            return
        src = Path(event.src_path)
        if src.suffix.lower() not in {".csv", ".json"}:
            return
        _logger.info("Realtime ingest: new file detected → %s", src.name)
        # Write a sentinel so the dashboard can pick up the latest file on next poll
        sentinel = _WATCH_DIR / ".latest_ingest"
        try:
            sentinel.write_text(str(src), encoding="utf-8")
        except Exception as exc:
            _logger.error("Realtime ingest: could not write sentinel: %s", exc)


def _start_ingest_watcher() -> "Observer | None":
    """Start a background watchdog observer once per Streamlit process."""
    if not _WATCHDOG_AVAILABLE:
        return None
    if st.session_state.get("_ingest_observer_started"):
        return None
    try:
        handler = _IngestHandler()
        observer = Observer()
        observer.schedule(handler, str(_WATCH_DIR), recursive=False)
        observer.daemon = True
        observer.start()
        st.session_state["_ingest_observer_started"] = True
        _logger.info("Realtime ingest watcher started on %s", _WATCH_DIR)
        return observer
    except Exception as exc:
        _logger.error("Could not start ingest watcher: %s", exc)
        return None


def poll_ingest_dir() -> Path | None:
    """Return the path of the latest ingested file if a new one arrived, else None.

    Reads the sentinel written by _IngestHandler and clears it so the same file
    is not processed twice.
    """
    sentinel = _WATCH_DIR / ".latest_ingest"
    if not sentinel.exists():
        return None
    try:
        raw = sentinel.read_text(encoding="utf-8").strip()
        sentinel.unlink(missing_ok=True)
        path = Path(raw)
        if path.exists():
            return path
    except Exception as exc:
        _logger.error("poll_ingest_dir: %s", exc)
    return None

# psycopg2 is optional — live telemetry tab is disabled if not installed.
try:
    import psycopg2
    import psycopg2.extras
    _PSYCOPG2_AVAILABLE = True
except ImportError:
    _PSYCOPG2_AVAILABLE = False

# ─── DB connection helper ────────────────────────────────────────────────────
_DB_DEFAULTS = dict(
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", "5433")),
    dbname=os.getenv("DB_NAME", "mass_ai"),
    user=os.getenv("DB_USER", "mass_ai"),
    password=os.getenv("DB_PASSWORD", "mass_ai_secret"),
)

def _db_connect():
    return psycopg2.connect(**_DB_DEFAULTS)

def fetch_alerts(limit: int = 50) -> pd.DataFrame:
    """Return the most recent *limit* unacknowledged alerts."""
    sql = """
        SELECT a.id, a.meter_id, a.anomaly_score, a.severity, a.created_at
        FROM   alerts a
        ORDER  BY a.created_at DESC
        LIMIT  %s
    """
    with _db_connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (limit,))
            rows = cur.fetchall()
    df = pd.DataFrame(rows)
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    return df


def fetch_meter_evidence(meter_id: str, hours: int = 2) -> dict:
    """Fetch raw telemetry + feature history for one meter (evidence report)."""
    sql_raw = """
        SELECT voltage, current, active_power, received_at
        FROM   raw_telemetry
        WHERE  meter_id = %s
          AND  received_at >= NOW() - INTERVAL '%s hours'
        ORDER  BY received_at ASC
        LIMIT  500
    """
    sql_feat = """
        SELECT p_avg_1h, v_std_1h, i_peak_1h, sample_count, computed_at
        FROM   processed_features
        WHERE  meter_id = %s
        ORDER  BY computed_at DESC
        LIMIT  10
    """
    sql_alerts = """
        SELECT anomaly_score, severity, created_at
        FROM   alerts
        WHERE  meter_id = %s
        ORDER  BY created_at DESC
        LIMIT  5
    """
    with _db_connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql_raw,   (meter_id, hours))
            raw_rows = cur.fetchall()
            cur.execute(sql_feat,  (meter_id,))
            feat_rows = cur.fetchall()
            cur.execute(sql_alerts, (meter_id,))
            alert_rows = cur.fetchall()

    df_raw   = pd.DataFrame(raw_rows)
    df_feat  = pd.DataFrame(feat_rows)
    df_alert = pd.DataFrame(alert_rows)
    for df in [df_raw, df_feat, df_alert]:
        if not df.empty:
            ts_col = [c for c in df.columns if "at" in c]
            if ts_col:
                df[ts_col[0]] = pd.to_datetime(df[ts_col[0]], utc=True)
    return {"raw": df_raw, "features": df_feat, "alerts": df_alert}


@st.dialog("Anomaly / Anomali", width="large")
def show_evidence_dialog(meter_id: str, score: float, severity: str, created_at) -> None:
    """Modal popup: evidence report for a flagged meter."""
    is_en = st.session_state.get("ui_lang", "tr") == "en"
    color = "#E74C3C" if severity == "KRITIK" else "#E67E22"

    meter_label = "Meter" if is_en else "Sayac"
    score_label = "Anomaly Score" if is_en else "Anomali Skoru"
    time_label = "Detected At" if is_en else "Tespit Zamani"

    st.markdown(
        f"<div style='background:{color}22;border-left:4px solid {color};padding:12px;border-radius:6px;margin-bottom:12px'>"
        f"<b style='color:{color};font-size:1.1rem'>{severity}</b><br>"
        f"<b>{meter_label}:</b> {meter_id} &nbsp;|&nbsp; "
        f"<b>{score_label}:</b> {score:.3f} &nbsp;|&nbsp; "
        f"<b>{time_label}:</b> {created_at}"
        f"</div>",
        unsafe_allow_html=True,
    )

    try:
        ev = fetch_meter_evidence(meter_id, hours=2)
    except Exception as exc:
        friendly_error(
            "Evidence fetch failed" if is_en else "Kanit verisi cekilemedi",
            "Check DB connection or telemetry availability." if is_en else "Veritabani baglantisini / telemetri varligini kontrol edin.",
            detail=exc,
        )
        return

    df_raw = ev["raw"]
    if df_raw.empty:
        st.warning("No raw telemetry in last 2 hours for this meter." if is_en else "Bu sayac icin son 2 saatte ham telemetri verisi yok.")
    else:
        st.markdown("#### Last 2 Hours Telemetry" if is_en else "#### Son 2 Saatlik Telemetri")
        fig = go.Figure()
        for col, color_line, yaxis in [
            ("active_power", "#E74C3C", "y1"),
            ("voltage", "#2E86C1", "y2"),
            ("current", "#27AE60", "y3"),
        ]:
            fig.add_trace(go.Scatter(
                x=df_raw["received_at"],
                y=df_raw[col],
                name={
                    "active_power": ("Power (W)" if is_en else "Guc (W)"),
                    "voltage": ("Voltage (V)" if is_en else "Voltaj (V)"),
                    "current": ("Current (A)" if is_en else "Akim (A)"),
                }[col],
                line=dict(width=1.5),
                yaxis=yaxis,
            ))
        fig.update_layout(
            height=320,
            margin=dict(l=20, r=20, t=10, b=20),
            legend=dict(orientation="h", y=-0.3),
            yaxis=dict(title=("Power (W)" if is_en else "Guc (W)"), side="left"),
            yaxis2=dict(title=("Voltage (V)" if is_en else "Voltaj (V)"), side="right", overlaying="y"),
            yaxis3=dict(title=("Current (A)" if is_en else "Akim (A)"), side="right", overlaying="y", anchor="free", position=1.0),
        )
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Power (W)" if is_en else "Ort. Guc (W)", f"{df_raw['active_power'].mean():.1f}")
        c2.metric("Voltage Std (V)" if is_en else "Voltaj Std (V)", f"{df_raw['voltage'].std():.2f}")
        c3.metric("Max Current (A)" if is_en else "Maks. Akim (A)", f"{df_raw['current'].max():.3f}")

    df_feat = ev["features"]
    if not df_feat.empty:
        st.markdown("#### Computed Features (Last 10 Passes)" if is_en else "#### Hesaplanan Ozellikler (Son 10 Pass)")
        st.dataframe(
            df_feat.rename(columns={
                "p_avg_1h": "P_avg (W)",
                "v_std_1h": "V_std",
                "i_peak_1h": "I_peak (A)",
                "sample_count": ("Sample" if is_en else "Ornek"),
                "computed_at": ("Computed At" if is_en else "Hesap Zamani"),
            }),
            use_container_width=True,
            hide_index=True,
        )

    if not df_raw.empty:
        st.markdown("#### Download Evidence Report" if is_en else "#### Kanit Raporu Indir")
        csv_bytes = df_raw.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=("Download CSV" if is_en else "CSV olarak indir"),
            data=csv_bytes,
            file_name=f"evidence_{meter_id}_{time.strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )


def fetch_live_telemetry(limit: int = 100) -> pd.DataFrame:
    """Return the most recent *limit* rows from raw_telemetry."""
    sql = """
        SELECT id, meter_id, voltage, current, active_power, received_at
        FROM   raw_telemetry
        ORDER  BY received_at DESC
        LIMIT  %s
    """
    with _db_connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (limit,))
            rows = cur.fetchall()
    df = pd.DataFrame(rows)
    if not df.empty:
        df["received_at"] = pd.to_datetime(df["received_at"], utc=True)
    return df

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ========== SAYFA AYARLARI ==========
st.set_page_config(
    page_title="MASS-AI | Akilli Sayac Anomali Tespiti",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CUSTOM CSS ==========
st.markdown("""
<style>
    /* ── Design tokens ──────────────────────────────────────────────── */
    :root {
        --accent:        #38bdf8;
        --accent-soft:   rgba(56, 189, 248, 0.18);
        --accent-glow:   rgba(56, 189, 248, 0.35);
        --danger:        rgba(251, 113, 113, 1);
        --danger-soft:   rgba(251, 113, 113, 0.18);
        --success:       rgba(52, 211, 153, 1);
        --success-soft:  rgba(52, 211, 153, 0.15);

        --glass-bg:        rgba(255,255,255,0.042);
        --glass-border:    rgba(255,255,255,0.11);
        --glass-hi:        inset 0 1px 0 rgba(255,255,255,0.18);
        --glass-lo:        inset 0 -1px 0 rgba(0,0,0,0.22);
        --glass-shadow:    0 20px 70px rgba(0,0,0,0.44);

        --text-main:   #f0f4ff;
        --text-muted:  rgba(240,244,255,0.52);
        --text-dim:    rgba(240,244,255,0.30);
        --line-soft:   rgba(255,255,255,0.07);

        --radius-card: 26px;
        --radius-btn:  16px;
        --ease-smooth: cubic-bezier(0.34, 1.22, 0.64, 1);
        --ease-out:    cubic-bezier(0.16, 1, 0.3, 1);
    }

    /* ── Scrollbar ──────────────────────────────────────────────────── */
    * { scrollbar-width: thin; scrollbar-color: rgba(255,255,255,0.12) transparent; }
    *::-webkit-scrollbar { width: 5px; height: 5px; }
    *::-webkit-scrollbar-track { background: transparent; }
    *::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.14); border-radius: 99px; }
    *::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.24); }

    /* ── Base background ────────────────────────────────────────────── */
    html, body,
    [data-testid="stAppViewContainer"],
    [data-testid="stApp"] {
        background:
            radial-gradient(ellipse 80% 50% at 10% -5%,  rgba(56,189,248,0.07), transparent),
            radial-gradient(ellipse 60% 40% at 90%  5%,  rgba(129,140,248,0.06), transparent),
            radial-gradient(ellipse 70% 60% at 50% 110%, rgba(56,189,248,0.04), transparent),
            linear-gradient(175deg, #05070b 0%, #080c12 55%, #040609 100%);
        color: var(--text-main);
        font-feature-settings: "kern" 1, "liga" 1, "ss01" 1;
    }
    [data-testid="stAppViewContainer"] > .main { background: transparent; }

    /* ── Top header bar ─────────────────────────────────────────────── */
    [data-testid="stHeader"] {
        background: rgba(5,7,11,0.55);
        backdrop-filter: blur(28px);
        -webkit-backdrop-filter: blur(28px);
        border-bottom: 1px solid rgba(255,255,255,0.055);
    }

    /* ── Sidebar ────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background:
            linear-gradient(160deg, rgba(56,189,248,0.04) 0%, transparent 40%),
            linear-gradient(180deg, rgba(255,255,255,0.048), rgba(255,255,255,0.022)),
            rgba(7,9,14,0.82);
        border-right: 1px solid rgba(255,255,255,0.07);
        box-shadow: 20px 0 56px rgba(0,0,0,0.28), var(--glass-hi);
        backdrop-filter: blur(36px);
        -webkit-backdrop-filter: blur(36px);
    }
    [data-testid="stSidebar"] > div:first-child { background: transparent; }
    [data-testid="stSidebar"] * { color: var(--text-main) !important; }
    [data-testid="stSidebar"] hr, .stMarkdown hr { border-color: var(--line-soft); }

    /* ── Typography ─────────────────────────────────────────────────── */
    .main-header {
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        line-height: 1.12;
        margin: 0 0 0.2rem;
        background: linear-gradient(135deg, #ffffff 0%, rgba(255,255,255,0.72) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        filter: drop-shadow(0 0 28px rgba(255,255,255,0.14));
    }
    .sub-header {
        font-size: 0.93rem;
        color: var(--text-muted);
        margin: 0;
        letter-spacing: 0.01em;
    }
    h1, h2, h3, h4 { letter-spacing: -0.02em; }
    .stMarkdown h3 { font-size: 1.15rem; font-weight: 700; color: var(--text-main); }
    .stMarkdown h4 { font-size: 1rem;    font-weight: 600; color: var(--text-main); }

    /* ── Hero card ──────────────────────────────────────────────────── */
    .hero-shell {
        position: relative;
        overflow: hidden;
        border-radius: 32px;
        padding: 1.5rem 1.8rem 1.4rem;
        margin-bottom: 1.2rem;
        background:
            linear-gradient(135deg, rgba(56,189,248,0.06) 0%, transparent 50%),
            linear-gradient(180deg, rgba(255,255,255,0.075), rgba(255,255,255,0.018));
        border: 1px solid rgba(255,255,255,0.13);
        box-shadow:
            var(--glass-shadow),
            var(--glass-hi),
            var(--glass-lo),
            0 0 0 1px rgba(56,189,248,0.06);
        backdrop-filter: blur(36px);
        -webkit-backdrop-filter: blur(36px);
    }
    .hero-shell::before {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(135deg,
            rgba(255,255,255,0.13) 0%,
            transparent 38%,
            transparent 65%,
            rgba(56,189,248,0.07) 100%);
        pointer-events: none;
    }
    .hero-shell::after {
        content: "";
        position: absolute;
        top: -1px; left: 10%; right: 10%;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.30), transparent);
        pointer-events: none;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.73rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--accent);
        background: var(--accent-soft);
        border: 1px solid rgba(56,189,248,0.28);
        border-radius: 99px;
        padding: 3px 12px;
        margin-bottom: 0.55rem;
    }

    /* ── Metric cards ───────────────────────────────────────────────── */
    [data-testid="stMetric"] {
        position: relative;
        overflow: hidden;
        background:
            linear-gradient(180deg, rgba(255,255,255,0.065), rgba(255,255,255,0.020));
        border: 1px solid rgba(255,255,255,0.095);
        border-radius: var(--radius-card);
        padding: 1.1rem 1.2rem 1rem;
        box-shadow:
            0 16px 52px rgba(0,0,0,0.24),
            var(--glass-hi),
            var(--glass-lo);
        backdrop-filter: blur(30px);
        -webkit-backdrop-filter: blur(30px);
        transition: transform 240ms var(--ease-out),
                    box-shadow 240ms var(--ease-out),
                    border-color 240ms ease;
    }
    [data-testid="stMetric"]::after {
        content: "";
        position: absolute;
        top: 0; left: 12%; right: 12%;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(56,189,248,0.45), transparent);
        opacity: 0;
        transition: opacity 240ms ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-3px) scale(1.015);
        border-color: rgba(56,189,248,0.22);
        box-shadow:
            0 28px 72px rgba(0,0,0,0.32),
            0 0 0 1px rgba(56,189,248,0.12),
            var(--glass-hi), var(--glass-lo);
    }
    [data-testid="stMetric"]:hover::after { opacity: 1; }
    div[data-testid="stMetricValue"] {
        font-size: 1.95rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: var(--text-main);
    }
    div[data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: 0.82rem; }
    div[data-testid="stMetricDelta"]  { color: var(--text-muted) !important; font-size: 0.80rem; }

    /* ── Tab bar ────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(255,255,255,0.025);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 20px;
        padding: 5px;
        backdrop-filter: blur(28px);
        -webkit-backdrop-filter: blur(28px);
        box-shadow: var(--glass-hi), var(--glass-lo);
    }
    .stTabs [data-baseweb="tab"] {
        height: auto;
        padding: 9px 20px;
        font-size: 0.88rem;
        font-weight: 600;
        border-radius: 15px;
        color: var(--text-muted);
        transition: color 180ms ease, background 180ms ease;
        letter-spacing: 0.005em;
    }
    .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
        color: rgba(240,244,255,0.80);
        background: rgba(255,255,255,0.04);
    }
    .stTabs [aria-selected="true"] {
        background:
            linear-gradient(180deg, rgba(56,189,248,0.14), rgba(56,189,248,0.06)) !important;
        color: #ffffff !important;
        border: 1px solid rgba(56,189,248,0.22) !important;
        box-shadow:
            inset 0 1px 0 rgba(255,255,255,0.18),
            0 6px 20px rgba(0,0,0,0.22),
            0 0 14px rgba(56,189,248,0.10);
    }

    /* ── Glass panel (charts, dataframes, expanders) ────────────────── */
    .block-container { padding-top: 1.8rem; padding-bottom: 2rem; }

    div[data-testid="stPlotlyChart"],
    div[data-testid="stDataFrame"],
    div[data-testid="stTable"],
    div[data-testid="stExpander"],
    div[data-testid="stAlert"],
    div[data-testid="stForm"] {
        border-radius: var(--radius-card);
    }
    div[data-testid="stPlotlyChart"],
    div[data-testid="stDataFrame"],
    div[data-testid="stAlert"],
    div[data-baseweb="select"],
    div[data-testid="stMultiSelect"],
    div[data-testid="stSlider"],
    div[data-testid="stNumberInput"],
    div[data-testid="stTextInput"] {
        background:
            linear-gradient(180deg, rgba(255,255,255,0.055), rgba(255,255,255,0.018));
        border: 1px solid rgba(255,255,255,0.09) !important;
        box-shadow:
            0 14px 48px rgba(0,0,0,0.20),
            var(--glass-hi), var(--glass-lo);
        backdrop-filter: blur(28px);
        -webkit-backdrop-filter: blur(28px);
    }
    div[data-testid="stPlotlyChart"],
    div[data-testid="stDataFrame"] { padding: 0.6rem; }

    /* ── Inputs ─────────────────────────────────────────────────────── */
    div[data-baseweb="select"] > div,
    div[data-testid="stTextInput"] input,
    textarea,
    [data-baseweb="base-input"] {
        background: rgba(255,255,255,0.025) !important;
        color: var(--text-main) !important;
    }
    div[data-testid="stTextInput"]:focus-within,
    div[data-baseweb="select"]:focus-within {
        border-color: rgba(56,189,248,0.35) !important;
        box-shadow: 0 0 0 3px rgba(56,189,248,0.08) !important;
    }

    /* ── Slider ─────────────────────────────────────────────────────── */
    .stSlider [data-baseweb="slider"] > div > div {
        background: linear-gradient(90deg, var(--accent), rgba(56,189,248,0.40)) !important;
    }
    .stSlider [role="slider"] {
        background: #ffffff !important;
        box-shadow: 0 0 0 3px rgba(56,189,248,0.40), 0 2px 8px rgba(0,0,0,0.30) !important;
    }

    /* ── Buttons ────────────────────────────────────────────────────── */
    div[data-testid="stButton"] > button,
    div[data-testid="stDownloadButton"] > button {
        background:
            linear-gradient(180deg, rgba(255,255,255,0.075), rgba(255,255,255,0.028)) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        box-shadow: 0 10px 32px rgba(0,0,0,0.20), var(--glass-hi) !important;
        backdrop-filter: blur(20px);
    }
    .stButton > button,
    .stDownloadButton > button {
        color: var(--text-main) !important;
        border-radius: var(--radius-btn) !important;
        min-height: 2.75rem;
        font-weight: 600;
        font-size: 0.88rem;
        letter-spacing: 0.01em;
        transition:
            transform 200ms var(--ease-out),
            box-shadow 200ms ease,
            border-color 200ms ease;
    }
    .stButton > button:hover,
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        border-color: rgba(56,189,248,0.32) !important;
        box-shadow:
            0 18px 48px rgba(0,0,0,0.28),
            0 0 0 1px rgba(56,189,248,0.16),
            var(--glass-hi) !important;
    }
    .stButton > button:active,
    .stDownloadButton > button:active { transform: translateY(0) scale(0.98); }
    button[kind="primary"] {
        background: linear-gradient(135deg, rgba(56,189,248,0.25), rgba(56,189,248,0.10)) !important;
        border-color: rgba(56,189,248,0.40) !important;
        box-shadow: 0 10px 32px rgba(56,189,248,0.12), var(--glass-hi) !important;
    }
    button[kind="primary"]:hover {
        background: linear-gradient(135deg, rgba(56,189,248,0.35), rgba(56,189,248,0.15)) !important;
        box-shadow: 0 18px 48px rgba(56,189,248,0.20), 0 0 0 1px rgba(56,189,248,0.35), var(--glass-hi) !important;
    }

    /* ── Alert / Safe boxes ─────────────────────────────────────────── */
    .alert-box {
        background: var(--danger-soft);
        border-left: 3px solid var(--danger);
        padding: 11px 16px;
        border-radius: 16px;
        margin: 8px 0;
        box-shadow: 0 4px 20px rgba(251,113,113,0.12);
        backdrop-filter: blur(20px);
    }
    .safe-box {
        background: var(--success-soft);
        border-left: 3px solid var(--success);
        padding: 11px 16px;
        border-radius: 16px;
        margin: 8px 0;
        box-shadow: 0 4px 20px rgba(52,211,153,0.10);
        backdrop-filter: blur(20px);
    }

    /* ── Live indicator ─────────────────────────────────────────────── */
    .live-indicator {
        display: inline-block;
        width: 9px;
        height: 9px;
        background: var(--accent);
        border-radius: 50%;
        margin-right: 7px;
        box-shadow: 0 0 0 0 var(--accent-glow);
        animation: pulse-ring 1.6s cubic-bezier(0.4,0,0.6,1) infinite;
    }
    @keyframes pulse-ring {
        0%   { box-shadow: 0 0 0 0   var(--accent-glow); }
        60%  { box-shadow: 0 0 0 7px rgba(56,189,248,0); }
        100% { box-shadow: 0 0 0 0   rgba(56,189,248,0); }
    }

    /* ── Glass divider ──────────────────────────────────────────────── */
    .glass-divider {
        height: 1px;
        margin: 1.1rem 0 1.5rem;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.13), transparent);
    }

    /* ── Misc text ──────────────────────────────────────────────────── */
    .stMarkdown, p, label, span, .stCaption, .stText { color: var(--text-main); }
    .stCaption { color: var(--text-muted) !important; font-size: 0.80rem; }
    [data-testid="stDataFrame"] * { color: var(--text-main) !important; }

    /* ── Expander ───────────────────────────────────────────────────── */
    div[data-testid="stExpander"] {
        background:
            linear-gradient(180deg, rgba(255,255,255,0.048), rgba(255,255,255,0.015));
        border: 1px solid rgba(255,255,255,0.08) !important;
        backdrop-filter: blur(24px);
    }
    div[data-testid="stExpander"]:hover { border-color: rgba(56,189,248,0.18) !important; }

    /* ── Streamlit overrides ────────────────────────────────────────── */
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"]
      > [data-testid="stVerticalBlockBorderWrapper"] { background: transparent; }
    div[data-testid="stMarkdownContainer"]:has(h3),
    div[data-testid="stMarkdownContainer"]:has(h4) { border-radius: var(--radius-card); }

    /* ── Fade-in animation for page load ────────────────────────────── */
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(14px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .block-container > div { animation: fadeUp 420ms var(--ease-out) both; }
</style>
""", unsafe_allow_html=True)


def inject_liquid_spotlight() -> None:
    """Inject a dual-layer cursor spotlight (outer soft halo + inner accent core)."""
    components.html(
        """
        <div id="spot-outer"></div>
        <div id="spot-inner"></div>
        <style>
          #spot-outer, #spot-inner {
            position: fixed;
            left: 0; top: 0;
            border-radius: 999px;
            pointer-events: none;
            will-change: transform;
          }
          #spot-outer {
            width: 600px; height: 600px;
            z-index: 0;
            opacity: 0.45;
            transform: translate3d(-9999px,-9999px,0);
            background: radial-gradient(circle,
              rgba(56,189,248,0.14) 0%,
              rgba(56,189,248,0.07) 25%,
              rgba(56,189,248,0.02) 50%,
              transparent 70%);
            filter: blur(22px);
            mix-blend-mode: screen;
          }
          #spot-inner {
            width: 220px; height: 220px;
            z-index: 1;
            opacity: 0.30;
            transform: translate3d(-9999px,-9999px,0);
            background: radial-gradient(circle,
              rgba(255,255,255,0.22) 0%,
              rgba(255,255,255,0.08) 40%,
              transparent 70%);
            filter: blur(8px);
            mix-blend-mode: screen;
          }
        </style>
        <script>
          const outer = document.getElementById("spot-outer");
          const inner = document.getElementById("spot-inner");
          let tx = window.innerWidth/2, ty = window.innerHeight/3;
          let ox = tx, oy = ty, ix = tx, iy = ty;
          const lerp = (a,b,t) => a + (b-a)*t;
          const tick = () => {
            ox = lerp(ox, tx, 0.08); oy = lerp(oy, ty, 0.08);
            ix = lerp(ix, tx, 0.18); iy = lerp(iy, ty, 0.18);
            outer.style.transform = `translate3d(${ox-300}px,${oy-300}px,0)`;
            inner.style.transform = `translate3d(${ix-110}px,${iy-110}px,0)`;
            requestAnimationFrame(tick);
          };
          window.addEventListener("mousemove", e => { tx=e.clientX; ty=e.clientY; }, {passive:true});
          requestAnimationFrame(tick);
        </script>
        """,
        height=0,
        width=0,
    )


TRANSLATIONS = {
    "tr": {
        "subtitle": "Milli Akilli Sayac Sistemleri | Yapay Zeka Tabanli Anomali Tespit ve Kacak Elektrik Siniflandirma v2.0",
        "sidebar_tagline": "Akilli Sayac Anomali Tespiti",
        "appearance": "Gorunum",
        "theme": "Tema",
        "language": "Dil",
        "theme_dark": "Siyah Cam",
        "theme_light": "Beyaz Cam",
        "filters": "Filtreler",
        "customer_profile": "Musteri Profili",
        "risk_band": "Risk Bandi",
        "threshold": "Kacak Olasilik Esigi",
        "shown": "Gosterilen",
        "project_info": "Proje Bilgisi",
        "total_customers": "Toplam Musteri",
        "detected": "Tespit Edilen",
        "urgent": "Acil",
        "critical": "Kritik",
        "estimated_loss": "Tahmini Aylik Kayip",
        "field_team": "Saha ekibi",
        "clean": "Temiz",
        "regional_map": "Bolgesel Anomali Haritasi",
        "risk_distribution": "Risk Seviyesi Dagilimi",
        "prob_distribution": "Kacak Olasilik Dagilimi",
        "risk_mix": "Risk Karmasi",
        "alerts": "Anomali Alarmlari",
        "no_alerts": "Secilen esik degerinde alarm bulunmuyor.",
        "suspicion_alarm": "musteri icin kacak suphe alarmi",
        "live_detector_alerts": "Canli Anomali Uyarilari",
        "refresh": "Yenile",
        "refresh_live_alerts": "Canli uyarilari yenile",
        "new_critical_anomaly": "yeni KRITIK anomali",
        "db_unavailable": "Veritabani kullanilamiyor",
        "no_live_alert": "Henuz canli uyari yok.",
        "open": "Ac",
        "histogram_y_axis": "Musteri Sayisi",
        "risk_mix_center": "Risk Karmasi",
        "customer_id": "Musteri ID",
        "anomaly_score": "Anomali Skoru",
        "currency_symbol": "₺",
        "id": "ID",
        "profile": "Profil",
        "probability": "Kacak Olasiligi",
        "risk": "Risk",
        "avg_consumption": "Ort. Tuketim",
        "zero_pct": "Sifir %",
        "sudden_change": "Ani Degisim",
        "tab_overview": "Genel Bakis",
        "tab_customers": "Musteri Listesi",
        "tab_timeseries": "Zaman Serisi",
        "tab_models": "Model Performansi",
        "tab_detail": "Musteri Detay",
        "tab_simulation": "Canli Simulasyon",
        "tab_telemetry": "Canli Telemetri (DB)",
        "telemetry_title": "Canli Telemetri - Son 100 Kayit (Postgres)",
        "telemetry_refresh_help": "Yenileme ile son 100 kayit Postgres'ten cekilir.",
        "telemetry_last_updated": "Son guncelleme",
        "telemetry_total_records": "Toplam Kayit",
        "telemetry_avg_voltage": "Ort. Voltaj (V)",
        "telemetry_avg_current": "Ort. Akim (A)",
        "telemetry_avg_power": "Ort. Guc (W)",
        "telemetry_chart_title": "Aktif Guc (W) - Son 100 Olcum",
        "telemetry_time_axis": "Zaman (UTC)",
        "telemetry_power_axis": "Aktif Guc (W)",
        "telemetry_time_col": "Zaman",
        "telemetry_meter_col": "Sayac ID",
        "telemetry_voltage_col": "Voltaj (V)",
        "telemetry_current_col": "Akim (A)",
        "telemetry_power_col": "Guc (W)",
        "telemetry_no_data": "Henuz veri yok - gateway ve sensor_mock calisiyor mu?",
        "telemetry_psycopg_missing": "psycopg2 kurulu degil. Canli telemetri icin su komutu calistir:",
        "telemetry_db_error": "Postgres baglantisi kurulamadi",
        "telemetry_db_help": "Kontrol et: `docker compose up -d mass-ai-db` calisiyor mu?",
        "active_model": "Aktif Model",
        "light_mode": "Acik Mod",
        "dark_mode": "Koyu Mod",
        "data_source": "Veri Kaynagi",
        "data_source_sample": "Hazir Ornek Veri",
        "data_source_external": "Dis CSV Dosyasi",
        "external_csv_path": "CSV Dosya Yolu",
        "external_csv_hint": "Ornek: C:\\Users\\...\\data_extracted\\data.csv",
        "external_loaded": "Dis veri yuklendi",
        "external_preview": "Onizleme musteri",
        "external_load_error": "Dis veri yuklenemedi, hazir ornek veriye donuldu",
        "external_missing_path": "Dis kaynak secili ama CSV yolu bos",
        "external_csv_upload": "CSV Yukle (Klasorden Sec)",
        "external_selected_file": "Secilen dosya",
        "external_saved_path": "Kayit yolu",
    },
    "en": {
        "subtitle": "National Smart Meter Systems | AI-Powered Anomaly Detection and Electricity Theft Classification v2.0",
        "sidebar_tagline": "Smart Meter Anomaly Detection",
        "appearance": "Appearance",
        "theme": "Theme",
        "language": "Language",
        "theme_dark": "Black Glass",
        "theme_light": "White Glass",
        "filters": "Filters",
        "customer_profile": "Customer Profile",
        "risk_band": "Risk Band",
        "threshold": "Theft Probability Threshold",
        "shown": "Showing",
        "project_info": "Project Info",
        "total_customers": "Total Customers",
        "detected": "Detected",
        "urgent": "Urgent",
        "critical": "Critical",
        "estimated_loss": "Estimated Monthly Loss",
        "field_team": "Field team",
        "clean": "Clean",
        "regional_map": "Regional Anomaly Map",
        "risk_distribution": "Risk Distribution",
        "prob_distribution": "Theft Probability Distribution",
        "risk_mix": "Risk Mix",
        "alerts": "Anomaly Alerts",
        "no_alerts": "No alerts found for the selected threshold.",
        "suspicion_alarm": "customers flagged for suspected theft",
        "live_detector_alerts": "Live Detector Alerts",
        "refresh": "Refresh",
        "refresh_live_alerts": "Refresh live alerts",
        "new_critical_anomaly": "new CRITICAL anomaly",
        "db_unavailable": "Database unavailable",
        "no_live_alert": "No live alert yet.",
        "open": "Open",
        "histogram_y_axis": "Customer Count",
        "risk_mix_center": "Risk Mix",
        "customer_id": "Customer ID",
        "anomaly_score": "Anomaly Score",
        "currency_symbol": "TRY",
        "id": "ID",
        "profile": "Profile",
        "probability": "Theft Probability",
        "risk": "Risk",
        "avg_consumption": "Avg. Consumption",
        "zero_pct": "Zero %",
        "sudden_change": "Sudden Change",
        "tab_overview": "Overview",
        "tab_customers": "Customer List",
        "tab_timeseries": "Time Series",
        "tab_models": "Model Performance",
        "tab_detail": "Customer Detail",
        "tab_simulation": "Live Simulation",
        "tab_telemetry": "Live Telemetry (DB)",
        "telemetry_title": "Live Telemetry - Last 100 Records (Postgres)",
        "telemetry_refresh_help": "Each refresh pulls the latest 100 records from Postgres.",
        "telemetry_last_updated": "Last update",
        "telemetry_total_records": "Total Records",
        "telemetry_avg_voltage": "Avg. Voltage (V)",
        "telemetry_avg_current": "Avg. Current (A)",
        "telemetry_avg_power": "Avg. Power (W)",
        "telemetry_chart_title": "Active Power (W) - Last 100 Samples",
        "telemetry_time_axis": "Time (UTC)",
        "telemetry_power_axis": "Active Power (W)",
        "telemetry_time_col": "Time",
        "telemetry_meter_col": "Meter ID",
        "telemetry_voltage_col": "Voltage (V)",
        "telemetry_current_col": "Current (A)",
        "telemetry_power_col": "Power (W)",
        "telemetry_no_data": "No data yet - are gateway and sensor_mock running?",
        "telemetry_psycopg_missing": "psycopg2 is not installed. Run this command for live telemetry:",
        "telemetry_db_error": "Postgres connection failed",
        "telemetry_db_help": "Check: is `docker compose up -d mass-ai-db` running?",
        "active_model": "Active Model",
        "light_mode": "Light Mode",
        "dark_mode": "Dark Mode",
        "data_source": "Data Source",
        "data_source_sample": "Built-in Sample",
        "data_source_external": "External CSV",
        "external_csv_path": "CSV File Path",
        "external_csv_hint": "Example: C:\\Users\\...\\data_extracted\\data.csv",
        "external_loaded": "External data loaded",
        "external_preview": "Preview customers",
        "external_load_error": "External data failed, switched back to built-in sample",
        "external_missing_path": "External source selected but CSV path is empty",
        "external_csv_upload": "Upload CSV (Choose from folder)",
        "external_selected_file": "Selected file",
        "external_saved_path": "Saved path",
    },
}


def tr(key: str) -> str:
    lang = st.session_state.get("ui_lang", "tr")
    return TRANSLATIONS.get(lang, TRANSLATIONS["tr"]).get(key, key)


def ensure_ui_state() -> None:
    """Initialize persistent UI state once to avoid theme/language drift on reruns."""
    if "ui_theme" not in st.session_state:
        st.session_state["ui_theme"] = "light"
    if "ui_lang" not in st.session_state:
        st.session_state["ui_lang"] = "en"
    if "data_source" not in st.session_state:
        st.session_state["data_source"] = "sample"
    if "external_csv_path" not in st.session_state:
        default_external = os.path.join(
            os.path.expanduser("~"),
            "OneDrive",
            "Desktop",
            "Electric Theft Data",
            "data_extracted",
            "data.csv",
        )
        st.session_state["external_csv_path"] = default_external if os.path.exists(default_external) else ""
    if "external_upload_sig" not in st.session_state:
        st.session_state["external_upload_sig"] = ""


def canonical_risk(value: str) -> str:
    text = str(value)
    if any(token in text for token in ["Düs", "Düş", "Dus", "DÃ¼", "Low"]):
        return "low"
    if any(token in text for token in ["Yük", "Yuk", "YÃ¼", "High"]):
        return "high"
    if any(token in text for token in ["Krit", "Crit"]):
        return "critical"
    if any(token in text for token in ["Acil", "Urg"]):
        return "urgent"
    return "medium"


def risk_label(value: str) -> str:
    labels = {
        "tr": {"low": "Dusuk", "medium": "Orta", "high": "Yuksek", "critical": "Kritik", "urgent": "Acil"},
        "en": {"low": "Low", "medium": "Medium", "high": "High", "critical": "Critical", "urgent": "Urgent"},
    }
    return labels.get(st.session_state.get("ui_lang", "tr"), labels["tr"])[canonical_risk(value)]


def profile_label(value: str) -> str:
    labels = {
        "tr": {"residential": "Konut", "commercial": "Ticari", "industrial": "Sanayi"},
        "en": {"residential": "Residential", "commercial": "Commercial", "industrial": "Industrial"},
    }
    return labels.get(st.session_state.get("ui_lang", "tr"), labels["tr"]).get(value, str(value))


def inject_theme_overrides(theme: str) -> None:
    if theme == "light":
        css = """
        <style>
            :root {
                --accent:        #0ea5e9;
                --accent-soft:   rgba(14,165,233,0.12);
                --accent-glow:   rgba(14,165,233,0.28);
                --glass-shadow:  0 20px 60px rgba(15,23,42,0.10);
                --glass-hi:      inset 0 1px 0 rgba(255,255,255,0.88);
                --glass-lo:      inset 0 -1px 0 rgba(15,23,42,0.06);
                --text-main:     #0f172a;
                --text-muted:    rgba(15,23,42,0.52);
                --line-soft:     rgba(15,23,42,0.08);
                --radius-card:   26px;
            }
            html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
                background:
                    radial-gradient(ellipse 80% 50% at 10% -5%, rgba(14,165,233,0.07), transparent),
                    radial-gradient(ellipse 60% 40% at 90%  5%, rgba(129,140,248,0.05), transparent),
                    linear-gradient(175deg, #f8fafc 0%, #eef2f9 55%, #e9eef6 100%) !important;
                color: var(--text-main) !important;
            }
            [data-testid="stHeader"] {
                background: rgba(255,255,255,0.58) !important;
                border-bottom: 1px solid rgba(15,23,42,0.07) !important;
            }
            [data-testid="stSidebar"] {
                background:
                    linear-gradient(160deg, rgba(14,165,233,0.04) 0%, transparent 40%),
                    linear-gradient(180deg, rgba(255,255,255,0.85), rgba(255,255,255,0.72)) !important;
                border-right: 1px solid rgba(15,23,42,0.07) !important;
                box-shadow: 20px 0 48px rgba(15,23,42,0.08), inset 0 1px 0 rgba(255,255,255,0.90) !important;
            }
            [data-testid="stSidebar"] * { color: #0f172a !important; }
            [data-testid="stPlotlyChart"],
            [data-testid="stDataFrame"],
            [data-testid="stAlert"],
            [data-testid="stMetric"],
            div[data-baseweb="select"],
            div[data-testid="stMultiSelect"],
            div[data-testid="stSlider"],
            div[data-testid="stTextInput"],
            div[data-testid="stFileUploader"],
            [data-testid="stFileUploaderDropzone"],
            div[data-testid="stButton"] > button,
            div[data-testid="stDownloadButton"] > button,
            .hero-shell, .alert-box, .safe-box {
                background:
                    linear-gradient(180deg, rgba(255,255,255,0.90), rgba(255,255,255,0.65)) !important;
                border-color: rgba(15,23,42,0.08) !important;
                box-shadow:
                    0 12px 36px rgba(15,23,42,0.08),
                    inset 0 1px 0 rgba(255,255,255,0.95),
                    inset 0 -1px 0 rgba(15,23,42,0.04) !important;
            }
            .hero-shell { border-color: rgba(15,23,42,0.09) !important; }
            .main-header {
                background: linear-gradient(135deg, #0f172a 0%, #334155 100%) !important;
                -webkit-background-clip: text !important;
                -webkit-text-fill-color: transparent !important;
                background-clip: text !important;
                filter: none !important;
            }
            .stMarkdown, p, label, span, .stCaption, .stText, h1,h2,h3,h4 { color: #0f172a !important; }
            div[data-testid="stMetricValue"]  { color: #0f172a !important; }
            div[data-testid="stMetricLabel"],
            div[data-testid="stMetricDelta"]  { color: rgba(15,23,42,0.55) !important; }
            [data-testid="stDataFrame"] * { color: #0f172a !important; }
            div[data-testid="stTextInput"] input,
            [data-baseweb="base-input"] input,
            [data-baseweb="base-input"] {
                background: rgba(255,255,255,0.96) !important;
                color: #0f172a !important;
                -webkit-text-fill-color: #0f172a !important;
                border-color: rgba(15,23,42,0.14) !important;
            }
            .stTabs [aria-selected="true"] {
                background: linear-gradient(180deg, rgba(14,165,233,0.16), rgba(14,165,233,0.06)) !important;
                border-color: rgba(14,165,233,0.28) !important;
            }
            .stSlider [data-baseweb="slider"] > div > div {
                background: linear-gradient(90deg, var(--accent), rgba(14,165,233,0.40)) !important;
            }
            .stSlider [role="slider"] {
                background: #0ea5e9 !important;
                box-shadow: 0 0 0 3px rgba(14,165,233,0.30) !important;
            }
            [data-baseweb="popover"],
            [data-baseweb="menu"],
            [role="listbox"] {
                background: rgba(255,255,255,0.98) !important;
                color: #0f172a !important;
                border: 1px solid rgba(15,23,42,0.10) !important;
                box-shadow: 0 18px 48px rgba(15,23,42,0.14) !important;
            }
            [role="option"] {
                background: transparent !important;
                color: #0f172a !important;
            }
            [role="option"][aria-selected="true"] {
                background: rgba(14,165,233,0.10) !important;
            }
            div[data-testid="stFileUploader"] button,
            [data-testid="stFileUploaderDropzone"] button {
                background: linear-gradient(135deg, #f8fbff, #dff2ff) !important;
                color: #0f172a !important;
                border: 1px solid rgba(14,165,233,0.28) !important;
                box-shadow: 0 10px 24px rgba(14,165,233,0.10) !important;
            }
            #spot-outer {
                opacity: 0.28 !important;
                background: radial-gradient(circle, rgba(14,165,233,0.18) 0%, rgba(14,165,233,0.07) 40%, transparent 70%) !important;
                mix-blend-mode: multiply !important;
            }
            #spot-inner {
                opacity: 0.18 !important;
                mix-blend-mode: multiply !important;
            }
        </style>
        """
    else:
        css = ""  # dark mode uses the base CSS as-is
    if css:
        st.markdown(css, unsafe_allow_html=True)


def apply_plotly_theme(fig: go.Figure, theme: str) -> go.Figure:
    if theme == "light":
        tc = "#0f172a"
        axis_common = dict(
            color=tc,
            tickfont=dict(color=tc),
            title=dict(font=dict(color=tc)),
            gridcolor="rgba(15,23,42,0.08)",
            zerolinecolor="rgba(15,23,42,0.15)",
            linecolor="rgba(15,23,42,0.12)",
        )
        fig.update_layout(
            paper_bgcolor="rgba(255,255,255,0.80)",
            plot_bgcolor="rgba(255,255,255,0.35)",
            font=dict(color=tc, family="Inter, system-ui, sans-serif"),
            legend=dict(bgcolor="rgba(255,255,255,0.70)", font=dict(color=tc),
                        bordercolor="rgba(15,23,42,0.10)", borderwidth=1),
        )
        fig.update_xaxes(**axis_common)
        fig.update_yaxes(**axis_common)
    else:
        tc = "#f0f4ff"
        axis_common = dict(
            color=tc,
            tickfont=dict(color=tc),
            title=dict(font=dict(color=tc)),
            gridcolor="rgba(255,255,255,0.06)",
            zerolinecolor="rgba(255,255,255,0.10)",
            linecolor="rgba(255,255,255,0.07)",
        )
        fig.update_layout(
            paper_bgcolor="rgba(8,12,18,0.55)",
            plot_bgcolor="rgba(255,255,255,0.012)",
            font=dict(color=tc, family="Inter, system-ui, sans-serif"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=tc)),
        )
        fig.update_xaxes(**axis_common)
        fig.update_yaxes(**axis_common)
    return fig


# ========== VERI YUKLEME ==========
def _pick_column(columns, candidates):
    lower_map = {str(col).lower(): col for col in columns}
    for candidate in candidates:
        key = candidate.lower()
        if key in lower_map:
            return lower_map[key]
    return None


def _date_columns_from_dataframe(df: pd.DataFrame, skip_cols=None):
    skip = set(skip_cols or [])
    parsed = {}
    for col in df.columns:
        if col in skip:
            continue
        ts = pd.to_datetime(str(col), errors="coerce")
        if pd.notna(ts):
            parsed[col] = ts
    ordered = sorted(parsed.keys(), key=lambda c: parsed[c])
    return ordered, parsed


def _load_sample_data():
    base = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    features = pd.read_csv(os.path.join(base, "features.csv"))
    raw = pd.read_csv(os.path.join(base, "raw_consumption_sample.csv"))
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    meta = {
        "source": "sample",
        "rows": int(len(features)),
        "raw_customers": int(raw["customer_id"].nunique()) if "customer_id" in raw.columns else 0,
    }
    return features, raw, meta


def resolve_external_csv_path(path_value: str) -> str:
    raw_path = str(path_value or "").strip().strip('"').strip("'")
    if not raw_path:
        return ""

    candidate = Path(raw_path).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())

    if candidate.is_dir():
        for name in ["data.csv", "dataset.csv", "features.csv"]:
            possible = candidate / name
            if possible.exists():
                return str(possible.resolve())
        csv_files = sorted(candidate.glob("*.csv"))
        if csv_files:
            return str(csv_files[0].resolve())
    return str(candidate)


def _build_external_dataset(csv_path: str):
    ext_df = pd.read_csv(csv_path)

    id_col = _pick_column(ext_df.columns, ["CONS_NO", "customer_id", "meter_id", "id"])
    label_col = _pick_column(ext_df.columns, ["FLAG", "label", "target", "is_theft", "is_fraud"])

    if id_col is None:
        raise ValueError("ID column not found. Expected one of: CONS_NO, customer_id, meter_id, id")

    date_cols, parsed_dates = _date_columns_from_dataframe(ext_df, skip_cols=[id_col, label_col] if label_col else [id_col])
    if len(date_cols) < 30:
        raise ValueError("Date-like columns not found. Expected wide time-series columns like 2014/1/1, 2014/1/2, ...")

    work = ext_df[[id_col] + ([label_col] if label_col else []) + date_cols].copy()
    work = work.sample(frac=1.0, random_state=42).reset_index(drop=True)

    source_ids = work[id_col].astype(str)
    source_ids = source_ids.replace({"nan": "", "None": ""})
    missing_mask = source_ids.str.len() == 0
    if missing_mask.any():
        source_ids.loc[missing_mask] = [f"row_{idx}" for idx in source_ids[missing_mask].index]

    daily = work[date_cols].apply(pd.to_numeric, errors="coerce")
    valid_rows = daily.notna().sum(axis=1) > 0
    if not valid_rows.all():
        work = work.loc[valid_rows].reset_index(drop=True)
        daily = daily.loc[valid_rows].reset_index(drop=True)
        source_ids = source_ids.loc[valid_rows].reset_index(drop=True)
    if len(work) == 0:
        raise ValueError("No usable consumption rows found in CSV.")

    arr = daily.to_numpy(dtype=np.float32)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_consumption = np.nanmean(arr, axis=1)
        std_consumption = np.nanstd(arr, axis=1)
        min_consumption = np.nanmin(arr, axis=1)
        max_consumption = np.nanmax(arr, axis=1)
        median_consumption = np.nanmedian(arr, axis=1)
        q25 = np.nanpercentile(arr, 25, axis=1)
        q75 = np.nanpercentile(arr, 75, axis=1)

    mean_consumption = np.nan_to_num(mean_consumption, nan=0.0)
    std_consumption = np.nan_to_num(std_consumption, nan=0.0)
    min_consumption = np.nan_to_num(min_consumption, nan=0.0)
    max_consumption = np.nan_to_num(max_consumption, nan=0.0)
    median_consumption = np.nan_to_num(median_consumption, nan=0.0)
    q25 = np.nan_to_num(q25, nan=0.0)
    q75 = np.nan_to_num(q75, nan=0.0)
    iqr = q75 - q25

    skewness = daily.skew(axis=1, numeric_only=True).fillna(0.0).to_numpy(dtype=np.float32)
    kurtosis = daily.kurt(axis=1, numeric_only=True).fillna(0.0).to_numpy(dtype=np.float32)

    with np.errstate(invalid="ignore"):
        zero_measurement_pct = np.nanmean((arr <= 1e-4).astype(np.float32), axis=1)
    zero_measurement_pct = np.nan_to_num(zero_measurement_pct, nan=0.0)

    filled = np.where(np.isnan(arr), np.nanmedian(arr, axis=1, keepdims=True), arr)
    filled = np.nan_to_num(filled, nan=0.0)

    if filled.shape[1] > 1:
        diffs = np.abs(np.diff(filled, axis=1))
        with np.errstate(invalid="ignore", divide="ignore"):
            sudden_change_ratio = np.nanmean(diffs, axis=1) / (np.abs(mean_consumption) + 1e-6)
        sudden_change_ratio = np.clip(np.nan_to_num(sudden_change_ratio, nan=0.0), 0.0, 1.0)
    else:
        sudden_change_ratio = np.zeros(len(filled), dtype=np.float32)

    x = np.arange(filled.shape[1], dtype=np.float32)
    x_centered = x - x.mean()
    denom = float((x_centered ** 2).sum()) + 1e-8
    trend_slope = (filled @ x_centered) / denom
    trend_slope = np.nan_to_num(trend_slope, nan=0.0, posinf=0.0, neginf=0.0)

    date_index = pd.DatetimeIndex([parsed_dates[c] for c in date_cols])
    is_weekend = np.array([d.weekday() >= 5 for d in date_index], dtype=bool)
    if is_weekend.any() and (~is_weekend).any():
        weekend_mean = np.nanmean(arr[:, is_weekend], axis=1)
        weekday_mean = np.nanmean(arr[:, ~is_weekend], axis=1)
        weekend_weekday_ratio = np.nan_to_num(weekend_mean / (weekday_mean + 1e-6), nan=1.0, posinf=1.0, neginf=1.0)
    else:
        weekend_weekday_ratio = np.ones(len(arr), dtype=np.float32)

    night_day_ratio = np.clip(1.0 + (zero_measurement_pct - np.nanmean(zero_measurement_pct)) * 2.0, 0.2, 3.5)
    peak_hour = np.full(len(arr), 12.0, dtype=np.float32)
    mean_daily_total = mean_consumption.copy()
    std_daily_total = std_consumption.copy()
    cv_daily = np.nan_to_num(std_daily_total / (mean_daily_total + 1e-6), nan=0.0, posinf=0.0, neginf=0.0)

    # ── Gelişmiş ayırt edici özellikler (SGCC/gerçek veri için) ────────────────

    # 1. En uzun ardışık sıfır serisi — hırsızlıkta uzun kesintiler olur
    def _max_zero_run(row: np.ndarray) -> int:
        best = cur = 0
        for v in row:
            if np.isnan(v) or v <= 1e-4:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        return best

    max_zero_run = np.array([_max_zero_run(filled[i]) for i in range(len(filled))], dtype=np.float32)

    # 2. Tüketim trendi: son %33 ortalama / ilk %33 ortalama
    #    Hırsızlıkta zaman içinde tüketim düşer → oran < 1
    n3 = max(1, filled.shape[1] // 3)
    early_mean = np.nanmean(filled[:, :n3], axis=1)
    late_mean  = np.nanmean(filled[:, -n3:], axis=1)
    relative_trend = np.nan_to_num(late_mean / (early_mean + 1e-6), nan=1.0, posinf=1.0, neginf=0.0)
    relative_trend = np.clip(relative_trend, 0.0, 5.0).astype(np.float32)

    # 3. Aylık toplam tüketimlerin değişkenlik katsayısı
    #    Ani düşüş/çıkış olan aylarda CV yüksek olur
    date_dt = pd.DatetimeIndex([parsed_dates[c] for c in date_cols])
    months = date_dt.to_period("M")
    unique_months = sorted(set(months))
    if len(unique_months) >= 2:
        monthly_totals = np.zeros((len(filled), len(unique_months)), dtype=np.float32)
        for mi, m in enumerate(unique_months):
            mask = np.array([mo == m for mo in months])
            monthly_totals[:, mi] = np.nanmean(filled[:, mask], axis=1)
        monthly_cv = np.nan_to_num(
            np.nanstd(monthly_totals, axis=1) / (np.nanmean(monthly_totals, axis=1) + 1e-6),
            nan=0.0, posinf=0.0
        ).astype(np.float32)
    else:
        monthly_cv = np.zeros(len(filled), dtype=np.float32)

    # 4. En yüksek %5'lik tüketim ortalaması — pik kırpma tespiti
    #    Normal müşterilerde pik/ortalama oranı yüksek, kırpmalıda düşük
    p95 = np.nanpercentile(filled, 95, axis=1)
    peak_to_mean = np.nan_to_num(p95 / (mean_consumption + 1e-6), nan=1.0, posinf=1.0).astype(np.float32)

    # 5. Değişim noktası skoru — ardışık günler arası büyük sıçrama sayısı
    #    Sayaç manipülasyonunda ani seviye değişimleri olur
    if filled.shape[1] > 1:
        day_diffs = np.diff(filled, axis=1)
        threshold_per_row = np.nanstd(filled, axis=1, keepdims=True) * 2.0
        changepoints = np.nan_to_num(
            (np.abs(day_diffs) > threshold_per_row).sum(axis=1) / filled.shape[1],
            nan=0.0
        ).astype(np.float32)
    else:
        changepoints = np.zeros(len(filled), dtype=np.float32)

    if label_col:
        labels = pd.to_numeric(work[label_col], errors="coerce").fillna(0).astype(int).to_numpy()
        labels = (labels > 0).astype(int)
    else:
        labels = np.zeros(len(work), dtype=int)

    q_33 = np.nanquantile(mean_consumption, 0.33) if len(mean_consumption) > 2 else np.nanmedian(mean_consumption)
    q_66 = np.nanquantile(mean_consumption, 0.66) if len(mean_consumption) > 2 else np.nanmedian(mean_consumption)
    profiles = np.where(
        mean_consumption <= q_33,
        "residential",
        np.where(mean_consumption <= q_66, "commercial", "industrial"),
    )

    theft_types = np.array(["constant_reduction", "night_zeroing", "random_zeros", "gradual_decrease", "peak_clipping"])
    theft_type_values = np.full(len(labels), "none", dtype=object)
    pos_idx = np.where(labels == 1)[0]
    if len(pos_idx) > 0:
        theft_type_values[pos_idx] = theft_types[np.mod(np.arange(len(pos_idx)), len(theft_types))]

    features = pd.DataFrame(
        {
            "customer_id": np.arange(len(work), dtype=int),
            "profile": profiles,
            "label": labels.astype(int),
            "theft_type": theft_type_values,
            "source_customer_ref": source_ids.values,
            "mean_consumption": mean_consumption,
            "std_consumption": std_consumption,
            "min_consumption": min_consumption,
            "max_consumption": max_consumption,
            "median_consumption": median_consumption,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "mean_daily_total": mean_daily_total,
            "std_daily_total": std_daily_total,
            "cv_daily": cv_daily,
            "night_day_ratio": night_day_ratio,
            "weekend_weekday_ratio": weekend_weekday_ratio,
            "peak_hour": peak_hour,
            "zero_measurement_pct": zero_measurement_pct,
            "sudden_change_ratio": sudden_change_ratio,
            "trend_slope": trend_slope,
            "q25": q25,
            "q75": q75,
            "iqr": iqr,
            # Gelişmiş özellikler
            "max_zero_run": max_zero_run,
            "relative_trend": relative_trend,
            "monthly_cv": monthly_cv,
            "peak_to_mean": peak_to_mean,
            "changepoint_rate": changepoints,
        }
    )

    numeric_cols = features.select_dtypes(include=[np.number]).columns
    features[numeric_cols] = features[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    raw_customer_limit = min(260, len(work))
    raw_slice = work.iloc[:raw_customer_limit][[id_col] + date_cols].copy()
    raw_long = raw_slice.melt(id_vars=[id_col], var_name="timestamp", value_name="consumption_kw")
    raw_long["timestamp"] = pd.to_datetime(raw_long["timestamp"], errors="coerce")
    raw_long["consumption_kw"] = pd.to_numeric(raw_long["consumption_kw"], errors="coerce")
    raw_long = raw_long.dropna(subset=["timestamp", "consumption_kw"])

    id_map = dict(zip(source_ids.astype(str), features["customer_id"]))
    raw_long["customer_id"] = raw_long[id_col].astype(str).map(id_map)
    raw_long = raw_long.dropna(subset=["customer_id"])
    raw_long["customer_id"] = raw_long["customer_id"].astype(int)
    label_map = features.set_index("customer_id")["label"].to_dict()
    raw_long["label"] = raw_long["customer_id"].map(label_map).fillna(0).astype(int)
    raw = raw_long[["customer_id", "timestamp", "consumption_kw", "label"]].sort_values(["customer_id", "timestamp"])

    meta = {
        "source": "external",
        "rows": int(len(features)),
        "raw_customers": int(raw_customer_limit),
        "path": csv_path,
        "id_col": id_col,
        "label_col": label_col or "N/A",
        "date_cols": int(len(date_cols)),
    }
    return features, raw, meta


@st.cache_data
def load_data(source: str = "sample", external_csv_path: str = ""):
    if source == "external":
        path = resolve_external_csv_path(external_csv_path)
        if not path:
            raise ValueError("External CSV path is empty.")
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV file not found: {path}")
        return _build_external_dataset(path)

    return _load_sample_data()


@st.cache_data
def run_models(features_df):
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, roc_auc_score, f1_score

    def safe_roc_curve(y_true, y_score):
        if len(np.unique(y_true)) < 2:
            return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])
        return roc_curve(y_true, y_score)

    def safe_pr_curve(y_true, y_score):
        if len(np.unique(y_true)) < 2:
            return np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.5])
        return precision_recall_curve(y_true, y_score)

    def safe_auc(y_true, y_score):
        if len(np.unique(y_true)) < 2:
            return 0.5
        return roc_auc_score(y_true, y_score)

    from sklearn.preprocessing import RobustScaler

    features_df = features_df.copy()
    meta_cols = ["customer_id", "profile", "label", "theft_type", "source_customer_ref"]
    feature_cols = [c for c in features_df.columns if c not in meta_cols]
    features_df[feature_cols] = (
        features_df[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    X = features_df[feature_cols].to_numpy(dtype=np.float32)
    y = pd.to_numeric(features_df["label"], errors="coerce").fillna(0).astype(int).to_numpy()
    y = (y > 0).astype(int)

    # RobustScaler: outlier'lara (800k gibi) karşı StandardScaler'dan çok daha dayanıklı
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.clip(X_scaled, -10, 10)  # aşırı uçları sınırla

    class_counts = pd.Series(y).value_counts()
    pos_rate = float(np.mean(y)) if len(y) else 0.05
    stratify_y = y if (len(class_counts) > 1 and class_counts.min() >= 2) else None
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_scaled, y, np.arange(len(y)), test_size=0.25, random_state=42, stratify=stratify_y
    )

    # Isolation Forest: gerçek pozitif orana göre contamination ayarla
    iso_contamination = float(np.clip(pos_rate, 0.01, 0.20))
    iso = IsolationForest(n_estimators=300, contamination=iso_contamination,
                          max_features=0.8, random_state=42)
    iso.fit(X_train)  # sadece train seti üzerinde fit et (veri sızıntısını önle)
    iso_scores = -iso.score_samples(X_scaled)
    iso_scores_test = -iso.score_samples(X_test)
    iso_preds_test = (iso.predict(X_test) == -1).astype(int)

    can_train_supervised = len(np.unique(y_train)) > 1
    if can_train_supervised:
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        # Sınıf dengesizliği için ağırlık: azınlık sınıfını daha güçlü ağırlıkla
        cw = {0: 1.0, 1: max(2.0, neg_count / max(pos_count, 1) * 0.7)}
        rf = RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=5,
            class_weight=cw, max_features="sqrt", random_state=42, n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        rf_probs_all = rf.predict_proba(X_scaled)[:, 1]
        rf_probs_test = rf.predict_proba(X_test)[:, 1]
        # Eşiği class oranına göre otomatik ayarla (0.5 yerine)
        optimal_thresh = np.clip(pos_rate * 1.5, 0.20, 0.60)
        rf_preds_test = (rf_probs_test >= optimal_thresh).astype(int)
        importance = dict(zip(feature_cols, rf.feature_importances_))
    else:
        iso_min = float(np.min(iso_scores))
        iso_span = float(np.max(iso_scores) - iso_min) + 1e-8
        rf_probs_all = (iso_scores - iso_min) / iso_span
        rf_probs_test = rf_probs_all[idx_test]
        rf_preds_test = (rf_probs_test >= 0.5).astype(int)
        importance = {name: 0.0 for name in feature_cols}

    rf_fpr, rf_tpr, _ = safe_roc_curve(y_test, rf_probs_test)
    iso_fpr, iso_tpr, _ = safe_roc_curve(y_test, iso_scores_test)
    rf_prec, rf_rec, _ = safe_pr_curve(y_test, rf_probs_test)
    iso_prec, iso_rec, _ = safe_pr_curve(y_test, iso_scores_test)

    rf_cm = confusion_matrix(y_test, rf_preds_test, labels=[0, 1])
    iso_cm = confusion_matrix(y_test, iso_preds_test, labels=[0, 1])

    rf_auc = safe_auc(y_test, rf_probs_test)
    iso_auc = safe_auc(y_test, iso_scores_test)
    rf_f1 = f1_score(y_test, rf_preds_test, zero_division=0)
    iso_f1 = f1_score(y_test, iso_preds_test, zero_division=0)

    xgb_probs_all = None
    xgb_auc = 0.0
    xgb_f1 = 0.0
    xgb_fpr = np.array([])
    xgb_tpr = np.array([])
    primary_probs = rf_probs_all
    best_model_name = f"Random Forest (AUC {rf_auc:.3f})"

    if can_train_supervised:
        try:
            from xgboost import XGBClassifier

            neg_c, pos_c = (y_train == 0).sum(), (y_train == 1).sum()
            xgb = XGBClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=neg_c / max(pos_c, 1),
                min_child_weight=5,
                eval_metric="aucpr",  # imbalanced için PR-AUC daha anlamlı
                random_state=42,
                verbosity=0,
                n_jobs=-1,
            )
            xgb.fit(X_train, y_train)
            xgb_probs_all = xgb.predict_proba(X_scaled)[:, 1]
            xgb_probs_test = xgb.predict_proba(X_test)[:, 1]
            xgb_preds_test = (xgb_probs_test >= optimal_thresh).astype(int)
            xgb_auc = safe_auc(y_test, xgb_probs_test)
            xgb_f1 = f1_score(y_test, xgb_preds_test, zero_division=0)
            xgb_fpr, xgb_tpr, _ = safe_roc_curve(y_test, xgb_probs_test)
            if xgb_auc > rf_auc:
                primary_probs = xgb_probs_all
                best_model_name = f"XGBoost (AUC {xgb_auc:.3f})"
        except Exception as _xgb_exc:
            _logger.warning("run_models: XGBoost training skipped: %s", _xgb_exc)

    features_df["anomaly_score"] = iso_scores
    features_df["theft_probability"] = primary_probs
    features_df["predicted_theft"] = (primary_probs >= 0.5).astype(int)

    # 5-tier risk band (desktop uygulamasıyla aynı eşikler)
    features_df['risk_band'] = pd.cut(
        features_df['theft_probability'],
        bins=[-0.001, 0.30, 0.45, 0.70, 0.85, 1.001],
        labels=['Düşük', 'Orta', 'Yüksek', 'Kritik', 'Acil']
    )
    # Risk Score 0-100
    features_df['risk_score'] = (features_df['theft_probability'] * 100).round(1)

    # Tahmini aylık kayıp (TRY)
    tariff = features_df['profile'].map(
        {'residential': 2.28, 'commercial': 2.85, 'industrial': 1.92}
    ).fillna(2.15)
    base_loss = features_df.get('mean_consumption', pd.Series(np.ones(len(features_df)) * 10.0))
    features_df['est_monthly_loss'] = np.where(
        features_df['theft_probability'] >= 0.5,
        (base_loss * 30 * tariff * features_df['theft_probability']).round(0),
        0.0
    )

    # Eski risk_level uyumluluk için tut
    features_df["risk_level"] = features_df["risk_band"]

    metrics = {
        "rf_fpr": rf_fpr,
        "rf_tpr": rf_tpr,
        "rf_auc": rf_auc,
        "rf_f1": rf_f1,
        "rf_cm": rf_cm,
        "iso_fpr": iso_fpr,
        "iso_tpr": iso_tpr,
        "iso_auc": iso_auc,
        "iso_f1": iso_f1,
        "iso_cm": iso_cm,
        "rf_prec": rf_prec,
        "rf_rec": rf_rec,
        "iso_prec": iso_prec,
        "iso_rec": iso_rec,
        "importance": importance,
        "feature_cols": feature_cols,
        "y_test": y_test,
        "rf_probs_test": rf_probs_test,
        "iso_scores_test": iso_scores_test,
        "xgb_fpr": xgb_fpr,
        "xgb_tpr": xgb_tpr,
        "xgb_auc": xgb_auc,
        "xgb_f1": xgb_f1,
        "best_model": best_model_name,
    }

    return features_df, metrics


@st.cache_data
def run_models_v2(features_df):
    from sklearn.ensemble import GradientBoostingClassifier, IsolationForest, RandomForestClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import RobustScaler

    def safe_roc_curve(y_true, y_score):
        if len(np.unique(y_true)) < 2:
            return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])
        return roc_curve(y_true, y_score)

    def safe_pr_curve(y_true, y_score):
        if len(np.unique(y_true)) < 2:
            return np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.5])
        return precision_recall_curve(y_true, y_score)

    def safe_auc(y_true, y_score):
        if len(np.unique(y_true)) < 2:
            return 0.5
        return roc_auc_score(y_true, y_score)

    def safe_ap(y_true, y_score):
        if len(np.unique(y_true)) < 2:
            return 0.0
        return average_precision_score(y_true, y_score)

    features_df = features_df.copy()
    meta_cols = ["customer_id", "profile", "label", "theft_type", "source_customer_ref"]
    feature_cols = [c for c in features_df.columns if c not in meta_cols]
    features_df[feature_cols] = (
        features_df[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    X = features_df[feature_cols].to_numpy(dtype=np.float32)
    y = pd.to_numeric(features_df["label"], errors="coerce").fillna(0).astype(int).to_numpy()
    y = (y > 0).astype(int)

    scaler = RobustScaler()
    X_scaled = np.clip(scaler.fit_transform(X), -10, 10)

    class_counts = pd.Series(y).value_counts()
    pos_rate = float(np.mean(y)) if len(y) else 0.05
    stratify_y = y if (len(class_counts) > 1 and class_counts.min() >= 2) else None
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_scaled, y, np.arange(len(y)), test_size=0.25, random_state=42, stratify=stratify_y
    )

    iso_contamination = float(np.clip(pos_rate, 0.01, 0.20))
    iso = IsolationForest(n_estimators=300, contamination=iso_contamination, max_features=0.8, random_state=42)
    iso.fit(X_train)
    iso_scores = -iso.score_samples(X_scaled)
    iso_scores_test = -iso.score_samples(X_test)
    iso_preds_test = (iso.predict(X_test) == -1).astype(int)

    train_class_counts = pd.Series(y_train).value_counts()
    can_train_supervised = len(np.unique(y_train)) > 1 and train_class_counts.min() >= 2
    optimal_thresh = 0.5
    if can_train_supervised:
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        cw = {0: 1.0, 1: max(2.0, neg_count / max(pos_count, 1) * 0.7)}
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=5,
            class_weight=cw,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        rf_probs_all = rf.predict_proba(X_scaled)[:, 1]
        rf_probs_test = rf.predict_proba(X_test)[:, 1]
        optimal_thresh = float(np.clip(pos_rate * 1.5, 0.20, 0.60))
        rf_preds_test = (rf_probs_test >= optimal_thresh).astype(int)
        importance = dict(zip(feature_cols, rf.feature_importances_))
    else:
        iso_min = float(np.min(iso_scores))
        iso_span = float(np.max(iso_scores) - iso_min) + 1e-8
        rf_probs_all = (iso_scores - iso_min) / iso_span
        rf_probs_test = rf_probs_all[idx_test]
        rf_preds_test = (rf_probs_test >= 0.5).astype(int)
        importance = {name: 0.0 for name in feature_cols}

    rf_fpr, rf_tpr, _ = safe_roc_curve(y_test, rf_probs_test)
    iso_fpr, iso_tpr, _ = safe_roc_curve(y_test, iso_scores_test)
    rf_prec, rf_rec, _ = safe_pr_curve(y_test, rf_probs_test)
    iso_prec, iso_rec, _ = safe_pr_curve(y_test, iso_scores_test)
    rf_cm = confusion_matrix(y_test, rf_preds_test, labels=[0, 1])
    iso_cm = confusion_matrix(y_test, iso_preds_test, labels=[0, 1])
    rf_auc = safe_auc(y_test, rf_probs_test)
    iso_auc = safe_auc(y_test, iso_scores_test)
    rf_f1 = f1_score(y_test, rf_preds_test, zero_division=0)
    iso_f1 = f1_score(y_test, iso_preds_test, zero_division=0)

    primary_probs = rf_probs_all
    best_model_name = f"Random Forest (AUC {rf_auc:.3f})"
    best_model_key = "Random Forest"
    best_preds_test = rf_preds_test
    model_curves = {}
    model_rows = []
    xgb_probs_test = np.array([])

    def register_model(name, model_type, y_score, y_pred, importances=None):
        fpr, tpr, _ = safe_roc_curve(y_test, y_score)
        precision, recall, _ = safe_pr_curve(y_test, y_score)
        row = {
            "Model": name,
            "Type": model_type,
            "ROC-AUC": safe_auc(y_test, y_score),
            "PR-AUC": safe_ap(y_test, y_score),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0),
        }
        model_rows.append(row)
        model_curves[name] = {
            "fpr": fpr,
            "tpr": tpr,
            "precision": precision,
            "recall": recall,
            "auc": row["ROC-AUC"],
            "ap": row["PR-AUC"],
            "pred": y_pred,
        }
        if importances is not None:
            model_curves[name]["importance"] = importances

    if can_train_supervised:
        try:
            from xgboost import XGBClassifier

            neg_c, pos_c = (y_train == 0).sum(), (y_train == 1).sum()
            xgb = XGBClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=neg_c / max(pos_c, 1),
                min_child_weight=5,
                eval_metric="aucpr",
                random_state=42,
                verbosity=0,
                n_jobs=-1,
            )
            xgb.fit(X_train, y_train)
            xgb_probs_all = xgb.predict_proba(X_scaled)[:, 1]
            xgb_probs_test = xgb.predict_proba(X_test)[:, 1]
            xgb_preds_test = (xgb_probs_test >= optimal_thresh).astype(int)
            xgb_auc = safe_auc(y_test, xgb_probs_test)
            if xgb_auc > rf_auc:
                primary_probs = xgb_probs_all
                best_model_name = f"XGBoost (AUC {xgb_auc:.3f})"
                best_model_key = "XGBoost"
                best_preds_test = xgb_preds_test
        except Exception as _xgb_exc:
            _logger.warning("run_models_v2: XGBoost training skipped: %s", _xgb_exc)

        log_reg = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
        log_reg.fit(X_train, y_train)
        lr_probs_test = log_reg.predict_proba(X_test)[:, 1]
        lr_preds_test = (lr_probs_test >= optimal_thresh).astype(int)

        gb = GradientBoostingClassifier(n_estimators=220, learning_rate=0.05, subsample=0.82, random_state=42)
        gb.fit(X_train, y_train)
        gb_probs_test = gb.predict_proba(X_test)[:, 1]
        gb_preds_test = (gb_probs_test >= optimal_thresh).astype(int)

        stack_cv = int(max(2, min(5, train_class_counts.min())))
        stack = StackingClassifier(
            estimators=[
                ("lr", LogisticRegression(class_weight="balanced", max_iter=1200, random_state=42)),
                ("rf", RandomForestClassifier(n_estimators=150, max_depth=8, class_weight="balanced", random_state=42)),
                ("gb", GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, random_state=42)),
            ],
            final_estimator=LogisticRegression(max_iter=1000),
            cv=stack_cv,
            stack_method="predict_proba",
            n_jobs=-1,
        )
        stack.fit(X_train, y_train)
        stack_probs_all = stack.predict_proba(X_scaled)[:, 1]
        stack_probs_test = stack.predict_proba(X_test)[:, 1]
        stack_preds_test = (stack_probs_test >= optimal_thresh).astype(int)
        stack_auc = safe_auc(y_test, stack_probs_test)
        if stack_auc > safe_auc(y_test, primary_probs[idx_test]):
            primary_probs = stack_probs_all
            best_model_name = f"Stacking Ensemble (AUC {stack_auc:.3f})"
            best_model_key = "Stacking Ensemble"
            best_preds_test = stack_preds_test

        register_model("Random Forest", "Supervised", rf_probs_test, rf_preds_test, importance)
        if len(xgb_probs_test):
            register_model("XGBoost", "Supervised", xgb_probs_test, xgb_preds_test)
        register_model("Logistic Regression", "Supervised", lr_probs_test, lr_preds_test)
        register_model("Gradient Boosting", "Supervised", gb_probs_test, gb_preds_test)
        register_model("Stacking Ensemble", "Meta Learning", stack_probs_test, stack_preds_test)
    else:
        register_model("Random Forest", "Fallback", rf_probs_test, rf_preds_test, importance)

    register_model("Isolation Forest", "Unsupervised", iso_scores_test, iso_preds_test)

    features_df["anomaly_score"] = iso_scores
    features_df["theft_probability"] = primary_probs
    features_df["predicted_theft"] = (primary_probs >= optimal_thresh).astype(int)
    features_df["risk_band"] = pd.cut(
        features_df["theft_probability"],
        bins=[-0.001, 0.30, 0.45, 0.70, 0.85, 1.001],
        labels=["DÃ¼ÅŸÃ¼k", "Orta", "YÃ¼ksek", "Kritik", "Acil"],
    )
    features_df["risk_score"] = (features_df["theft_probability"] * 100).round(1)

    tariff = features_df["profile"].map({"residential": 2.28, "commercial": 2.85, "industrial": 1.92}).fillna(2.15)
    base_loss = features_df.get("mean_consumption", pd.Series(np.ones(len(features_df)) * 10.0))
    features_df["est_monthly_loss"] = np.where(
        features_df["theft_probability"] >= optimal_thresh,
        (base_loss * 30 * tariff * features_df["theft_probability"]).round(0),
        0.0,
    )
    features_df["risk_level"] = features_df["risk_band"]
    features_df["risk_category"] = features_df["risk_band"]
    features_df["priority_index"] = np.round(
        features_df["risk_score"] * 0.65
        + features_df["est_monthly_loss"].clip(upper=5000) * 0.01,
        2,
    )

    try:
        from mass_ai_engine import MassAIEngine

        features_df = MassAIEngine()._build_explainability_columns(features_df)
    except Exception as exc:
        _logger.warning("run_models_v2: explainability enrichment skipped: %s", exc)
        fallback_reason = np.where(
            features_df["theft_probability"] >= optimal_thresh,
            "high theft probability",
            "variance monitor",
        )
        features_df["risk_reason_1"] = fallback_reason
        features_df["risk_reason_2"] = "-"
        features_df["risk_reason_3"] = "-"
        features_df["risk_drivers"] = fallback_reason
        features_df["risk_summary"] = np.where(
            features_df["theft_probability"] >= optimal_thresh,
            "High risk case should be reviewed first.",
            "Customer remains in monitoring queue.",
        )

    metrics = {
        "rf_fpr": rf_fpr,
        "rf_tpr": rf_tpr,
        "rf_auc": rf_auc,
        "rf_f1": rf_f1,
        "rf_cm": rf_cm,
        "iso_fpr": iso_fpr,
        "iso_tpr": iso_tpr,
        "iso_auc": iso_auc,
        "iso_f1": iso_f1,
        "iso_cm": iso_cm,
        "rf_prec": rf_prec,
        "rf_rec": rf_rec,
        "iso_prec": iso_prec,
        "iso_rec": iso_rec,
        "importance": importance,
        "feature_cols": feature_cols,
        "y_test": y_test,
        "rf_probs_test": rf_probs_test,
        "iso_scores_test": iso_scores_test,
        "best_model": best_model_name,
        "best_model_key": best_model_key,
        "best_threshold": float(optimal_thresh),
        "best_preds_test": best_preds_test,
        "model_curves": model_curves,
        "model_table": pd.DataFrame(model_rows).sort_values(["ROC-AUC", "PR-AUC"], ascending=False).reset_index(drop=True),
    }

    return features_df, metrics


# ========== SIDEBAR ==========
def render_sidebar(features_df, data_meta):
    ensure_ui_state()

    st.sidebar.markdown("## MASS-AI v2.0")
    st.sidebar.markdown(f"*{tr('sidebar_tagline')}*")
    st.sidebar.markdown("---")

    st.sidebar.markdown(f"### {tr('appearance')}")
    st.sidebar.selectbox(
        tr("theme"),
        options=["dark", "light"],
        format_func=lambda x: tr("theme_dark") if x == "dark" else tr("theme_light"),
        key="ui_theme",
    )
    st.sidebar.selectbox(
        tr("language"),
        options=["tr", "en"],
        format_func=lambda x: "Turkce" if x == "tr" else "English",
        key="ui_lang",
    )
    st.sidebar.markdown("---")

    st.sidebar.markdown(f"### {tr('data_source')}")
    st.sidebar.selectbox(
        tr("data_source"),
        options=["sample", "external"],
        format_func=lambda x: tr("data_source_sample") if x == "sample" else tr("data_source_external"),
        key="data_source",
    )
    if st.session_state.get("data_source") == "external":
        external_path_input = st.sidebar.text_input(
            tr("external_csv_path"),
            value=st.session_state.get("external_csv_path", ""),
        )
        resolved_input = resolve_external_csv_path(external_path_input)
        if st.session_state.get("external_csv_path") != resolved_input:
            st.session_state["external_csv_path"] = resolved_input
            load_data.clear()
            st.session_state["_need_rerun"] = True
        st.sidebar.caption(tr("external_csv_hint"))
        uploaded_csv = st.sidebar.file_uploader(
            tr("external_csv_upload"),
            type=["csv"],
            accept_multiple_files=False,
            key="external_csv_upload_widget",
        )
        if uploaded_csv is not None:
            uploads_dir = os.path.join(os.path.dirname(__file__), "..", "data", "uploads")
            os.makedirs(uploads_dir, exist_ok=True)
            safe_name = os.path.basename(uploaded_csv.name) or "external_upload.csv"
            target_path = os.path.join(uploads_dir, safe_name)
            upload_sig = f"{safe_name}:{uploaded_csv.size}"

            if st.session_state.get("external_upload_sig") != upload_sig or not os.path.exists(target_path):
                with open(target_path, "wb") as fout:
                    fout.write(uploaded_csv.getbuffer())
                st.session_state["external_upload_sig"] = upload_sig

            st.sidebar.caption(f"{tr('external_selected_file')}: {safe_name}")
            st.sidebar.caption(f"{tr('external_saved_path')}: {target_path}")
            if st.session_state.get("external_csv_path") != target_path:
                st.session_state["external_csv_path"] = target_path
                load_data.clear()
                st.session_state["_need_rerun"] = True
    if st.session_state.get("data_source") == "external":
        path_now = resolve_external_csv_path(st.session_state.get("external_csv_path", ""))
        if not path_now:
            st.sidebar.warning("CSV yolu girilmedi — örnek veri gösteriliyor." if st.session_state.get("ui_lang","tr") == "tr" else "No CSV path set — showing sample data.")
        elif not os.path.exists(path_now):
            st.sidebar.error("Dosya bulunamadı." if st.session_state.get("ui_lang","tr") == "tr" else "File not found.")
        elif data_meta.get("source") == "external":
            st.sidebar.success(
                f"{tr('external_loaded')}: {data_meta.get('rows', 0):,} | "
                f"{tr('external_preview')}: {data_meta.get('raw_customers', 0)}"
            )
            st.sidebar.caption(f"{tr('external_saved_path')}: {path_now}")
    st.sidebar.markdown("---")

    st.sidebar.markdown(f"### {tr('filters')}")
    profile_filter = st.sidebar.multiselect(
        tr("customer_profile"),
        options=["residential", "commercial", "industrial"],
        default=["residential", "commercial", "industrial"],
        format_func=profile_label,
    )

    risk_options = sorted(
        features_df["risk_band"].astype(str).unique().tolist(),
        key=lambda item: ["low", "medium", "high", "critical", "urgent"].index(canonical_risk(item)),
    )
    risk_filter = st.sidebar.multiselect(
        tr("risk_band"),
        options=risk_options,
        default=risk_options,
        format_func=risk_label,
    )

    prob_threshold = st.sidebar.slider(tr("threshold"), 0.0, 1.0, 0.5, 0.05)

    filtered = features_df[
        (features_df["profile"].isin(profile_filter)) &
        (features_df["risk_band"].isin(risk_filter))
    ]

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**{tr('shown')}:** {len(filtered)} / {len(features_df)}")
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### {tr('project_info')}")
    st.sidebar.markdown("**Omer Burak Kocak**")
    st.sidebar.markdown("Marmara Uni. EEE | 2026")

    return filtered, prob_threshold


@st.cache_data
def _make_city_coords(n: int) -> tuple:
    """Return (lats, lons, city_names) lists of length *n* with a fixed random seed."""
    rng = np.random.default_rng(42)
    cities = {
        "Istanbul": (41.01, 28.97, 0.35), "Ankara": (39.93, 32.86, 0.15),
        "Izmir": (38.42, 27.14, 0.10), "Diyarbakir": (37.91, 40.22, 0.10),
        "Antalya": (36.90, 30.69, 0.08), "Adana": (37.00, 35.32, 0.07),
        "Bursa": (40.19, 29.06, 0.08), "Gaziantep": (37.06, 37.38, 0.07),
    }
    names = list(cities.keys())
    probs = [v[2] for v in cities.values()]
    chosen = rng.choice(names, size=n, p=probs)
    lats = [cities[c][0] + rng.normal(0, 0.3) for c in chosen]
    lons = [cities[c][1] + rng.normal(0, 0.3) for c in chosen]
    return lats, lons, list(chosen)


# ========== TAB 1: GENEL BAKIS ==========
def render_overview(df, threshold, raw_df):
    theme = st.session_state.get("ui_theme", "dark")

    total = len(df)
    detected = (df["theft_probability"] >= threshold).sum()
    detection_rate = detected / total * 100 if total else 0
    _risk_keys = df["risk_band"].astype(str).apply(canonical_risk)
    urgent_count = (_risk_keys == "urgent").sum()
    critical_count = (_risk_keys == "critical").sum()
    monthly_loss = df["est_monthly_loss"].sum() if "est_monthly_loss" in df.columns else 0

    # ── Row 1 — Action summary (decision-first hierarchy) ────────────────────
    hot_region = "-"
    hot_region_score = 0.0
    if "region" in df.columns and len(df) > 0:
        region_stats = df.groupby("region")["risk_score"].mean().sort_values(ascending=False)
        if not region_stats.empty:
            hot_region = str(region_stats.index[0])
            hot_region_score = float(region_stats.iloc[0])

    pending_investigations = int(critical_count + urgent_count)
    high_risk_tone = "danger" if detected > 0 else "ok"
    region_tone = "warn" if hot_region_score >= 60 else "neutral"
    pending_tone = "danger" if pending_investigations > 0 else "ok"

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        action_card(
            label=tr("high_risk_customers") if "high_risk_customers" in (_I18N.get(st.session_state.get("ui_lang", "tr"), {}) if "_I18N" in globals() else {}) else "Yuksek riskli musteri" if st.session_state.get("ui_lang", "tr") == "tr" else "High-risk customers",
            value=f"{int(detected)}",
            subtitle=f"%{detection_rate:.1f} " + ("of portfolio - open worklist" if st.session_state.get("ui_lang", "tr") == "en" else "portfoy - worklist'te incele"),
            tone=high_risk_tone,
        )
    with col_b:
        action_card(
            label="En sicak bolge" if st.session_state.get("ui_lang", "tr") == "tr" else "Hottest region",
            value=hot_region.title() if hot_region != "-" else "-",
            subtitle=("ortalama risk skoru " if st.session_state.get("ui_lang", "tr") == "tr" else "avg risk score ") + f"{hot_region_score:.0f}",
            tone=region_tone,
        )
    with col_c:
        action_card(
            label="Bekleyen inceleme" if st.session_state.get("ui_lang", "tr") == "tr" else "Pending investigations",
            value=f"{pending_investigations}",
            subtitle=f"{critical_count} critical + {urgent_count} urgent",
            tone=pending_tone,
        )

    with st.expander("KPI detay" if st.session_state.get("ui_lang", "tr") == "tr" else "KPI detail", expanded=False):
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric(tr("total_customers"), f"{total:,}")
        col2.metric(tr("detected"), f"{detected}", delta=f"%{detection_rate:.1f}", delta_color="inverse")
        col3.metric(tr("urgent"), f"{urgent_count}", delta=tr("field_team") if urgent_count > 0 else tr("clean"), delta_color="inverse" if urgent_count > 0 else "normal")
        col4.metric(tr("critical"), f"{critical_count}")
        col5.metric(tr("estimated_loss"), f"{tr('currency_symbol')} {monthly_loss:,.0f}", delta_color="inverse")

    st.markdown('<div class="glass-divider"></div>', unsafe_allow_html=True)
    st.markdown(f"### {tr('regional_map')}")

    lats, lons, city_names = _make_city_coords(len(df))
    map_df = df.copy()
    map_df["lat"] = lats
    map_df["lon"] = lons
    map_df["city"] = city_names
    map_df["profile_label"] = map_df["profile"].apply(profile_label)
    map_df["risk_label"] = map_df["risk_level"].astype(str).apply(risk_label)

    map_scale = [
        [0.0, "#27ae60"],
        [0.25, "#7fcd6c"],
        [0.5, "#f1c40f"],
        [0.75, "#e67e22"],
        [1.0, "#e74c3c"],
    ]
    # TODO: limit hover_data to essential columns — reduces Plotly JSON payload
    map_fig = px.scatter_mapbox(
        map_df,
        lat="lat",
        lon="lon",
        color="theft_probability",
        size="risk_score",
        color_continuous_scale=map_scale,
        range_color=[0, 1],
        size_max=12,
        zoom=5,
        center={"lat": 39.0, "lon": 35.0},
        mapbox_style="carto-positron" if theme == "light" else "carto-darkmatter",
        hover_data={
            "customer_id": True,
            "theft_probability": ":.2f",
            "risk_label": True,
            "city": True,
            "lat": False,
            "lon": False,
            "risk_score": False,
            "profile": False,
            "risk_level": False,
            "profile_label": False,
        },
        labels={
            "customer_id": tr("customer_id"),
            "theft_probability": tr("probability"),
            "risk_label": tr("risk"),
            "city": "Sehir" if st.session_state.get("ui_lang", "tr") == "tr" else "City",
        },
        height=450,
    )
    map_fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), coloraxis_colorbar=dict(title=tr("probability")))
    apply_plotly_theme(map_fig, theme)
    st.plotly_chart(map_fig, use_container_width=True)

    st.markdown('<div class="glass-divider"></div>', unsafe_allow_html=True)
    with st.expander("Istatistik detayi" if st.session_state.get("ui_lang", "tr") == "tr" else "Statistical detail", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"### {tr('risk_distribution')}")
            risk_counts = df["risk_level"].astype(str).value_counts()
            pie_labels = [risk_label(item) for item in risk_counts.index]
            if theme == "light":
                pie_colors = ["#27ae60", "#3498db", "#f1c40f", "#e67e22", "#e74c3c"]
                center_fill = "rgba(255,255,255,0.76)"
            else:
                pie_colors = ["#2ecc71", "#5dade2", "#f4d03f", "#eb984e", "#ec7063"]
                center_fill = "rgba(10,13,17,0.88)"
            donut_fig = go.Figure(
                data=[go.Pie(
                    labels=pie_labels,
                    values=risk_counts.values,
                    hole=0.58,
                    sort=False,
                    direction="clockwise",
                    marker=dict(colors=pie_colors[: len(risk_counts)], line=dict(color="rgba(255,255,255,0.08)", width=1.2)),
                    textinfo="label+percent",
                    textfont_size=13,
                    hovertemplate="%{label}: %{value} <extra></extra>",
                )]
            )
            donut_fig.add_annotation(
                text=tr("risk_mix"),
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=18, color="#101418" if theme == "light" else "#f5f7fb"),
            )
            donut_fig.add_shape(type="circle", xref="paper", yref="paper", x0=0.36, y0=0.36, x1=0.64, y1=0.64, fillcolor=center_fill, line=dict(color="rgba(255,255,255,0.06)", width=1))
            donut_fig.update_layout(height=360, margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
            apply_plotly_theme(donut_fig, theme)
            st.plotly_chart(donut_fig, use_container_width=True)

        with col2:
            st.markdown(f"### {tr('prob_distribution')}")
            histogram_fig = go.Figure()
            histogram_colors = {
                "residential": "#4f46e5",
                "commercial": "#06b6d4",
                "industrial": "#f97316",
            }
            for profile in ["residential", "commercial", "industrial"]:
                subset = df[df["profile"] == profile]
                histogram_fig.add_trace(go.Histogram(
                    x=subset["theft_probability"],
                    name=profile_label(profile),
                    marker_color=histogram_colors[profile],
                    opacity=0.72,
                    nbinsx=30,
                ))
            histogram_fig.update_layout(
                barmode="overlay",
                height=360,
                xaxis_title=tr("probability"),
                yaxis_title=tr("histogram_y_axis"),
                margin=dict(l=20, r=20, t=20, b=20),
            )
            apply_plotly_theme(histogram_fig, theme)
            st.plotly_chart(histogram_fig, use_container_width=True)

    # ── Row 3 — Triage list (top alerts) ─────────────────────────────────────
    st.markdown('<div class="glass-divider"></div>', unsafe_allow_html=True)
    st.markdown(f"### {tr('alerts')}")
    alerts = df[df["theft_probability"] >= threshold].sort_values("theft_probability", ascending=False).head(10)
    if len(alerts) == 0:
        st.success(tr("no_alerts"))
    else:
        is_en = st.session_state.get("ui_lang", "tr") == "en"
        st.caption(("Top 10 cases sorted by probability. Click " if is_en else "Olasiliga gore ilk 10 vaka. ")
                   + "**Open** " + ("to jump to customer detail." if is_en else "tusu detay sayfasina gider."))
        header_cols = st.columns([1.2, 1, 1.2, 2.5, 0.8])
        header_cols[0].markdown(f"**{tr('id')}**")
        header_cols[1].markdown(f"**{tr('probability')}**")
        header_cols[2].markdown(f"**{tr('risk')}**")
        header_cols[3].markdown(f"**{'Reason' if is_en else 'Neden'}**")
        header_cols[4].markdown("**" + ("Open" if is_en else "Ac") + "**")
        for _, row in alerts.iterrows():
            rc = st.columns([1.2, 1, 1.2, 2.5, 0.8])
            rc[0].write(f"#{int(row['customer_id'])}")
            rc[1].write(f"{row['theft_probability']:.1%}")
            rc[2].write(risk_label(str(row.get("risk_level", "-"))))
            reason_text = str(row.get("risk_reason_1", "-")) or "-"
            rc[3].write(reason_text[:72])
            if rc[4].button("Ac" if not is_en else "Open", key=f"triage_open_{int(row['customer_id'])}"):
                st.session_state["selected_customer"] = int(row["customer_id"])
                st.session_state["_need_rerun"] = True

    if _PSYCOPG2_AVAILABLE:
        st.markdown('<div class="glass-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"### {tr('live_detector_alerts')}")
        col_alert, col_refresh = st.columns([5, 1])
        with col_refresh:
            do_refresh = st.button(tr("refresh"), key="alert_refresh", help=tr("refresh_live_alerts"))

        if "db_alerts" not in st.session_state or do_refresh:
            try:
                prev_ids = set(st.session_state.get("db_alerts", pd.DataFrame()).get("id", pd.Series()).tolist())
                st.session_state["db_alerts"] = fetch_alerts(limit=20)
                st.session_state["db_alerts_ts"] = time.strftime("%H:%M:%S")
                df_al = st.session_state["db_alerts"]
                if not df_al.empty:
                    new_rows = df_al[~df_al["id"].isin(prev_ids)]
                    new_kritik = new_rows[new_rows["severity"] == "KRITIK"]
                    if not new_kritik.empty:
                        st.toast(f"{len(new_kritik)} {tr('new_critical_anomaly')}", icon="!")
                        row = new_kritik.iloc[0]
                        st.session_state["popup_meter"] = row["meter_id"]
                        st.session_state["popup_score"] = row["anomaly_score"]
                        st.session_state["popup_severity"] = row["severity"]
                        st.session_state["popup_time"] = str(row["created_at"])
            except Exception as exc:
                st.session_state["db_alerts"] = pd.DataFrame()
                friendly_error(
                    tr("db_unavailable"),
                    "Start the realtime ingest stack or load demo telemetry." if is_en else "Realtime ingest yigitini baslatin veya demo telemetri yukleyin.",
                    detail=exc,
                )

        if st.session_state.get("popup_meter"):
            show_evidence_dialog(
                st.session_state.pop("popup_meter"),
                st.session_state.pop("popup_score"),
                st.session_state.pop("popup_severity"),
                st.session_state.pop("popup_time"),
            )

        df_al = st.session_state.get("db_alerts", pd.DataFrame())
        last_al_ts = st.session_state.get("db_alerts_ts", "-")

        if df_al.empty:
            st.info(tr("no_live_alert"))
            st.caption("No streaming telemetry yet. Load the bundled demo CSV from the **Telemetry** tab to populate this panel." if is_en else "Henuz canli telemetri yok. **Telemetri** sekmesinden demo CSV'yi yukleyerek bu paneli besleyebilirsiniz.")
        else:
            kritik = df_al[df_al["severity"] == "KRITIK"]
            if not kritik.empty:
                st.error(f"**{len(kritik)} CRITICAL** | {last_al_ts}")
            else:
                st.warning(f"**{len(df_al)} alerts** | {last_al_ts}")

            for _, row in df_al.iterrows():
                c1, c2, c3, c4, c5 = st.columns([2, 2, 1, 1, 1])
                c1.write(str(row["created_at"])[:19])
                c2.write(row["meter_id"])
                c3.write(f"{row['anomaly_score']:.3f}")
                badge_color = "#111827" if theme == "light" else "#f3f4f6"
                badge_text = "#ffffff" if theme == "light" else "#101418"
                c4.markdown(
                    f"<span style='background:{badge_color};color:{badge_text};padding:2px 8px;border-radius:999px;font-size:0.8rem'>{row['severity']}</span>",
                    unsafe_allow_html=True,
                )
                if c5.button(tr("open"), key=f"ev_{row['id']}"):
                    show_evidence_dialog(
                        row["meter_id"],
                        row["anomaly_score"],
                        row["severity"],
                        str(row["created_at"])[:19],
                    )


# ========== TAB 2: ZAMAN SERISI KARSILASTIRMA ==========
def render_timeseries_comparison(df, raw_df):
    theme = st.session_state.get("ui_theme", "dark")
    is_en = st.session_state.get("ui_lang", "tr") == "en"

    title = "Time Series Comparison: Normal vs Theft" if is_en else "Zaman Serisi Karsilastirma: Normal vs Kacak"
    subtitle = "Compare normal and theft consumption patterns side by side." if is_en else "Ayni profildeki normal ve kacak musterilerin tuketim paternlerini yan yana karsilastirin."
    profile_label_txt = "Customer Profile" if is_en else "Musteri Profili"
    days_label_txt = "Time Window" if is_en else "Gosterilecek Sure"
    no_data_txt = "Insufficient sample in first 200 customers for this profile." if is_en else "Bu profil icin yeterli veri yok (ilk 200 musteri icinde)."
    stats_title = "Statistics Comparison" if is_en else "Istatistik Karsilastirma"
    patterns_title = "All Theft Types - Sample Consumption Patterns" if is_en else "Tum Kacak Turleri - Ornek Tuketim Paternleri"

    st.markdown(f"### {title}")
    st.markdown(f"*{subtitle}*")

    col1, col2 = st.columns([1, 1])

    with col1:
        profile_sel = st.selectbox(
            profile_label_txt,
            ["residential", "commercial", "industrial"],
            format_func=profile_label,
        )

    with col2:
        days_sel = st.selectbox(
            days_label_txt,
            [3, 7, 14, 30],
            index=1,
            format_func=(lambda x: f"{x} days" if is_en else f"{x} Gun"),
        )

    # Ham verisi olan müşterilerle sınırla (ilk 260 müşteri)
    available_ids = set(raw_df["customer_id"].unique())
    normal_pool = df[
        (df["label"] == 0) &
        (df["profile"] == profile_sel) &
        (df["customer_id"].isin(available_ids))
    ].sort_values("customer_id")
    theft_pool = df[
        (df["label"] == 1) &
        (df["profile"] == profile_sel) &
        (df["customer_id"].isin(available_ids))
    ].sort_values("customer_id")

    if len(normal_pool) == 0 or len(theft_pool) == 0:
        st.info(no_data_txt)
        return

    # Müşteri seçici
    sel_col1, sel_col2 = st.columns(2)
    _prob_map = df.set_index("customer_id")["theft_probability"].to_dict()
    normal_label = "Normal Customer" if is_en else "Normal Musteri"
    theft_label  = "Theft Customer"  if is_en else "Kacak Musteri"

    with sel_col1:
        normal_id = st.selectbox(
            normal_label,
            options=normal_pool["customer_id"].tolist(),
            format_func=lambda x: f"#{x}  —  Risk: {_prob_map.get(x, 0):.0%}",
            key="ts_normal_sel",
        )
    with sel_col2:
        theft_id = st.selectbox(
            theft_label,
            options=theft_pool["customer_id"].tolist(),
            format_func=lambda x: f"#{x}  —  Risk: {_prob_map.get(x, 0):.0%}",
            key="ts_theft_sel",
        )

    normal_cust = df[df["customer_id"] == normal_id].iloc[0]
    theft_cust  = df[df["customer_id"] == theft_id].iloc[0]
    n_points = days_sel * 96

    normal_raw = raw_df[raw_df["customer_id"] == normal_cust["customer_id"]].head(n_points)
    theft_raw = raw_df[raw_df["customer_id"] == theft_cust["customer_id"]].head(n_points)

    norm_title = (
        f"Normal Customer #{int(normal_cust['customer_id'])} (Risk: {normal_cust['theft_probability']:.0%})"
        if is_en else
        f"Normal Musteri #{int(normal_cust['customer_id'])} (Risk: {normal_cust['theft_probability']:.0%})"
    )
    theft_title = (
        f"Theft Customer #{int(theft_cust['customer_id'])} - {theft_cust['theft_type']} (Risk: {theft_cust['theft_probability']:.0%})"
        if is_en else
        f"Kacak Musteri #{int(theft_cust['customer_id'])} - {theft_cust['theft_type']} (Risk: {theft_cust['theft_probability']:.0%})"
    )

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[norm_title, theft_title],
    )

    fig.add_trace(
        go.Scatter(
            x=normal_raw["timestamp"],
            y=normal_raw["consumption_kw"],
            mode="lines",
            line=dict(color="#27AE60", width=1),
            fill="tozeroy",
            fillcolor="rgba(39,174,96,0.1)",
            name="Normal",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=theft_raw["timestamp"],
            y=theft_raw["consumption_kw"],
            mode="lines",
            line=dict(color="#E74C3C", width=1),
            fill="tozeroy",
            fillcolor="rgba(231,76,60,0.1)",
            name="Theft" if is_en else "Kacak",
            showlegend=True,
        ),
        row=2,
        col=1,
    )

    zero_points = theft_raw[theft_raw["consumption_kw"] < 0.01]
    if len(zero_points) > 0:
        fig.add_trace(
            go.Scatter(
                x=zero_points["timestamp"],
                y=zero_points["consumption_kw"],
                mode="markers",
                marker=dict(color="#F39C12", size=4, symbol="x"),
                name="Zero Consumption" if is_en else "Sifir Tuketim",
                showlegend=True,
            ),
            row=2,
            col=1,
        )

    fig.update_layout(height=550, hovermode="x unified", margin=dict(l=20, r=20, t=40, b=20))
    fig.update_yaxes(title_text="kW", row=1, col=1)
    fig.update_yaxes(title_text="kW", row=2, col=1)
    apply_plotly_theme(fig, theme)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"#### {stats_title}")
    metric_col_name = "Metric" if is_en else "Metrik"
    normal_col_name = f"Normal #{int(normal_cust['customer_id'])}"
    theft_col_name = f"Theft #{int(theft_cust['customer_id'])}" if is_en else f"Kacak #{int(theft_cust['customer_id'])}"

    metrics_labels = [
        "Mean Consumption (kW)" if is_en else "Ort. Tuketim (kW)",
        "Std Consumption" if is_en else "Std Tuketim",
        "Min Consumption" if is_en else "Min Tuketim",
        "Max Consumption" if is_en else "Max Tuketim",
        "Zero Measurement %" if is_en else "Sifir Olcum %",
        "Sudden Change Ratio" if is_en else "Ani Degisim Orani",
        "Night/Day Ratio" if is_en else "Gece/Gunduz Orani",
        "Theft Probability" if is_en else "Kacak Olasiligi",
    ]

    comp_data = {
        metric_col_name: metrics_labels,
        normal_col_name: [
            f"{normal_cust['mean_consumption']:.3f}",
            f"{normal_cust['std_consumption']:.3f}",
            f"{normal_cust['min_consumption']:.3f}",
            f"{normal_cust['max_consumption']:.3f}",
            f"{normal_cust['zero_measurement_pct']:.1%}",
            f"{normal_cust['sudden_change_ratio']:.4f}",
            f"{normal_cust['night_day_ratio']:.3f}",
            f"{normal_cust['theft_probability']:.1%}",
        ],
        theft_col_name: [
            f"{theft_cust['mean_consumption']:.3f}",
            f"{theft_cust['std_consumption']:.3f}",
            f"{theft_cust['min_consumption']:.3f}",
            f"{theft_cust['max_consumption']:.3f}",
            f"{theft_cust['zero_measurement_pct']:.1%}",
            f"{theft_cust['sudden_change_ratio']:.4f}",
            f"{theft_cust['night_day_ratio']:.3f}",
            f"{theft_cust['theft_probability']:.1%}",
        ],
    }
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

    st.markdown('<div class="glass-divider"></div>', unsafe_allow_html=True)
    st.markdown(f"### {patterns_title}")

    theft_types = ["constant_reduction", "night_zeroing", "random_zeros", "gradual_decrease", "peak_clipping"]
    type_labels_tr = {
        "constant_reduction": "Sabit Azaltma (Sayac Manipulasyonu)",
        "night_zeroing": "Gece Sifirlamasi (Kablo Bypass)",
        "random_zeros": "Rastgele Sifirlar (Sayac Durdurma)",
        "gradual_decrease": "Kademeli Azalma (Yavas Hirsizlik)",
        "peak_clipping": "Pik Kirpma (Akim Sinirlandirma)",
    }
    type_labels_en = {
        "constant_reduction": "Constant Reduction (Meter Tampering)",
        "night_zeroing": "Night Zeroing (Cable Bypass)",
        "random_zeros": "Random Zeros (Meter Stop)",
        "gradual_decrease": "Gradual Decrease (Slow Theft)",
        "peak_clipping": "Peak Clipping (Current Limiting)",
    }
    type_labels = type_labels_en if is_en else type_labels_tr

    fig2 = make_subplots(rows=1, cols=5, subplot_titles=[type_labels[t] for t in theft_types])
    colors = ["#E74C3C", "#E67E22", "#F39C12", "#8E44AD", "#2980B9"]

    for i, tt in enumerate(theft_types):
        tt_cust = df[(df["theft_type"] == tt) & (df["customer_id"] < 200)]
        if len(tt_cust) > 0:
            cid = tt_cust.iloc[0]["customer_id"]
            cust_raw = raw_df[raw_df["customer_id"] == cid].head(96 * 3)
            fig2.add_trace(
                go.Scatter(
                    y=cust_raw["consumption_kw"].values,
                    mode="lines",
                    line=dict(color=colors[i], width=1),
                    fill="tozeroy",
                    fillcolor=f"rgba({int(colors[i][1:3],16)},{int(colors[i][3:5],16)},{int(colors[i][5:7],16)},0.1)",
                    name=tt,
                    showlegend=False,
                ),
                row=1,
                col=i + 1,
            )

    fig2.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
    for i in range(1, 6):
        fig2.update_yaxes(title_text="kW" if i == 1 else "", row=1, col=i)
        fig2.update_xaxes(title_text="", showticklabels=False, row=1, col=i)
    apply_plotly_theme(fig2, theme)
    st.plotly_chart(fig2, use_container_width=True)


# ========== TAB 3: MODEL PERFORMANSI ==========
def render_model_performance(df, metrics):
    theme = st.session_state.get("ui_theme", "dark")
    is_en = st.session_state.get("ui_lang", "tr") == "en"

    title = "Model Performance Analysis" if is_en else "Model Performans Analizi"
    roc_title = "ROC Curve"
    pr_title = "Precision-Recall Curve"
    rf_cm_title = "Random Forest - Confusion Matrix"
    iso_cm_title = "Isolation Forest - Confusion Matrix"
    importance_title = "Feature Importance (Random Forest)" if is_en else "Ozellik Onemliligi (Random Forest)"
    detect_title = "Detection Success by Theft Type" if is_en else "Kacak Turune Gore Tespit Basarisi"
    model_table_title = "Model Comparison Table" if is_en else "Model Karsilastirma Tablosu"

    st.markdown(f"### {title}")

    model_table = metrics.get("model_table", pd.DataFrame()).copy()
    if not model_table.empty:
        best_model_key = metrics.get("best_model_key", model_table.iloc[0]["Model"])
        comparison_df = model_table.sort_values(["ROC-AUC", "PR-AUC", "F1"], ascending=False).reset_index(drop=True)
        best_auc = float(comparison_df["ROC-AUC"].max())
        comparison_df["Gap vs Best ROC-AUC"] = (best_auc - comparison_df["ROC-AUC"]).round(4)

        def model_use_case(name: str) -> str:
            mapping_en = {
                "Stacking Ensemble": "Best overall ranking for analyst queue",
                "Random Forest": "Reliable baseline and feature importance",
                "XGBoost": "Strong precision on labeled fraud patterns",
                "Gradient Boosting": "Secondary challenger with different error shape",
                "Logistic Regression": "Simple calibrated benchmark",
                "Isolation Forest": "Cold-start anomaly screen without labels",
            }
            mapping_tr = {
                "Stacking Ensemble": "Analist kuyrugu icin en guclu genel siralama",
                "Random Forest": "Guvenilir baseline ve ozellik onemi",
                "XGBoost": "Etiketli kacak paternlerinde guclu hassasiyet",
                "Gradient Boosting": "Farkli hata profiline sahip challenger",
                "Logistic Regression": "Basit kalibrasyon benchmark'i",
                "Isolation Forest": "Etiketsiz soguk baslangic anomali ekrani",
            }
            return (mapping_en if is_en else mapping_tr).get(name, "-" if is_en else "-")

        comparison_df["Use Case"] = comparison_df["Model"].map(model_use_case)

        st.markdown("#### Champion vs Challengers" if is_en else "#### Champion ve Challenger'lar")
        top_cards = comparison_df.head(min(3, len(comparison_df)))
        card_cols = st.columns(len(top_cards))
        for col, (_, row) in zip(card_cols, top_cards.iterrows()):
            tone = "danger" if row["Model"] == best_model_key else ("warn" if row["Gap vs Best ROC-AUC"] <= 0.02 else "neutral")
            subtitle = (
                f"PR-AUC {row['PR-AUC']:.3f} | F1 {row['F1']:.3f}"
                if is_en else
                f"PR-AUC {row['PR-AUC']:.3f} | F1 {row['F1']:.3f}"
            )
            with col:
                action_card(
                    row["Model"],
                    f"ROC-AUC {row['ROC-AUC']:.3f}",
                    subtitle,
                    tone=tone,
                )

        champion_note = (
            f"Champion: {best_model_key}. Best threshold for the current split: {metrics.get('best_threshold', 0.5):.2f}."
            if is_en else
            f"Champion: {best_model_key}. Guncel split icin en iyi esik: {metrics.get('best_threshold', 0.5):.2f}."
        )
        st.caption(champion_note)

        comparison_display = comparison_df[
            ["Model", "Type", "ROC-AUC", "PR-AUC", "Precision", "Recall", "F1", "Gap vs Best ROC-AUC", "Use Case"]
        ].copy()
        if not is_en:
            comparison_display = comparison_display.rename(
                columns={
                    "Model": "Model",
                    "Type": "Tip",
                    "Precision": "Precision",
                    "Recall": "Recall",
                    "Use Case": "Kullanim Amaci",
                    "Gap vs Best ROC-AUC": "En Iyi ROC-AUC Farki",
                }
            )
        for col_name in ["ROC-AUC", "PR-AUC", "Precision", "Recall", "F1", "Gap vs Best ROC-AUC", "En Iyi ROC-AUC Farki"]:
            if col_name in comparison_display.columns:
                comparison_display[col_name] = comparison_display[col_name].map(lambda value: f"{float(value):.4f}")
        st.dataframe(comparison_display, use_container_width=True, hide_index=True)

        st.markdown('<div class="glass-divider"></div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("RF ROC-AUC", f"{metrics['rf_auc']:.4f}")
    col2.metric("RF F1", f"{metrics['rf_f1']:.4f}")
    col3.metric("IF ROC-AUC", f"{metrics['iso_auc']:.4f}")
    col4.metric("IF F1", f"{metrics['iso_f1']:.4f}")

    st.markdown('<div class="glass-divider"></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    curve_palette = ["#2563eb", "#ea580c", "#16a34a", "#7c3aed", "#dc2626", "#0891b2"]

    with col1:
        st.markdown(f"#### {roc_title}")
        fig = go.Figure()
        for idx, (model_name, curve) in enumerate(metrics.get("model_curves", {}).items()):
            fig.add_trace(go.Scatter(
                x=curve["fpr"],
                y=curve["tpr"],
                mode="lines",
                name=f"{model_name} (AUC={curve['auc']:.3f})",
                line=dict(color=curve_palette[idx % len(curve_palette)], width=2.5),
            ))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random (AUC=0.500)", line=dict(color="gray", width=1, dash="dash")))
        fig.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(x=0.35, y=0.1),
        )
        apply_plotly_theme(fig, theme)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(f"#### {pr_title}")
        fig = go.Figure()
        for idx, (model_name, curve) in enumerate(metrics.get("model_curves", {}).items()):
            fig.add_trace(go.Scatter(
                x=curve["recall"],
                y=curve["precision"],
                mode="lines",
                name=f"{model_name} (AP={curve['ap']:.3f})",
                line=dict(color=curve_palette[idx % len(curve_palette)], width=2.5),
            ))
        baseline = (metrics["y_test"] == 1).sum() / len(metrics["y_test"])
        fig.add_trace(go.Scatter(x=[0, 1], y=[baseline, baseline], mode="lines", name=f"Baseline ({baseline:.2f})", line=dict(color="gray", width=1, dash="dash")))
        fig.update_layout(
            xaxis_title="Recall",
            yaxis_title="Precision",
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(x=0.02, y=0.1),
        )
        apply_plotly_theme(fig, theme)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="glass-divider"></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    cm_txt_color = "#0f172a" if theme == "light" else "#f0f4ff"
    cm_axis = dict(color=cm_txt_color, tickfont=dict(color=cm_txt_color), title=dict(font=dict(color=cm_txt_color)))

    with col1:
        st.markdown(f"#### {rf_cm_title}")
        cm = metrics["rf_cm"]
        # Hücre değerine göre otomatik kontrast renk (açık hücre → koyu yazı, koyu hücre → beyaz)
        cm_max = cm.max()
        cell_colors = [["#ffffff" if v < cm_max * 0.55 else "#000000" for v in row] for row in cm]
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=["Normal", "Theft" if is_en else "Kacak"],
            y=["Normal", "Theft" if is_en else "Kacak"],
            colorscale="Blues",
            showscale=False,
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 22, "color": cm_txt_color},
        ))
        fig.update_layout(
            xaxis_title="Predicted" if is_en else "Tahmin",
            yaxis_title="Actual" if is_en else "Gercek",
            xaxis=cm_axis, yaxis={**cm_axis, "autorange": "reversed"},
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        apply_plotly_theme(fig, theme)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(f"#### {iso_cm_title}")
        cm = metrics["iso_cm"]
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=["Normal", "Theft" if is_en else "Kacak"],
            y=["Normal", "Theft" if is_en else "Kacak"],
            colorscale="Oranges",
            showscale=False,
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 22, "color": cm_txt_color},
        ))
        fig.update_layout(
            xaxis_title="Predicted" if is_en else "Tahmin",
            yaxis_title="Actual" if is_en else "Gercek",
            xaxis=cm_axis, yaxis={**cm_axis, "autorange": "reversed"},
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        apply_plotly_theme(fig, theme)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="glass-divider"></div>', unsafe_allow_html=True)
    st.markdown(f"#### {importance_title}")
    imp = metrics["importance"]
    top_feats = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:12]
    feat_names = [f[0] for f in top_feats][::-1]
    feat_vals = [f[1] for f in top_feats][::-1]

    fig = go.Figure(go.Bar(
        x=feat_vals,
        y=feat_names,
        orientation="h",
        marker=dict(color=feat_vals, colorscale="Blues", showscale=False),
        text=[f"{v:.3f}" for v in feat_vals],
        textposition="outside",
    ))
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=80, t=20, b=20),
        xaxis_title="Importance Score" if is_en else "Onem Skoru",
    )
    apply_plotly_theme(fig, theme)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="glass-divider"></div>', unsafe_allow_html=True)
    st.markdown(f"#### {detect_title}")
    col1, col2 = st.columns(2)

    with col1:
        theft_df = df[df["label"] == 1]
        type_detection = theft_df.groupby("theft_type").agg(toplam=("label", "count"), tespit=("predicted_theft", "sum")).reset_index()
        type_detection["oran"] = type_detection["tespit"] / type_detection["toplam"] * 100

        fig = go.Figure(go.Bar(
            x=type_detection["theft_type"],
            y=type_detection["oran"],
            marker_color=["#1B4F72", "#2E86C1", "#5DADE2", "#85C1E9", "#AED6F1"],
            text=[f"{v:.0f}%" for v in type_detection["oran"]],
            textposition="outside",
        ))
        fig.update_layout(
            yaxis_title="Detection Rate (%)" if is_en else "Tespit Orani (%)",
            yaxis_range=[0, 110],
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        apply_plotly_theme(fig, theme)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(f"#### {model_table_title}")
        if not model_table.empty:
            model_table = model_table.rename(columns={"Type": ("Type" if is_en else "Tip")})
            for col in ["ROC-AUC", "PR-AUC", "Precision", "Recall", "F1"]:
                if col in model_table.columns:
                    model_table[col] = model_table[col].map(lambda x: f"{float(x):.4f}")
            st.dataframe(model_table, use_container_width=True, hide_index=True)
            st.caption(
                (
                    f"Computed from the current test split. Best threshold: {metrics.get('best_threshold', 0.5):.2f}"
                    if is_en else
                    f"Guncel test ayrimindan hesaplandi. En iyi esik: {metrics.get('best_threshold', 0.5):.2f}"
                )
            )
        else:
            st.info("Only fallback scoring is available for this dataset." if is_en else "Bu veri setinde yalnizca fallback skorlama mevcut.")


# ========== TAB 4: MUSTERI DETAY ==========
def render_customer_detail(df, raw_df):
    theme = st.session_state.get("ui_theme", "dark")
    is_en = st.session_state.get("ui_lang", "tr") == "en"

    st.markdown("### Customer Detail" if is_en else "### Musteri Detay Inceleme")

    col1, col2 = st.columns([1, 3])

    with col1:
        view_label = "View" if is_en else "Gosterim"
        high_risk_opt = "High Risk" if is_en else "Yuksek Riskli"
        all_opt = "All Customers" if is_en else "Tum Musteriler"
        view_mode = st.radio(view_label, [high_risk_opt, all_opt], label_visibility="collapsed")

        if view_mode == high_risk_opt:
            pool = df[df["theft_probability"] > 0.5].sort_values("theft_probability", ascending=False)
        else:
            pool = df.sort_values("customer_id")

        if len(pool) == 0:
            st.info("No customer in this filter." if is_en else "Bu filtrede musteri yok.")
            return

        select_label = "Select Customer" if is_en else "Musteri Sec"
        _prob_map = df.set_index("customer_id")["theft_probability"].to_dict()
        selected_id = st.selectbox(
            select_label,
            options=pool["customer_id"].tolist(),
            format_func=lambda x: f"#{x} ({_prob_map.get(x, 0):.0%})",
        )

        cust = df[df["customer_id"] == selected_id].iloc[0]
        st.markdown(f"**{tr('profile')}:** {profile_label(cust['profile'])}")
        st.markdown(f"**{tr('risk')}:** {risk_label(cust['risk_level'])}")
        st.markdown(f"**{tr('probability')}:** {cust['theft_probability']:.1%}")
        st.markdown(f"**{tr('anomaly_score')}:** {cust['anomaly_score']:.3f}")

        if cust["label"] == 1:
            theft_type_label = "Theft Type" if is_en else "Kacak Turu"
            st.markdown(f"**{theft_type_label}:** {cust['theft_type']}")

        if cust["theft_probability"] > 0.7:
            st.markdown('<div class="alert-box">WARNING <strong>HIGH RISK</strong></div>' if is_en else '<div class="alert-box">UYARI <strong>YUKSEK RISK</strong></div>', unsafe_allow_html=True)
        elif cust["theft_probability"] < 0.3:
            st.markdown('<div class="safe-box">SAFE</div>' if is_en else '<div class="safe-box">NORMAL</div>', unsafe_allow_html=True)

    with col2:
        cust_raw = raw_df[raw_df["customer_id"] == selected_id]
        if len(cust_raw) > 0:
            subplot_1 = "Consumption Profile" if is_en else "Tuketim Profili"
            subplot_2 = "Daily Total Consumption" if is_en else "Gunluk Toplam Tuketim"
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                subplot_titles=[subplot_1, subplot_2],
                row_heights=[0.6, 0.4],
            )

            fig.add_trace(
                go.Scatter(
                    x=cust_raw["timestamp"],
                    y=cust_raw["consumption_kw"],
                    mode="lines",
                    line=dict(color="#2E86C1", width=1),
                    fill="tozeroy",
                    fillcolor="rgba(46,134,193,0.1)",
                    name="Consumption" if is_en else "Tuketim",
                ),
                row=1,
                col=1,
            )

            zeros = cust_raw[cust_raw["consumption_kw"] < 0.01]
            if len(zeros) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=zeros["timestamp"],
                        y=zeros["consumption_kw"],
                        mode="markers",
                        marker=dict(color="red", size=3),
                        name="Zero" if is_en else "Sifir",
                    ),
                    row=1,
                    col=1,
                )

            daily = cust_raw.set_index("timestamp").resample("D")["consumption_kw"].sum().reset_index()
            _daily_mean_thresh = daily["consumption_kw"].mean() * 0.3
            bar_colors = ["#E74C3C" if v < _daily_mean_thresh else "#2E86C1" for v in daily["consumption_kw"]]
            fig.add_trace(
                go.Bar(
                    x=daily["timestamp"],
                    y=daily["consumption_kw"],
                    marker_color=bar_colors,
                    name="Daily" if is_en else "Gunluk",
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

            fig.update_layout(height=500, margin=dict(l=20, r=20, t=40, b=20), hovermode="x unified")
            fig.update_yaxes(title_text="kW", row=1, col=1)
            fig.update_yaxes(title_text="kWh/day" if is_en else "kWh/gun", row=2, col=1)
            apply_plotly_theme(fig, theme)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Raw consumption data not available (first 200 customers)." if is_en else "Ham tuketim verisi mevcut degil (ilk 200 musteri).")

    st.markdown('<div class="glass-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### Customer Feature Profile" if is_en else "#### Musteri Ozellik Profili")

    features_to_show = ["mean_consumption", "std_consumption", "zero_measurement_pct", "sudden_change_ratio", "night_day_ratio", "cv_daily"]
    labels = [
        "Mean Consumption" if is_en else "Ort. Tuketim",
        "Std Dev" if is_en else "Std Sapma",
        "Zero %" if is_en else "Sifir %",
        "Sudden Change" if is_en else "Ani Degisim",
        "Night/Day" if is_en else "Gece/Gunduz",
        "Daily CV" if is_en else "Gunluk CV",
    ]

    cust_vals, pop_vals = [], []
    for f in features_to_show:
        f_min = df[f].min()
        f_max = df[f].max()
        cust_vals.append((cust[f] - f_min) / (f_max - f_min + 1e-8))
        pop_vals.append((df[f].mean() - f_min) / (f_max - f_min + 1e-8))

    radar = go.Figure()
    radar.add_trace(go.Scatterpolar(
        r=pop_vals + [pop_vals[0]],
        theta=labels + [labels[0]],
        fill="toself",
        fillcolor="rgba(46,134,193,0.1)",
        line=dict(color="#2E86C1"),
        name="Population Avg" if is_en else "Populasyon Ort.",
    ))
    radar.add_trace(go.Scatterpolar(
        r=cust_vals + [cust_vals[0]],
        theta=labels + [labels[0]],
        fill="toself",
        fillcolor="rgba(231,76,60,0.15)",
        line=dict(color="#E74C3C"),
        name=(f"Customer #{int(selected_id)}" if is_en else f"Musteri #{int(selected_id)}"),
    ))
    radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=400,
        margin=dict(l=60, r=60, t=20, b=20),
    )
    apply_plotly_theme(radar, theme)
    st.plotly_chart(radar, use_container_width=True)


# ========== TAB 5: CANLI SIMULASYON ==========
def render_live_simulation(df, raw_df):
    is_en = st.session_state.get("ui_lang", "tr") == "en"
    theme = st.session_state.get("ui_theme", "dark")

    title = "Live Simulation Mode" if is_en else "Canli Simulasyon Modu"
    desc = "Watch streaming telemetry and anomaly signals in real time." if is_en else "Gercek zamanli akan veri simulasyonu - MASS sayaclarindan gelen veriyi canli izleyin."

    st.markdown(f'<p style="font-size:1.3rem; font-weight:600;"><span class="live-indicator"></span> {title}</p>', unsafe_allow_html=True)
    st.markdown(f"*{desc}*")

    col1, col2, col3 = st.columns(3)
    with col1:
        sim_speed = st.selectbox("Speed" if is_en else "Hiz", [0.05, 0.1, 0.2, 0.5], index=1, format_func=lambda x: f"{x}s")
    with col2:
        n_customers = st.selectbox("Customer Count" if is_en else "Musteri Sayisi", [3, 5, 8], index=1)
    with col3:
        sim_points = st.selectbox("Data Points" if is_en else "Veri Noktasi", [50, 100, 200], index=1)

    normal_sample = df[(df["label"] == 0) & (df["customer_id"] < 200)].head(max(n_customers - 1, 1))
    theft_sample = df[(df["label"] == 1) & (df["customer_id"] < 200)].head(max(1, n_customers - len(normal_sample)))
    sim_customers = pd.concat([normal_sample, theft_sample]).drop_duplicates(subset=["customer_id"]).head(n_customers)
    if sim_customers.empty:
        st.warning("No customers available for simulation." if is_en else "Simulasyon icin musteri bulunamadi.")
        return

    start_btn = "Start Simulation" if is_en else "Simulasyonu Baslat"
    if st.button(start_btn, type="primary", use_container_width=True):
        metric_cols = st.columns(4)
        metric_cols[0].markdown("**Streaming Samples**" if is_en else "**Akan Olcum**")
        metric_cols[1].markdown("**Detected Anomalies**" if is_en else "**Anomali Tespit**")
        metric_cols[2].markdown("**Avg Consumption**" if is_en else "**Ort. Tuketim**")
        metric_cols[3].markdown("**Alarm Count**" if is_en else "**Alarm Sayisi**")

        m_count = metric_cols[0].empty()
        m_anomaly = metric_cols[1].empty()
        m_avg = metric_cols[2].empty()
        m_alarm = metric_cols[3].empty()

        chart_placeholder = st.empty()
        alert_placeholder = st.empty()
        progress = st.progress(0)

        alarm_count = 0
        anomaly_count = 0
        total_consumption = 0

        customer_data = {}
        for _, c in sim_customers.iterrows():
            cid = c["customer_id"]
            craw = raw_df[raw_df["customer_id"] == cid].head(sim_points)
            customer_data[cid] = {
                "values": craw["consumption_kw"].values,
                "label": c["label"],
                "theft_type": c.get("theft_type", "none"),
                "profile": c["profile"],
                "buffer_x": [],
                "buffer_y": [],
            }

        for step in range(sim_points):
            fig = make_subplots(rows=len(sim_customers), cols=1, shared_xaxes=True, vertical_spacing=0.04)

            for i, (cid, cdata) in enumerate(customer_data.items()):
                if step < len(cdata["values"]):
                    val = cdata["values"][step]
                    cdata["buffer_x"].append(step)
                    cdata["buffer_y"].append(val)
                    total_consumption += val

                    color = "#E74C3C" if cdata["label"] == 1 else "#27AE60"
                    name = f"#{int(cid)} {'THEFT' if cdata['label'] == 1 else 'NORMAL'}" if is_en else f"#{int(cid)} {'KACAK' if cdata['label'] == 1 else 'NORMAL'}"

                    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
                    fig.add_trace(
                        go.Scatter(
                            x=cdata["buffer_x"],
                            y=cdata["buffer_y"],
                            mode="lines",
                            line=dict(color=color, width=1.5),
                            fill="tozeroy",
                            fillcolor=f"rgba({r},{g},{b},0.08)",
                            name=name,
                            showlegend=True,
                        ),
                        row=i + 1,
                        col=1,
                    )

                    if val < 0.01 and cdata["label"] == 1:
                        anomaly_count += 1
                        fig.add_trace(
                            go.Scatter(
                                x=[step],
                                y=[val],
                                mode="markers",
                                marker=dict(color="red", size=8, symbol="x"),
                                showlegend=False,
                            ),
                            row=i + 1,
                            col=1,
                        )

                    if val < 0.01 and step > 5:
                        alarm_count += 1

            fig.update_layout(height=80 * len(sim_customers) + 100, margin=dict(l=20, r=20, t=20, b=20), showlegend=True, legend=dict(orientation="h", y=1.02))
            for i in range(len(sim_customers)):
                fig.update_yaxes(title_text="kW", row=i + 1, col=1)

            apply_plotly_theme(fig, theme)
            chart_placeholder.plotly_chart(fig, use_container_width=True)

            m_count.metric("Streaming Samples" if is_en else "Akan Olcum", f"{step + 1}/{sim_points}")
            m_anomaly.metric("Detected Anomalies" if is_en else "Anomali Tespit", f"{anomaly_count}")
            avg = total_consumption / ((step + 1) * len(sim_customers))
            m_avg.metric("Avg Consumption" if is_en else "Ort. Tuketim", f"{avg:.2f} kW")
            m_alarm.metric("Alarms" if is_en else "Alarm", f"{alarm_count}", delta=("Warning" if is_en and alarm_count > 0 else ""))

            if alarm_count > 0 and step % 20 == 0 and step > 0:
                msg = f"{alarm_count} zero-consumption alerts detected" if is_en else f"{alarm_count} sifir tuketim alarmi tespit edildi"
                alert_placeholder.warning(msg)

            progress.progress((step + 1) / sim_points)
            time.sleep(sim_speed)

        progress.empty()
        done = f"Simulation completed: {sim_points} samples processed, {anomaly_count} anomalies detected." if is_en else f"Simulasyon tamamlandi: {sim_points} olcum islendi, {anomaly_count} anomali tespit edildi."
        st.success(done)

    else:
        st.markdown('<div class="glass-divider"></div>', unsafe_allow_html=True)
        info = "Press start to run real-time simulation preview." if is_en else "Yukaridaki butona basarak canli simulasyonu baslatin."
        st.info(info)

        fig = make_subplots(rows=len(sim_customers), cols=1, shared_xaxes=True, vertical_spacing=0.04)
        preview_colors = ["#2563eb", "#16a34a", "#ea580c", "#7c3aed", "#dc2626", "#0891b2", "#65a30d", "#0f766e"]
        plotted = 0
        for row_idx, (_, preview_cust) in enumerate(sim_customers.iterrows(), start=1):
            preview_raw = raw_df[raw_df["customer_id"] == preview_cust["customer_id"]].head(96 * 3)
            if preview_raw.empty:
                continue
            plotted += 1
            fig.add_trace(go.Scatter(
                x=preview_raw["timestamp"],
                y=preview_raw["consumption_kw"],
                mode="lines",
                line=dict(color=preview_colors[(row_idx - 1) % len(preview_colors)], width=1.2),
                fill="tozeroy",
                fillcolor="rgba(37,99,235,0.08)",
                name=f"#{int(preview_cust['customer_id'])}",
                showlegend=True,
            ), row=row_idx, col=1)
            fig.update_yaxes(title_text=f"#{int(preview_cust['customer_id'])}", row=row_idx, col=1)
        if plotted > 0:
            fig.update_layout(
                height=max(260, 150 * len(sim_customers)),
                margin=dict(l=20, r=20, t=20, b=20),
                title=("Preview - Multiple Customers" if is_en else "Onizleme - Coklu Musteri"),
                legend=dict(orientation="h", y=1.02),
            )
            apply_plotly_theme(fig, theme)
            st.plotly_chart(fig, use_container_width=True)


def _trend_label(row: pd.Series, is_en: bool) -> str:
    relative_trend = float(row.get("relative_trend", 1.0) or 1.0)
    trend_slope = float(row.get("trend_slope", 0.0) or 0.0)
    if relative_trend <= 0.82 or trend_slope <= -0.10:
        return "Sharp drop" if is_en else "Sert dusus"
    if relative_trend <= 0.95 or trend_slope < -0.01:
        return "Falling" if is_en else "Dusuyor"
    if relative_trend >= 1.12 or trend_slope >= 0.10:
        return "Rising" if is_en else "Yukseliyor"
    if 0.95 <= relative_trend <= 1.05 and abs(trend_slope) <= 0.01:
        return "Stable" if is_en else "Stabil"
    return "Volatile" if is_en else "Dalgali"


def _priority_label(row: pd.Series, is_en: bool) -> str:
    priority_index = float(row.get("priority_index", row.get("risk_score", 0.0)) or 0.0)
    risk_key = canonical_risk(str(row.get("risk_band", row.get("risk_category", "medium"))))
    if risk_key == "urgent" or priority_index >= 80:
        return "P1 - Immediate" if is_en else "P1 - Hemen"
    if risk_key == "critical" or priority_index >= 65:
        return "P2 - Today" if is_en else "P2 - Bugun"
    if risk_key == "high" or priority_index >= 50:
        return "P3 - Queue" if is_en else "P3 - Sirada"
    return "P4 - Monitor" if is_en else "P4 - Izle"


def _recommended_action(row: pd.Series, is_en: bool) -> str:
    risk_key = canonical_risk(str(row.get("risk_band", row.get("risk_category", "medium"))))
    zero_pct = float(row.get("zero_measurement_pct", 0.0) or 0.0)
    if risk_key in {"urgent", "critical"}:
        return "Escalate to field review" if is_en else "Saha incelemesine gonder"
    if zero_pct >= 0.15:
        return "Check meter / outage logs" if is_en else "Sayac ve kesinti kayitlarini kontrol et"
    return "Keep in analyst queue" if is_en else "Analist kuyrugunda tut"


def _prepare_customer_worklist(df: pd.DataFrame, raw_df: pd.DataFrame, is_en: bool) -> pd.DataFrame:
    view = df.copy()
    view["risk_key"] = view["risk_band"].astype(str).apply(canonical_risk)
    if "priority_index" not in view.columns:
        view["priority_index"] = np.round(
            view["risk_score"] * 0.65 + view["est_monthly_loss"].clip(upper=5000) * 0.01,
            2,
        )
    if "risk_reason_1" not in view.columns:
        fallback_reason = np.where(
            view["theft_probability"] >= 0.5,
            "high theft probability",
            "monitor consumption variance",
        )
        view["risk_reason_1"] = fallback_reason

    last_seen_map = pd.Series(dtype="datetime64[ns]")
    if not raw_df.empty and {"customer_id", "timestamp"}.issubset(raw_df.columns):
        seen = raw_df[["customer_id", "timestamp"]].copy()
        seen["timestamp"] = pd.to_datetime(seen["timestamp"], errors="coerce")
        seen = seen.dropna(subset=["timestamp"])
        if not seen.empty:
            last_seen_map = seen.groupby("customer_id")["timestamp"].max()

    view["last_seen_ts"] = view["customer_id"].map(last_seen_map)
    view["last_seen"] = view["last_seen_ts"].apply(
        lambda ts: ts.strftime("%Y-%m-%d %H:%M") if pd.notna(ts) else "-"
    )
    view["trend_label"] = view.apply(lambda row: _trend_label(row, is_en), axis=1)
    view["priority_label"] = view.apply(lambda row: _priority_label(row, is_en), axis=1)
    view["recommended_action"] = view.apply(lambda row: _recommended_action(row, is_en), axis=1)
    return view


def render_customer_list(df, raw_df):
    is_en = st.session_state.get("ui_lang", "tr") == "en"

    title = "Investigation Worklist" if is_en else "Inceleme Is Listesi"
    st.markdown(f"### {title}")

    worklist = _prepare_customer_worklist(df, raw_df, is_en)
    if worklist.empty:
        st.info("No customers available." if is_en else "Gosterilecek musteri yok.")
        return

    c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
    with c1:
        search_label = "Search Customer ID" if is_en else "Musteri ID ara"
        search = st.text_input(search_label, placeholder=("e.g. 1080" if is_en else "orn: 1080"))

    all_risk_keys = ["urgent", "critical", "high", "medium", "low"]
    with c2:
        band_sel = st.multiselect(
            tr("risk_band"),
            all_risk_keys,
            default=all_risk_keys,
            format_func=lambda x: risk_label(x),
        )

    with c3:
        profile_sel = st.multiselect(
            tr("profile"),
            sorted(worklist["profile"].astype(str).unique().tolist()),
            default=sorted(worklist["profile"].astype(str).unique().tolist()),
            format_func=profile_label,
        )

    sort_opts = {
        ("Priority Desc" if is_en else "Oncelik Azalan"): ("priority_index", False),
        ("Risk Score Desc" if is_en else "Risk Skoru Azalan"): ("risk_score", False),
        ("Monthly Loss Desc" if is_en else "Aylik Kayip Azalan"): ("est_monthly_loss", False),
        ("Last Seen Desc" if is_en else "Son Gorus Azalan"): ("last_seen_ts", False),
        ("Customer ID Asc" if is_en else "Musteri ID Artan"): ("customer_id", True),
    }
    with c4:
        sort_col = st.selectbox("Sort" if is_en else "Siralama", list(sort_opts.keys()))

    view = worklist.copy()

    if search:
        view = view[view["customer_id"].astype(str).str.contains(search)]
    if band_sel:
        view = view[view["risk_key"].isin(band_sel)]
    if profile_sel:
        view = view[view["profile"].isin(profile_sel)]

    sc, asc = sort_opts[sort_col]
    if sc in view.columns:
        view = view.sort_values(sc, ascending=asc)

    shown_txt = "Showing" if is_en else "Gosterilen"
    customer_txt = "customers" if is_en else "musteri"
    st.caption(f"{shown_txt}: **{len(view)}** / {len(df)} {customer_txt}")

    headline_cols = st.columns(3)
    with headline_cols[0]:
        urgent_cases = int(view["risk_key"].isin(["urgent", "critical"]).sum())
        action_card(
            "Immediate cases" if is_en else "Hemen bakilacak",
            str(urgent_cases),
            "Critical + urgent customers" if is_en else "Kritik ve acil musteriler",
            tone="danger" if urgent_cases else "ok",
        )
    with headline_cols[1]:
        top_reason = str(view.iloc[0].get("risk_reason_1", "-")) if not view.empty else "-"
        action_card(
            "Top driver" if is_en else "Ana neden",
            top_reason[:44] if top_reason and top_reason != "-" else ("Stable queue" if is_en else "Kuyruk stabil"),
            "Most urgent work item" if is_en else "En acil kayit",
            tone="warn",
        )
    with headline_cols[2]:
        latest_seen = view["last_seen"].iloc[0] if not view.empty else "-"
        action_card(
            "Latest signal" if is_en else "Son sinyal",
            latest_seen,
            "Newest customer activity in queue" if is_en else "Kuyruktaki en yeni musteri aktivitesi",
            tone="neutral",
        )

    st.markdown("#### Worklist" if is_en else "#### Is Kuyrugu")
    header_cols = st.columns([1.0, 1.1, 2.7, 1.1, 1.3, 1.0])
    header_cols[0].markdown(f"**{tr('customer_id')}**")
    header_cols[1].markdown(f"**{'Priority' if is_en else 'Oncelik'}**")
    header_cols[2].markdown(f"**{'Reason' if is_en else 'Neden'}**")
    header_cols[3].markdown(f"**{'Trend' if is_en else 'Trend'}**")
    header_cols[4].markdown(f"**{'Last Seen' if is_en else 'Last Seen'}**")
    header_cols[5].markdown(f"**{'Action' if is_en else 'Aksiyon'}**")

    top_rows = view.sort_values(["priority_index", "theft_probability"], ascending=False).head(14)
    for _, row in top_rows.iterrows():
        rc = st.columns([1.0, 1.1, 2.7, 1.1, 1.3, 1.0])
        rc[0].write(f"#{int(row['customer_id'])}")
        rc[1].write(str(row.get("priority_label", "-")))
        rc[2].write(str(row.get("risk_reason_1", "-"))[:90])
        rc[2].caption(str(row.get("recommended_action", "-")))
        rc[3].write(str(row.get("trend_label", "-")))
        rc[4].write(str(row.get("last_seen", "-")))
        if rc[5].button("Open" if is_en else "Ac", key=f"worklist_open_{int(row['customer_id'])}"):
            st.session_state["selected_customer"] = int(row["customer_id"])
            st.toast(
                f"Customer #{int(row['customer_id'])} pinned for detail view."
                if is_en else
                f"Musteri #{int(row['customer_id'])} detay icin secildi.",
            )

    full_title = "Full comparison table" if is_en else "Tam karsilastirma tablosu"
    with st.expander(full_title):
        table = view[
            [
                "customer_id",
                "profile",
                "risk_score",
                "priority_index",
                "risk_key",
                "theft_probability",
                "est_monthly_loss",
                "risk_reason_1",
                "trend_label",
                "last_seen",
            ]
        ].copy()
        table.columns = [
            tr("customer_id"),
            tr("profile"),
            "Risk Score" if is_en else "Risk Skoru",
            "Priority Index" if is_en else "Oncelik Indeksi",
            tr("risk_band"),
            tr("probability"),
            ("Monthly Loss" if is_en else "Aylik Kayip") + f" ({tr('currency_symbol')})",
            "Reason" if is_en else "Neden",
            "Trend",
            "Last Seen" if is_en else "Last Seen",
        ]
        table[tr("profile")] = table[tr("profile")].map(profile_label)
        table[tr("risk_band")] = table[tr("risk_band")].map(risk_label)
        table[tr("probability")] = (table[tr("probability")] * 100).round(1).astype(str) + "%"
        loss_col = ("Monthly Loss" if is_en else "Aylik Kayip") + f" ({tr('currency_symbol')})"
        table[loss_col] = table[loss_col].apply(lambda x: f"{tr('currency_symbol')}{x:,.0f}" if x > 0 else "-")
        st.dataframe(table, use_container_width=True, hide_index=True, height=420)

        csv = view.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV" if is_en else "CSV Indir",
            csv,
            file_name=f"customers_{time.strftime('%Y%m%d_%H%M')}.csv" if is_en else f"musteriler_{time.strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )

    exp_title = "Risk Band Summary" if is_en else "Risk Bandi Ozeti"
    with st.expander(exp_title):
        risk_order = ["urgent", "critical", "high", "medium", "low"]
        band_counts = view["risk_key"].value_counts().reindex(risk_order, fill_value=0)
        band_loss = view.groupby("risk_key")["est_monthly_loss"].sum().reindex(risk_order, fill_value=0)

        summary = pd.DataFrame({
            tr("risk_band"): [risk_label(x) for x in risk_order],
            ("Customer Count" if is_en else "Musteri Sayisi"): band_counts.values,
            ("Total Monthly Loss" if is_en else "Toplam Aylik Kayip") + f" ({tr('currency_symbol')})": [f"{tr('currency_symbol')}{v:,.0f}" for v in band_loss.values],
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)


_DEMO_TELEMETRY_CSV = Path(__file__).resolve().parent.parent / "data" / "demo" / "demo_telemetry.csv"


def _load_demo_telemetry_csv(limit: int = 500) -> pd.DataFrame:
    """Read the bundled demo telemetry CSV and align it to the live schema."""
    if not _DEMO_TELEMETRY_CSV.exists():
        raise FileNotFoundError(
            f"Demo telemetry not found at {_DEMO_TELEMETRY_CSV}. "
            "Run: python -m project.synthetic.export_telemetry --customers 100 --days 7 "
            "--telemetry-output project/data/demo/demo_telemetry.csv"
        )
    df = pd.read_csv(_DEMO_TELEMETRY_CSV)
    df["received_at"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("received_at", ascending=False).head(limit).reset_index(drop=True)
    return df[["received_at", "meter_id", "voltage", "current", "active_power"]]


def render_live_telemetry():
    """Show the latest telemetry rows from Postgres or the bundled demo CSV."""
    theme = st.session_state.get("ui_theme", "dark")

    st.markdown(f"### {tr('telemetry_title')}")

    col_btn, col_demo, col_info = st.columns([1, 1.3, 4])
    with col_btn:
        refresh = st.button(tr("refresh"), use_container_width=True, disabled=not _PSYCOPG2_AVAILABLE)
    with col_demo:
        load_demo = st.button("Load demo CSV", use_container_width=True, key="tele_load_demo")
    with col_info:
        if _PSYCOPG2_AVAILABLE:
            st.caption(tr("telemetry_refresh_help"))
        else:
            st.caption("Postgres driver not installed — using demo CSV fallback.")

    if load_demo:
        try:
            st.session_state["live_df"] = _load_demo_telemetry_csv(limit=500)
            st.session_state["live_ts"] = time.strftime("%H:%M:%S")
            st.session_state["live_source"] = "demo_csv"
        except Exception as exc:
            friendly_error(
                "Demo CSV load failed",
                "Re-generate it with: python -m project.synthetic.export_telemetry",
                detail=exc,
            )
            return
    elif _PSYCOPG2_AVAILABLE and ("live_df" not in st.session_state or refresh):
        try:
            st.session_state["live_df"] = fetch_live_telemetry(limit=100)
            st.session_state["live_ts"] = time.strftime("%H:%M:%S")
            st.session_state["live_source"] = "postgres"
        except Exception as exc:
            friendly_error(
                tr("telemetry_db_error"),
                tr("telemetry_db_help"),
                detail=exc,
            )
            return
    elif not _PSYCOPG2_AVAILABLE and "live_df" not in st.session_state:
        st.info("Click **Load demo CSV** to preview bundled telemetry without a database.")
        return

    df = st.session_state.get("live_df", pd.DataFrame())
    last_ts = st.session_state.get("live_ts", "-")
    source_label = st.session_state.get("live_source", "postgres")
    if source_label == "demo_csv":
        st.caption("Source: bundled demo CSV (project/data/demo/demo_telemetry.csv)")

    if df.empty:
        st.info(tr("telemetry_no_data"))
        return

    st.caption(f"{tr('telemetry_last_updated')}: {last_ts}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(tr("telemetry_total_records"), f"{len(df)}")
    col2.metric(tr("telemetry_avg_voltage"), f"{df['voltage'].mean():.1f}")
    col3.metric(tr("telemetry_avg_current"), f"{df['current'].mean():.3f}")
    col4.metric(tr("telemetry_avg_power"), f"{df['active_power'].mean():.1f}")

    fig = go.Figure()
    for meter in df["meter_id"].unique()[:10]:
        sub = df[df["meter_id"] == meter].sort_values("received_at")
        fig.add_trace(go.Scatter(
            x=sub["received_at"],
            y=sub["active_power"],
            mode="lines+markers",
            name=meter,
            line=dict(width=1.5),
            marker=dict(size=4),
        ))

    fig.update_layout(
        title=tr("telemetry_chart_title"),
        xaxis_title=tr("telemetry_time_axis"),
        yaxis_title=tr("telemetry_power_axis"),
        height=380,
        legend=dict(orientation="h", y=-0.25),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    apply_plotly_theme(fig, theme)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        df[["received_at", "meter_id", "voltage", "current", "active_power"]].rename(
            columns={
                "received_at": tr("telemetry_time_col"),
                "meter_id": tr("telemetry_meter_col"),
                "voltage": tr("telemetry_voltage_col"),
                "current": tr("telemetry_current_col"),
                "active_power": tr("telemetry_power_col"),
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


# ========== ANA UYGULAMA ==========
def main():
    ensure_ui_state()

    # Rerun isteklerini güvenli şekilde render döngüsü başında işle
    if st.session_state.pop("_need_rerun", False):
        st.rerun()

    _start_ingest_watcher()
    new_ingest_path = poll_ingest_dir()
    if new_ingest_path is not None:
        _logger.info("main: new ingest file picked up → %s", new_ingest_path)
        st.session_state["data_source"] = "external"
        st.session_state["external_csv_path"] = str(new_ingest_path)
        load_data.clear()
        st.toast(f"New data loaded: {new_ingest_path.name}", icon="⚡")
        st.rerun()

    data_source = st.session_state.get("data_source", "sample")
    external_csv_path = resolve_external_csv_path(st.session_state.get("external_csv_path", ""))
    load_error = None

    # Eğer external seçili ama path yoksa veya dosya mevcut değilse, hemen sample'a düş
    if data_source == "external" and (not external_csv_path or not os.path.exists(external_csv_path)):
        if external_csv_path and not os.path.exists(external_csv_path):
            load_error = f"Dosya bulunamadı: {external_csv_path}"
        data_source = "sample"
        external_csv_path = ""

    try:
        features_df, raw_df, data_meta = load_data(data_source, external_csv_path)
    except Exception as exc:
        _logger.error("load_data failed (source=%s): %s", data_source, exc, exc_info=True)
        load_error = f"{tr('external_load_error')}: {exc}"
        try:
            features_df, raw_df, data_meta = load_data("sample", "")
        except Exception as exc2:
            _logger.error("Sample data fallback also failed: %s", exc2, exc_info=True)
            friendly_error(
                "Critical error — even sample data could not be loaded",
                "Check mass_ai_engine.py and that scikit-learn / numpy are installed.",
                detail=exc2,
            )
            st.stop()
        data_meta["source"] = "sample"

    features_df, metrics = run_models_v2(features_df)
    filtered_df, threshold = render_sidebar(features_df, data_meta)

    theme = st.session_state.get("ui_theme", "dark")
    inject_liquid_spotlight()
    inject_theme_overrides(theme)
    is_en = st.session_state.get("ui_lang", "tr") == "en"
    badge_txt = "AI · Anomaly Detection · v2.0" if is_en else "Yapay Zeka · Anomali Tespiti · v2.0"
    st.markdown(
        f"""
        <div class="hero-shell">
            <div class="hero-badge">⚡ {badge_txt}</div>
            <p class="main-header">MASS-AI Dashboard</p>
            <p class="sub-header">{tr('subtitle')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if load_error:
        st.warning(load_error)

    if "best_model" in metrics:
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"### {tr('active_model')}")
        st.sidebar.success(metrics["best_model"])

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        f"⚡ {tr('tab_overview')}",
        f"👥 {tr('tab_customers')}",
        f"📈 {tr('tab_timeseries')}",
        f"🎯 {tr('tab_models')}",
        f"🔍 {tr('tab_detail')}",
        f"🔴 {tr('tab_simulation')}",
        f"📡 {tr('tab_telemetry')}",
    ])

    with tab1:
        render_overview(filtered_df, threshold, raw_df)

    with tab2:
        render_customer_list(filtered_df, raw_df)

    with tab3:
        render_timeseries_comparison(features_df, raw_df)

    with tab4:
        render_model_performance(features_df, metrics)

    with tab5:
        render_customer_detail(features_df, raw_df)

    with tab6:
        render_live_simulation(features_df, raw_df)

    with tab7:
        render_live_telemetry()

    st.markdown('<div class="glass-divider" style="margin-top:2.5rem;"></div>', unsafe_allow_html=True)
    st.markdown(
        "<div style='"
        "text-align:center;"
        "color:var(--text-dim,rgba(240,244,255,0.28));"
        "font-size:0.78rem;"
        "letter-spacing:0.06em;"
        "text-transform:uppercase;"
        "padding-bottom:1.2rem;"
        "'>"
        "MASS-AI v2.0 &nbsp;·&nbsp; Omer Burak Kocak"
        "&nbsp;·&nbsp; Marmara Universitesi EEE &nbsp;·&nbsp; 2026"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
