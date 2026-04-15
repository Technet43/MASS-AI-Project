"""
MASS-AI desktop application.

Desktop-first analyst workspace for anomaly detection, scoring, and persistent
Ops Center case management.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

MISSING = []
try:
    import matplotlib
except ImportError:
    MISSING.append("matplotlib")

try:
    import numpy as np
except ImportError:
    MISSING.append("numpy")

try:
    import pandas as pd
except ImportError:
    MISSING.append("pandas")

try:
    from sklearn.ensemble import GradientBoostingClassifier, IsolationForest, RandomForestClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
except ImportError:
    MISSING.append("scikit-learn")

try:
    from xgboost import XGBClassifier
except ImportError:
    MISSING.append("xgboost")

if MISSING:
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror(
        "MASS-AI",
        "Missing packages were detected:\n\n"
        + "\n".join(f"- {name}" for name in MISSING)
        + "\n\nInstall them with:\npython -m pip install -r requirements.txt",
    )
    sys.exit(1)

try:
    import tkintermapview
except ImportError:
    tkintermapview = None

ENABLE_EXPERIMENTAL_MAP = os.environ.get("MASS_AI_ENABLE_MAP", "").strip().lower() in {"1", "true", "yes", "on"}

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from app_prefs import load_theme_preference, save_theme_preference
from app_metadata import APP_NAME, APP_VERSION, BUILD_CHANNEL, support_bundle_name, version_label
from mass_ai_domain import (
    CRITICAL_RISK_LABELS as DOMAIN_CRITICAL_RISK_LABELS,
    RISK_LABELS as DOMAIN_RISK_LABELS,
    build_case_recommendation as domain_build_case_recommendation,
    build_executive_brief_html as domain_build_executive_brief_html,
    build_executive_brief_text as domain_build_executive_brief_text,
    filter_case_dataframe as domain_filter_case_dataframe,
    fmt_currency as domain_fmt_currency,
    fmt_percent as domain_fmt_percent,
    format_local_datetime as domain_format_local_datetime,
    is_case_overdue as domain_is_case_overdue,
    normalize_follow_up_input as domain_normalize_follow_up_input,
    priority_for_risk as domain_priority_for_risk,
    safe_text as domain_safe_text,
    summarize_case_notes as domain_summarize_case_notes,
)
from mass_ai_engine import MassAIEngine
from ops_store import (
    CASE_PRIORITIES,
    CASE_STATUSES,
    RESOLUTION_REASONS,
    OpsStore,
    parse_datetime,
)
from support_bundle import create_support_bundle, support_failure_message
from ui_kit import (
    DEFAULT_THEME_NAME,
    GlassButton,
    GlassCard,
    GlassTheme,
    ScrollablePanel,
    SegmentedTabs,
    ThemePreviewPicker,
    apply_glass_ttk_theme,
    apply_glass_window_theme,
    build_glass_theme,
    bind_wrap_to_width,
    normalize_theme_name,
    THEME_CHOICES,
)

np.random.seed(42)

CRITICAL_RISK_LABELS = DOMAIN_CRITICAL_RISK_LABELS
RISK_LABELS = DOMAIN_RISK_LABELS
fmt_currency = domain_fmt_currency
fmt_percent = domain_fmt_percent
safe_text = domain_safe_text
format_local_datetime = domain_format_local_datetime
normalize_follow_up_input = domain_normalize_follow_up_input
priority_for_risk = domain_priority_for_risk
is_case_overdue = domain_is_case_overdue
build_case_recommendation = domain_build_case_recommendation
summarize_case_notes = domain_summarize_case_notes
filter_case_dataframe = domain_filter_case_dataframe
build_executive_brief_html = domain_build_executive_brief_html
build_executive_brief_text = domain_build_executive_brief_text

class LoginWindow:
    def __init__(self, parent, ops_store, theme, on_success):
        self.top = tk.Toplevel(parent)
        self.top.title("MASS-AI Login")
        self.top.geometry("380x440")
        self.top.resizable(False, False)
        self.top.configure(bg=theme.bg)
        apply_glass_window_theme(self.top, theme)
        self.top.grab_set()
        self.ops_store = ops_store
        self.on_success = on_success
        self.theme = theme
        
        parent.withdraw()
        
        container = tk.Frame(self.top, bg=theme.bg)
        container.place(relx=0.5, rely=0.5, anchor="center")
        
        tk.Label(container, text="MASS-AI", font=theme.hero_font, bg=theme.bg, fg=theme.text).pack(pady=(0,5))
        tk.Label(container, text="Enterprise Sec-Ops", font=theme.body_bold_font, bg=theme.bg, fg=theme.blue).pack(pady=(0,30))
        
        tk.Label(container, text="Username", font=theme.body_font, bg=theme.bg, fg=theme.muted).pack(anchor="w")
        self.user_var = tk.StringVar()
        ttk.Entry(container, textvariable=self.user_var, font=theme.body_font, width=28).pack(pady=(0, 15), ipady=5)
        
        tk.Label(container, text="Password", font=theme.body_font, bg=theme.bg, fg=theme.muted).pack(anchor="w")
        self.pass_var = tk.StringVar()
        ttk.Entry(container, textvariable=self.pass_var, show="*", font=theme.body_font, width=28).pack(pady=(0, 30), ipady=5)
        
        GlassButton(container, theme, text="Authenticate", command=self.attempt_login, fill=theme.blue, ink="white").pack(pady=(15, 0))
        
        tk.Label(container, text="Demo defaults: admin/admin, analyst/analyst, field/field", font=("Segoe UI", 8), bg=theme.bg, fg=theme.muted).pack(pady=(20,0))
        
        self.top.protocol("WM_DELETE_WINDOW", parent.destroy)
        self.top.bind("<Return>", lambda e: self.attempt_login())
        
    def attempt_login(self):
        user = self.user_var.get().strip()
        pwd = self.pass_var.get().strip()
        record = self.ops_store.authenticate(user, pwd)
        if record:
            self.top.destroy()
            self.on_success(record)
        else:
            messagebox.showerror("Authentication Failed", "Incorrect username or password.", parent=self.top)


class MassAIApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(f"{APP_NAME} {version_label()}")
        self.root.geometry("1320x820")
        self.root.minsize(1024, 680)
        initial_theme_name = normalize_theme_name(os.environ.get("MASS_AI_THEME") or load_theme_preference() or DEFAULT_THEME_NAME)
        self.theme_mode_var = tk.StringVar(value=initial_theme_name)
        self.theme = build_glass_theme(self.theme_mode_var.get())
        self.root.configure(bg=self.theme.bg)

        self.engine = MassAIEngine()
        self.ops_store = OpsStore()
        self.current_df = None
        self.ops_case_df = pd.DataFrame()
        self.selected_case_id = None
        self.current_figure = None
        self.current_canvas = None
        self.action_buttons = []
        self.tab_buttons = {}
        self.tab_frames = {}
        self.tab_scroll_panels = {}
        self.active_tab = "operations"
        self.customer_sort_column = "priority_index"
        self.customer_sort_reverse = True
        self.busy = False
        self._rebuilding_theme = False
        self._closing = False
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.status_var = tk.StringVar(value="Ready")
        self.workspace_meta_var = tk.StringVar(value=f"{version_label()} | {BUILD_CHANNEL}")
        self.ops_meta_var = tk.StringVar(value="Queue loaded")
        self.data_source_var = tk.StringVar(value="No data source")
        self.summary_title_var = tk.StringVar(value="Ops Center")
        self.summary_subtitle_var = tk.StringVar(value="Queue loaded")
        self.synthetic_preset_var = tk.StringVar(value=self.engine.synthetic_preset_names()[0])
        self.customer_search_var = tk.StringVar(value="")
        self.risk_filter_var = tk.StringVar(value="All risk bands")
        self.table_count_var = tk.StringVar(value="0 records")

        self.ops_search_var = tk.StringVar(value="")
        self.ops_status_filter_var = tk.StringVar(value="All open")
        self.ops_risk_filter_var = tk.StringVar(value="All risk bands")
        self.ops_priority_filter_var = tk.StringVar(value="All priorities")
        self.ops_overdue_only_var = tk.BooleanVar(value=False)
        self.ops_queue_count_var = tk.StringVar(value="0 cases")

        self.detail_customer_var = tk.StringVar(value="-")
        self.detail_profile_var = tk.StringVar(value="-")
        self.detail_risk_var = tk.StringVar(value="-")
        self.detail_probability_var = tk.StringVar(value="-")
        self.detail_loss_var = tk.StringVar(value="-")
        self.detail_theft_type_var = tk.StringVar(value="-")
        self.detail_recommendation_var = tk.StringVar(value="Select a customer")

        self.ops_case_title_var = tk.StringVar(value="No case selected")
        self.ops_case_subtitle_var = tk.StringVar(value="Pick a case")
        self.ops_customer_var = tk.StringVar(value="-")
        self.ops_profile_var = tk.StringVar(value="-")
        self.ops_risk_var = tk.StringVar(value="-")
        self.ops_probability_var = tk.StringVar(value="-")
        self.ops_exposure_var = tk.StringVar(value="-")
        self.ops_pattern_var = tk.StringVar(value="-")
        self.ops_last_analysis_var = tk.StringVar(value="-")
        self.ops_follow_up_display_var = tk.StringVar(value="-")
        self.ops_overdue_var = tk.StringVar(value="-")
        self.ops_case_status_display_var = tk.StringVar(value="-")
        self.ops_case_priority_display_var = tk.StringVar(value="-")
        self.ops_recommendation_var = tk.StringVar(value="Case recommendations will appear here.")

        self.case_status_var = tk.StringVar(value=CASE_STATUSES[0])
        self.case_priority_var = tk.StringVar(value=CASE_PRIORITIES[-1])
        self.case_follow_up_var = tk.StringVar(value="")
        self.case_resolution_var = tk.StringVar(value="")

        self.current_user = None

        def on_login(user):
            self.current_user = user
            self.root.deiconify()
            self.configure_styles()
            self.setup_ui()
            self.build_menu()
            self.bind_events()
            self.bind_shortcuts()
            self.refresh_ops_view(preserve_selection=False)
            self.update_log()
            self.log(f"Authenticated as '{user['username']}' with role '{user['role']}'")

        LoginWindow(self.root, self.ops_store, self.theme, on_login)

    def configure_styles(self):
        style = ttk.Style()
        apply_glass_ttk_theme(style, self.theme)
        apply_glass_window_theme(self.root, self.theme)

    def bind_events(self):
        self.customer_search_var.trace_add("write", lambda *_: self.refresh_customer_table())
        self.risk_filter_var.trace_add("write", lambda *_: self.refresh_customer_table())
        self.ops_search_var.trace_add("write", lambda *_: self.refresh_ops_view())
        self.ops_status_filter_var.trace_add("write", lambda *_: self.refresh_ops_view())
        self.ops_risk_filter_var.trace_add("write", lambda *_: self.refresh_ops_view())
        self.ops_priority_filter_var.trace_add("write", lambda *_: self.refresh_ops_view())
        self.ops_overdue_only_var.trace_add("write", lambda *_: self.refresh_ops_view())

    def bind_shortcuts(self):
        self.root.bind("<Control-o>", lambda _event: self.load_csv())
        self.root.bind("<Control-r>", lambda _event: self.run_demo())
        self.root.bind("<Control-s>", lambda _event: self.export_csv())
        self.root.bind("<Control-e>", lambda _event: self.export_report())
        self.root.bind("<F5>", lambda _event: self.run_demo())

    def build_menu(self):
        menu = tk.Menu(self.root)
        file_menu = tk.Menu(menu, tearoff=0)
        file_menu.add_command(label="Open CSV or Excel...", command=self.load_csv, accelerator="Ctrl+O")
        file_menu.add_command(label="Run synthetic dataset", command=self.run_demo, accelerator="Ctrl+R")
        file_menu.add_separator()
        file_menu.add_command(label="Export results CSV...", command=self.export_csv, accelerator="Ctrl+S")
        file_menu.add_command(label="Export charts PNG...", command=self.export_charts)
        file_menu.add_command(label="Export executive brief...", command=self.export_report, accelerator="Ctrl+E")
        file_menu.add_command(label="Export support bundle...", command=self.export_support_bundle)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.destroy)

        help_menu = tk.Menu(menu, tearoff=0)
        help_menu.add_command(label="About MASS-AI Desktop", command=self.show_about_dialog)
        menu.add_cascade(label="File", menu=file_menu)
        menu.add_cascade(label="Help", menu=help_menu)
        self.root.config(menu=menu)

    def post_ui(self, callback, *args, **kwargs):
        if self._closing:
            return
        try:
            self.root.after(0, lambda: self._safe_ui_callback(callback, *args, **kwargs))
        except (tk.TclError, RuntimeError):
            pass

    def _safe_ui_callback(self, callback, *args, **kwargs):
        if self._closing:
            return
        try:
            callback(*args, **kwargs)
        except (tk.TclError, RuntimeError):
            pass

    def progress_callback(self):
        def inner(percent, message):
            self.post_ui(self.update_progress, percent, message)
        return inner

    def setup_ui(self):
        self.root.grid_columnconfigure(0, minsize=312)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        self.build_titlebar()
        self.build_sidebar()
        self.build_workspace()

    def build_titlebar(self):
        bar = self.make_card(self.root, padding=(16, 12), tint=self.theme.card)
        bar.grid(row=0, column=0, columnspan=2, sticky="ew", padx=14, pady=(12, 8))
        bar.body.grid_columnconfigure(1, weight=1)

        dots = tk.Frame(bar.body, bg=self.theme.card)
        dots.grid(row=0, column=0, sticky="nw", padx=(0, 12), pady=(2, 0))
        for color in (self.theme.red, self.theme.orange, self.theme.green):
            tk.Label(dots, bg=color, width=2, height=1).pack(side="left", padx=5, pady=6)

        titles = tk.Frame(bar.body, bg=self.theme.card)
        titles.grid(row=0, column=1, sticky="ew", padx=(0, 10))
        titles.grid_columnconfigure(0, weight=1)
        tk.Label(titles, text="MASS-AI Desktop", bg=self.theme.card, fg=self.theme.text, font=self.theme.hero_font).grid(row=0, column=0, sticky="w")
        subtitle = tk.Label(
            titles,
            text="Fraud ops workspace",
            bg=self.theme.card,
            fg=self.theme.muted,
            font=self.theme.body_font,
            justify="left",
        )
        subtitle.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        bind_wrap_to_width(subtitle, extra_padding=12, min_wrap=280)

        meta = self.make_surface(bar.body, tint=self.theme.blue_soft, border="#203041")
        meta.grid(row=0, column=2, sticky="e")
        tk.Label(meta, text="Workspace", bg=self.theme.blue_soft, fg=self.theme.blue, font=self.theme.small_font).pack(anchor="e", padx=12, pady=(8, 2))
        meta_text = tk.Label(meta, textvariable=self.workspace_meta_var, bg=self.theme.blue_soft, fg=self.theme.text, font=self.theme.body_bold_font, justify="right")
        meta_text.pack(anchor="e", padx=12, pady=(0, 8))
        bind_wrap_to_width(meta_text, extra_padding=20, min_wrap=170)

    def make_card(self, parent, *, padding=(18, 18), tint=None, border=None):
        return GlassCard(parent, self.theme, padding=padding, tint=tint, border=border)

    def make_surface(self, parent, *, tint=None, border=None):
        return tk.Frame(
            parent,
            bg=tint or self.theme.card_alt,
            highlightthickness=1,
            highlightbackground=border or self.theme.border,
            highlightcolor=border or self.theme.border,
            bd=0,
        )

    def create_action_button(self, parent, text, fill, command, *, ink=None):
        light_fills = {
            self.theme.card,
            self.theme.card_alt,
            self.theme.frosted,
            self.theme.blue_soft,
            self.theme.green_soft,
            self.theme.orange_soft,
            self.theme.purple_soft,
            self.theme.red_soft,
        }
        button = GlassButton(
            parent,
            self.theme,
            text=text,
            command=command,
            fill=fill,
            ink=ink or (self.theme.text if fill in light_fills else "white"),
            pad_x=16,
            pad_y=12,
        )
        button.pack(fill="x", pady=6)
        self.action_buttons.append(button)
        return button

    def create_info_block(self, parent, label, variable, *, bg, border=None):
        box = tk.Frame(parent, bg=bg, highlightthickness=1 if border else 0, highlightbackground=border or bg, bd=0)
        tk.Label(box, text=label, bg=bg, fg=self.theme.soft, font=self.theme.small_font).pack(anchor="w", padx=12, pady=(10, 2))
        value = tk.Label(box, textvariable=variable, bg=bg, fg=self.theme.text, font=self.theme.body_bold_font, justify="left")
        value.pack(anchor="w", fill="x", padx=12, pady=(0, 10))
        bind_wrap_to_width(value, extra_padding=26, min_wrap=120)
        return box

    def theme_is_dark(self):
        rgb = self.theme.bg.lstrip("#")
        if len(rgb) != 6:
            return False
        red = int(rgb[0:2], 16)
        green = int(rgb[2:4], 16)
        blue = int(rgb[4:6], 16)
        brightness = (red * 299 + green * 587 + blue * 114) / 1000
        return brightness < 128

    def chart_palette(self):
        if self.theme_is_dark():
            return {
                "face": self.theme.card,
                "panel": self.theme.card_alt,
                "grid": self.theme.border,
                "spine": self.theme.border,
                "text": self.theme.text,
                "muted": self.theme.muted,
                "accent": [self.theme.red, self.theme.orange, self.theme.blue, self.theme.green, self.theme.purple],
                "hist_edge": "#ffffff",
            }
        if self.theme.bg == "#c8d8ec":
            return {
                "face": "#d6e2f0",
                "panel": "#dde8f4",
                "grid": "#b8cce2",
                "spine": "#a0b8d4",
                "text": "#1a2840",
                "muted": "#526a88",
                "accent": ["#e84040", "#f0960a", "#2d7cf6", "#34b864", "#7c60e8"],
                "hist_edge": "#e8f0f8",
            }
        return {
            "face": "#ffffff",
            "panel": "#fbfcff",
            "grid": "#e6ebf3",
            "spine": "#d5ddea",
            "text": "#1a2234",
            "muted": "#667089",
            "accent": ["#ef6a62", "#f0a43a", "#2d84f3", "#39b86a", "#7566ff"],
            "hist_edge": "#ffffff",
        }

    def reflow_card_items(self, container, items, *, narrow_threshold=520, max_columns=2):
        width = container.winfo_width() or container.winfo_reqwidth()
        columns = 1 if width < narrow_threshold else max_columns
        for index in range(max_columns):
            container.grid_columnconfigure(index, weight=0, uniform="")
        for index in range(columns):
            container.grid_columnconfigure(index, weight=1, uniform="glass")
        for item in items:
            item.grid_forget()
        for index, item in enumerate(items):
            item.grid(row=index // columns, column=index % columns, sticky="ew", padx=6, pady=6)

    def build_sidebar(self):
        sidebar = ScrollablePanel(self.root, self.theme, bg=self.theme.bg)
        sidebar.grid(row=1, column=0, sticky="nsew", padx=(14, 10), pady=(0, 14))
        sidebar.body.grid_columnconfigure(0, weight=1)

        brand_card = self.make_card(sidebar.body, padding=(16, 14), tint=self.theme.card)
        brand_card.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        tk.Label(brand_card.body, text="MASS-AI", bg=self.theme.card, fg=self.theme.text, font=("Segoe UI Semibold", 24, "bold")).pack(anchor="w")
        brand_copy = tk.Label(
            brand_card.body,
            text="Fraud ops",
            bg=self.theme.card,
            fg=self.theme.muted,
            font=self.theme.body_font,
            justify="left",
        )
        brand_copy.pack(anchor="w", fill="x", pady=(4, 10))
        bind_wrap_to_width(brand_copy, extra_padding=16, min_wrap=240)

        state_chip = self.make_surface(brand_card.body, tint=self.theme.blue_soft, border="#1d2e40")
        state_chip.pack(fill="x")
        tk.Label(state_chip, text="Workspace status", bg=self.theme.blue_soft, fg=self.theme.blue, font=self.theme.body_bold_font).pack(anchor="w", padx=12, pady=(8, 2))
        state_text = tk.Label(state_chip, textvariable=self.data_source_var, bg=self.theme.blue_soft, fg=self.theme.muted, font=self.theme.body_font, justify="left")
        state_text.pack(anchor="w", fill="x", padx=12, pady=(0, 8))
        bind_wrap_to_width(state_text, extra_padding=20, min_wrap=230)

        appearance_row = tk.Frame(brand_card.body, bg=self.theme.card)
        appearance_row.pack(fill="x", pady=(10, 0))
        tk.Label(appearance_row, text="Appearance", bg=self.theme.card, fg=self.theme.soft, font=self.theme.small_font).pack(anchor="w")
        self.theme_selector = ThemePreviewPicker(
            appearance_row,
            self.theme,
            [(choice, "Glass" if choice == "Liquid Glass" else choice) for choice in THEME_CHOICES],
            self.on_theme_selected,
        )
        self.theme_selector.pack(fill="x", pady=(4, 0))
        self.theme_selector.set_active(self.theme_mode_var.get())

        primary_card = self.make_card(sidebar.body, padding=(14, 14), tint=self.theme.card)
        primary_card.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        tk.Label(primary_card.body, text="Run", bg=self.theme.card, fg=self.theme.text, font=self.theme.section_font).pack(anchor="w")
        tk.Label(primary_card.body, text="Synthetic preset", bg=self.theme.card, fg=self.theme.soft, font=self.theme.small_font).pack(anchor="w", pady=(8, 4))
        self.synthetic_preset_combo = ttk.Combobox(
            primary_card.body,
            textvariable=self.synthetic_preset_var,
            values=self.engine.synthetic_preset_names(),
            state="readonly",
            style="Glass.TCombobox",
        )
        self.synthetic_preset_combo.pack(fill="x", pady=(0, 6))
        
        role = self.current_user.get("role", "analyst").lower() if self.current_user else "analyst"
        if role in {"admin", "analyst"}:
            self.create_action_button(primary_card.body, "Run synthetic dataset", self.theme.dark, self.run_demo, ink="white")
            self.create_action_button(primary_card.body, "Load and score CSV/XLSX", self.theme.blue, self.load_csv, ink="white")
        else:
            tk.Label(primary_card.body, text="Run permissions restricted", bg=self.theme.card, fg=self.theme.orange, font=self.theme.small_font).pack(pady=10)

        exports_card = self.make_card(sidebar.body, padding=(14, 14), tint=self.theme.card)
        exports_card.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        tk.Label(exports_card.body, text="Export", bg=self.theme.card, fg=self.theme.text, font=self.theme.section_font).pack(anchor="w")
        self.create_action_button(exports_card.body, "Export results CSV", self.theme.card_alt, self.export_csv, ink=self.theme.text)
        self.create_action_button(exports_card.body, "Export charts PNG", self.theme.card_alt, self.export_charts, ink=self.theme.text)
        self.create_action_button(exports_card.body, "Export executive brief", self.theme.card_alt, self.export_report, ink=self.theme.text)
        self.create_action_button(exports_card.body, "Export support bundle", self.theme.card_alt, self.export_support_bundle, ink=self.theme.text)

        support_card = self.make_card(sidebar.body, padding=(14, 14), tint=self.theme.card)
        support_card.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        tk.Label(support_card.body, text="Run Status", bg=self.theme.card, fg=self.theme.text, font=self.theme.section_font).pack(anchor="w")
        status_copy = tk.Label(support_card.body, textvariable=self.status_var, bg=self.theme.card, fg=self.theme.muted, font=self.theme.body_font, justify="left")
        status_copy.pack(anchor="w", fill="x", pady=(4, 8))
        bind_wrap_to_width(status_copy, extra_padding=18, min_wrap=230)
        self.progress = ttk.Progressbar(support_card.body, style="Glass.Horizontal.TProgressbar", mode="determinate")
        self.progress.pack(fill="x")

        session_card = self.make_card(sidebar.body, padding=(14, 14), tint=self.theme.card)
        session_card.grid(row=4, column=0, sticky="ew", pady=(0, 10))
        tk.Label(session_card.body, text="Session", bg=self.theme.card, fg=self.theme.text, font=self.theme.section_font).pack(anchor="w")
        self.session_info = tk.Label(
            session_card.body,
            text="No metrics",
            bg=self.theme.card,
            fg=self.theme.muted,
            font=self.theme.body_font,
            justify="left",
        )
        self.session_info.pack(anchor="w", fill="x", pady=(4, 0))
        bind_wrap_to_width(self.session_info, extra_padding=18, min_wrap=230)

        log_card = self.make_card(sidebar.body, padding=(14, 14), tint=self.theme.card)
        log_card.grid(row=5, column=0, sticky="nsew", pady=(0, 4))
        log_card.body.grid_rowconfigure(1, weight=1)
        log_card.body.grid_columnconfigure(0, weight=1)
        tk.Label(log_card.body, text="Log", bg=self.theme.card, fg=self.theme.text, font=self.theme.section_font).grid(row=0, column=0, sticky="w", pady=(0, 10))
        self.log_text = tk.Text(
            log_card.body,
            bg=self.theme.card_alt,
            fg=self.theme.text,
            relief="flat",
            font=("Consolas", 9),
            wrap="word",
            insertbackground=self.theme.text,
            padx=12,
            pady=12,
        )
        self.log_text.grid(row=1, column=0, sticky="nsew")
        self.log_text.configure(state="disabled")

    def show_tab(self, name):
        self.active_tab = name
        if name in self.tab_scroll_panels:
            self.tab_scroll_panels[name].tkraise()
        if hasattr(self, "segmented_tabs"):
            self.segmented_tabs.set_active(name)

    def build_workspace(self):
        workspace = tk.Frame(self.root, bg=self.theme.bg)
        workspace.grid(row=1, column=1, sticky="nsew", padx=(0, 14), pady=(0, 14))
        workspace.grid_rowconfigure(2, weight=1)
        workspace.grid_columnconfigure(0, weight=1)

        hero = self.make_card(workspace, padding=(16, 14), tint=self.theme.card)
        hero.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        hero.body.grid_columnconfigure(0, weight=1)

        hero_left = tk.Frame(hero.body, bg=self.theme.card)
        hero_left.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        hero_left.grid_columnconfigure(0, weight=1)
        tk.Label(hero_left, textvariable=self.summary_title_var, bg=self.theme.card, fg=self.theme.text, font=self.theme.hero_font).grid(row=0, column=0, sticky="w")
        hero_subtitle = tk.Label(
            hero_left,
            textvariable=self.summary_subtitle_var,
            bg=self.theme.card,
            fg=self.theme.muted,
            font=self.theme.body_font,
            justify="left",
        )
        hero_subtitle.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        bind_wrap_to_width(hero_subtitle, extra_padding=18, min_wrap=320)

        hero_meta = self.make_surface(hero.body, tint=self.theme.green_soft, border="#1a2c23")
        hero_meta.grid(row=0, column=1, sticky="e")
        tk.Label(hero_meta, text="Ops status", bg=self.theme.green_soft, fg=self.theme.green, font=self.theme.small_font).pack(anchor="e", padx=12, pady=(8, 2))
        hero_meta_text = tk.Label(hero_meta, textvariable=self.ops_meta_var, bg=self.theme.green_soft, fg=self.theme.text, font=self.theme.body_bold_font, justify="right")
        hero_meta_text.pack(anchor="e", padx=12, pady=(0, 8))
        bind_wrap_to_width(hero_meta_text, extra_padding=18, min_wrap=160)

        nav_card = self.make_card(workspace, padding=(8, 8), tint=self.theme.card_alt)
        nav_card.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        role = self.current_user.get("role", "analyst").lower() if self.current_user else "analyst"

        all_tabs = [
            ("operations", "Operations"),
            ("overview", "Overview"),
            ("customers", "Customers"),
            ("charts", "Charts"),
            ("tools", "Tools"),
        ]
        if ENABLE_EXPERIMENTAL_MAP:
            all_tabs.insert(3, ("map", "Live Map"))
        
        # RBAC Filter for tabs
        if role == "admin":
            active_tabs = all_tabs
        elif role == "analyst":
            active_tabs = [t for t in all_tabs if t[0] != "tools"]
        else: # field
            active_tabs = [t for t in all_tabs if t[0] in {"operations", "map"}]

        self.segmented_tabs = SegmentedTabs(
            nav_card.body,
            self.theme,
            active_tabs,
            self.show_tab,
        )
        self.segmented_tabs.pack(fill="x")

        self.content_host = tk.Frame(workspace, bg=self.theme.bg)
        self.content_host.grid(row=2, column=0, sticky="nsew")
        self.content_host.grid_rowconfigure(0, weight=1)
        self.content_host.grid_columnconfigure(0, weight=1)

        for tab_name, _ in active_tabs:
            panel = ScrollablePanel(
                self.content_host,
                self.theme,
                bg=self.theme.bg,
                lock_x=(tab_name != "charts"),
                allow_horizontal_wheel=(tab_name == "charts"),
                show_horizontal_scrollbar=(tab_name == "charts"),
                horizontal_wheel_factor=8,
            )
            panel.grid(row=0, column=0, sticky="nsew")
            panel.body.grid_columnconfigure(0, weight=1)
            panel.body.grid_rowconfigure(0, weight=1)
            frame = tk.Frame(panel.body, bg=self.theme.bg)
            frame.grid(row=0, column=0, sticky="nsew")
            self.tab_scroll_panels[tab_name] = panel
            self.tab_frames[tab_name] = frame

        if "operations" in self.tab_frames: self.build_operations_tab()
        if "overview" in self.tab_frames: self.build_overview_tab()
        if "customers" in self.tab_frames: self.build_customers_tab()
        if ENABLE_EXPERIMENTAL_MAP and "map" in self.tab_frames:
            self.build_map_tab()
        if "charts" in self.tab_frames: self.build_charts_tab()
        if "tools" in self.tab_frames: self.build_tools_tab()
        
        self.show_tab(active_tabs[0][0])

    def build_detail_row(self, parent, row, label, variable):
        box = self.create_info_block(parent, label, variable, bg=self.theme.card_alt, border=self.theme.border)
        box.grid(row=row, column=0, sticky="ew", pady=5)

    def build_grid_detail_row(self, parent, row, column, label, variable):
        bg = parent.cget("bg")
        box = self.create_info_block(parent, label, variable, bg=bg, border=self.theme.border if bg != self.theme.card else None)
        box.grid(row=row, column=column, sticky="ew", padx=6, pady=6)
        return box

    def configure_tree_tags(self, tree):
        if self.theme_is_dark():
            rows = {
                "Low": "#0d1a12",
                "Moderate": "#171208",
                "High": "#1d1209",
                "Critical": "#241012",
                "Urgent": "#2a1012",
            }
        else:
            rows = {
                "Low": "#eff8f2",
                "Moderate": "#fff7ea",
                "High": "#fff0e6",
                "Critical": "#ffe8e5",
                "Urgent": "#ffe2e0",
            }
        for tag, color in rows.items():
            tree.tag_configure(tag, background=color, foreground=self.theme.text)

    def build_operations_tab(self):
        frame = self.tab_frames["operations"]
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_rowconfigure(1, weight=1)

        self.ops_kpi_row = tk.Frame(frame, bg=self.theme.bg)
        self.ops_kpi_row.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        for idx in range(4):
            self.ops_kpi_row.grid_columnconfigure(idx, weight=1)

        self.ops_kpi_labels = []
        specs = [
            ("Open Cases", "--", self.theme.blue, self.theme.blue_soft),
            ("Urgent", "--", self.theme.red, self.theme.red_soft),
            ("Overdue", "--", self.theme.orange, self.theme.orange_soft),
            ("Resolved", "--", self.theme.green, self.theme.green_soft),
        ]
        for idx, (title, value, color, soft_bg) in enumerate(specs):
            card = self.make_card(self.ops_kpi_row, padding=(16, 14), tint=soft_bg)
            card.grid(row=0, column=idx, sticky="nsew", padx=(0 if idx == 0 else 6, 0 if idx == 3 else 6))
            tk.Label(card.body, text=title, bg=soft_bg, fg=self.theme.soft, font=self.theme.body_font).pack(anchor="w")
            value_label = tk.Label(card.body, text=value, bg=soft_bg, fg=color, font=("Segoe UI Semibold", 20, "bold"))
            value_label.pack(anchor="w", pady=(8, 0))
            self.ops_kpi_labels.append(value_label)

        queue_card = self.make_card(frame, padding=(18, 18), tint=self.theme.card)
        queue_card.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        queue_card.body.grid_rowconfigure(2, weight=1)
        queue_card.body.grid_columnconfigure(0, weight=1)

        queue_header = tk.Frame(queue_card.body, bg=self.theme.card)
        queue_header.grid(row=0, column=0, sticky="ew")
        queue_header.grid_columnconfigure(0, weight=1)
        tk.Label(queue_header, text="Case Queue", bg=self.theme.card, fg=self.theme.text, font=self.theme.hero_font).grid(row=0, column=0, sticky="w")
        tk.Label(queue_header, textvariable=self.ops_queue_count_var, bg=self.theme.card, fg=self.theme.muted, font=self.theme.body_font).grid(row=0, column=1, sticky="e")

        filter_card = self.make_surface(queue_card.body, tint=self.theme.frosted)
        filter_card.grid(row=1, column=0, sticky="ew", pady=(12, 12))
        for col, weight in enumerate([3, 1, 1, 1]):
            filter_card.grid_columnconfigure(col, weight=weight)

        self.ops_search_label = tk.Label(filter_card, text="Search", bg=self.theme.frosted, fg=self.theme.text, font=self.theme.small_font)
        self.ops_status_label = tk.Label(filter_card, text="Status", bg=self.theme.frosted, fg=self.theme.text, font=self.theme.small_font)
        self.ops_risk_label = tk.Label(filter_card, text="Risk Band", bg=self.theme.frosted, fg=self.theme.text, font=self.theme.small_font)
        self.ops_priority_label = tk.Label(filter_card, text="Priority", bg=self.theme.frosted, fg=self.theme.text, font=self.theme.small_font)
        self.ops_search_entry = tk.Entry(filter_card, textvariable=self.ops_search_var, relief="flat", bg=self.theme.card, fg=self.theme.text, insertbackground=self.theme.text, font=self.theme.body_font)
        self.ops_status_combo = ttk.Combobox(filter_card, textvariable=self.ops_status_filter_var, values=["All open", "All statuses"] + CASE_STATUSES, state="readonly", style="Glass.TCombobox")
        self.ops_risk_combo = ttk.Combobox(filter_card, textvariable=self.ops_risk_filter_var, values=["All risk bands"] + list(reversed(DOMAIN_RISK_LABELS)), state="readonly", style="Glass.TCombobox")
        self.ops_priority_combo = ttk.Combobox(filter_card, textvariable=self.ops_priority_filter_var, values=["All priorities"] + CASE_PRIORITIES, state="readonly", style="Glass.TCombobox")
        self.ops_overdue_check = ttk.Checkbutton(filter_card, text="Overdue only", variable=self.ops_overdue_only_var, style="Glass.TCheckbutton", command=self.refresh_ops_view)

        self.ops_search_label.grid(row=0, column=0, sticky="w", padx=(14, 10), pady=(12, 4))
        self.ops_status_label.grid(row=0, column=1, sticky="w", padx=(0, 10), pady=(12, 4))
        self.ops_risk_label.grid(row=0, column=2, sticky="w", padx=(0, 10), pady=(12, 4))
        self.ops_priority_label.grid(row=0, column=3, sticky="w", padx=(0, 14), pady=(12, 4))
        self.ops_search_entry.grid(row=1, column=0, sticky="ew", padx=(14, 10), pady=(0, 10))
        self.ops_status_combo.grid(row=1, column=1, sticky="ew", padx=(0, 10), pady=(0, 10))
        self.ops_risk_combo.grid(row=1, column=2, sticky="ew", padx=(0, 10), pady=(0, 10))
        self.ops_priority_combo.grid(row=1, column=3, sticky="ew", padx=(0, 14), pady=(0, 10))
        self.ops_overdue_check.grid(row=2, column=0, sticky="w", padx=14, pady=(0, 12))

        queue_host = self.make_surface(queue_card.body, tint=self.theme.card_alt)
        queue_host.grid(row=2, column=0, sticky="nsew")
        queue_host.grid_rowconfigure(0, weight=1)
        queue_host.grid_columnconfigure(0, weight=1)

        self.ops_tree = ttk.Treeview(
            queue_host,
            columns=("customer_id", "profile", "risk_band", "fraud_probability", "priority", "status", "follow_up_at", "est_monthly_loss"),
            show="headings",
            style="Glass.Treeview",
        )
        labels = {"customer_id": "Customer ID", "profile": "Profile", "risk_band": "Risk Band", "fraud_probability": "Fraud Probability", "priority": "Priority", "status": "Status", "follow_up_at": "Follow-up", "est_monthly_loss": "Monthly Exposure"}
        for col, width in [("customer_id", 110), ("profile", 120), ("risk_band", 105), ("fraud_probability", 120), ("priority", 82), ("status", 110), ("follow_up_at", 135), ("est_monthly_loss", 130)]:
            self.ops_tree.heading(col, text=labels[col])
            self.ops_tree.column(col, width=width, anchor="center")
        self.configure_tree_tags(self.ops_tree)
        if self.theme_is_dark():
            self.ops_tree.tag_configure("OverdueCase", background="#20170b", foreground=self.theme.text)
            self.ops_tree.tag_configure("ResolvedCase", background="#0d1711", foreground=self.theme.muted)
        else:
            self.ops_tree.tag_configure("OverdueCase", background="#fff7de", foreground=self.theme.text)
            self.ops_tree.tag_configure("ResolvedCase", background="#eef8f1", foreground=self.theme.muted)
        self.ops_tree.grid(row=0, column=0, sticky="nsew")
        self.ops_tree.bind("<<TreeviewSelect>>", self.on_ops_case_selected)
        ops_scrollbar = ttk.Scrollbar(queue_host, orient="vertical", style="Glass.Vertical.TScrollbar", command=self.ops_tree.yview)
        ops_xscroll = ttk.Scrollbar(queue_host, orient="horizontal", command=self.ops_tree.xview)
        self.ops_tree.configure(yscrollcommand=ops_scrollbar.set, xscrollcommand=ops_xscroll.set)
        ops_scrollbar.grid(row=0, column=1, sticky="ns")
        ops_xscroll.grid(row=1, column=0, sticky="ew")

        detail_card = self.make_card(frame, padding=(0, 0), tint=self.theme.card)
        detail_card.grid(row=1, column=1, sticky="nsew", padx=(8, 0))
        detail_card.body.grid_rowconfigure(0, weight=1)
        detail_card.body.grid_columnconfigure(0, weight=1)
        detail_scroll = ScrollablePanel(detail_card.body, self.theme, bg=self.theme.card)
        detail_scroll.grid(row=0, column=0, sticky="nsew")
        detail_body = detail_scroll.body
        detail_body.grid_columnconfigure(0, weight=1)
        detail_body.grid_rowconfigure(6, weight=1)

        tk.Label(detail_body, text="Case", bg=self.theme.card, fg=self.theme.text, font=self.theme.section_font).grid(row=0, column=0, sticky="w", padx=18, pady=(18, 4))
        tk.Label(detail_body, textvariable=self.ops_case_title_var, bg=self.theme.card, fg=self.theme.text, font=self.theme.hero_font, justify="left").grid(row=1, column=0, sticky="ew", padx=18)
        case_subtitle = tk.Label(detail_body, textvariable=self.ops_case_subtitle_var, bg=self.theme.card, fg=self.theme.muted, font=self.theme.body_font, justify="left")
        case_subtitle.grid(row=2, column=0, sticky="ew", padx=18, pady=(6, 14))
        bind_wrap_to_width(case_subtitle, extra_padding=28, min_wrap=240)

        snapshot_card = self.make_card(detail_body, padding=(14, 14), tint=self.theme.card_alt)
        snapshot_card.grid(row=3, column=0, sticky="ew", padx=18, pady=(0, 12))
        tk.Label(snapshot_card.body, text="Latest Score Snapshot", bg=self.theme.card_alt, fg=self.theme.text, font=self.theme.section_font).pack(anchor="w", pady=(0, 8))
        snapshot_grid = tk.Frame(snapshot_card.body, bg=self.theme.card_alt)
        snapshot_grid.pack(fill="x")
        self.ops_snapshot_items = [
            self.create_info_block(snapshot_grid, "Customer", self.ops_customer_var, bg=self.theme.frosted, border=self.theme.border),
            self.create_info_block(snapshot_grid, "Profile", self.ops_profile_var, bg=self.theme.frosted, border=self.theme.border),
            self.create_info_block(snapshot_grid, "Risk band", self.ops_risk_var, bg=self.theme.frosted, border=self.theme.border),
            self.create_info_block(snapshot_grid, "Fraud probability", self.ops_probability_var, bg=self.theme.frosted, border=self.theme.border),
            self.create_info_block(snapshot_grid, "Monthly exposure", self.ops_exposure_var, bg=self.theme.frosted, border=self.theme.border),
            self.create_info_block(snapshot_grid, "Fraud pattern", self.ops_pattern_var, bg=self.theme.frosted, border=self.theme.border),
            self.create_info_block(snapshot_grid, "Last analysis", self.ops_last_analysis_var, bg=self.theme.frosted, border=self.theme.border),
            self.create_info_block(snapshot_grid, "Follow-up status", self.ops_follow_up_display_var, bg=self.theme.frosted, border=self.theme.border),
        ]
        snapshot_grid.bind("<Configure>", lambda _event: self.reflow_card_items(snapshot_grid, self.ops_snapshot_items, narrow_threshold=540), add="+")

        state_card = self.make_card(detail_body, padding=(14, 14), tint=self.theme.card)
        state_card.grid(row=4, column=0, sticky="ew", padx=18, pady=(0, 12))
        tk.Label(state_card.body, text="Status", bg=self.theme.card, fg=self.theme.text, font=self.theme.section_font).pack(anchor="w", pady=(0, 8))
        state_grid = tk.Frame(state_card.body, bg=self.theme.card)
        state_grid.pack(fill="x")
        state_items = [
            self.create_info_block(state_grid, "Current status", self.ops_case_status_display_var, bg=self.theme.card_alt, border=self.theme.border),
            self.create_info_block(state_grid, "Current priority", self.ops_case_priority_display_var, bg=self.theme.card_alt, border=self.theme.border),
            self.create_info_block(state_grid, "Overdue indicator", self.ops_overdue_var, bg=self.theme.orange_soft, border=self.theme.border),
        ]
        state_grid.bind("<Configure>", lambda _event: self.reflow_card_items(state_grid, state_items, narrow_threshold=520), add="+")
        tk.Label(state_card.body, text="Action", bg=self.theme.card, fg=self.theme.text, font=self.theme.body_bold_font).pack(anchor="w", pady=(10, 4))
        recommendation = tk.Label(state_card.body, textvariable=self.ops_recommendation_var, bg=self.theme.card, fg=self.theme.muted, font=self.theme.body_font, justify="left")
        recommendation.pack(anchor="w", fill="x")
        bind_wrap_to_width(recommendation, extra_padding=24, min_wrap=240)

        editor_card = self.make_card(detail_body, padding=(14, 14), tint=self.theme.card_alt)
        editor_card.grid(row=5, column=0, sticky="ew", padx=18, pady=(0, 12))
        tk.Label(editor_card.body, text="Update", bg=self.theme.card_alt, fg=self.theme.text, font=self.theme.section_font).pack(anchor="w", pady=(0, 8))
        form_grid = tk.Frame(editor_card.body, bg=self.theme.card_alt)
        form_grid.pack(fill="x")
        status_field = tk.Frame(form_grid, bg=self.theme.card_alt)
        tk.Label(status_field, text="Status", bg=self.theme.card_alt, fg=self.theme.text, font=self.theme.small_font).pack(anchor="w", pady=(0, 4))
        ttk.Combobox(status_field, textvariable=self.case_status_var, values=CASE_STATUSES, state="readonly", style="Glass.TCombobox").pack(fill="x")
        priority_field = tk.Frame(form_grid, bg=self.theme.card_alt)
        tk.Label(priority_field, text="Priority", bg=self.theme.card_alt, fg=self.theme.text, font=self.theme.small_font).pack(anchor="w", pady=(0, 4))
        ttk.Combobox(priority_field, textvariable=self.case_priority_var, values=CASE_PRIORITIES, state="readonly", style="Glass.TCombobox").pack(fill="x")
        follow_up_field = tk.Frame(form_grid, bg=self.theme.card_alt)
        tk.Label(follow_up_field, text="Follow-up date", bg=self.theme.card_alt, fg=self.theme.text, font=self.theme.small_font).pack(anchor="w", pady=(0, 4))
        tk.Entry(follow_up_field, textvariable=self.case_follow_up_var, relief="flat", bg=self.theme.card, fg=self.theme.text, insertbackground=self.theme.text, font=self.theme.body_font).pack(fill="x")
        resolution_field = tk.Frame(form_grid, bg=self.theme.card_alt)
        tk.Label(resolution_field, text="Resolution reason", bg=self.theme.card_alt, fg=self.theme.text, font=self.theme.small_font).pack(anchor="w", pady=(0, 4))
        ttk.Combobox(resolution_field, textvariable=self.case_resolution_var, values=[""] + RESOLUTION_REASONS, state="readonly", style="Glass.TCombobox").pack(fill="x")
        workflow_items = [status_field, priority_field, follow_up_field, resolution_field]
        form_grid.bind("<Configure>", lambda _event: self.reflow_card_items(form_grid, workflow_items, narrow_threshold=520), add="+")
        tk.Label(editor_card.body, text="YYYY-MM-DD or YYYY-MM-DD HH:MM", bg=self.theme.card_alt, fg=self.theme.muted, font=self.theme.small_font).pack(anchor="w", pady=(10, 10))
        GlassButton(editor_card.body, self.theme, text="Save case changes", command=self.save_case_changes, fill=self.theme.dark, ink="white", anchor="center").pack(anchor="e")

        notes_card = self.make_card(detail_body, padding=(14, 14), tint=self.theme.card)
        notes_card.grid(row=6, column=0, sticky="nsew", padx=18, pady=(0, 18))
        notes_card.body.grid_columnconfigure(0, weight=1)
        notes_card.body.grid_rowconfigure(3, weight=1)
        tk.Label(notes_card.body, text="Notes", bg=self.theme.card, fg=self.theme.text, font=self.theme.section_font).grid(row=0, column=0, sticky="w", pady=(0, 8))
        self.case_note_text = tk.Text(notes_card.body, height=4, relief="flat", bg=self.theme.card_alt, fg=self.theme.text, insertbackground=self.theme.text, font=self.theme.body_font, padx=10, pady=8)
        self.case_note_text.grid(row=1, column=0, sticky="ew")
        actions_frame = tk.Frame(notes_card.body, bg=self.theme.card)
        actions_frame.grid(row=2, column=0, sticky="ew", pady=(10, 10))
        GlassButton(actions_frame, self.theme, text="Escalate & Email Field Agent", command=self.escalate_and_email, fill=self.theme.red, ink="white").pack(side="left")
        GlassButton(actions_frame, self.theme, text="Add note", command=self.add_case_note, fill=self.theme.blue, ink="white").pack(side="right")
        self.case_timeline_text = tk.Text(notes_card.body, relief="flat", bg=self.theme.card_alt, fg=self.theme.text, insertbackground=self.theme.text, font=("Consolas", 9), wrap="word", padx=10, pady=10)
        self.case_timeline_text.grid(row=3, column=0, sticky="nsew")

        self.ops_queue_card = queue_card
        self.ops_detail_card = detail_card
        self.ops_filter_card = filter_card
        frame.bind("<Configure>", self.reflow_operations_layout, add="+")
        filter_card.bind("<Configure>", self.reflow_ops_filter_layout, add="+")
        self.reflow_operations_layout()
        self.reflow_ops_filter_layout()

    def escalate_and_email(self):
        if not self.selected_case_id:
            messagebox.showwarning("Warning", "Please select a case to escalate.")
            return
            
        case_details = self.ops_store.get_case(self.selected_case_id)
        if not case_details:
            return
            
        # Update case state to escalated
        self.case_status_var.set("Escalated")
        self.case_priority_var.set("P1")
        self.ops_store.update_case(
            self.selected_case_id,
            status="Escalated",
            priority="P1",
        )
        # Audit Log
        username = self.current_user.get("username", "Unknown") if self.current_user else "Unknown"
        self.ops_store.add_case_note(self.selected_case_id, f"AUDIT: Case escalated by User '{username}' via automated notification dispatch.")
        
        # Simulate Email Dispatch
        customer_id = case_details.get("customer_id", "Unknown")
        risk_band = case_details.get("risk_band", "High")
        prob = case_details.get("fraud_probability", 0.0)
        
        email_body = (
            f"\n{'='*60}\n"
            f"TO: field.ops@tedas.gov.tr\n"
            f"FROM: MASS-AI Dispatch <noreply@mass-ai.local>\n"
            f"SUBJECT: [URGENT] Investigation Required - Customer {customer_id}\n\n"
            f"Field Operations Team,\n\n"
            f"Please prioritize an investigation for Customer {customer_id}.\n"
            f"Risk Level: {risk_band} ({prob:.2f} probability of anomaly/theft).\n"
            f"Pattern Detected: {case_details.get('fraud_pattern', 'N/A')}\n"
            f"Potential Exposure: {case_details.get('est_monthly_loss', 0.0)} TL/mo.\n\n"
            f"Please refer to the secure portal for full case history.\n"
            f"{'='*60}\n"
        )
        self.log(email_body)
        self.refresh_ops_view()
        messagebox.showinfo("Email Dispatched", f"Urgent investigation email has been dispatched for {customer_id}.")


    def reflow_operations_layout(self, _event=None):
        frame = self.tab_frames.get("operations")
        if frame is None:
            return
        width = frame.winfo_width() or frame.winfo_reqwidth()
        stacked = width < 1320
        if stacked:
            frame.grid_columnconfigure(0, weight=1)
            frame.grid_columnconfigure(1, weight=0)
            if hasattr(self, "ops_queue_card"):
                self.ops_queue_card.grid_configure(row=1, column=0, columnspan=1, padx=(0, 0), pady=(0, 10))
            if hasattr(self, "ops_detail_card"):
                self.ops_detail_card.grid_configure(row=2, column=0, columnspan=1, padx=(0, 0), pady=(0, 0))
        else:
            frame.grid_columnconfigure(0, weight=5)
            frame.grid_columnconfigure(1, weight=4)
            if hasattr(self, "ops_queue_card"):
                self.ops_queue_card.grid_configure(row=1, column=0, columnspan=1, padx=(0, 8), pady=(0, 0))
            if hasattr(self, "ops_detail_card"):
                self.ops_detail_card.grid_configure(row=1, column=1, columnspan=1, padx=(8, 0), pady=(0, 0))

    def reflow_ops_filter_layout(self, _event=None):
        if not hasattr(self, "ops_filter_card"):
            return
        card = self.ops_filter_card
        width = card.winfo_width() or card.winfo_reqwidth()
        compact = width < 820
        for col in range(4):
            card.grid_columnconfigure(col, weight=1)
        if compact:
            # Search
            self.ops_search_label.grid_configure(row=0, column=0, columnspan=2, padx=(14, 10), pady=(10, 4), sticky="w")
            self.ops_search_entry.grid_configure(row=1, column=0, columnspan=2, padx=(14, 10), pady=(0, 8), sticky="ew")
            # Status / Risk
            self.ops_status_label.grid_configure(row=2, column=0, columnspan=1, padx=(14, 10), pady=(0, 4), sticky="w")
            self.ops_status_combo.grid_configure(row=3, column=0, columnspan=1, padx=(14, 10), pady=(0, 8), sticky="ew")
            self.ops_risk_label.grid_configure(row=2, column=1, columnspan=1, padx=(0, 14), pady=(0, 4), sticky="w")
            self.ops_risk_combo.grid_configure(row=3, column=1, columnspan=1, padx=(0, 14), pady=(0, 8), sticky="ew")
            # Priority / overdue
            self.ops_priority_label.grid_configure(row=4, column=0, columnspan=1, padx=(14, 10), pady=(0, 4), sticky="w")
            self.ops_priority_combo.grid_configure(row=5, column=0, columnspan=1, padx=(14, 10), pady=(0, 10), sticky="ew")
            self.ops_overdue_check.grid_configure(row=5, column=1, columnspan=1, padx=(0, 14), pady=(0, 10), sticky="e")
        else:
            self.ops_search_label.grid_configure(row=0, column=0, columnspan=1, padx=(14, 10), pady=(12, 4), sticky="w")
            self.ops_search_entry.grid_configure(row=1, column=0, columnspan=1, padx=(14, 10), pady=(0, 10), sticky="ew")
            self.ops_status_label.grid_configure(row=0, column=1, columnspan=1, padx=(0, 10), pady=(12, 4), sticky="w")
            self.ops_status_combo.grid_configure(row=1, column=1, columnspan=1, padx=(0, 10), pady=(0, 10), sticky="ew")
            self.ops_risk_label.grid_configure(row=0, column=2, columnspan=1, padx=(0, 10), pady=(12, 4), sticky="w")
            self.ops_risk_combo.grid_configure(row=1, column=2, columnspan=1, padx=(0, 10), pady=(0, 10), sticky="ew")
            self.ops_priority_label.grid_configure(row=0, column=3, columnspan=1, padx=(0, 14), pady=(12, 4), sticky="w")
            self.ops_priority_combo.grid_configure(row=1, column=3, columnspan=1, padx=(0, 14), pady=(0, 10), sticky="ew")
            self.ops_overdue_check.grid_configure(row=2, column=0, columnspan=1, padx=14, pady=(0, 12), sticky="w")

    def build_overview_tab(self):
        frame = self.tab_frames["overview"]
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_rowconfigure(2, weight=1)

        self.kpi_row = tk.Frame(frame, bg=self.theme.bg)
        self.kpi_row.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        for idx in range(4):
            self.kpi_row.grid_columnconfigure(idx, weight=1)

        self.kpi_cards = []
        specs = [("Total Customers", "--", self.theme.blue, self.theme.blue_soft), ("High Risk", "--", self.theme.red, self.theme.red_soft), ("Monthly Exposure", "--", self.theme.green, self.theme.green_soft), ("Average Fraud Score", "--", self.theme.orange, self.theme.orange_soft)]
        for idx, (title, value, color, soft_bg) in enumerate(specs):
            card = self.make_card(self.kpi_row, padding=(16, 14), tint=soft_bg)
            card.grid(row=0, column=idx, sticky="nsew", padx=(0 if idx == 0 else 6, 0 if idx == 3 else 6))
            tk.Label(card.body, text=title, bg=soft_bg, fg=self.theme.soft, font=self.theme.body_font).pack(anchor="w")
            value_label = tk.Label(card.body, text=value, bg=soft_bg, fg=color, font=("Segoe UI Semibold", 20, "bold"))
            value_label.pack(anchor="w", pady=(8, 0))
            self.kpi_cards.append(value_label)

        self.model_card = self.make_card(frame, padding=(18, 18), tint=self.theme.card)
        self.model_card.grid(row=1, column=0, sticky="nsew", padx=(0, 6), pady=(0, 12))
        self.model_card.body.grid_columnconfigure(0, weight=1)
        tk.Label(self.model_card.body, text="Model Performance", bg=self.theme.card, fg=self.theme.text, font=self.theme.section_font).grid(row=0, column=0, sticky="w", pady=(0, 10))
        self.model_rows_frame = tk.Frame(self.model_card.body, bg=self.theme.card)
        self.model_rows_frame.grid(row=1, column=0, sticky="nsew")

        self.ops_brief_card = self.make_card(frame, padding=(18, 18), tint=self.theme.card)
        self.ops_brief_card.grid(row=1, column=1, sticky="nsew", padx=(6, 0), pady=(0, 12))
        tk.Label(self.ops_brief_card.body, text="Ops Center Brief", bg=self.theme.card, fg=self.theme.text, font=self.theme.section_font).pack(anchor="w")
        self.ops_text = tk.Label(self.ops_brief_card.body, text="Run analysis to refresh.", bg=self.theme.card, fg=self.theme.muted, font=self.theme.body_font, justify="left")
        self.ops_text.pack(anchor="w", fill="x", pady=(8, 0))
        bind_wrap_to_width(self.ops_text, extra_padding=20, min_wrap=220)

        self.top_card = self.make_card(frame, padding=(18, 18), tint=self.theme.card)
        self.top_card.grid(row=2, column=0, columnspan=2, sticky="nsew")
        self.top_card.body.grid_rowconfigure(1, weight=1)
        self.top_card.body.grid_columnconfigure(0, weight=1)
        tk.Label(self.top_card.body, text="Top Priority Customers", bg=self.theme.card, fg=self.theme.text, font=self.theme.section_font).grid(row=0, column=0, sticky="w", pady=(0, 10))
        self.top_tree = ttk.Treeview(self.top_card.body, columns=("customer_id", "profile", "risk_score", "risk_category", "est_monthly_loss"), show="headings", style="Glass.Treeview", height=10)
        labels = {"customer_id": "Customer ID", "profile": "Profile", "risk_score": "Risk Score", "risk_category": "Risk Band", "est_monthly_loss": "Monthly Exposure"}
        for col, width in [("customer_id", 120), ("profile", 120), ("risk_score", 110), ("risk_category", 120), ("est_monthly_loss", 150)]:
            self.top_tree.heading(col, text=labels[col])
            self.top_tree.column(col, width=width, anchor="center")
        self.configure_tree_tags(self.top_tree)
        self.top_tree.grid(row=1, column=0, sticky="nsew")
        self.top_tree.bind("<Double-1>", self.jump_to_selected_top_customer)

    def build_customers_tab(self):
        frame = self.tab_frames["customers"]
        frame.grid_columnconfigure(0, weight=5)
        frame.grid_columnconfigure(1, weight=3)
        frame.grid_rowconfigure(1, weight=1)

        toolbar = self.make_card(frame, padding=(14, 12), tint=self.theme.card)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        toolbar.body.grid_columnconfigure(0, weight=3)
        toolbar.body.grid_columnconfigure(1, weight=2)
        self.customer_search_label = tk.Label(toolbar.body, text="Customer Search", bg=self.theme.card, fg=self.theme.text, font=self.theme.small_font)
        self.customer_risk_label = tk.Label(toolbar.body, text="Risk Filter", bg=self.theme.card, fg=self.theme.text, font=self.theme.small_font)
        self.customer_search_entry = tk.Entry(toolbar.body, textvariable=self.customer_search_var, relief="flat", bg=self.theme.card_alt, fg=self.theme.text, insertbackground=self.theme.text, font=self.theme.body_font)
        self.customer_risk_combo = ttk.Combobox(toolbar.body, textvariable=self.risk_filter_var, values=["All risk bands"] + list(reversed(DOMAIN_RISK_LABELS)), state="readonly", style="Glass.TCombobox")
        self.customer_count_label = tk.Label(toolbar.body, textvariable=self.table_count_var, bg=self.theme.card, fg=self.theme.muted, font=self.theme.body_font)
        self.customer_search_label.grid(row=0, column=0, sticky="w", pady=(0, 4))
        self.customer_risk_label.grid(row=0, column=1, sticky="w", padx=(12, 0), pady=(0, 4))
        self.customer_search_entry.grid(row=1, column=0, sticky="ew", padx=(0, 12))
        self.customer_risk_combo.grid(row=1, column=1, sticky="ew")
        self.customer_count_label.grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))

        table_card = self.make_card(frame, padding=(14, 14), tint=self.theme.card)
        table_card.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        table_card.body.grid_rowconfigure(1, weight=1)
        table_card.body.grid_columnconfigure(0, weight=1)
        tk.Label(table_card.body, text="Customers", bg=self.theme.card, fg=self.theme.text, font=self.theme.section_font).grid(row=0, column=0, sticky="w", pady=(0, 10))
        table_host = self.make_surface(table_card.body, tint=self.theme.card_alt)
        table_host.grid(row=1, column=0, sticky="nsew")
        table_host.grid_rowconfigure(0, weight=1)
        table_host.grid_columnconfigure(0, weight=1)

        self.customer_tree = ttk.Treeview(
            table_host,
            columns=("customer_id", "profile", "risk_score", "risk_category", "theft_probability", "est_monthly_loss", "priority_index"),
            show="headings",
            style="Glass.Treeview",
        )
        labels = {"customer_id": "Customer ID", "profile": "Profile", "risk_score": "Risk Score", "risk_category": "Risk Band", "theft_probability": "Fraud Probability", "est_monthly_loss": "Monthly Exposure", "priority_index": "Priority Index"}
        for col, width in [("customer_id", 140), ("profile", 170), ("risk_score", 120), ("risk_category", 130), ("theft_probability", 160), ("est_monthly_loss", 175), ("priority_index", 145)]:
            self.customer_tree.heading(col, text=labels[col], command=lambda key=col: self.set_customer_sort(key))
            self.customer_tree.column(col, width=width, minwidth=110, anchor="center", stretch=False)
        self.configure_tree_tags(self.customer_tree)
        self.customer_tree.grid(row=0, column=0, sticky="nsew")
        self.customer_tree.bind("<<TreeviewSelect>>", self.on_customer_selected)
        self.customer_tree.bind("<Double-1>", self.jump_to_selected_customer_case)
        scrollbar = ttk.Scrollbar(table_host, orient="vertical", style="Glass.Vertical.TScrollbar", command=self.customer_tree.yview)
        xscroll = ttk.Scrollbar(table_host, orient="horizontal", style="Glass.Horizontal.TScrollbar", command=self.customer_tree.xview)
        self.customer_tree.configure(yscrollcommand=scrollbar.set, xscrollcommand=xscroll.set)
        scrollbar.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        self.customer_xscroll = xscroll
        self.bind_customer_horizontal_navigation()

        detail_card = self.make_card(frame, padding=(14, 14), tint=self.theme.card)
        detail_card.grid(row=1, column=1, sticky="nsew", padx=(8, 0))
        detail_card.body.grid_columnconfigure(0, weight=1)
        tk.Label(detail_card.body, text="Customer Detail", bg=self.theme.card, fg=self.theme.text, font=self.theme.section_font).grid(row=0, column=0, sticky="w", pady=(0, 10))
        detail_body = tk.Frame(detail_card.body, bg=self.theme.card)
        detail_body.grid(row=1, column=0, sticky="nsew")
        detail_body.grid_columnconfigure(0, weight=1)
        self.build_detail_row(detail_body, 0, "Customer", self.detail_customer_var)
        self.build_detail_row(detail_body, 1, "Profile", self.detail_profile_var)
        self.build_detail_row(detail_body, 2, "Risk", self.detail_risk_var)
        self.build_detail_row(detail_body, 3, "Probability", self.detail_probability_var)
        self.build_detail_row(detail_body, 4, "Monthly exposure", self.detail_loss_var)
        self.build_detail_row(detail_body, 5, "Fraud pattern", self.detail_theft_type_var)

        recommendation_card = self.make_surface(detail_body, tint=self.theme.card_alt)
        recommendation_card.grid(row=6, column=0, sticky="ew", pady=(12, 0))
        tk.Label(recommendation_card, text="Action", bg=self.theme.card_alt, fg=self.theme.text, font=self.theme.body_bold_font).pack(anchor="w", padx=12, pady=(12, 6))
        recommendation_text = tk.Label(recommendation_card, textvariable=self.detail_recommendation_var, bg=self.theme.card_alt, fg=self.theme.muted, font=self.theme.body_font, justify="left")
        recommendation_text.pack(anchor="w", fill="x", padx=12, pady=(0, 12))
        bind_wrap_to_width(recommendation_text, extra_padding=24, min_wrap=180)
        self.customers_toolbar = toolbar
        self.customers_table_card = table_card
        self.customers_detail_card = detail_card
        frame.bind("<Configure>", self.reflow_customers_layout, add="+")
        toolbar.body.bind("<Configure>", self.reflow_customers_toolbar, add="+")
        self.reflow_customers_layout()
        self.reflow_customers_toolbar()

    def normalized_wheel_steps(self, event) -> int:
        if getattr(event, "delta", 0):
            delta = int(-event.delta / 120)
            if delta == 0:
                return -1 if event.delta > 0 else 1
            return delta
        num = getattr(event, "num", 0)
        if num == 4:
            return -1
        if num == 5:
            return 1
        return 0

    def bind_customer_horizontal_navigation(self):
        bindings = (
            ("<Shift-MouseWheel>", self.on_customer_horizontal_wheel),
            ("<Shift-Button-4>", self.on_customer_horizontal_wheel),
            ("<Shift-Button-5>", self.on_customer_horizontal_wheel),
        )
        for event_name, handler in bindings:
            self.customer_tree.bind(event_name, handler, add="+")
            self.customer_xscroll.bind(event_name, handler, add="+")
        self.customer_xscroll.bind("<MouseWheel>", self.on_customer_horizontal_wheel, add="+")
        self.customer_xscroll.bind("<Button-4>", self.on_customer_horizontal_wheel, add="+")
        self.customer_xscroll.bind("<Button-5>", self.on_customer_horizontal_wheel, add="+")

    def on_customer_horizontal_wheel(self, event):
        steps = self.normalized_wheel_steps(event)
        if steps == 0:
            return None
        self.customer_tree.xview_scroll(steps * 6, "units")
        return "break"

    def reflow_customers_layout(self, _event=None):
        frame = self.tab_frames.get("customers")
        if frame is None:
            return
        width = frame.winfo_width() or frame.winfo_reqwidth()
        stacked = width < 1160
        if stacked:
            frame.grid_columnconfigure(0, weight=1)
            frame.grid_columnconfigure(1, weight=0)
            if hasattr(self, "customers_table_card"):
                self.customers_table_card.grid_configure(row=1, column=0, padx=(0, 0), pady=(0, 10))
            if hasattr(self, "customers_detail_card"):
                self.customers_detail_card.grid_configure(row=2, column=0, padx=(0, 0), pady=(0, 0))
        else:
            frame.grid_columnconfigure(0, weight=5)
            frame.grid_columnconfigure(1, weight=3)
            if hasattr(self, "customers_table_card"):
                self.customers_table_card.grid_configure(row=1, column=0, padx=(0, 8), pady=(0, 0))
            if hasattr(self, "customers_detail_card"):
                self.customers_detail_card.grid_configure(row=1, column=1, padx=(8, 0), pady=(0, 0))

    def reflow_customers_toolbar(self, _event=None):
        if not hasattr(self, "customers_toolbar"):
            return
        body = self.customers_toolbar.body
        width = body.winfo_width() or body.winfo_reqwidth()
        compact = width < 760
        if compact:
            self.customer_search_label.grid_configure(row=0, column=0, columnspan=2, padx=(0, 0), pady=(0, 4), sticky="w")
            self.customer_search_entry.grid_configure(row=1, column=0, columnspan=2, padx=(0, 0), sticky="ew")
            self.customer_risk_label.grid_configure(row=2, column=0, columnspan=2, padx=(0, 0), pady=(8, 4), sticky="w")
            self.customer_risk_combo.grid_configure(row=3, column=0, columnspan=2, padx=(0, 0), sticky="ew")
            self.customer_count_label.grid_configure(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))
        else:
            self.customer_search_label.grid_configure(row=0, column=0, columnspan=1, padx=(0, 0), pady=(0, 4), sticky="w")
            self.customer_risk_label.grid_configure(row=0, column=1, columnspan=1, padx=(12, 0), pady=(0, 4), sticky="w")
            self.customer_search_entry.grid_configure(row=1, column=0, columnspan=1, padx=(0, 12), sticky="ew")
            self.customer_risk_combo.grid_configure(row=1, column=1, columnspan=1, padx=(0, 0), sticky="ew")
            self.customer_count_label.grid_configure(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))

    def build_charts_tab(self):
        frame = self.tab_frames["charts"]
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(0, weight=1)
        chart_card = self.make_card(frame, padding=(18, 18), tint=self.theme.card)
        chart_card.grid(row=0, column=0, sticky="nsew")
        chart_card.body.grid_rowconfigure(2, weight=1)
        chart_card.body.grid_columnconfigure(0, weight=1)
        tk.Label(chart_card.body, text="Charts", bg=self.theme.card, fg=self.theme.text, font=self.theme.hero_font).grid(row=0, column=0, sticky="w")
        chart_copy = tk.Label(chart_card.body, text="Visual brief", bg=self.theme.card, fg=self.theme.muted, font=self.theme.body_font, justify="left")
        chart_copy.grid(row=1, column=0, sticky="ew", pady=(6, 12))
        bind_wrap_to_width(chart_copy, extra_padding=20, min_wrap=280)
        self.chart_host = tk.Frame(chart_card.body, bg=self.theme.card)
        self.chart_host.grid(row=2, column=0, sticky="nsew")

    def build_map_tab(self):
        frame = self.tab_frames["map"]
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)

        header = self.make_card(frame)
        header.grid(row=0, column=0, sticky="ew", padx=16, pady=16)
        tk.Label(header.body, text="Live Geographic Heatmap", font=self.theme.hero_font, bg=self.theme.card, fg=self.theme.text).pack(side="left")
        
        self.api_poll_state = False
        self.api_btn = GlassButton(header.body, self.theme, text="Start Real-time API Stream", command=self.toggle_api_polling, fill=self.theme.blue, ink="white")
        self.api_btn.pack(side="right", padx=10, pady=4)

        map_host = tk.Frame(frame, bg=self.theme.bg)
        map_host.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 16))
        
        if tkintermapview:
            self.map_widget = tkintermapview.TkinterMapView(map_host, corner_radius=12)
            self.map_widget.pack(fill="both", expand=True)
            self.map_widget.set_position(39.0, 35.0)  # Turkey center
            self.map_widget.set_zoom(6)
            self.map_markers = []
        else:
            tk.Label(map_host, text="tkintermapview package not installed.", bg=self.theme.bg, fg=self.theme.muted).pack(pady=50)

    def toggle_api_polling(self):
        if not self.api_poll_state:
            if self.current_df is None or self.current_df.empty:
                messagebox.showerror("Error", "Load or generate data first to mock API polling.")
                return
            self.api_poll_state = True
            self.api_btn.config(text="Stop API Polling")
            self.log("API Polling started. Listening for incoming TEDAŞ webhooks...")
            self._polling = True
            def _mock_poll():
                if getattr(self, "_polling", False):
                    self.log(f"ALERT: Webhook from TEDAŞ Region-B received. Analyzed risk: HIGH.")
                    self.log(f"SYSTEM: Anomaly detected at Transformer TR-{(np.random.randint(1,50)):03d}.")
                    self.root.after(15000, _mock_poll)
            self.root.after(3000, _mock_poll)
        else:
            self._polling = False
            self.api_poll_state = False
            self.api_btn.config(text="Start Real-time API Stream")
            self.log("API Polling successfully stopped.")

    def render_map_markers(self):
        if not hasattr(self, "map_widget") or getattr(self.engine, "df_scored", None) is None:
            return
            
        for m in getattr(self, "map_markers", []):
            m.delete()
        self.map_markers = []
        
        df = self.engine.df_scored
        if "latitude" not in df.columns:
            return
            
        top_cases = df.head(50)
        for _, row in top_cases.iterrows():
            lat = row.get("latitude")
            lon = row.get("longitude")
            if pd.isna(lat) or pd.isna(lon):
                continue
            
            risk = str(row.get("risk_category", "Low"))
            color = self.theme.red if risk in {"Critical", "Urgent"} else (self.theme.orange if risk == "High" else self.theme.green)
            
            try:
                marker = self.map_widget.set_marker(lat, lon, text=f"{risk} Risk", marker_color_outside=color, marker_color_circle=self.theme.card)
                self.map_markers.append(marker)
            except Exception:
                pass


    def build_tools_tab(self):
        frame = self.tab_frames["tools"]
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=1)

        # Left column: Actions
        actions_card = self.make_card(frame, padding=(18, 18), tint=self.theme.card)
        actions_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=(0, 6))
        actions_card.body.grid_columnconfigure(0, weight=1)
        tk.Label(actions_card.body, text="Actions", bg=self.theme.card, fg=self.theme.text, font=self.theme.hero_font).grid(row=0, column=0, sticky="w")
        tk.Label(actions_card.body, text="Install, test, and build from here", bg=self.theme.card, fg=self.theme.muted, font=self.theme.body_font).grid(row=1, column=0, sticky="w", pady=(4, 12))

        btn_frame = tk.Frame(actions_card.body, bg=self.theme.card)
        btn_frame.grid(row=2, column=0, sticky="ew")
        btn_frame.grid_columnconfigure(0, weight=1)

        for idx, (label, cmd) in enumerate([
            ("Install dependencies", self._tools_install_deps),
            ("Run smoke tests", self._tools_run_tests),
            ("Build Windows EXE", self._tools_build_exe),
        ]):
            fill_color = self.theme.blue if idx == 0 else self.theme.card_alt
            ink_color = "#ffffff" if idx == 0 else self.theme.text
            btn = GlassButton(btn_frame, self.theme, text=label, command=cmd, fill=fill_color, ink=ink_color, anchor="center")
            btn.grid(row=idx, column=0, sticky="ew", pady=(0 if idx == 0 else 8, 0))

        # Right column: Quick folders
        assets_card = self.make_card(frame, padding=(18, 18), tint=self.theme.card)
        assets_card.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=(0, 6))
        assets_card.body.grid_columnconfigure(0, weight=1)
        tk.Label(assets_card.body, text="Quick Access", bg=self.theme.card, fg=self.theme.text, font=self.theme.hero_font).grid(row=0, column=0, sticky="w")
        tk.Label(assets_card.body, text="Open project folders", bg=self.theme.card, fg=self.theme.muted, font=self.theme.body_font).grid(row=1, column=0, sticky="w", pady=(4, 12))

        folder_frame = tk.Frame(assets_card.body, bg=self.theme.card)
        folder_frame.grid(row=2, column=0, sticky="ew")
        folder_frame.grid_columnconfigure(0, weight=1)

        base_dir = Path(__file__).resolve().parent.parent
        for idx, (label, folder) in enumerate([
            ("Project folder", Path(__file__).resolve().parent),
            ("Documentation", base_dir / "docs"),
            ("Images", base_dir / "images"),
            ("Business docs", base_dir / "business_docs"),
        ]):
            btn = GlassButton(folder_frame, self.theme, text=label, command=lambda p=folder: self._open_folder(p), fill=self.theme.card_alt, ink=self.theme.text, anchor="center")
            btn.grid(row=idx, column=0, sticky="ew", pady=(0 if idx == 0 else 8, 0))

        # Bottom: Output console
        console_card = self.make_card(frame, padding=(14, 14), tint=self.theme.card)
        console_card.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(6, 0))
        console_card.body.grid_rowconfigure(1, weight=1)
        console_card.body.grid_columnconfigure(0, weight=1)
        tk.Label(console_card.body, text="Console Output", bg=self.theme.card, fg=self.theme.text, font=self.theme.section_font).grid(row=0, column=0, sticky="w", pady=(0, 8))
        self.tools_console = tk.Text(
            console_card.body, bg=self.theme.card_alt, fg=self.theme.text, relief="flat",
            font=("Consolas", 9), wrap="word", insertbackground=self.theme.text, padx=12, pady=12, height=12,
        )
        self.tools_console.grid(row=1, column=0, sticky="nsew")
        self.tools_console.configure(state="disabled")
        frame.grid_rowconfigure(1, weight=1)

    def _tools_console_write(self, text):
        self.tools_console.configure(state="normal")
        self.tools_console.insert("end", text + "\n")
        self.tools_console.see("end")
        self.tools_console.configure(state="disabled")

    def _tools_run_subprocess(self, cmd, label):
        import threading
        self.tools_console.configure(state="normal")
        self.tools_console.delete("1.0", "end")
        self.tools_console.configure(state="disabled")
        self._tools_console_write(f"▶ {label}...")
        self._tools_console_write(f"  Command: {cmd}\n")

        def run():
            try:
                proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        cwd=str(Path(__file__).resolve().parent), text=True)
                for line in proc.stdout:
                    self.root.after(0, self._tools_console_write, line.rstrip())
                proc.wait()
                code = proc.returncode
                self.root.after(0, self._tools_console_write, f"\n{'✓ Done' if code == 0 else f'✗ Exited with code {code}'}")
            except Exception as exc:
                self.root.after(0, self._tools_console_write, f"✗ Error: {exc}")

        threading.Thread(target=run, daemon=True).start()

    def _tools_install_deps(self):
        req = Path(__file__).resolve().parent / "requirements.txt"
        self._tools_run_subprocess(f'"{sys.executable}" -m pip install -r "{req}"', "Installing dependencies")

    def _tools_run_tests(self):
        self._tools_run_subprocess(f'"{sys.executable}" -m unittest discover -s tests -t . -p "test_*.py" -v', "Running smoke tests")

    def _tools_build_exe(self):
        self._tools_run_subprocess(
            f'"{sys.executable}" -m PyInstaller --noconfirm --clean --onefile --windowed --name MASS_AI_Desktop '
            '--distpath artifacts/dist --workpath artifacts/build --specpath packaging '
            '--collect-all xgboost --collect-all sklearn --collect-data matplotlib mass_ai_desktop.py',
            "Building Windows EXE"
        )

    def _open_folder(self, path):
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(path))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:
            messagebox.showerror("MASS-AI", f"Could not open folder:\n{exc}")

    def format_column_value(self, column, value):
        if column in {"theft_probability", "fraud_probability"}:
            return fmt_percent(value, decimals=2)
        if column == "risk_score":
            return f"{float(value or 0):.1f}"
        if column == "priority_index":
            return f"{float(value or 0):.2f}"
        if column == "est_monthly_loss":
            return fmt_currency(value)
        if column in {"follow_up_at", "last_analysis_at", "updated_at"}:
            return format_local_datetime(value)
        return safe_text(value)

    def set_busy(self, busy):
        self.busy = busy
        for button in self.action_buttons:
            button.configure(state=("disabled" if busy else "normal"))
        if hasattr(self, "theme_selector"):
            self.theme_selector.set_enabled(not busy)
        if hasattr(self, "synthetic_preset_combo"):
            self.synthetic_preset_combo.configure(state=("disabled" if busy else "readonly"))
        self.root.configure(cursor=("watch" if busy else ""))

    def set_status(self, message):
        self.status_var.set(message)

    def update_progress(self, percent, message):
        if hasattr(self, "progress"):
            self.progress["value"] = percent
        self.set_status(message)

    def update_data_source(self):
        self.data_source_var.set(f"Data source - {self.engine.last_source}")

    def log(self, message):
        self.engine.log(message)
        if hasattr(self, "log_text") and not self._closing:
            self.update_log()

    def update_log(self):
        if not hasattr(self, "log_text"):
            return
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", tk.END)
        for line in self.engine.log_lines[-28:]:
            self.log_text.insert(tk.END, line + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")

    def on_theme_selected(self, selected=None, _event=None):
        if self._rebuilding_theme or self.busy:
            if self.busy:
                self.set_status("Wait for the current analysis to finish before changing appearance")
            return
        chosen_value = selected if isinstance(selected, str) else self.theme_mode_var.get()
        selected = normalize_theme_name(chosen_value)
        self.theme_mode_var.set(selected)
        self.apply_theme(selected)

    def apply_theme(self, theme_name):
        if self._rebuilding_theme or self.busy:
            return
        normalized_theme_name = normalize_theme_name(theme_name)
        self.theme_mode_var.set(normalized_theme_name)
        os.environ["MASS_AI_THEME"] = normalized_theme_name
        save_theme_preference(normalized_theme_name)
        active_tab = self.active_tab
        selected_case_id = self.selected_case_id
        progress_value = float(self.progress["value"]) if hasattr(self, "progress") else 0
        self._rebuilding_theme = True
        # Gecici arka plan ve gorsel stabilite icin cam/saydamlik hilesi (Titreme cozumu)
        self.root.wm_attributes("-alpha", 0.0) 
        try:
            self.theme = build_glass_theme(normalized_theme_name)
            self.root.configure(bg=self.theme.bg)
            if self.current_figure is not None:
                plt.close(self.current_figure)
                self.current_figure = None
            self.current_canvas = None
            for child in self.root.winfo_children():
                child.destroy()
            self.action_buttons = []
            self.tab_buttons = {}
            self.tab_frames = {}
            self.tab_scroll_panels = {}
            self.configure_styles()
            self.setup_ui()
            self.build_menu()
            if hasattr(self, "progress"):
                self.progress["value"] = progress_value
            self.update_log()
            if self.current_df is not None:
                self.show_results()
            self.refresh_ops_view(preferred_customer_id=selected_case_id)
            self.show_tab(active_tab if active_tab in self.tab_scroll_panels else "operations")
        finally:
            self.root.update_idletasks() # UI'nin cizilmesini bekle
            self.root.wm_attributes("-alpha", 1.0) # Gorunurlugu geri getir
            self._rebuilding_theme = False

    def run_in_background(self, worker):
        if self.busy:
            return
        self.set_busy(True)

        def wrapped():
            try:
                worker()
            except Exception as exc:
                self.engine.log(f"ERROR: {exc}")
                self.post_ui(self.update_log)
                self.post_ui(self.update_progress, 0, f"Error: {str(exc)[:120]}")
                self.post_ui(messagebox.showerror, "MASS-AI", str(exc))
            finally:
                self.post_ui(self.set_busy, False)

        threading.Thread(target=wrapped, daemon=True).start()

    def run_demo(self):
        if self.busy:
            return

        def worker():
            progress = self.progress_callback()
            self.engine.generate_synthetic(callback=progress, preset_name=self.synthetic_preset_var.get())
            self.engine.train_models(callback=progress)
            self.engine.score_customers(callback=progress)
            self.post_ui(self.after_analysis_complete, "Synthetic dataset analysis complete")

        self.run_in_background(worker)

    def load_csv(self):
        if self.busy:
            return

        path = filedialog.askopenfilename(
            title="Select a CSV or Excel file",
            filetypes=[("Supported files", "*.csv *.xlsx *.xls"), ("CSV files", "*.csv"), ("Excel files", "*.xlsx;*.xls"), ("All files", "*.*")],
        )
        if not path:
            return

        def worker():
            progress = self.progress_callback()
            self.engine.load_dataset(path, callback=progress)
            self.engine.train_models(callback=progress)
            self.engine.score_customers(callback=progress)
            self.post_ui(self.after_analysis_complete, f"Dataset scored: {os.path.basename(path)}")

        self.run_in_background(worker)

    def sync_ops_center(self):
        overview = self.engine.build_overview()
        if overview is None or self.engine.df_scored is None:
            return
        run_meta = {
            "created_at": overview["last_run_at_full"] or datetime.now().isoformat(timespec="minutes"),
            "source_name": overview["data_source"],
            "model_name": overview["best_model"],
            "customer_count": overview["customer_count"],
            "high_risk_count": overview["high_risk_count"],
            "total_exposure": overview["total_loss"],
        }
        self.ops_store.sync_run(self.engine.df_scored, run_meta)
        self.engine.log("Ops Center queue synced with the latest scoring run")

    def after_analysis_complete(self, status_message):
        self.sync_ops_center()
        self.update_progress(100, status_message)
        self.update_data_source()
        self.update_log()
        self.show_results()
        if ENABLE_EXPERIMENTAL_MAP:
            self.render_map_markers()
        self.refresh_ops_view(preserve_selection=False)
        self.show_tab("operations")

    def on_close(self):
        if self._closing:
            return
        if self.busy and not messagebox.askyesno("MASS-AI", "An analysis is still running. Close the app anyway?"):
            return
        self._closing = True
        try:
            if self.current_figure is not None:
                plt.close(self.current_figure)
                self.current_figure = None
        except Exception:
            pass
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def render_model_rows(self):
        for widget in self.model_rows_frame.winfo_children():
            widget.destroy()
        if not self.engine.results:
            tk.Label(self.model_rows_frame, text="No model results yet.", bg=self.theme.card, fg=self.theme.muted, font=self.theme.body_font).pack(anchor="w")
            return

        header = tk.Frame(self.model_rows_frame, bg=self.theme.card_alt)
        header.pack(fill="x", pady=(0, 6))
        for text, width in [("Model", 24), ("ROC-AUC", 12), ("F1", 10), ("Type", 14)]:
            tk.Label(header, text=text, width=width, anchor="w", bg=self.theme.card_alt, fg=self.theme.text, font=self.theme.body_bold_font).pack(side="left", padx=4, pady=8)

        ordered = sorted(self.engine.results, key=lambda item: self.engine.results[item].get("auc", 0), reverse=True)
        for name in ordered:
            result = self.engine.results[name]
            row = tk.Frame(self.model_rows_frame, bg=self.theme.card)
            row.pack(fill="x", pady=2)
            values = [(name, 24, self.theme.text), (f"{result.get('auc', 0):.4f}", 12, self.theme.blue), (f"{result.get('f1', 0):.4f}", 10, self.theme.text), (result.get("type", "-"), 14, self.theme.muted)]
            for text, width, color in values:
                tk.Label(row, text=text, width=width, anchor="w", bg=self.theme.card, fg=color, font=self.theme.small_font).pack(side="left", padx=4, pady=4)

    def render_ops_notes(self, overview, ops_metrics):
        if not overview:
            self.ops_text.configure(text="Run analysis to refresh.")
            return
        preset_note = (
            f"Preset context: {overview.get('preset_name')} - {overview.get('preset_summary')}"
            if overview.get("preset_name")
            else "Preset context: imported dataset"
        )
        notes = [
            f"Leading model: {overview['best_model']}.",
            preset_note,
            f"Open Ops cases: {ops_metrics.get('open_cases', 0)} with {ops_metrics.get('overdue', 0)} overdue follow-ups.",
            f"Critical or urgent customers in the latest run: {overview['critical_count']}.",
            safe_text(overview.get("explainability_summary") or "Explainability snapshot will appear after scoring."),
            f"Top customer in the latest run: {overview['top_customer']}.",
            "Recommended sequence: review overdue cases first, then move urgent and critical accounts into active dispatch.",
        ]
        self.ops_text.configure(text="\n".join(notes))

    def populate_tree(self, tree, rows, columns, risk_column="risk_category", extra_tag_func=None):
        tree.delete(*tree.get_children())
        for _, row in rows.iterrows():
            risk_tag = safe_text(row.get(risk_column, ""))
            tags = []
            if extra_tag_func is not None:
                extra_tag = extra_tag_func(row)
                if extra_tag:
                    tags.append(extra_tag)
            if risk_tag in DOMAIN_RISK_LABELS:
                tags.append(risk_tag)
            values = [self.format_column_value(col, row.get(col, "-")) for col in columns]
            item_id = safe_text(row.get("customer_id", ""))
            if tree is getattr(self, "ops_tree", None) and item_id:
                tree.insert("", tk.END, iid=item_id, values=values, tags=tuple(tags))
            else:
                tree.insert("", tk.END, values=values, tags=tuple(tags))

    def show_results(self):
        overview = self.engine.build_overview()
        if overview is None:
            return

        self.current_df = self.engine.df_scored.copy()
        source_label = safe_text(overview.get("preset_name") or overview["data_source"])
        self.summary_title_var.set(f"{overview['customer_count']} customers scored")
        self.summary_subtitle_var.set(f"{source_label}  |  {overview['best_model']}  |  {overview['last_run_at']}")
        self.workspace_meta_var.set(f"{version_label()} | {source_label}")
        self.ops_meta_var.set(f"{overview['best_model']} | {source_label} | {overview['last_run_at']}")
        ops_metrics = self.ops_store.case_metrics()
        self.session_info.configure(text=f"Open Ops cases: {ops_metrics.get('open_cases', 0)}\nUrgent cases: {ops_metrics.get('urgent', 0)}\nOverdue follow-ups: {ops_metrics.get('overdue', 0)}\nEstimated exposure: {fmt_currency(overview['total_loss'])}")

        for label, value in zip(
            self.kpi_cards,
            [str(overview["customer_count"]), str(overview["high_risk_count"]), fmt_currency(overview["total_loss"]), fmt_percent(overview["average_probability"], decimals=1)],
        ):
            label.configure(text=value)

        self.render_model_rows()
        self.render_ops_notes(overview, ops_metrics)
        self.populate_tree(self.top_tree, self.current_df.head(10), ["customer_id", "profile", "risk_score", "risk_category", "est_monthly_loss"])
        if not self.current_df.empty:
            self.update_customer_detail_panel(self.current_df.iloc[0])
        self.refresh_customer_table()
        self.render_chart_tab()

    def filtered_customers(self):
        if self.current_df is None:
            return pd.DataFrame()
        filtered = self.current_df.copy()
        search = self.customer_search_var.get().strip().lower()
        risk = self.risk_filter_var.get().strip()
        if search:
            mask = pd.Series(False, index=filtered.index)
            for col in ["customer_id", "profile", "risk_category", "theft_type"]:
                if col in filtered.columns:
                    mask = mask | filtered[col].astype(str).str.lower().str.contains(search, na=False)
            filtered = filtered[mask]
        if risk and risk != "All risk bands" and "risk_category" in filtered.columns:
            filtered = filtered[filtered["risk_category"].astype(str) == risk]
        if self.customer_sort_column in filtered.columns:
            filtered = filtered.sort_values(self.customer_sort_column, ascending=not self.customer_sort_reverse, kind="mergesort")
        return filtered

    def refresh_customer_table(self):
        if self.current_df is None:
            self.customer_tree.delete(*self.customer_tree.get_children())
            self.table_count_var.set("0 records")
            return

        filtered = self.filtered_customers()
        self.populate_tree(self.customer_tree, filtered.head(300), ["customer_id", "profile", "risk_score", "risk_category", "theft_probability", "est_monthly_loss", "priority_index"])
        self.table_count_var.set(f"{len(filtered)} {'records shown (first 300)' if len(filtered) > 300 else 'records shown'}")
        children = self.customer_tree.get_children()
        if children:
            self.customer_tree.selection_set(children[0])
            self.customer_tree.focus(children[0])
            self.on_customer_selected()
        else:
            self.detail_customer_var.set("-")
            self.detail_profile_var.set("-")
            self.detail_risk_var.set("-")
            self.detail_probability_var.set("-")
            self.detail_loss_var.set("-")
            self.detail_theft_type_var.set("-")
            self.detail_recommendation_var.set("No matching customers")

    def set_customer_sort(self, column):
        if self.customer_sort_column == column:
            self.customer_sort_reverse = not self.customer_sort_reverse
        else:
            self.customer_sort_column = column
            self.customer_sort_reverse = column not in {"customer_id", "profile", "risk_category", "theft_type"}
        self.refresh_customer_table()

    def on_customer_selected(self, _event=None):
        selection = self.customer_tree.selection()
        if not selection or self.current_df is None:
            return
        values = self.customer_tree.item(selection[0], "values")
        if not values:
            return
        customer_id = str(values[0])
        matches = self.current_df[self.current_df["customer_id"].astype(str) == customer_id]
        if not matches.empty:
            self.update_customer_detail_panel(matches.iloc[0])

    def update_customer_detail_panel(self, row):
        risk = safe_text(row.get("risk_category", "-"))
        case = self.ops_store.get_case(row.get("customer_id"))
        overdue = bool(case.get("is_overdue")) if case else False
        action = build_case_recommendation(risk, case.get("status") if case else None, overdue=overdue)
        risk_summary = safe_text(row.get("risk_summary"))
        self.detail_customer_var.set(safe_text(row.get("customer_id", "-")))
        self.detail_profile_var.set(safe_text(row.get("profile", "-")))
        self.detail_risk_var.set(f"{risk} | score {row.get('risk_score', 0)}")
        self.detail_probability_var.set(fmt_percent(row.get("theft_probability", 0), decimals=1))
        self.detail_loss_var.set(fmt_currency(row.get("est_monthly_loss", 0)))
        self.detail_theft_type_var.set(safe_text(row.get("theft_type", "-")))
        self.detail_recommendation_var.set(f"{action}\n\nWhy flagged: {risk_summary}" if risk_summary != "-" else action)

    def jump_to_selected_top_customer(self, _event=None):
        selection = self.top_tree.selection()
        if selection:
            values = self.top_tree.item(selection[0], "values")
            if values:
                self.jump_to_ops_customer(str(values[0]))

    def jump_to_selected_customer_case(self, _event=None):
        selection = self.customer_tree.selection()
        if selection:
            values = self.customer_tree.item(selection[0], "values")
            if values:
                self.jump_to_ops_customer(str(values[0]))

    def jump_to_ops_customer(self, customer_id):
        self.ops_search_var.set(str(customer_id))
        self.show_tab("operations")
        self.refresh_ops_view(preferred_customer_id=str(customer_id))
        if self.selected_case_id != str(customer_id):
            messagebox.showinfo("MASS-AI", "No Ops Center case was found for that customer. Cases are auto-created for fraud probability at or above 50%.")

    def build_ops_filters(self):
        return {
            "search": self.ops_search_var.get(),
            "status": self.ops_status_filter_var.get(),
            "risk_band": self.ops_risk_filter_var.get(),
            "priority": self.ops_priority_filter_var.get(),
            "overdue_only": self.ops_overdue_only_var.get(),
        }

    def refresh_ops_view(self, preserve_selection=True, preferred_customer_id=None):
        selection_target = preferred_customer_id if preferred_customer_id is not None else (self.selected_case_id if preserve_selection else None)
        self.ops_case_df = self.ops_store.list_cases(self.build_ops_filters())
        metrics = self.ops_store.case_metrics()
        for label, value in zip(self.ops_kpi_labels, [str(metrics.get("open_cases", 0)), str(metrics.get("urgent", 0)), str(metrics.get("overdue", 0)), str(metrics.get("resolved_this_week", 0))]):
            label.configure(text=value)

        self.ops_queue_count_var.set(f"{len(self.ops_case_df)} cases shown")
        self.populate_tree(
            self.ops_tree,
            self.ops_case_df.head(500),
            ["customer_id", "profile", "risk_band", "fraud_probability", "priority", "status", "follow_up_at", "est_monthly_loss"],
            risk_column="risk_band",
            extra_tag_func=lambda row: "ResolvedCase" if safe_text(row.get("status")) == "Resolved" else ("OverdueCase" if bool(row.get("is_overdue")) else None),
        )
        self.session_info.configure(text=f"Open Ops cases: {metrics.get('open_cases', 0)}\nUrgent cases: {metrics.get('urgent', 0)}\nOverdue follow-ups: {metrics.get('overdue', 0)}\nResolved this week: {metrics.get('resolved_this_week', 0)}")

        children = self.ops_tree.get_children()
        if selection_target and selection_target in children:
            self.ops_tree.selection_set(selection_target)
            self.ops_tree.focus(selection_target)
            self.ops_tree.see(selection_target)
            self.load_case_detail(selection_target)
        elif children:
            first = children[0]
            self.ops_tree.selection_set(first)
            self.ops_tree.focus(first)
            self.load_case_detail(first)
        else:
            self.clear_case_detail()

    def on_ops_case_selected(self, _event=None):
        selection = self.ops_tree.selection()
        if selection:
            self.load_case_detail(selection[0])

    def clear_case_detail(self):
        self.selected_case_id = None
        self.ops_case_title_var.set("No case selected")
        self.ops_case_subtitle_var.set("Pick a case")
        for variable in [self.ops_customer_var, self.ops_profile_var, self.ops_risk_var, self.ops_probability_var, self.ops_exposure_var, self.ops_pattern_var, self.ops_last_analysis_var, self.ops_follow_up_display_var, self.ops_overdue_var, self.ops_case_status_display_var, self.ops_case_priority_display_var]:
            variable.set("-")
        self.ops_recommendation_var.set("Case recommendations will appear here.")
        self.case_status_var.set(CASE_STATUSES[0])
        self.case_priority_var.set(CASE_PRIORITIES[-1])
        self.case_follow_up_var.set("")
        self.case_resolution_var.set("")
        self.case_note_text.delete("1.0", tk.END)
        self.case_timeline_text.configure(state="normal")
        self.case_timeline_text.delete("1.0", tk.END)
        self.case_timeline_text.configure(state="disabled")

    def load_case_detail(self, customer_id):
        case = self.ops_store.get_case(customer_id)
        if not case:
            self.clear_case_detail()
            return
        notes = self.ops_store.list_case_notes(customer_id)
        history = self.ops_store.list_case_history(customer_id)
        self.selected_case_id = str(case["customer_id"])
        overdue = bool(case.get("is_overdue"))
        self.ops_case_title_var.set(safe_text(case.get("case_title")))
        self.ops_case_subtitle_var.set(f"Latest source: {safe_text(case.get('source_name'))} | Last seen run ID: {safe_text(case.get('last_seen_run_id'))}")
        self.ops_customer_var.set(safe_text(case.get("customer_id")))
        self.ops_profile_var.set(safe_text(case.get("profile")))
        self.ops_risk_var.set(safe_text(case.get("risk_band")))
        self.ops_probability_var.set(fmt_percent(case.get("fraud_probability", 0), decimals=2))
        self.ops_exposure_var.set(fmt_currency(case.get("est_monthly_loss", 0)))
        self.ops_pattern_var.set(safe_text(case.get("fraud_pattern")))
        self.ops_last_analysis_var.set(format_local_datetime(case.get("last_analysis_at")))
        follow_up_text = format_local_datetime(case.get("follow_up_at"))
        self.ops_follow_up_display_var.set(f"Overdue | {follow_up_text}" if overdue else follow_up_text)
        self.ops_overdue_var.set("Overdue" if overdue else "On track")
        self.ops_case_status_display_var.set(safe_text(case.get("status")))
        self.ops_case_priority_display_var.set(safe_text(case.get("priority")))
        current_summary = "-"
        if self.current_df is not None and not self.current_df.empty:
            matches = self.current_df[self.current_df["customer_id"].astype(str) == str(case["customer_id"])]
            if not matches.empty:
                current_summary = safe_text(matches.iloc[0].get("risk_summary"))
        base_recommendation = safe_text(case.get("recommended_action")) or build_case_recommendation(case.get("risk_band"), case.get("status"), overdue=overdue)
        self.ops_recommendation_var.set(f"{base_recommendation}\n\nWhy flagged: {current_summary}" if current_summary != "-" else base_recommendation)
        self.case_status_var.set(safe_text(case.get("status")))
        self.case_priority_var.set(safe_text(case.get("priority")))
        self.case_follow_up_var.set(parse_datetime(case.get("follow_up_at")).strftime("%Y-%m-%d %H:%M") if parse_datetime(case.get("follow_up_at")) else "")
        self.case_resolution_var.set(safe_text(case.get("resolution_reason")) if case.get("resolution_reason") else "")
        self.case_note_text.delete("1.0", tk.END)

        self.case_timeline_text.configure(state="normal")
        self.case_timeline_text.delete("1.0", tk.END)
        if history:
            self.case_timeline_text.insert(tk.END, "History\n")
            self.case_timeline_text.insert(tk.END, "-------\n")
            for item in history:
                self.case_timeline_text.insert(tk.END, f"{format_local_datetime(item.get('created_at'))} | {safe_text(item.get('event_type'))}\n{safe_text(item.get('event_summary'))}\n\n")
        if notes:
            self.case_timeline_text.insert(tk.END, "Notes\n")
            self.case_timeline_text.insert(tk.END, "-----\n")
            for note in notes:
                self.case_timeline_text.insert(tk.END, f"{format_local_datetime(note.get('created_at'))}\n{safe_text(note.get('note_text'))}\n\n")
        elif not history:
            self.case_timeline_text.insert(tk.END, "No notes yet.")
        self.case_timeline_text.configure(state="disabled")

    def save_case_changes(self):
        if not self.selected_case_id:
            messagebox.showwarning("MASS-AI", "Select a case before saving changes.")
            return
        raw_follow_up = self.case_follow_up_var.get()
        follow_up_at = "" if not str(raw_follow_up).strip() else normalize_follow_up_input(raw_follow_up)
        self.ops_store.update_case(self.selected_case_id, status=self.case_status_var.get(), priority=self.case_priority_var.get(), follow_up_at=follow_up_at, resolution_reason=self.case_resolution_var.get())
        self.engine.log(f"Case updated: {self.selected_case_id}")
        self.update_log()
        self.refresh_ops_view(preferred_customer_id=self.selected_case_id)

    def add_case_note(self):
        if not self.selected_case_id:
            messagebox.showwarning("MASS-AI", "Select a case before adding a note.")
            return
        note = self.case_note_text.get("1.0", tk.END).strip()
        if not note:
            messagebox.showwarning("MASS-AI", "Write a note before saving it.")
            return
        self.ops_store.add_case_note(self.selected_case_id, note)
        self.engine.log(f"Case note added: {self.selected_case_id}")
        self.update_log()
        self.load_case_detail(self.selected_case_id)

    def render_chart_tab(self):
        if self.current_df is None:
            self.show_chart_placeholder("Run an analysis to generate charts.")
            return
        if self.current_canvas is not None:
            self.current_canvas.get_tk_widget().destroy()
            self.current_canvas = None
        if self.current_figure is not None:
            plt.close(self.current_figure)
            self.current_figure = None
        for widget in self.chart_host.winfo_children():
            widget.destroy()

        try:
            df = self.current_df
            palette = self.chart_palette()
            fig, axes = plt.subplots(
                4,
                1,
                figsize=(10.4, 12.6),
                gridspec_kw={"height_ratios": [1.0, 1.0, 1.0, 1.25]},
            )
            fig.patch.set_facecolor(palette["face"])
            performance_ax, risk_ax, curve_ax, top_ax = axes

            def style_axis(ax, *, grid_axis="x"):
                ax.set_facecolor(palette["panel"])
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_color(palette["spine"])
                ax.spines["bottom"].set_color(palette["spine"])
                ax.tick_params(colors=palette["muted"], labelsize=9)
                ax.title.set_color(palette["text"])
                ax.xaxis.label.set_color(palette["muted"])
                ax.yaxis.label.set_color(palette["muted"])
                ax.grid(axis=grid_axis, color=palette["grid"], linewidth=0.8)
                ax.set_axisbelow(True)

            def empty_axis(ax, title, message):
                ax.text(0.5, 0.5, message, ha="center", va="center", color=palette["muted"], transform=ax.transAxes)
                ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
                style_axis(ax)

            if self.engine.results:
                ordered = sorted(self.engine.results, key=lambda item: self.engine.results[item].get("auc", 0), reverse=True)
                aucs = [self.engine.results[name].get("auc", 0) for name in ordered]
                f1s = [self.engine.results[name].get("f1", 0) for name in ordered]
                rows = np.arange(len(ordered))
                for row, auc, f1 in zip(rows, aucs, f1s):
                    performance_ax.hlines(row, min(auc, f1), max(auc, f1), color=palette["grid"], linewidth=6, zorder=1)
                performance_ax.scatter(aucs, rows, s=88, color=palette["accent"][2], label="ROC-AUC", zorder=3)
                performance_ax.scatter(f1s, rows, s=88, color=palette["accent"][3], label="F1", zorder=3)
                for row, auc, f1 in zip(rows, aucs, f1s):
                    label_x = min(max(auc, f1) + 0.02, 0.985)
                    label_ha = "left"
                    if label_x >= 0.96:
                        label_x = max(max(auc, f1) - 0.015, 0.84)
                        label_ha = "right"
                    performance_ax.text(
                        label_x,
                        row,
                        f"{auc:.2f} / {f1:.2f}",
                        va="center",
                        ha=label_ha,
                        color=palette["muted"],
                        fontsize=9,
                    )
                performance_ax.set_yticks(rows)
                performance_ax.set_yticklabels(ordered)
                performance_ax.invert_yaxis()
                performance_ax.set_xlim(0, 1.04)
                performance_ax.set_title("Model quality", loc="left", fontsize=13, fontweight="bold")
                performance_ax.set_xlabel("Score")
                performance_ax.legend(frameon=False, labelcolor=palette["text"], loc="lower right", ncol=2)
                style_axis(performance_ax, grid_axis="x")
            else:
                empty_axis(performance_ax, "Model quality", "No model result")

            counts = df["risk_category"].astype(str).value_counts().reindex(DOMAIN_RISK_LABELS, fill_value=0)
            active_counts = counts[counts > 0]
            if not active_counts.empty:
                risk_exposure = (
                    df.groupby(df["risk_category"].astype(str))["est_monthly_loss"]
                    .sum()
                    .reindex(active_counts.index, fill_value=0)
                )
                risk_color_map = {
                    "Low": palette["accent"][3],
                    "Moderate": palette["accent"][1],
                    "High": palette["accent"][1],
                    "Critical": palette["accent"][0],
                    "Urgent": palette["accent"][0],
                }
                risk_colors = [risk_color_map.get(label, palette["accent"][2]) for label in active_counts.index]
                risk_ax.barh(active_counts.index, active_counts.values, color=risk_colors, edgecolor=palette["spine"], linewidth=0.6)
                total_customers = max(len(df), 1)
                max_count = float(active_counts.max()) if len(active_counts) else 1.0
                for idx, (label, value) in enumerate(active_counts.items()):
                    exposure_text = fmt_currency(risk_exposure.get(label, 0))
                    risk_ax.text(
                        value + max(max_count * 0.03, 1.0),
                        idx,
                        f"{int(value)} | {value / total_customers:.0%} | {exposure_text}",
                        va="center",
                        ha="left",
                        color=palette["muted"],
                        fontsize=9,
                    )
                risk_ax.set_xlim(0, max_count * 1.72 + 1)
                risk_ax.set_title("Risk mix", loc="left", fontsize=13, fontweight="bold")
                risk_ax.set_xlabel("Customers")
                style_axis(risk_ax, grid_axis="x")
            else:
                empty_axis(risk_ax, "Risk mix", "No risk distribution yet")

            probabilities = np.sort(df["theft_probability"].fillna(0).astype(float).to_numpy())[::-1]
            if len(probabilities):
                curve_x = np.arange(1, len(probabilities) + 1)
                curve_ax.fill_between(curve_x, probabilities, color=palette["accent"][2], alpha=0.15)
                curve_ax.plot(curve_x, probabilities, color=palette["accent"][2], linewidth=2.4)
                curve_ax.axhline(y=0.5, color=palette["accent"][0], linestyle="--", linewidth=1.3)
                high_risk_count = int((probabilities >= 0.5).sum())
                curve_ax.text(
                    0.99,
                    0.92,
                    f"{high_risk_count} accounts above 0.50",
                    transform=curve_ax.transAxes,
                    ha="right",
                    va="top",
                    color=palette["muted"],
                    fontsize=9,
                )
                curve_ax.set_title("Risk curve", loc="left", fontsize=13, fontweight="bold")
                curve_ax.set_xlabel("Customers ranked by fraud probability")
                curve_ax.set_ylabel("Fraud probability")
                curve_ax.set_xlim(1, len(curve_x))
                curve_ax.set_ylim(0, 1.02)
                style_axis(curve_ax, grid_axis="y")
            else:
                empty_axis(curve_ax, "Risk curve", "No score distribution yet")

            top_source = df.sort_values("priority_index", ascending=False).head(12)
            if not top_source.empty:
                watchlist = top_source.iloc[::-1].copy()
                profile_short = {
                    "residential": "res",
                    "commercial": "com",
                    "industrial": "ind",
                    "mixed_use": "mix",
                }
                watch_labels = watchlist.apply(
                    lambda row: f"ID-{safe_text(row.get('customer_id'))} | {profile_short.get(safe_text(row.get('profile')), safe_text(row.get('profile'))[:3].lower())}",
                    axis=1,
                )
                risk_color_map = {
                    "Low": palette["accent"][3],
                    "Moderate": palette["accent"][1],
                    "High": palette["accent"][1],
                    "Critical": palette["accent"][0],
                    "Urgent": palette["accent"][0],
                }
                watch_colors = watchlist["risk_category"].astype(str).map(risk_color_map).fillna(palette["accent"][2]).tolist()
                bars = top_ax.barh(watch_labels, watchlist["priority_index"], color=watch_colors, edgecolor=palette["spine"], linewidth=0.6)
                max_priority = float(watchlist["priority_index"].max()) if len(watchlist) else 1.0
                for bar, (_, row) in zip(bars, watchlist.iterrows()):
                    label = f"{safe_text(row.get('risk_category'))}  |  {fmt_percent(row.get('theft_probability', 0), decimals=1)}"
                    top_ax.text(
                        bar.get_width() + max(max_priority * 0.02, 0.8),
                        bar.get_y() + bar.get_height() / 2,
                        label,
                        va="center",
                        ha="left",
                        color=palette["muted"],
                        fontsize=9,
                    )
                top_ax.set_xlim(0, max_priority * 1.28 + 1)
                top_ax.set_title("Priority watchlist", loc="left", fontsize=13, fontweight="bold")
                top_ax.set_xlabel("Priority index")
                style_axis(top_ax, grid_axis="x")
            else:
                empty_axis(top_ax, "Priority watchlist", "No scored customers yet")

            insight_lines = []
            if getattr(self.engine, "last_explainability_summary", None):
                insight_lines.append(self.engine.last_explainability_summary)
            if "transformer_id" in df.columns and "transformer_loss_pct" in df.columns:
                hotspot = (
                    df.groupby("transformer_id")["transformer_loss_pct"]
                    .mean()
                    .sort_values(ascending=False)
                    .head(1)
                )
                if not hotspot.empty:
                    insight_lines.append(f"Highest transformer loss sits on {hotspot.index[0]} at {float(hotspot.iloc[0]):.1f}%.")
            if "peer_consumption_ratio" in df.columns:
                peer_low = int((df["peer_consumption_ratio"].fillna(1).astype(float) < 0.7).sum())
                insight_lines.append(f"{peer_low} customers are materially below their peer baseline.")

            if insight_lines:
                fig.text(
                    0.015,
                    0.012,
                    "Brief: " + " ".join(insight_lines[:3]),
                    ha="left",
                    va="bottom",
                    color=palette["muted"],
                    fontsize=9,
                )

            fig.subplots_adjust(left=0.24, right=0.94, top=0.975, bottom=0.06, hspace=0.42)
            self.current_figure = fig
            self.current_canvas = FigureCanvasTkAgg(fig, master=self.chart_host)
            self.current_canvas.draw()
            self.current_canvas.get_tk_widget().configure(bg=self.theme.card, highlightthickness=0, bd=0)
            self.current_canvas.get_tk_widget().pack(fill="both", expand=True)
        except Exception as exc:
            self.engine.log(f"Chart render error: {exc}")
            self.update_log()
            self.show_chart_placeholder(f"Charts could not be rendered.\n{exc}")

    def show_chart_placeholder(self, message):
        if hasattr(self, "chart_host"):
            for widget in self.chart_host.winfo_children():
                widget.destroy()
            placeholder = tk.Label(
                self.chart_host,
                text=message,
                bg=self.theme.card,
                fg=self.theme.muted,
                font=self.theme.body_font,
                justify="center",
            )
            placeholder.pack(fill="both", expand=True, padx=24, pady=24)

    def build_fallback_overview(self):
        cases = self.ops_store.list_cases({"status": "All statuses"})
        if cases.empty:
            return None
        latest_analysis = cases["last_analysis_at"].dropna().astype(str)
        last_seen = latest_analysis.max() if not latest_analysis.empty else None
        return {
            "best_model": "Ops Center snapshot",
            "customer_count": int(len(cases)),
            "high_risk_count": int((cases["fraud_probability"] >= 0.70).sum()),
            "critical_count": int(cases["risk_band"].astype(str).isin(DOMAIN_CRITICAL_RISK_LABELS).sum()),
            "average_probability": float(cases["fraud_probability"].fillna(0).mean()),
            "total_loss": float(cases["est_monthly_loss"].fillna(0).sum()),
            "top_customer": safe_text(cases.iloc[0]["customer_id"]),
            "data_source": "Persistent Ops Center",
            "last_run_at": format_local_datetime(last_seen),
            "last_run_at_full": last_seen,
            "preset_name": None,
            "preset_summary": None,
            "explainability_summary": "Explainability is available when a fresh scoring run is loaded in the desktop app.",
        }

    def export_csv(self):
        if self.engine.df_scored is None:
            messagebox.showwarning("MASS-AI", "Run an analysis before exporting results.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")], initialfile=f"mass_ai_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")
        if not path:
            return
        self.engine.df_scored.to_csv(path, index=False)
        self.engine.log(f"Results exported: {path}")
        self.update_log()
        messagebox.showinfo("MASS-AI", f"Results exported:\n{path}")

    def export_charts(self):
        if self.current_figure is None:
            messagebox.showwarning("MASS-AI", "Run an analysis before exporting charts.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")], initialfile=f"mass_ai_charts_{datetime.now().strftime('%Y%m%d_%H%M')}.png")
        if not path:
            return
        self.current_figure.savefig(path, dpi=160, bbox_inches="tight", facecolor=self.theme.card)
        self.engine.log(f"Charts exported: {path}")
        self.update_log()
        messagebox.showinfo("MASS-AI", f"Charts exported:\n{path}")

    def export_report(self):
        overview = self.engine.build_overview() or self.build_fallback_overview()
        if overview is None:
            messagebox.showwarning("MASS-AI", "Run an analysis or load existing Ops cases before exporting the executive brief.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".html",
            filetypes=[("HTML", "*.html"), ("Markdown", "*.md"), ("Text", "*.txt")],
            initialfile=f"mass_ai_executive_brief_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
        )
        if not path:
            return

        ops_metrics = self.ops_store.case_metrics()
        explain_columns = [
            "customer_id",
            "profile",
            "theft_type",
            "risk_category",
            "theft_probability",
            "est_monthly_loss",
            "risk_reason_1",
            "risk_reason_2",
            "risk_reason_3",
            "risk_drivers",
            "risk_summary",
        ]
        explain_lookup = None
        if self.current_df is not None and not self.current_df.empty:
            explain_lookup = self.current_df[[col for col in explain_columns if col in self.current_df.columns]].copy()
            explain_lookup["customer_id"] = explain_lookup["customer_id"].astype(str)
        if self.current_df is not None and not self.current_df.empty:
            report_rows = self.ops_store.list_cases({"status": "All open"}).head(10)
            if report_rows.empty:
                report_rows = self.current_df.head(10).rename(columns={"risk_category": "risk_band", "theft_type": "fraud_pattern", "theft_probability": "fraud_probability"})
                report_rows["priority"] = report_rows["risk_band"].astype(str).apply(priority_for_risk)
                report_rows["status"] = "New"
            elif explain_lookup is not None:
                report_rows["customer_id"] = report_rows["customer_id"].astype(str)
                report_rows = report_rows.merge(explain_lookup, on="customer_id", how="left", suffixes=("", "_current"))
                for base_col, alt_col in [("profile", "profile_current"), ("fraud_pattern", "theft_type"), ("risk_band", "risk_category"), ("fraud_probability", "theft_probability"), ("est_monthly_loss", "est_monthly_loss_current")]:
                    if alt_col in report_rows.columns:
                        report_rows[base_col] = report_rows[base_col].where(report_rows[base_col].notna(), report_rows[alt_col])
        else:
            report_rows = self.ops_store.list_cases({"status": "All open"}).head(10)

        selected_case = self.ops_store.get_case(self.selected_case_id) if self.selected_case_id else None
        if selected_case is None and not report_rows.empty:
            selected_case = self.ops_store.get_case(report_rows.iloc[0]["customer_id"])
        if selected_case is not None and explain_lookup is not None:
            match = explain_lookup[explain_lookup["customer_id"] == str(selected_case["customer_id"])]
            if not match.empty:
                enriched = match.iloc[0].to_dict()
                selected_case = {**selected_case, **enriched}
        selected_notes = self.ops_store.list_case_notes(selected_case["customer_id"]) if selected_case else []

        content = build_executive_brief_html(overview, ops_metrics, report_rows, selected_case, selected_notes) if path.lower().endswith(".html") else build_executive_brief_text(overview, ops_metrics, report_rows, selected_case, selected_notes)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)

        self.engine.log(f"Executive brief exported: {path}")
        self.update_log()
        messagebox.showinfo("MASS-AI", f"Executive brief exported:\n{path}")

    def export_support_bundle(self):
        overview = self.engine.build_overview() or self.build_fallback_overview() or {}
        ops_metrics = self.ops_store.case_metrics()
        selected_case = self.ops_store.get_case(self.selected_case_id) if self.selected_case_id else None
        selected_notes = self.ops_store.list_case_notes(selected_case["customer_id"]) if selected_case else []
        case_history = self.ops_store.list_case_history(selected_case["customer_id"]) if selected_case else []

        path = filedialog.asksaveasfilename(
            defaultextension=".zip",
            filetypes=[("ZIP bundle", "*.zip")],
            initialfile=support_bundle_name(),
        )
        if not path:
            return

        try:
            bundle_path = create_support_bundle(
                path,
                theme_name=self.theme_mode_var.get(),
                overview=overview,
                ops_metrics=ops_metrics,
                selected_case=selected_case,
                selected_notes=selected_notes,
                case_history=case_history,
                log_lines=self.engine.log_lines,
                current_df=self.current_df,
                extra_sections={
                    "app_version": APP_VERSION,
                    "build_channel": BUILD_CHANNEL,
                    "feature_columns": list(self.engine.feature_cols),
                    "schema_summary": self.engine.schema_summary() if hasattr(self.engine, "schema_summary") else {},
                },
            )
        except Exception as exc:
            messagebox.showerror("MASS-AI", f"Support bundle export failed:\n{support_failure_message(exc)}")
            return

        self.engine.log(f"Support bundle exported: {bundle_path}")
        self.update_log()
        messagebox.showinfo("MASS-AI", f"Support bundle exported:\n{bundle_path}")

    def show_about_dialog(self):
        messagebox.showinfo(
            "About MASS-AI Desktop",
            f"{APP_NAME}\n{version_label()} | {BUILD_CHANNEL}\n\nOps workspace for triage, case handling, reporting, and support export.",
        )

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    MassAIApp().run()
