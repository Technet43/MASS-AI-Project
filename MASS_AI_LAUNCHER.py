import os
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR / "project"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ui_kit import (
    DEFAULT_THEME_NAME,
    THEME_CHOICES,
    GlassButton,
    GlassCard,
    ScrollablePanel,
    ThemePreviewPicker,
    apply_glass_ttk_theme,
    bind_wrap_to_width,
    build_glass_theme,
    normalize_theme_name,
)
from app_prefs import load_theme_preference, save_theme_preference
from app_metadata import APP_NAME, BUILD_CHANNEL, support_bundle_name, version_label
from support_bundle import create_support_bundle, support_failure_message


class LauncherApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(f"{APP_NAME} Launcher {version_label()}")
        self.root.geometry("1080x720")
        self.root.minsize(900, 620)
        self.style = ttk.Style()
        initial_theme_name = normalize_theme_name(os.environ.get("MASS_AI_THEME") or load_theme_preference() or DEFAULT_THEME_NAME)
        self.theme_name_var = tk.StringVar(value=initial_theme_name)
        self.theme = build_glass_theme(self.theme_name_var.get())
        self.apply_theme(self.theme_name_var.get())

    def command_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["MASS_AI_THEME"] = normalize_theme_name(self.theme_name_var.get())
        return env

    def run_command(self, cmd, cwd=None):
        try:
            if sys.platform.startswith("win") and isinstance(cmd, str):
                subprocess.Popen(cmd, cwd=cwd, shell=True, env=self.command_env())
            else:
                subprocess.Popen(cmd, cwd=cwd, env=self.command_env())
        except Exception as exc:
            messagebox.showerror("MASS-AI", f"The command could not be launched:\n{exc}")

    def open_desktop(self):
        target = PROJECT_DIR / "mass_ai_desktop.py"
        if not target.exists():
            messagebox.showerror("MASS-AI", "mass_ai_desktop.py was not found.")
            return
        self.run_command([sys.executable, str(target)], cwd=PROJECT_DIR)

    def open_dashboard(self):
        target = PROJECT_DIR / "dashboard" / "app.py"
        if not target.exists():
            messagebox.showerror("MASS-AI", "dashboard/app.py was not found.")
            return
        self.run_command([sys.executable, "-m", "streamlit", "run", str(target)], cwd=PROJECT_DIR)

    def install_requirements(self):
        req = PROJECT_DIR / "requirements.txt"
        if not req.exists():
            messagebox.showerror("MASS-AI", "requirements.txt was not found.")
            return
        self.run_command([sys.executable, "-m", "pip", "install", "-r", str(req)], cwd=PROJECT_DIR)

    def run_smoke_tests(self):
        target = BASE_DIR / "RUN_SMOKE_TESTS.bat"
        if sys.platform.startswith("win") and target.exists():
            self.run_command(f'"{target}"', cwd=BASE_DIR)
            return
        self.run_command(
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                str(PROJECT_DIR / "tests"),
                "-t",
                str(PROJECT_DIR),
                "-p",
                "test_*.py",
                "-v",
            ],
            cwd=BASE_DIR,
        )

    def build_windows_executable(self):
        target = BASE_DIR / "BUILD_DESKTOP_EXE.bat"
        if sys.platform.startswith("win") and target.exists():
            self.run_command(f'"{target}"', cwd=BASE_DIR)
            return
        messagebox.showinfo("MASS-AI", "Windows executable packaging is available on Windows via BUILD_DESKTOP_EXE.bat.")

    def export_support_bundle(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".zip",
            filetypes=[("ZIP bundle", "*.zip")],
            initialfile=support_bundle_name("mass_ai_launcher_support"),
        )
        if not path:
            return
        try:
            bundle_path = create_support_bundle(
                path,
                theme_name=self.theme_name_var.get(),
                overview={"data_source": "Launcher", "best_model": "N/A"},
                ops_metrics={},
                selected_case=None,
                selected_notes=[],
                case_history=[],
                log_lines=[
                    "Launcher support bundle generated.",
                    f"Theme: {self.theme_name_var.get()}",
                    f"Workspace: {BASE_DIR}",
                ],
                current_df=None,
                extra_sections={"launcher_only": True},
            )
        except Exception as exc:
            messagebox.showerror("MASS-AI", f"Support bundle export failed:\n{support_failure_message(exc)}")
            return
        messagebox.showinfo("MASS-AI", f"Support bundle exported:\n{bundle_path}")

    def open_folder(self, folder_name):
        path = BASE_DIR / folder_name
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:
            messagebox.showerror("MASS-AI", f"The folder could not be opened:\n{exc}")

    def make_section_title(self, parent, title, description=""):
        title_label = tk.Label(parent, text=title, bg=parent["bg"], fg=self.theme.text, font=self.theme.section_font)
        title_label.grid(row=0, column=0, sticky="w")
        if not description:
            return None
        description_label = tk.Label(
            parent,
            text=description,
            bg=parent["bg"],
            fg=self.theme.muted,
            font=self.theme.body_font,
            justify="left",
        )
        description_label.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        bind_wrap_to_width(description_label, extra_padding=12, min_wrap=240)
        return description_label

    def on_theme_selected(self, selected=None, _event=None):
        chosen_value = selected if isinstance(selected, str) else self.theme_name_var.get()
        selected = normalize_theme_name(chosen_value)
        if selected != self.theme_name_var.get():
            self.theme_name_var.set(selected)
        self.apply_theme(selected)

    def apply_theme(self, theme_name: str):
        normalized = normalize_theme_name(theme_name)
        self.theme_name_var.set(normalized)
        os.environ["MASS_AI_THEME"] = normalized
        save_theme_preference(normalized)
        self.theme = build_glass_theme(normalized)
        self.root.configure(bg=self.theme.bg)
        apply_glass_ttk_theme(self.style, self.theme)
        for child in self.root.winfo_children():
            child.destroy()
        self.build_ui()

    def build_ui(self):
        shell = ScrollablePanel(self.root, self.theme, bg=self.theme.bg)
        shell.pack(fill="both", expand=True)
        shell.body.grid_columnconfigure(0, weight=1)
        shell.body.grid_rowconfigure(2, weight=1)

        chrome = tk.Frame(shell.body, bg=self.theme.bg)
        chrome.grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 8))
        for color in (self.theme.red, self.theme.orange, self.theme.green):
            tk.Label(chrome, bg=color, width=2, height=1).pack(side="left", padx=5, pady=5)

        hero = GlassCard(shell.body, self.theme, padding=(18, 16))
        hero.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 10))
        hero.body.grid_columnconfigure(0, weight=3)
        hero.body.grid_columnconfigure(1, weight=2)

        hero_left = tk.Frame(hero.body, bg=hero.body["bg"])
        hero_left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        tk.Label(hero_left, text="MASS-AI", bg=hero_left["bg"], fg=self.theme.text, font=self.theme.hero_font).pack(anchor="w")
        subtitle = tk.Label(
            hero_left,
            text=f"{version_label()} | {BUILD_CHANNEL}",
            bg=hero_left["bg"],
            fg=self.theme.muted,
            font=self.theme.body_font,
            justify="left",
        )
        subtitle.pack(anchor="w", pady=(4, 0))
        bind_wrap_to_width(subtitle, extra_padding=20, min_wrap=220)

        hero_right = tk.Frame(hero.body, bg=hero.body["bg"])
        hero_right.grid(row=0, column=1, sticky="ne")
        status_chip = GlassCard(hero_right, self.theme, padding=(12, 10), tint=self.theme.frosted)
        status_chip.pack(anchor="e", fill="x")
        tk.Label(status_chip.body, text="Status", bg=status_chip.body["bg"], fg=self.theme.blue, font=self.theme.body_bold_font).pack(anchor="e")
        tk.Label(status_chip.body, text=f"Ready | {version_label()}", bg=status_chip.body["bg"], fg=self.theme.text, font=self.theme.body_bold_font).pack(anchor="e", pady=(2, 0))

        appearance_row = tk.Frame(hero_right, bg=hero_right["bg"])
        appearance_row.pack(anchor="e", fill="x", pady=(10, 0))
        tk.Label(appearance_row, text="Appearance", bg=appearance_row["bg"], fg=self.theme.soft, font=self.theme.small_font).pack(anchor="e")
        theme_selector = ThemePreviewPicker(
            appearance_row,
            self.theme,
            [(choice, "Glass" if choice == "Liquid Glass" else choice) for choice in THEME_CHOICES],
            self.on_theme_selected,
        )
        theme_selector.pack(fill="x", pady=(4, 0))
        theme_selector.set_active(self.theme_name_var.get())

        content = tk.Frame(shell.body, bg=self.theme.bg)
        content.grid(row=2, column=0, sticky="nsew", padx=14, pady=(0, 14))
        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=1)

        left_column = GlassCard(content, self.theme, padding=(18, 16))
        left_column.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left_column.body.grid_columnconfigure(0, weight=1)
        self.make_section_title(left_column.body, "Actions")

        workflow_buttons = tk.Frame(left_column.body, bg=left_column.body["bg"])
        workflow_buttons.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        workflow_buttons.grid_columnconfigure(0, weight=1)
        action_buttons = [
            GlassButton(workflow_buttons, self.theme, text="Open desktop", command=self.open_desktop, fill=self.theme.blue, anchor="center"),
            GlassButton(workflow_buttons, self.theme, text="Open dashboard", command=self.open_dashboard, fill=self.theme.card_alt, ink=self.theme.text, anchor="center"),
            GlassButton(workflow_buttons, self.theme, text="Install deps", command=self.install_requirements, fill=self.theme.card_alt, ink=self.theme.text, anchor="center"),
            GlassButton(workflow_buttons, self.theme, text="Smoke tests", command=self.run_smoke_tests, fill=self.theme.card_alt, ink=self.theme.text, anchor="center"),
            GlassButton(workflow_buttons, self.theme, text="Build EXE", command=self.build_windows_executable, fill=self.theme.frosted, ink=self.theme.text, anchor="center"),
            GlassButton(workflow_buttons, self.theme, text="Support bundle", command=self.export_support_bundle, fill=self.theme.card_alt, ink=self.theme.text, anchor="center"),
        ]
        for row, button in enumerate(action_buttons):
            button.grid(row=row, column=0, sticky="ew", pady=(0 if row == 0 else 8, 0))

        tips = GlassCard(left_column.body, self.theme, padding=(12, 12), tint=self.theme.frosted)
        tips.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        tips.body.grid_columnconfigure(0, weight=1)
        tk.Label(tips.body, text="Order", bg=tips.body["bg"], fg=self.theme.text, font=self.theme.body_bold_font).grid(row=0, column=0, sticky="w")
        tips_text = tk.Label(
            tips.body,
            text="1. Install deps\n2. Open desktop\n3. Run tests\n4. Build EXE",
            bg=tips.body["bg"],
            fg=self.theme.muted,
            font=self.theme.body_font,
            justify="left",
        )
        tips_text.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        bind_wrap_to_width(tips_text, extra_padding=12, min_wrap=240)

        right_column = GlassCard(content, self.theme, padding=(18, 16))
        right_column.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        right_column.body.grid_columnconfigure(0, weight=1)
        self.make_section_title(right_column.body, "Assets")

        asset_buttons = tk.Frame(right_column.body, bg=right_column.body["bg"])
        asset_buttons.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        asset_buttons.grid_columnconfigure(0, weight=1)
        folder_buttons = [
            GlassButton(asset_buttons, self.theme, text="Project folder", command=lambda: self.open_folder("project"), fill=self.theme.card_alt, ink=self.theme.text, anchor="center"),
            GlassButton(asset_buttons, self.theme, text="Open docs", command=lambda: self.open_folder("docs"), fill=self.theme.card_alt, ink=self.theme.text, anchor="center"),
            GlassButton(asset_buttons, self.theme, text="Open images", command=lambda: self.open_folder("images"), fill=self.theme.card_alt, ink=self.theme.text, anchor="center"),
            GlassButton(asset_buttons, self.theme, text="Business docs", command=lambda: self.open_folder("business_docs"), fill=self.theme.card_alt, ink=self.theme.text, anchor="center"),
        ]
        for row, button in enumerate(folder_buttons):
            button.grid(row=row, column=0, sticky="ew", pady=(0 if row == 0 else 8, 0))

        asset_note = GlassCard(right_column.body, self.theme, padding=(12, 12), tint=self.theme.card_alt)
        asset_note.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        asset_note.body.grid_columnconfigure(0, weight=1)
        tk.Label(asset_note.body, text="Shortcuts", bg=asset_note.body["bg"], fg=self.theme.text, font=self.theme.body_bold_font).grid(row=0, column=0, sticky="w")
        release_text = tk.Label(
            asset_note.body,
            text=f"{version_label()} | {BUILD_CHANNEL}\nSTART_MASS_AI_DESKTOP.bat launches the app.\nBUILD_DESKTOP_EXE.bat packages the Windows build.\nSupport bundle exports a ZIP for troubleshooting.",
            bg=asset_note.body["bg"],
            fg=self.theme.muted,
            font=self.theme.body_font,
            justify="left",
        )
        release_text.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        bind_wrap_to_width(release_text, extra_padding=12, min_wrap=240)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    LauncherApp().run()
