from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass, replace
from tkinter import ttk


@dataclass(frozen=True)
class GlassTheme:
    bg: str = "#f3f5fb"
    shell: str = "#eef1f7"
    card: str = "#fbfcff"
    card_alt: str = "#f6f8fd"
    frosted: str = "#f8f9fd"
    border: str = "#dbe2ef"
    border_strong: str = "#c6cfdf"
    text: str = "#192235"
    muted: str = "#667089"
    soft: str = "#8891a9"
    line: str = "#e8edf7"
    blue: str = "#2082f3"
    blue_soft: str = "#ebf3ff"
    green: str = "#3bc66a"
    green_soft: str = "#edf9f0"
    orange: str = "#ffab2d"
    orange_soft: str = "#fff6e8"
    red: str = "#ff5b57"
    red_soft: str = "#fff0ef"
    purple: str = "#6d63ff"
    purple_soft: str = "#f1efff"
    dark: str = "#1c1c1f"
    shadow: str = "#edf2fb"
    radius: int = 20

    @property
    def title_font(self) -> tuple[str, int, str]:
        return ("Segoe UI Semibold", 16, "bold")

    @property
    def hero_font(self) -> tuple[str, int, str]:
        return ("Segoe UI Semibold", 22, "bold")

    @property
    def section_font(self) -> tuple[str, int, str]:
        return ("Segoe UI Semibold", 11, "bold")

    @property
    def body_font(self) -> tuple[str, int]:
        return ("Segoe UI", 10)

    @property
    def body_bold_font(self) -> tuple[str, int, str]:
        return ("Segoe UI Semibold", 10, "bold")

    @property
    def small_font(self) -> tuple[str, int]:
        return ("Segoe UI", 8)


DEFAULT_THEME_NAME = "White"
THEME_CHOICES = ("White", "Black", "Liquid Glass")


def normalize_theme_name(name: str | None) -> str:
    key = str(name or DEFAULT_THEME_NAME).strip().lower()
    for choice in THEME_CHOICES:
        if choice.lower() == key:
            return choice
    return DEFAULT_THEME_NAME


def build_glass_theme(name: str | None = None) -> GlassTheme:
    key = normalize_theme_name(name).lower()
    base = GlassTheme()
    if key == "black":
        return replace(
            base,
            bg="#000000",
            shell="#040506",
            card="#0a0c0f",
            card_alt="#101318",
            frosted="#0d1014",
            border="#1c222b",
            border_strong="#323b47",
            text="#f5f7fb",
            muted="#9aa3b2",
            soft="#788293",
            line="#1d222b",
            blue="#4c9dff",
            blue_soft="#0b1624",
            green="#59d98e",
            green_soft="#0c1812",
            orange="#ffb24a",
            orange_soft="#1e140b",
            red="#ff6b66",
            red_soft="#210f11",
            purple="#8d7dff",
            purple_soft="#121022",
            dark="#05070a",
            shadow="#030405",
        )
    if key == "liquid glass":
        return replace(
            base,
            bg="#c8d8ec",
            shell="#bfd2e8",
            card="#dfe8f4",
            card_alt="#d4dfef",
            frosted="#d8e4f2",
            border="#a8bdd6",
            border_strong="#90aac8",
            text="#1a2840",
            muted="#526a88",
            soft="#7a94b0",
            line="#b8cce2",
            blue="#2d7cf6",
            blue_soft="#d0e2f8",
            green="#34b864",
            green_soft="#d4f0de",
            orange="#f0960a",
            orange_soft="#f8ecd0",
            red="#e84040",
            red_soft="#f8d4d4",
            purple="#7c60e8",
            purple_soft="#e0d8f8",
            dark="#2a3e58",
            shadow="#b8cade",
        )
    return base


def _theme_is_dark(theme: GlassTheme) -> bool:
    rgb = theme.bg.lstrip("#")
    if len(rgb) != 6:
        return False
    red = int(rgb[0:2], 16)
    green = int(rgb[2:4], 16)
    blue = int(rgb[4:6], 16)
    brightness = (red * 299 + green * 587 + blue * 114) / 1000
    return brightness < 128


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    value = color.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected a 6-digit hex color, got: {color}")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    red, green, blue = rgb
    return f"#{red:02x}{green:02x}{blue:02x}"


def _mix_color(start: str, end: str, ratio: float) -> str:
    ratio = max(0.0, min(1.0, ratio))
    start_rgb = _hex_to_rgb(start)
    end_rgb = _hex_to_rgb(end)
    mixed = tuple(int(start_value + (end_value - start_value) * ratio) for start_value, end_value in zip(start_rgb, end_rgb))
    return _rgb_to_hex(mixed)


def apply_glass_ttk_theme(style: ttk.Style, theme: GlassTheme) -> None:
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    style.configure(
        "Glass.Treeview",
        background=theme.card,
        fieldbackground=theme.card,
        foreground=theme.text,
        rowheight=32,
        borderwidth=0,
        relief="flat",
        font=theme.body_font,
    )
    style.configure(
        "Glass.Treeview.Heading",
        background=theme.frosted,
        foreground=theme.text,
        relief="flat",
        font=theme.body_bold_font,
        borderwidth=0,
        padding=(8, 10),
    )
    selected_bg = "#1d3047" if _theme_is_dark(theme) else "#dfeeff"
    style.map("Glass.Treeview", background=[("selected", selected_bg)], foreground=[("selected", theme.text)])

    style.configure(
        "Glass.TCombobox",
        padding=8,
        relief="flat",
        borderwidth=1,
        fieldbackground=theme.card_alt,
        background=theme.card_alt,
        foreground=theme.text,
        selectbackground=theme.card_alt,
        selectforeground=theme.text,
        arrowsize=14,
    )
    style.map(
        "Glass.TCombobox",
        bordercolor=[("focus", theme.blue), ("readonly", theme.border_strong)],
        lightcolor=[("focus", theme.blue), ("readonly", theme.border_strong)],
        darkcolor=[("focus", theme.blue), ("readonly", theme.border_strong)],
        fieldbackground=[("readonly", theme.card_alt), ("disabled", theme.card)],
        background=[("readonly", theme.card_alt), ("disabled", theme.card)],
        foreground=[("readonly", theme.text), ("disabled", theme.soft)],
        selectforeground=[("readonly", theme.text)],
        selectbackground=[("readonly", theme.card_alt)],
        arrowcolor=[("focus", theme.blue), ("readonly", theme.soft)],
    )

    style.configure(
        "Glass.Vertical.TScrollbar",
        background=theme.card_alt,
        troughcolor=theme.shell,
        bordercolor=theme.shell,
        arrowcolor=theme.soft,
        relief="flat",
    )
    style.configure(
        "Glass.Horizontal.TScrollbar",
        background=theme.card_alt,
        troughcolor=theme.shell,
        bordercolor=theme.shell,
        arrowcolor=theme.soft,
        relief="flat",
    )
    style.configure(
        "Glass.Horizontal.TProgressbar",
        troughcolor=theme.shadow,
        background=theme.blue,
        borderwidth=0,
        lightcolor=theme.blue,
        darkcolor=theme.blue,
    )
    style.configure("Glass.TCheckbutton", background=theme.frosted, foreground=theme.text, font=theme.body_font)


def apply_glass_window_theme(root: tk.Misc, theme: GlassTheme) -> None:
    root.option_add("*TCombobox*Listbox.background", theme.card_alt)
    root.option_add("*TCombobox*Listbox.foreground", theme.text)
    root.option_add("*TCombobox*Listbox.selectBackground", theme.blue)
    root.option_add("*TCombobox*Listbox.selectForeground", theme.text)


class GlassCard(tk.Frame):
    def __init__(
        self,
        parent,
        theme: GlassTheme,
        *,
        padding: tuple[int, int] | tuple[int, int, int, int] = (18, 18),
        tint: str | None = None,
        border: str | None = None,
    ):
        super().__init__(parent, bg=theme.shadow, highlightthickness=0, bd=0)
        self.theme = theme
        self.surface = tk.Frame(
            self,
            bg=tint or theme.card,
            highlightthickness=1,
            highlightbackground=border or theme.border,
            highlightcolor=border or theme.border,
            bd=0,
        )
        self.surface.pack(fill="both", expand=True, padx=1, pady=1)
        self.body = tk.Frame(self.surface, bg=tint or theme.card, highlightthickness=0, bd=0)
        left, top, right, bottom = _normalize_padding(padding)
        self.body.pack(fill="both", expand=True, padx=(left, right), pady=(top, bottom))


class GlassButton(tk.Button):
    def __init__(
        self,
        parent,
        theme: GlassTheme,
        *,
        text: str,
        command,
        fill: str,
        ink: str = "white",
        anchor: str = "w",
        pad_x: int = 16,
        pad_y: int = 12,
    ):
        super().__init__(
            parent,
            text=text,
            command=command,
            bg=fill,
            fg=ink,
            relief="flat",
            bd=0,
            activebackground=fill,
            activeforeground=ink,
            cursor="hand2",
            font=theme.body_bold_font,
            padx=pad_x,
            pady=pad_y,
            anchor=anchor,
            justify="left",
            highlightthickness=1,
            highlightbackground=theme.border,
            highlightcolor=theme.border,
            disabledforeground=theme.soft,
        )


class SegmentedTabs(tk.Frame):
    def __init__(self, parent, theme: GlassTheme, items: list[tuple[str, str]], command, *, compact: bool = False):
        super().__init__(parent, bg=theme.card, highlightthickness=0, bd=0)
        self.theme = theme
        self.command = command
        self.buttons: dict[str, tk.Button] = {}
        self.container = tk.Frame(self, bg=theme.frosted, highlightthickness=1, highlightbackground=theme.border, bd=0)
        self.container.pack(fill="x")
        self.items = items
        button_padx = 12 if compact else 16
        button_pady = 7 if compact else 9
        button_font = theme.small_font if compact else theme.body_bold_font
        for index, (key, label) in enumerate(items):
            self.container.grid_columnconfigure(index, weight=1, uniform="segment")
            button = tk.Button(
                self.container,
                text=label,
                command=lambda value=key: self.command(value),
                relief="flat",
                bd=0,
                cursor="hand2",
                padx=button_padx,
                pady=button_pady,
                font=button_font,
                highlightthickness=0,
                anchor="center",
            )
            button.grid(row=0, column=index, sticky="ew", padx=4, pady=4)
            self.buttons[key] = button
        self.set_active(items[0][0] if items else "")

    def set_active(self, active_key: str) -> None:
        for key, button in self.buttons.items():
            if key == active_key:
                button.configure(bg=self.theme.dark, fg="white", activebackground=self.theme.dark, activeforeground="white")
            else:
                button.configure(bg=self.theme.frosted, fg=self.theme.muted, activebackground=self.theme.card_alt, activeforeground=self.theme.text)


class ThemePreviewPicker(tk.Frame):
    def __init__(self, parent, theme: GlassTheme, items: list[tuple[str, str]], command):
        super().__init__(parent, bg=theme.card, highlightthickness=0, bd=0)
        self.theme = theme
        self.command = command
        self.active_key = items[0][0] if items else ""
        self.hover_key: str | None = None
        self.enabled = True
        self.cards: dict[str, tk.Frame] = {}
        self.labels: dict[str, tk.Label] = {}
        self.canvases: dict[str, tk.Canvas] = {}
        self.badges: dict[str, tk.Label] = {}
        self.preview_themes: dict[str, GlassTheme] = {}
        self._animation_jobs: dict[str, str] = {}
        self.items = items
        self.bind("<Destroy>", self._on_destroy, add="+")
        for index, (key, label) in enumerate(items):
            self.grid_columnconfigure(index, weight=1, uniform="theme-preview")
            preview_theme = build_glass_theme(key)
            self.preview_themes[key] = preview_theme
            card = tk.Frame(
                self,
                bg=theme.frosted,
                highlightthickness=1,
                highlightbackground=theme.border,
                highlightcolor=theme.border,
                bd=0,
                cursor="hand2",
            )
            card.grid(row=0, column=index, sticky="nsew", padx=(0 if index == 0 else 6, 0), pady=0)
            canvas = tk.Canvas(
                card,
                height=54,
                bg=preview_theme.shell,
                highlightthickness=1,
                highlightbackground=preview_theme.border,
                highlightcolor=preview_theme.border,
                bd=0,
                relief="flat",
                cursor="hand2",
            )
            canvas.pack(fill="x", padx=8, pady=(8, 6))
            canvas.bind("<Configure>", lambda _event, target=canvas, palette=preview_theme: self._draw_preview(target, palette), add="+")
            caption = tk.Label(card, text=label, bg=theme.frosted, fg=theme.muted, font=theme.small_font, cursor="hand2")
            caption.pack(fill="x", padx=6, pady=(0, 7))
            badge = tk.Label(card, text="Active", bg=theme.blue_soft, fg=theme.blue, font=theme.small_font, padx=6, pady=1, bd=0)
            self._bind_interaction(card, key)
            self._bind_interaction(canvas, key)
            self._bind_interaction(caption, key)
            self.cards[key] = card
            self.labels[key] = caption
            self.canvases[key] = canvas
            self.badges[key] = badge
            self._draw_preview(canvas, preview_theme)
        self.set_active(self.active_key)

    def _bind_interaction(self, widget: tk.Widget, key: str) -> None:
        widget.bind("<Button-1>", lambda _event, value=key: self._handle_click(value), add="+")
        widget.bind("<Enter>", lambda _event, value=key: self._handle_enter(value), add="+")
        widget.bind("<Leave>", lambda _event, value=key: self.after(12, lambda target=value: self._clear_hover_if_needed(target)), add="+")

    def _handle_click(self, key: str) -> None:
        if not self.enabled:
            return
        self.command(key)

    def _handle_enter(self, key: str) -> None:
        if not self.enabled:
            return
        self._set_hover(key)

    def _set_hover(self, key: str) -> None:
        if self.hover_key == key:
            return
        previous = self.hover_key
        self.hover_key = key
        if previous and previous in self.cards:
            self._refresh_card(previous)
        self._refresh_card(key)

    def _clear_hover_if_needed(self, key: str) -> None:
        card = self.cards.get(key)
        if card is None:
            return
        hovered_widget = self.winfo_containing(self.winfo_pointerx(), self.winfo_pointery())
        if hovered_widget is not None and self._is_descendant(hovered_widget, card):
            return
        if self.hover_key == key:
            self.hover_key = None
            self._refresh_card(key)

    def _is_descendant(self, widget: tk.Widget, ancestor: tk.Widget) -> bool:
        current = widget
        while current is not None:
            if current == ancestor:
                return True
            current = current.master
        return False

    def _draw_preview(self, canvas: tk.Canvas, preview_theme: GlassTheme) -> None:
        canvas.delete("all")
        width = max(canvas.winfo_width(), 78)
        height = max(canvas.winfo_height(), 54)
        pad = 6
        inner_left = pad
        inner_top = pad
        inner_right = width - pad
        inner_bottom = height - pad
        canvas.create_rectangle(
            inner_left,
            inner_top,
            inner_right,
            inner_bottom,
            fill=preview_theme.card,
            outline=preview_theme.border,
            width=1,
        )
        for index, color in enumerate((preview_theme.red, preview_theme.orange, preview_theme.green)):
            dot_left = inner_left + 6 + index * 10
            canvas.create_oval(dot_left, inner_top + 5, dot_left + 5, inner_top + 10, fill=color, outline=color)
        canvas.create_rectangle(
            inner_left + 6,
            inner_top + 15,
            inner_left + max(int((inner_right - inner_left) * 0.42), 24),
            inner_top + 18,
            fill=preview_theme.text,
            outline=preview_theme.text,
        )
        canvas.create_rectangle(
            inner_left + 6,
            inner_top + 23,
            inner_left + max(int((inner_right - inner_left) * 0.58), 34),
            inner_top + 26,
            fill=preview_theme.blue,
            outline=preview_theme.blue,
        )
        canvas.create_rectangle(
            inner_left + 6,
            inner_top + 31,
            inner_right - 6,
            inner_bottom - 6,
            fill=preview_theme.card_alt,
            outline=preview_theme.border,
            width=1,
        )

    def set_active(self, active_key: str) -> None:
        self.active_key = active_key
        for key, card in self.cards.items():
            self._refresh_card(key)

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        if not self.enabled:
            self.hover_key = None
        cursor = "hand2" if self.enabled else "arrow"
        for key in self.cards:
            self.cards[key].configure(cursor=cursor)
            self.labels[key].configure(cursor=cursor)
            self.canvases[key].configure(cursor=cursor)
            self._refresh_card(key)

    def _state_colors(self, key: str) -> dict[str, str]:
        preview_theme = self.preview_themes[key]
        selected = key == self.active_key
        hovered = key == self.hover_key
        if not self.enabled:
            card_bg = _mix_color(self.theme.frosted, self.theme.card, 0.35)
            border = _mix_color(self.theme.border, self.theme.card_alt, 0.5)
            text = _mix_color(self.theme.muted, self.theme.border, 0.4)
            preview_border = _mix_color(preview_theme.border, self.theme.border, 0.4)
        elif selected:
            card_bg = self.theme.card
            border = self.theme.blue
            text = self.theme.text
            preview_border = _mix_color(preview_theme.border, self.theme.blue, 0.55)
        elif hovered:
            card_bg = _mix_color(self.theme.frosted, self.theme.card, 0.65)
            border = _mix_color(self.theme.border, self.theme.blue, 0.4)
            text = _mix_color(self.theme.muted, self.theme.text, 0.7)
            preview_border = _mix_color(preview_theme.border, self.theme.blue, 0.2)
        else:
            card_bg = self.theme.frosted
            border = self.theme.border
            text = self.theme.muted
            preview_border = preview_theme.border
        return {
            "card_bg": card_bg,
            "border": border,
            "text": text,
            "preview_border": preview_border,
        }

    def _refresh_card(self, key: str) -> None:
        if key not in self.cards:
            return
        state = self._state_colors(key)
        self._animate_card(key, state)
        badge = self.badges[key]
        if key == self.active_key:
            badge.place(relx=1.0, x=-8, y=8, anchor="ne")
        else:
            badge.place_forget()

    def _animate_card(self, key: str, target: dict[str, str]) -> None:
        card = self.cards[key]
        label = self.labels[key]
        canvas = self.canvases[key]
        selected = key == self.active_key
        label.configure(font=self.theme.body_bold_font if selected else self.theme.small_font)
        start = {
            "card_bg": str(card.cget("bg")),
            "border": str(card.cget("highlightbackground")),
            "text": str(label.cget("fg")),
            "preview_border": str(canvas.cget("highlightbackground")),
        }
        existing_job = self._animation_jobs.get(key)
        if existing_job:
            try:
                self.after_cancel(existing_job)
            except tk.TclError:
                pass

        steps = 6

        def run(step_index: int = 1) -> None:
            progress = step_index / steps
            card_bg = _mix_color(start["card_bg"], target["card_bg"], progress)
            border = _mix_color(start["border"], target["border"], progress)
            text = _mix_color(start["text"], target["text"], progress)
            preview_border = _mix_color(start["preview_border"], target["preview_border"], progress)
            card.configure(bg=card_bg, highlightbackground=border, highlightcolor=border)
            label.configure(bg=card_bg, fg=text)
            canvas.configure(highlightbackground=preview_border, highlightcolor=preview_border)
            if step_index < steps:
                self._animation_jobs[key] = self.after(18, lambda: run(step_index + 1))
            else:
                self._animation_jobs.pop(key, None)

        run()

    def _on_destroy(self, _event=None) -> None:
        for job in list(self._animation_jobs.values()):
            try:
                self.after_cancel(job)
            except tk.TclError:
                pass
        self._animation_jobs.clear()


class ScrollablePanel(tk.Frame):
    def __init__(
        self,
        parent,
        theme: GlassTheme,
        *,
        bg: str | None = None,
        lock_x: bool = True,
        allow_horizontal_wheel: bool = False,
        show_horizontal_scrollbar: bool = False,
        horizontal_wheel_factor: int = 4,
    ):
        super().__init__(parent, bg=bg or theme.bg, highlightthickness=0, bd=0)
        self.theme = theme
        self.lock_x = lock_x
        self.allow_horizontal_wheel = allow_horizontal_wheel
        self.show_horizontal_scrollbar = show_horizontal_scrollbar
        self.horizontal_wheel_factor = max(1, int(horizontal_wheel_factor))
        self.canvas = tk.Canvas(self, bg=bg or theme.bg, highlightthickness=0, bd=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", style="Glass.Vertical.TScrollbar", command=self.canvas.yview)
        self.hscrollbar = None
        if self.show_horizontal_scrollbar:
            self.hscrollbar = ttk.Scrollbar(self, orient="horizontal", style="Glass.Horizontal.TScrollbar", command=self.canvas.xview)
        self.body = tk.Frame(self.canvas, bg=bg or theme.bg, highlightthickness=0, bd=0)
        self.window_id = self.canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        if self.hscrollbar is not None:
            self.canvas.configure(xscrollcommand=self.hscrollbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        if self.hscrollbar is not None:
            self.hscrollbar.grid(row=1, column=0, sticky="ew")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.body.bind("<Configure>", self._on_body_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self._bind_wheel_to_widget(self)
        self._bind_wheel_to_widget(self.canvas)
        self._bind_wheel_to_widget(self.body)
        self.canvas.bind("<Enter>", self._bind_descendant_wheel, add="+")
        self.body.bind("<Enter>", self._bind_descendant_wheel, add="+")

    def _on_body_configure(self, _event=None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        if self.lock_x:
            self.canvas.xview_moveto(0)
        self._bind_descendant_wheel()

    def _on_canvas_configure(self, event) -> None:
        if self.lock_x:
            self.canvas.itemconfigure(self.window_id, width=event.width)
        if self.lock_x:
            self.canvas.xview_moveto(0)

    def _bind_descendant_wheel(self, _event=None) -> None:
        stack = list(self.body.winfo_children())
        while stack:
            widget = stack.pop()
            self._bind_wheel_to_widget(widget)
            stack.extend(widget.winfo_children())

    def _bind_wheel_to_widget(self, widget: tk.Widget) -> None:
        if getattr(widget, "_glass_panel_wheel_bound", False):
            return
        if isinstance(widget, (tk.Text, tk.Listbox, ttk.Treeview)):
            return
        widget.bind("<MouseWheel>", self._on_mousewheel, add="+")
        widget.bind("<Button-4>", self._on_mousewheel, add="+")
        widget.bind("<Button-5>", self._on_mousewheel, add="+")
        setattr(widget, "_glass_panel_wheel_bound", True)

    def _on_mousewheel(self, event) -> None:
        steps = self._normalized_wheel_steps(event)
        if steps == 0:
            return None
        if self.allow_horizontal_wheel and not self.lock_x and (getattr(event, "state", 0) & 0x0001):
            self.canvas.xview_scroll(steps * self.horizontal_wheel_factor, "units")
            return "break"
        self.canvas.yview_scroll(steps, "units")
        return "break"

    def _normalized_wheel_steps(self, event) -> int:
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


def bind_wrap_to_width(widget: tk.Widget, *, extra_padding: int = 0, min_wrap: int = 120) -> None:
    def on_resize(event):
        width = max(event.width - extra_padding, min_wrap)
        widget.configure(wraplength=width)

    widget.bind("<Configure>", on_resize, add="+")
    if widget.master is not None:
        widget.master.bind("<Configure>", on_resize, add="+")


def _normalize_padding(padding):
    if len(padding) == 2:
        x, y = padding
        return x, y, x, y
    if len(padding) == 4:
        return padding
    raise ValueError("Padding must have 2 or 4 values.")
