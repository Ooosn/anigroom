from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import uuid

import numpy as np
from PIL import Image, ImageTk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anigroom.flow_annotations import sha256_file  # noqa: E402
from anigroom.seed_flow_annotations import (  # noqa: E402
    FlowSeed,
    SeedFlowAnnotations,
    SeedNeighborGraph,
    build_seed_neighbor_graph,
    load_seed_flow_annotations,
    make_flow_seed,
    nearest_seed_direction,
    propagate_follower_directions,
    save_seed_flow_annotations,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PROJECT_SCHEMA = "anigroom.seed_flow.project.v1"
TOOL_MODES = ("seed", "comb", "relax", "erase")


@dataclass(frozen=True)
class CanvasSeed:
    id: str
    position_px: tuple[float, float]
    direction_px: tuple[float, float]
    manual: bool


@dataclass(frozen=True)
class HistoryEntry:
    image_name: str
    before: tuple[CanvasSeed, ...]
    after: tuple[CanvasSeed, ...]


class ToolTip:
    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget = widget
        self.text = text
        self.window: tk.Toplevel | None = None
        self.after_id: str | None = None
        widget.bind("<Enter>", self._schedule, add=True)
        widget.bind("<Leave>", self._hide, add=True)
        widget.bind("<ButtonPress>", self._hide, add=True)

    def _schedule(self, _event: tk.Event | None = None) -> None:
        self._hide()
        self.after_id = self.widget.after(500, self._show)

    def _show(self) -> None:
        self.after_id = None
        if self.window is not None or not self.text:
            return
        x = self.widget.winfo_rootx() + self.widget.winfo_width() // 2
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 7
        self.window = tk.Toplevel(self.widget)
        self.window.wm_overrideredirect(True)
        self.window.wm_geometry(f"+{x}+{y}")
        tk.Label(
            self.window,
            text=self.text,
            background="#11151b",
            foreground="#e6eaf0",
            borderwidth=1,
            relief="solid",
            padx=7,
            pady=4,
            font=("Segoe UI", 9),
        ).pack()

    def _hide(self, _event: tk.Event | None = None) -> None:
        if self.after_id is not None:
            self.widget.after_cancel(self.after_id)
            self.after_id = None
        if self.window is not None:
            self.window.destroy()
            self.window = None


class FlowAnnotatorApp:
    def __init__(self, root: tk.Tk, *, input_dir: Path | None, output_dir: Path | None) -> None:
        self.root = root
        self.root.title("AniGroom Flow Brush")
        self.root.geometry("1500x920")
        self.root.minsize(1100, 700)
        self.root.configure(background="#0f1318")
        self._configure_style()

        self.input_dir: Path | None = None
        self.output_dir: Path | None = output_dir.resolve() if output_dir else None
        self.image_paths: list[Path] = []
        self.current_index = -1
        self.current_image: Image.Image | None = None
        self.current_photo: ImageTk.PhotoImage | None = None
        self.current_image_item: int | None = None
        self.image_hash_cache: dict[str, str] = {}
        self.annotations: dict[str, list[CanvasSeed]] = {}
        self.dirty_images: set[str] = set()
        self.failed_annotations: dict[str, str] = {}

        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.fit_mode = True
        self.pan_start: tuple[float, float] | None = None
        self.pan_origin: tuple[float, float] | None = None
        self.space_down = False
        self.cursor_image_point: tuple[float, float] | None = None

        self.stroke_before: tuple[CanvasSeed, ...] | None = None
        self.stroke_last_point: tuple[float, float] | None = None
        self.stroke_changed_ids: set[str] = set()
        self.stroke_positions_changed = False
        self.last_propagation_time = 0.0
        self.seed_items: dict[str, tuple[int, int]] = {}
        self.graph: SeedNeighborGraph | None = None
        self.graph_signature: tuple[tuple[str, float, float], ...] | None = None
        self.rng = np.random.default_rng(29)

        self.undo_stack: list[HistoryEntry] = []
        self.redo_stack: list[HistoryEntry] = []
        self.status_var = tk.StringVar(value="No folder open")
        self.path_var = tk.StringVar(value="")
        self.count_var = tk.StringVar(value="0 seeds")
        self.mode_var = tk.StringVar(value="seed")
        self.radius_var = tk.DoubleVar(value=54.0)
        self.density_var = tk.DoubleVar(value=0.45)
        self.strength_var = tk.DoubleVar(value=0.72)
        self.arrow_size_var = tk.DoubleVar(value=28.0)
        self.auto_smooth_var = tk.BooleanVar(value=True)
        self.control_values: dict[str, ttk.Label] = {}
        self.mode_buttons: dict[str, ttk.Button] = {}

        self._build_layout()
        self._bind_events()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._update_history_buttons()
        self._set_mode("seed")
        if input_dir is not None:
            self.root.after(30, lambda: self.open_folder(input_dir, output_dir=output_dir))
        else:
            self.root.after(150, self.choose_input_folder)

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("App.TFrame", background="#0f1318")
        style.configure("Panel.TFrame", background="#171c23")
        style.configure("Toolbar.TButton", background="#232a34", foreground="#e7ebf0", bordercolor="#323b48", padding=(10, 7), font=("Segoe UI", 9))
        style.map("Toolbar.TButton", background=[("active", "#303946"), ("pressed", "#1f252e")], foreground=[("disabled", "#687180")])
        style.configure("Mode.TButton", background="#222933", foreground="#c8cfd8", bordercolor="#323b48", padding=(11, 9), font=("Segoe UI Semibold", 9))
        style.configure("ModeActive.TButton", background="#aa275f", foreground="#ffffff", bordercolor="#e3488b", padding=(11, 9), font=("Segoe UI Semibold", 9))
        style.map("Mode.TButton", background=[("active", "#303946")])
        style.map("ModeActive.TButton", background=[("active", "#c22c6c")])
        style.configure("Accent.TButton", background="#d72d73", foreground="#ffffff", bordercolor="#ec4a91", padding=(12, 7), font=("Segoe UI Semibold", 9))
        style.map("Accent.TButton", background=[("active", "#e43a82"), ("pressed", "#b82461")])
        style.configure("Muted.TLabel", background="#171c23", foreground="#8993a1", font=("Segoe UI", 9))
        style.configure("Title.TLabel", background="#171c23", foreground="#f3f5f8", font=("Segoe UI Semibold", 10))
        style.configure("Value.TLabel", background="#171c23", foreground="#e6ebf2", font=("Cascadia Mono", 9))
        style.configure("Status.TFrame", background="#10151b")
        style.configure("Status.TLabel", background="#10151b", foreground="#aab2bd", font=("Segoe UI", 9))
        style.configure("TScale", background="#171c23", troughcolor="#303946")
        style.configure("TCheckbutton", background="#171c23", foreground="#c8cfd8", font=("Segoe UI", 9))

    def _build_layout(self) -> None:
        top = ttk.Frame(self.root, style="Panel.TFrame", padding=(10, 8))
        top.pack(side=tk.TOP, fill=tk.X)
        self._toolbar_button(top, "Open", self.choose_input_folder, "Open an image folder")
        self._toolbar_button(top, "Output", self.choose_output_folder, "Choose the annotation folder")
        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        self._toolbar_button(top, "Prev", lambda: self.navigate(-1), "Previous image")
        self._toolbar_button(top, "Next", lambda: self.navigate(1), "Next image")
        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        self.undo_button = self._toolbar_button(top, "Undo", self.undo, "Undo the last brush stroke")
        self.redo_button = self._toolbar_button(top, "Redo", self.redo, "Redo the last brush stroke")
        self._toolbar_button(top, "Clear", self.clear_current, "Remove all seeds from this image")
        self._toolbar_button(top, "Save", self.save_current, "Smooth followers and save", accent=True)
        tk.Label(top, textvariable=self.path_var, background="#171c23", foreground="#7f8997", anchor="e", font=("Segoe UI", 9)).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(18, 4))

        body = ttk.Frame(self.root, style="App.TFrame")
        body.pack(fill=tk.BOTH, expand=True)
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)

        left = ttk.Frame(body, style="Panel.TFrame", width=250, padding=(9, 10))
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 1))
        left.grid_propagate(False)
        ttk.Label(left, text="Images", style="Title.TLabel").pack(anchor="w", padx=4, pady=(0, 8))
        image_frame = ttk.Frame(left, style="Panel.TFrame")
        image_frame.pack(fill=tk.BOTH, expand=True)
        self.image_list = tk.Listbox(image_frame, activestyle="none", background="#141920", foreground="#cfd5dd", selectbackground="#9d2557", selectforeground="#ffffff", highlightthickness=1, highlightbackground="#29313c", borderwidth=0, font=("Cascadia Mono", 9), exportselection=False)
        image_scroll = ttk.Scrollbar(image_frame, orient=tk.VERTICAL, command=self.image_list.yview)
        self.image_list.configure(yscrollcommand=image_scroll.set)
        self.image_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        image_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        center = ttk.Frame(body, style="App.TFrame")
        center.grid(row=0, column=1, sticky="nsew")
        center.grid_rowconfigure(0, weight=1)
        center.grid_columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(center, background="#0b0f14", highlightthickness=0, cursor="crosshair")
        self.canvas.grid(row=0, column=0, sticky="nsew")

        right = ttk.Frame(body, style="Panel.TFrame", width=320, padding=(14, 14))
        right.grid(row=0, column=2, sticky="nsew", padx=(1, 0))
        right.grid_propagate(False)
        header = ttk.Frame(right, style="Panel.TFrame")
        header.pack(fill=tk.X, pady=(0, 12))
        ttk.Label(header, text="Flow Brush", style="Title.TLabel").pack(side=tk.LEFT)
        ttk.Label(header, textvariable=self.count_var, style="Muted.TLabel").pack(side=tk.RIGHT)
        mode_grid = ttk.Frame(right, style="Panel.TFrame")
        mode_grid.pack(fill=tk.X, pady=(0, 18))
        mode_specs = [
            ("seed", "Seed", "Scatter follower seeds; density is point count"),
            ("comb", "Comb", "Brush nearby arrows toward the stroke direction"),
            ("relax", "Relax", "Release anchors so neighbors can interpolate them"),
            ("erase", "Erase", "Remove seeds inside the brush"),
        ]
        for index, (mode, label, tooltip) in enumerate(mode_specs):
            button = ttk.Button(mode_grid, text=label, command=lambda value=mode: self._set_mode(value), style="Mode.TButton")
            button.grid(row=index // 2, column=index % 2, sticky="ew", padx=3, pady=3)
            mode_grid.grid_columnconfigure(index % 2, weight=1)
            self.mode_buttons[mode] = button
            ToolTip(button, tooltip)
        self._control_slider(right, "Radius", self.radius_var, 8.0, 180.0, "Brush radius in source-image pixels")
        self._control_slider(right, "Density", self.density_var, 0.05, 1.0, "Seed spacing inside the brush")
        self._control_slider(right, "Strength", self.strength_var, 0.05, 1.0, "Comb and interpolation blend strength")
        self._control_slider(right, "Arrow size", self.arrow_size_var, 8.0, 72.0, "One global display size; not saved as hair length")
        ttk.Checkbutton(right, text="Auto smooth followers", variable=self.auto_smooth_var).pack(fill=tk.X, pady=(12, 8))
        smooth_button = ttk.Button(right, text="Smooth all followers", command=self.smooth_all, style="Toolbar.TButton")
        smooth_button.pack(fill=tk.X, pady=(0, 18))
        ToolTip(smooth_button, "Keep manual anchors fixed and diffuse their direction through all followers")
        legend = ttk.Frame(right, style="Panel.TFrame")
        legend.pack(fill=tk.X, side=tk.BOTTOM, pady=(12, 4))
        self._legend_item(legend, "#ff2f85", "Manual")
        self._legend_item(legend, "#39c9df", "Inferred")

        status = ttk.Frame(self.root, style="Status.TFrame", padding=(10, 5))
        status.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(status, textvariable=self.status_var, style="Status.TLabel").pack(side=tk.LEFT)
        self.zoom_label = ttk.Label(status, text="100%", style="Status.TLabel")
        self.zoom_label.pack(side=tk.RIGHT)

    def _control_slider(self, parent, label, variable, minimum, maximum, tooltip) -> None:
        panel = ttk.Frame(parent, style="Panel.TFrame")
        panel.pack(fill=tk.X, pady=(0, 13))
        header = ttk.Frame(panel, style="Panel.TFrame")
        header.pack(fill=tk.X)
        ttk.Label(header, text=label, style="Muted.TLabel").pack(side=tk.LEFT)
        value_label = ttk.Label(header, text="", style="Value.TLabel")
        value_label.pack(side=tk.RIGHT)
        self.control_values[label] = value_label
        scale = ttk.Scale(panel, from_=minimum, to=maximum, variable=variable, command=lambda _value: self._controls_changed())
        scale.pack(fill=tk.X, pady=(6, 0))
        ToolTip(scale, tooltip)
        self._controls_changed()

    def _legend_item(self, parent, color: str, label: str) -> None:
        item = ttk.Frame(parent, style="Panel.TFrame")
        item.pack(side=tk.LEFT, padx=(0, 18))
        swatch = tk.Canvas(item, width=12, height=12, background="#171c23", highlightthickness=0)
        swatch.pack(side=tk.LEFT)
        swatch.create_oval(2, 2, 10, 10, fill=color, outline="#080a0d")
        ttk.Label(item, text=label, style="Muted.TLabel").pack(side=tk.LEFT, padx=(4, 0))

    def _toolbar_button(self, parent, text, command, tooltip, *, accent: bool = False):
        button = ttk.Button(parent, text=text, command=command, style="Accent.TButton" if accent else "Toolbar.TButton")
        button.pack(side=tk.LEFT, padx=2)
        ToolTip(button, tooltip)
        return button

    def _controls_changed(self) -> None:
        if not self.control_values:
            return
        values = {"Radius": f"{self.radius_var.get():.0f}px", "Density": f"{self.density_var.get():.2f}", "Strength": f"{self.strength_var.get():.2f}", "Arrow size": f"{self.arrow_size_var.get():.0f}px"}
        for label, text in values.items():
            if label in self.control_values:
                self.control_values[label].configure(text=text)
        if hasattr(self, "canvas") and self.current_image is not None:
            self._sync_seed_items(full=True)
            self._draw_brush_cursor(self.cursor_image_point)

    def _bind_events(self) -> None:
        self.image_list.bind("<<ListboxSelect>>", self._on_image_list_select)
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.bind("<Motion>", self._on_canvas_motion)
        self.canvas.bind("<Leave>", lambda _event: self._draw_brush_cursor(None))
        self.canvas.bind("<ButtonPress-1>", self._on_brush_press)
        self.canvas.bind("<B1-Motion>", self._on_brush_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_brush_release)
        self.canvas.bind("<ButtonPress-2>", self._on_pan_press)
        self.canvas.bind("<B2-Motion>", self._on_pan_motion)
        self.canvas.bind("<ButtonRelease-2>", self._on_pan_release)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", lambda event: self._zoom_at(1.15, event.x, event.y))
        self.canvas.bind("<Button-5>", lambda event: self._zoom_at(1.0 / 1.15, event.x, event.y))
        self.root.bind("<Control-s>", lambda _event: self.save_current())
        self.root.bind("<Control-z>", lambda _event: self.undo())
        self.root.bind("<Control-y>", lambda _event: self.redo())
        self.root.bind("<Left>", lambda _event: self.navigate(-1))
        self.root.bind("<Right>", lambda _event: self.navigate(1))
        self.root.bind("<KeyPress-space>", self._on_space_press)
        self.root.bind("<KeyRelease-space>", self._on_space_release)
        self.root.bind("<KeyPress-f>", lambda _event: self.fit_image())
        for key, mode in zip(("1", "2", "3", "4"), TOOL_MODES):
            self.root.bind(f"<KeyPress-{key}>", lambda _event, value=mode: self._set_mode(value))

    def _set_mode(self, mode: str) -> None:
        if mode not in TOOL_MODES:
            raise ValueError(f"unknown tool mode: {mode}")
        self.mode_var.set(mode)
        for name, button in self.mode_buttons.items():
            button.configure(style="ModeActive.TButton" if name == mode else "Mode.TButton")
        if hasattr(self, "canvas"):
            self.canvas.configure(cursor="crosshair")
        self.status_var.set(mode.capitalize())

    def choose_input_folder(self) -> None:
        value = filedialog.askdirectory(parent=self.root, title="Open image folder", initialdir=str(self.input_dir or Path.cwd()))
        if value:
            self.open_folder(Path(value))

    def choose_output_folder(self) -> None:
        value = filedialog.askdirectory(parent=self.root, title="Choose annotation folder", initialdir=str(self.output_dir or self.input_dir or Path.cwd()))
        if not value:
            return
        self._save_current(silent=True)
        self.output_dir = Path(value).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._load_all_annotations()
        self._refresh_image_list()
        self._load_current_image(force=True)

    def open_folder(self, directory: Path, *, output_dir: Path | None = None) -> None:
        directory = directory.resolve()
        if not directory.is_dir():
            messagebox.showerror("Folder not found", str(directory), parent=self.root)
            return
        images = sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            messagebox.showerror("No images", f"No supported images found in {directory}", parent=self.root)
            return
        self._save_current(silent=True)
        self.input_dir = directory
        self.image_paths = images
        self.current_index = 0
        self.output_dir = output_dir.resolve() if output_dir is not None else directory.parent / f"{directory.name}_flow_guidance"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.annotations = {path.name: [] for path in images}
        self.failed_annotations.clear()
        self.dirty_images.clear()
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.image_hash_cache.clear()
        self._load_all_annotations()
        self._refresh_image_list()
        self._select_image_list_row(0)
        self._load_current_image(force=True)

    def _load_all_annotations(self) -> None:
        if self.output_dir is None:
            return
        self.annotations = {path.name: [] for path in self.image_paths}
        self.failed_annotations.clear()
        for image_path in self.image_paths:
            annotation_path = self._annotation_path(image_path)
            if not annotation_path.is_file():
                continue
            try:
                loaded = load_seed_flow_annotations(annotation_path, image_path=image_path, verify_image=True)
                self.annotations[image_path.name] = [CanvasSeed(seed.id, seed.position_px, seed.direction_px, seed.manual) for seed in loaded.seeds]
            except Exception as exc:
                self.failed_annotations[image_path.name] = str(exc)

    def _annotation_path(self, image_path: Path) -> Path:
        if self.output_dir is None:
            raise RuntimeError("annotation output directory is not set")
        return self.output_dir / f"{image_path.stem}.flow.json"

    def _current_path(self) -> Path | None:
        return self.image_paths[self.current_index] if 0 <= self.current_index < len(self.image_paths) else None

    def _current_seeds(self) -> list[CanvasSeed]:
        path = self._current_path()
        return self.annotations.setdefault(path.name, []) if path else []

    def _refresh_image_list(self) -> None:
        selected = self.current_index
        self.image_list.delete(0, tk.END)
        for path in self.image_paths:
            count = len(self.annotations.get(path.name, []))
            marker = "!" if path.name in self.failed_annotations else ("*" if path.name in self.dirty_images else " ")
            self.image_list.insert(tk.END, f"{marker} {path.name:<18} {count:>5}")
        self._select_image_list_row(selected)

    def _select_image_list_row(self, index: int) -> None:
        if 0 <= index < self.image_list.size():
            self.image_list.selection_clear(0, tk.END)
            self.image_list.selection_set(index)
            self.image_list.activate(index)
            self.image_list.see(index)

    def _on_image_list_select(self, _event=None) -> None:
        selection = self.image_list.curselection()
        if not selection or int(selection[0]) == self.current_index:
            return
        self._save_current(silent=True)
        self.current_index = int(selection[0])
        self._load_current_image(force=True)

    def navigate(self, delta: int) -> None:
        if not self.image_paths:
            return
        index = max(0, min(len(self.image_paths) - 1, self.current_index + int(delta)))
        if index != self.current_index:
            self._save_current(silent=True)
            self.current_index = index
            self._select_image_list_row(index)
            self._load_current_image(force=True)

    def _load_current_image(self, *, force: bool = False) -> None:
        path = self._current_path()
        if path is None:
            return
        if force or self.current_image is None:
            with Image.open(path) as source:
                self.current_image = source.convert("RGB")
        self.fit_mode = True
        self._invalidate_graph()
        self._fit_transform()
        self._render_scene(rebuild_image=True)
        self._refresh_counts()
        self.path_var.set(f"{path.name}    ->    {self.output_dir}")
        self.status_var.set(f"Invalid existing JSON: {self.failed_annotations[path.name]}" if path.name in self.failed_annotations else f"Image {self.current_index + 1} of {len(self.image_paths)}")
        self._update_title()
        self.root.after(80, self.fit_image)

    def _fit_transform(self) -> None:
        if self.current_image is None:
            return
        canvas_width = max(self.canvas.winfo_width(), 2)
        canvas_height = max(self.canvas.winfo_height(), 2)
        width, height = self.current_image.size
        margin = 24
        self.scale = max(0.02, min((canvas_width - margin * 2) / width, (canvas_height - margin * 2) / height))
        self.offset_x = (canvas_width - width * self.scale) * 0.5
        self.offset_y = (canvas_height - height * self.scale) * 0.5
        self.fit_mode = True

    def fit_image(self) -> None:
        self._fit_transform()
        self._render_scene(rebuild_image=True)

    def _render_scene(self, *, rebuild_image: bool) -> None:
        if self.current_image is None:
            self.canvas.delete("all")
            return
        if rebuild_image:
            display_width = max(1, int(round(self.current_image.width * self.scale)))
            display_height = max(1, int(round(self.current_image.height * self.scale)))
            self.current_photo = ImageTk.PhotoImage(self.current_image.resize((display_width, display_height), Image.Resampling.LANCZOS))
            self.canvas.delete("all")
            self.seed_items.clear()
            self.current_image_item = self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.current_photo, tags=("source_image",))
        elif self.current_image_item is not None:
            self.canvas.coords(self.current_image_item, self.offset_x, self.offset_y)
        self._sync_seed_items(full=True)
        self._draw_brush_cursor(self.cursor_image_point)
        self.canvas.tag_lower("source_image")
        self.zoom_label.configure(text=f"{self.scale * 100:.0f}%")

    def _sync_seed_items(self, *, full: bool, changed_ids: set[str] | None = None) -> None:
        seed_by_id = {seed.id: seed for seed in self._current_seeds()}
        if full:
            for items in self.seed_items.values():
                for item in items:
                    self.canvas.delete(item)
            self.seed_items.clear()
            ids = set(seed_by_id)
        else:
            ids = set(changed_ids or ())
            for seed_id in set(self.seed_items) - set(seed_by_id):
                for item in self.seed_items.pop(seed_id):
                    self.canvas.delete(item)
        arrow_length = float(self.arrow_size_var.get())
        for seed_id in ids:
            seed = seed_by_id.get(seed_id)
            if seed is None:
                continue
            x0, y0 = self.image_to_canvas(*seed.position_px)
            end = (seed.position_px[0] + seed.direction_px[0] * arrow_length, seed.position_px[1] + seed.direction_px[1] * arrow_length)
            x1, y1 = self.image_to_canvas(*end)
            color = "#ff2f85" if seed.manual else "#39c9df"
            width = 2.2 if seed.manual else 1.6
            if seed_id not in self.seed_items:
                line = self.canvas.create_line(x0, y0, x1, y1, fill=color, width=width, arrow=tk.LAST, arrowshape=(10, 12, 4), tags=("seed", f"seed:{seed_id}"))
                dot = self.canvas.create_oval(x0 - 2.5, y0 - 2.5, x0 + 2.5, y0 + 2.5, fill=color, outline="#080a0d", width=1, tags=("seed", f"seed:{seed_id}"))
                self.seed_items[seed_id] = line, dot
            else:
                line, dot = self.seed_items[seed_id]
                self.canvas.coords(line, x0, y0, x1, y1)
                self.canvas.itemconfigure(line, fill=color, width=width)
                self.canvas.coords(dot, x0 - 2.5, y0 - 2.5, x0 + 2.5, y0 + 2.5)
                self.canvas.itemconfigure(dot, fill=color)

    def _draw_brush_cursor(self, point: tuple[float, float] | None) -> None:
        self.canvas.delete("brush_cursor")
        if point is None or self.current_image is None:
            return
        x, y = self.image_to_canvas(*point)
        radius = float(self.radius_var.get()) * self.scale
        color = {"seed": "#39c9df", "comb": "#ff2f85", "relax": "#ffd166", "erase": "#ff625f"}[self.mode_var.get()]
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, outline=color, width=2, dash=(5, 4), tags=("brush_cursor",))

    def image_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        return self.offset_x + float(x) * self.scale, self.offset_y + float(y) * self.scale

    def canvas_to_image(self, x: float, y: float, *, clamp: bool = False) -> tuple[float, float] | None:
        if self.current_image is None or self.scale <= 0:
            return None
        image_x = (float(x) - self.offset_x) / self.scale
        image_y = (float(y) - self.offset_y) / self.scale
        width, height = self.current_image.size
        if clamp:
            return max(0.0, min(width - 1.0, image_x)), max(0.0, min(height - 1.0, image_y))
        return (image_x, image_y) if 0.0 <= image_x <= width - 1 and 0.0 <= image_y <= height - 1 else None

    def _on_canvas_motion(self, event: tk.Event) -> None:
        self.cursor_image_point = self.canvas_to_image(event.x, event.y)
        self._draw_brush_cursor(self.cursor_image_point)

    def _on_brush_press(self, event: tk.Event) -> None:
        self.canvas.focus_set()
        if self.space_down:
            self._start_pan(event.x, event.y)
            return
        point = self.canvas_to_image(event.x, event.y)
        if point is None:
            return
        self.stroke_before = tuple(self._current_seeds())
        self.stroke_last_point = point
        self.stroke_changed_ids.clear()
        self.stroke_positions_changed = False
        self._apply_mode_stamp(point, previous=None)

    def _on_brush_motion(self, event: tk.Event) -> None:
        if self.pan_start is not None:
            self._move_pan(event.x, event.y)
            return
        point = self.canvas_to_image(event.x, event.y, clamp=True)
        self.cursor_image_point = point
        self._draw_brush_cursor(point)
        if self.stroke_before is None or self.stroke_last_point is None or point is None:
            return
        spacing = max(2.0, float(self.radius_var.get()) * 0.28)
        for sample in self.stroke_samples(self.stroke_last_point, point, spacing=spacing):
            previous = self.stroke_last_point
            self._apply_mode_stamp(sample, previous=previous)
            self.stroke_last_point = sample
        if self.mode_var.get() == "comb" and self.auto_smooth_var.get() and time.perf_counter() - self.last_propagation_time > 0.045:
            self._propagate_incremental(self.stroke_changed_ids, iterations=4)
            self.last_propagation_time = time.perf_counter()

    def _on_brush_release(self, event: tk.Event) -> None:
        if self.pan_start is not None:
            self._finish_pan()
            return
        if self.stroke_before is None:
            return
        point = self.canvas_to_image(event.x, event.y, clamp=True)
        if point is not None and self.stroke_last_point is not None:
            self._apply_mode_stamp(point, previous=self.stroke_last_point)
        if self.stroke_positions_changed:
            self._invalidate_graph()
        if self.auto_smooth_var.get() and self.stroke_changed_ids and self.mode_var.get() in {"comb", "relax"}:
            self._propagate_incremental(self.stroke_changed_ids, iterations=8)
        before, after = self.stroke_before, tuple(self._current_seeds())
        self.stroke_before = None
        self.stroke_last_point = None
        if before != after:
            self._record_history(before, after)
            self._mark_dirty()
            self._refresh_counts()
            self._sync_seed_items(full=self.stroke_positions_changed, changed_ids=self.stroke_changed_ids)
        self.stroke_changed_ids.clear()
        self.stroke_positions_changed = False

    def _apply_mode_stamp(self, point: tuple[float, float], previous: tuple[float, float] | None) -> None:
        mode = self.mode_var.get()
        if mode == "seed":
            self._scatter_seeds(point)
        elif mode == "comb" and previous is not None and math.dist(previous, point) >= 0.75:
            self._comb_seeds(point, (point[0] - previous[0], point[1] - previous[1]))
        elif mode == "relax":
            self._relax_seeds(point)
        elif mode == "erase":
            self._erase_seeds(point)

    @staticmethod
    def stroke_samples(start: tuple[float, float], end: tuple[float, float], *, spacing: float) -> list[tuple[float, float]]:
        distance = math.dist(start, end)
        if distance < spacing:
            return [end]
        count = max(1, int(math.ceil(distance / max(spacing, 1.0e-6))))
        return [(start[0] + (end[0] - start[0]) * i / count, start[1] + (end[1] - start[1]) * i / count) for i in range(1, count + 1)]

    def _scatter_seeds(self, center: tuple[float, float]) -> None:
        if self.current_image is None:
            return
        seeds = self._current_seeds()
        radius = float(self.radius_var.get())
        minimum_spacing = 22.0 - 17.0 * float(self.density_var.get())
        target_count = min(650, max(1, int(math.pi * radius * radius / (minimum_spacing * minimum_spacing) * 1.35)))
        cell = max(1.0, minimum_spacing)
        grid: dict[tuple[int, int], list[tuple[float, float]]] = {}
        for seed in seeds:
            position = seed.position_px
            grid.setdefault((int(position[0] // cell), int(position[1] // cell)), []).append(position)

        def clear(candidate: tuple[float, float]) -> bool:
            key = int(candidate[0] // cell), int(candidate[1] // cell)
            return all(math.dist(candidate, other) >= minimum_spacing for gx in range(key[0] - 1, key[0] + 2) for gy in range(key[1] - 1, key[1] + 2) for other in grid.get((gx, gy), ()))

        fallback = self._nearest_direction(center)
        width, height = self.current_image.size
        created: list[CanvasSeed] = []
        for _ in range(target_count):
            angle = float(self.rng.uniform(0.0, math.tau))
            radial = radius * math.sqrt(float(self.rng.uniform(0.0, 1.0)))
            candidate = center[0] + radial * math.cos(angle), center[1] + radial * math.sin(angle)
            if 0.0 <= candidate[0] <= width - 1 and 0.0 <= candidate[1] <= height - 1 and clear(candidate):
                seed = CanvasSeed(uuid.uuid4().hex[:16], candidate, fallback, False)
                created.append(seed)
                grid.setdefault((int(candidate[0] // cell), int(candidate[1] // cell)), []).append(candidate)
        if not created and not seeds:
            candidate = max(0.0, min(width - 1.0, center[0])), max(0.0, min(height - 1.0, center[1]))
            created.append(CanvasSeed(uuid.uuid4().hex[:16], candidate, fallback, False))
        if created:
            seeds.extend(created)
            ids = {seed.id for seed in created}
            self.stroke_changed_ids.update(ids)
            self.stroke_positions_changed = True
            self._invalidate_graph()
            self._sync_seed_items(full=False, changed_ids=ids)

    def _nearest_direction(self, position: tuple[float, float]) -> tuple[float, float]:
        seeds = self._current_seeds()
        if not seeds:
            return 0.0, 1.0
        return nearest_seed_direction(position, np.asarray([seed.position_px for seed in seeds]), np.asarray([seed.direction_px for seed in seeds]))

    def _indices_in_brush(self, center: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
        seeds = self._current_seeds()
        if not seeds:
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float64)
        positions = np.asarray([seed.position_px for seed in seeds], dtype=np.float64)
        distances = np.linalg.norm(positions - np.asarray(center, dtype=np.float64)[None], axis=1)
        indices = np.flatnonzero(distances <= float(self.radius_var.get()))
        return indices, distances[indices]

    def _comb_seeds(self, center: tuple[float, float], stroke_direction: tuple[float, float]) -> None:
        norm = math.hypot(*stroke_direction)
        if norm <= 1.0e-8:
            return
        target = np.asarray(stroke_direction) / norm
        indices, distances = self._indices_in_brush(center)
        if indices.size == 0:
            return
        seeds = self._current_seeds()
        radius = max(float(self.radius_var.get()), 1.0e-6)
        falloff = np.square(np.clip(1.0 - distances / radius, 0.0, 1.0)) * float(self.strength_var.get())
        changed: set[str] = set()
        for index, weight in zip(indices, falloff):
            if weight <= 1.0e-4:
                continue
            old = seeds[int(index)]
            mixed = (1.0 - weight) * np.asarray(old.direction_px) + weight * target
            mixed_norm = float(np.linalg.norm(mixed))
            if mixed_norm > 1.0e-8:
                seeds[int(index)] = CanvasSeed(old.id, old.position_px, (float(mixed[0] / mixed_norm), float(mixed[1] / mixed_norm)), True)
                changed.add(old.id)
        self.stroke_changed_ids.update(changed)
        self._sync_seed_items(full=False, changed_ids=changed)

    def _relax_seeds(self, center: tuple[float, float]) -> None:
        indices, _ = self._indices_in_brush(center)
        seeds = self._current_seeds()
        changed: set[str] = set()
        for index in indices:
            old = seeds[int(index)]
            if old.manual:
                seeds[int(index)] = CanvasSeed(old.id, old.position_px, old.direction_px, False)
                changed.add(old.id)
        self.stroke_changed_ids.update(changed)
        self._sync_seed_items(full=False, changed_ids=changed)

    def _erase_seeds(self, center: tuple[float, float]) -> None:
        indices, _ = self._indices_in_brush(center)
        if indices.size == 0:
            return
        remove = set(int(index) for index in indices)
        seeds = self._current_seeds()
        removed_ids = {seeds[index].id for index in remove}
        seeds[:] = [seed for index, seed in enumerate(seeds) if index not in remove]
        self.stroke_changed_ids.update(removed_ids)
        self.stroke_positions_changed = True
        self._invalidate_graph()
        self._sync_seed_items(full=False, changed_ids=removed_ids)

    def _graph_for_current(self) -> SeedNeighborGraph:
        seeds = self._current_seeds()
        signature = tuple((seed.id, seed.position_px[0], seed.position_px[1]) for seed in seeds)
        if self.graph is None or signature != self.graph_signature:
            self.graph = build_seed_neighbor_graph(np.asarray([seed.position_px for seed in seeds], dtype=np.float64).reshape(-1, 2), neighbor_count=8)
            self.graph_signature = signature
        return self.graph

    def _invalidate_graph(self) -> None:
        self.graph = None
        self.graph_signature = None

    def _propagate_incremental(self, changed_ids: set[str], *, iterations: int) -> None:
        seeds = self._current_seeds()
        if len(seeds) <= 1 or not changed_ids:
            return
        id_to_index = {seed.id: index for index, seed in enumerate(seeds)}
        changed_indices = [id_to_index[seed_id] for seed_id in changed_ids if seed_id in id_to_index]
        if not changed_indices:
            return
        output, active = propagate_follower_directions(
            np.asarray([seed.direction_px for seed in seeds]),
            np.asarray([seed.manual for seed in seeds]),
            self._graph_for_current(),
            changed_indices=changed_indices,
            rings=6,
            iterations=iterations,
            relaxation=float(self.strength_var.get()),
        )
        changed: set[str] = set()
        for index in active:
            old = seeds[int(index)]
            seeds[int(index)] = CanvasSeed(old.id, old.position_px, (float(output[index, 0]), float(output[index, 1])), old.manual)
            changed.add(old.id)
        self.stroke_changed_ids.update(changed)
        self._sync_seed_items(full=False, changed_ids=changed)

    def smooth_all(self, *, record_history: bool = True) -> bool:
        seeds = self._current_seeds()
        if len(seeds) <= 1 or not any(seed.manual for seed in seeds):
            return False
        before = tuple(seeds)
        output, active = propagate_follower_directions(
            np.asarray([seed.direction_px for seed in seeds]),
            np.asarray([seed.manual for seed in seeds]),
            self._graph_for_current(),
            iterations=24,
            relaxation=0.78,
        )
        for index in active:
            old = seeds[int(index)]
            seeds[int(index)] = CanvasSeed(old.id, old.position_px, (float(output[index, 0]), float(output[index, 1])), old.manual)
        after = tuple(seeds)
        if before == after:
            return False
        if record_history:
            self._record_history(before, after)
            self._mark_dirty()
        self._sync_seed_items(full=False, changed_ids={seeds[int(index)].id for index in active})
        self.status_var.set(f"Smoothed {len(active)} followers")
        return True

    def _refresh_counts(self) -> None:
        seeds = self._current_seeds()
        self.count_var.set(f"{len(seeds)} seeds  |  {sum(seed.manual for seed in seeds)} manual")

    def clear_current(self) -> None:
        seeds = self._current_seeds()
        if not seeds or not messagebox.askyesno("Clear seeds", "Remove every seed from this image?", parent=self.root):
            return
        before = tuple(seeds)
        seeds.clear()
        self._invalidate_graph()
        self._record_history(before, tuple())
        self._mark_dirty()
        self._refresh_counts()
        self._sync_seed_items(full=True)

    def _record_history(self, before: tuple[CanvasSeed, ...], after: tuple[CanvasSeed, ...]) -> None:
        path = self._current_path()
        if path is None or before == after:
            return
        self.undo_stack.append(HistoryEntry(path.name, before, after))
        if len(self.undo_stack) > 60:
            del self.undo_stack[0]
        self.redo_stack.clear()
        self._update_history_buttons()

    def undo(self) -> None:
        if self.undo_stack:
            entry = self.undo_stack.pop()
            self.redo_stack.append(entry)
            self._apply_history(entry.image_name, entry.before)

    def redo(self) -> None:
        if self.redo_stack:
            entry = self.redo_stack.pop()
            self.undo_stack.append(entry)
            self._apply_history(entry.image_name, entry.after)

    def _apply_history(self, image_name: str, seeds: tuple[CanvasSeed, ...]) -> None:
        self.annotations[image_name] = list(seeds)
        self.dirty_images.add(image_name)
        self._invalidate_graph()
        if self._current_path() is None or self._current_path().name != image_name:
            self.current_index = next(i for i, path in enumerate(self.image_paths) if path.name == image_name)
            self._select_image_list_row(self.current_index)
            self._load_current_image(force=True)
        else:
            self._refresh_counts()
            self._sync_seed_items(full=True)
        self._refresh_image_list()
        self._update_history_buttons()
        self._update_title()

    def _update_history_buttons(self) -> None:
        self.undo_button.state(["!disabled"] if self.undo_stack else ["disabled"])
        self.redo_button.state(["!disabled"] if self.redo_stack else ["disabled"])

    def _mark_dirty(self) -> None:
        path = self._current_path()
        if path is not None:
            self.dirty_images.add(path.name)
            self._refresh_image_list()
            self._update_title()

    def save_current(self) -> None:
        self._save_current(silent=False)

    def _save_current(self, *, silent: bool) -> bool:
        path = self._current_path()
        if path is None or self.current_image is None or self.output_dir is None:
            return False
        try:
            if self.auto_smooth_var.get():
                self.smooth_all(record_history=False)
            width, height = self.current_image.size
            flow_seeds: list[FlowSeed] = [make_flow_seed(seed.id, seed.position_px, seed.direction_px, width, height, manual=seed.manual) for seed in self._current_seeds()]
            annotations = SeedFlowAnnotations(
                image_filename=path.name,
                width=width,
                height=height,
                sha256=self.image_hash_cache.setdefault(path.name, sha256_file(path)),
                seeds=tuple(flow_seeds),
                updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
            save_seed_flow_annotations(annotations, self._annotation_path(path))
            self.dirty_images.discard(path.name)
            self.failed_annotations.pop(path.name, None)
            self._save_project_index()
            self._refresh_image_list()
            self.status_var.set(f"Saved {self._annotation_path(path).name}")
            self._update_title()
            return True
        except Exception as exc:
            if not silent:
                messagebox.showerror("Save failed", str(exc), parent=self.root)
            self.status_var.set(f"Save failed: {exc}")
            return False

    def _save_project_index(self) -> None:
        if self.input_dir is None or self.output_dir is None:
            return
        payload = {
            "schema": PROJECT_SCHEMA,
            "input_dir": str(self.input_dir),
            "output_dir": str(self.output_dir),
            "seed_semantics": "directed_unit_flow",
            "display_arrow_size_px": float(self.arrow_size_var.get()),
            "images": [
                {"filename": path.name, "annotation": f"{path.stem}.flow.json", "seed_count": len(self.annotations.get(path.name, [])), "manual_count": sum(seed.manual for seed in self.annotations.get(path.name, []))}
                for path in self.image_paths
            ],
        }
        destination = self.output_dir / "project.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(destination)

    def _on_pan_press(self, event) -> None:
        self._start_pan(event.x, event.y)

    def _on_pan_motion(self, event) -> None:
        self._move_pan(event.x, event.y)

    def _on_pan_release(self, _event) -> None:
        self._finish_pan()

    def _start_pan(self, x: float, y: float) -> None:
        self.pan_start = float(x), float(y)
        self.pan_origin = self.offset_x, self.offset_y
        self.fit_mode = False
        self.canvas.configure(cursor="fleur")

    def _move_pan(self, x: float, y: float) -> None:
        if self.pan_start is not None and self.pan_origin is not None:
            self.offset_x = self.pan_origin[0] + float(x) - self.pan_start[0]
            self.offset_y = self.pan_origin[1] + float(y) - self.pan_start[1]
            self._render_scene(rebuild_image=False)

    def _finish_pan(self) -> None:
        self.pan_start = None
        self.pan_origin = None
        self.canvas.configure(cursor="crosshair")

    def _on_space_press(self, _event) -> None:
        self.space_down = True
        self.canvas.configure(cursor="fleur")

    def _on_space_release(self, _event) -> None:
        self.space_down = False
        if self.pan_start is None:
            self.canvas.configure(cursor="crosshair")

    def _on_mousewheel(self, event) -> None:
        self._zoom_at(1.15 if event.delta > 0 else 1.0 / 1.15, event.x, event.y)

    def _zoom_at(self, factor: float, canvas_x: float, canvas_y: float) -> None:
        if self.current_image is None:
            return
        old_scale = self.scale
        new_scale = max(0.03, min(4.0, old_scale * float(factor)))
        if abs(new_scale - old_scale) < 1.0e-9:
            return
        image_x = (float(canvas_x) - self.offset_x) / old_scale
        image_y = (float(canvas_y) - self.offset_y) / old_scale
        self.scale = new_scale
        self.offset_x = float(canvas_x) - image_x * new_scale
        self.offset_y = float(canvas_y) - image_y * new_scale
        self.fit_mode = False
        self._render_scene(rebuild_image=True)

    def _on_canvas_resize(self, _event=None) -> None:
        if self.current_image is not None and self.fit_mode:
            self.root.after_idle(self.fit_image)

    @staticmethod
    def _point_segment_distance(point, start, end) -> float:
        px, py = point
        x0, y0 = start
        x1, y1 = end
        dx, dy = x1 - x0, y1 - y0
        length_sq = dx * dx + dy * dy
        if length_sq <= 1.0e-12:
            return math.hypot(px - x0, py - y0)
        t = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / length_sq))
        return math.hypot(px - (x0 + t * dx), py - (y0 + t * dy))

    def _update_title(self) -> None:
        path = self._current_path()
        dirty = " *" if path is not None and path.name in self.dirty_images else ""
        self.root.title(f"AniGroom Flow Brush{' - ' + path.name + dirty if path else ''}")

    def _on_close(self) -> None:
        path = self._current_path()
        if path is not None and path.name in self.dirty_images:
            if not self._save_current(silent=False) and not messagebox.askyesno("Unsaved annotations", "Close without saving?", parent=self.root):
                return
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Brush sparse directed seed flow over an image folder.")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def configure_windows_dpi() -> None:
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass


def main() -> None:
    args = parse_args()
    configure_windows_dpi()
    root = tk.Tk()
    FlowAnnotatorApp(root, input_dir=args.input_dir, output_dir=args.output_dir)
    root.mainloop()


if __name__ == "__main__":
    main()
