from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import uuid

from PIL import Image, ImageTk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anigroom.flow_annotations import (  # noqa: E402
    FlowArrow,
    ImageFlowAnnotations,
    load_flow_annotations,
    make_flow_arrow,
    save_flow_annotations,
    sha256_file,
)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PROJECT_SCHEMA = "anigroom.sparse_flow.project.v1"


@dataclass(frozen=True)
class CanvasArrow:
    id: str
    start_px: tuple[float, float]
    end_px: tuple[float, float]
    confidence: float = 1.0


@dataclass(frozen=True)
class HistoryEntry:
    image_name: str
    before: tuple[CanvasArrow, ...]
    after: tuple[CanvasArrow, ...]


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
        self.after_id = self.widget.after(550, self._show)

    def _show(self) -> None:
        self.after_id = None
        if self.window is not None or not self.text:
            return
        x = self.widget.winfo_rootx() + self.widget.winfo_width() // 2
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 7
        window = tk.Toplevel(self.widget)
        window.wm_overrideredirect(True)
        window.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            window,
            text=self.text,
            background="#11151b",
            foreground="#e6eaf0",
            borderwidth=1,
            relief="solid",
            padx=7,
            pady=4,
            font=("Segoe UI", 9),
        )
        label.pack()
        self.window = window

    def _hide(self, _event: tk.Event | None = None) -> None:
        if self.after_id is not None:
            self.widget.after_cancel(self.after_id)
            self.after_id = None
        if self.window is not None:
            self.window.destroy()
            self.window = None


class FlowAnnotatorApp:
    def __init__(
        self,
        root: tk.Tk,
        *,
        input_dir: Path | None,
        output_dir: Path | None,
    ) -> None:
        self.root = root
        self.root.title("AniGroom Flow Anchors")
        self.root.geometry("1500x920")
        self.root.minsize(1040, 680)
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
        self.annotations: dict[str, list[CanvasArrow]] = {}
        self.dirty_images: set[str] = set()
        self.failed_annotations: dict[str, str] = {}
        self.selected_arrow_id: str | None = None

        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.fit_mode = True
        self.preview_start: tuple[float, float] | None = None
        self.preview_end: tuple[float, float] | None = None
        self.preview_items: list[int] = []
        self.pan_start: tuple[float, float] | None = None
        self.pan_origin: tuple[float, float] | None = None
        self.space_down = False

        self.undo_stack: list[HistoryEntry] = []
        self.redo_stack: list[HistoryEntry] = []
        self._updating_confidence = False

        self.status_var = tk.StringVar(value="No folder open")
        self.path_var = tk.StringVar(value="")
        self.count_var = tk.StringVar(value="0 arrows")
        self.confidence_var = tk.DoubleVar(value=1.0)

        self._build_layout()
        self._bind_events()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

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
        style.configure(
            "Toolbar.TButton",
            background="#232a34",
            foreground="#e7ebf0",
            bordercolor="#323b48",
            padding=(10, 7),
            font=("Segoe UI", 9),
        )
        style.map(
            "Toolbar.TButton",
            background=[("active", "#303946"), ("pressed", "#1f252e")],
            foreground=[("disabled", "#687180")],
        )
        style.configure(
            "Accent.TButton",
            background="#d72d73",
            foreground="#ffffff",
            bordercolor="#ec4a91",
            padding=(12, 7),
            font=("Segoe UI Semibold", 9),
        )
        style.map("Accent.TButton", background=[("active", "#e43a82"), ("pressed", "#b82461")])
        style.configure("Muted.TLabel", background="#171c23", foreground="#8993a1", font=("Segoe UI", 9))
        style.configure("Title.TLabel", background="#171c23", foreground="#f3f5f8", font=("Segoe UI Semibold", 10))
        style.configure("Status.TFrame", background="#10151b")
        style.configure("Status.TLabel", background="#10151b", foreground="#aab2bd", font=("Segoe UI", 9))
        style.configure("TScale", background="#171c23", troughcolor="#303946")

    def _build_layout(self) -> None:
        top = ttk.Frame(self.root, style="Panel.TFrame", padding=(10, 8))
        top.pack(side=tk.TOP, fill=tk.X)

        self._toolbar_button(top, "Open", self.choose_input_folder, "Open an image folder")
        self._toolbar_button(top, "Output", self.choose_output_folder, "Choose where per-image JSON files are saved")
        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        self._toolbar_button(top, "Prev", lambda: self.navigate(-1), "Previous image (A or Left)")
        self._toolbar_button(top, "Next", lambda: self.navigate(1), "Next image (D or Right)")
        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        self.undo_button = self._toolbar_button(top, "Undo", self.undo, "Undo the last edit (Ctrl+Z)")
        self.redo_button = self._toolbar_button(top, "Redo", self.redo, "Redo the last edit (Ctrl+Y)")
        self.delete_button = self._toolbar_button(top, "Delete", self.delete_selected, "Delete the selected arrow")
        self.clear_button = self._toolbar_button(top, "Clear", self.clear_current, "Remove every arrow from this image")
        self.save_button = self._toolbar_button(top, "Save", self.save_current, "Save this image and the project index", accent=True)

        path_label = tk.Label(
            top,
            textvariable=self.path_var,
            background="#171c23",
            foreground="#7f8997",
            anchor="e",
            font=("Segoe UI", 9),
        )
        path_label.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(18, 4))

        body = ttk.Frame(self.root, style="App.TFrame")
        body.pack(fill=tk.BOTH, expand=True)
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)

        left = ttk.Frame(body, style="Panel.TFrame", width=270, padding=(9, 10))
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 1))
        left.grid_propagate(False)
        ttk.Label(left, text="Images", style="Title.TLabel").pack(anchor="w", padx=4, pady=(0, 8))
        image_list_frame = ttk.Frame(left, style="Panel.TFrame")
        image_list_frame.pack(fill=tk.BOTH, expand=True)
        self.image_list = tk.Listbox(
            image_list_frame,
            activestyle="none",
            background="#141920",
            foreground="#cfd5dd",
            selectbackground="#9d2557",
            selectforeground="#ffffff",
            highlightthickness=1,
            highlightbackground="#29313c",
            borderwidth=0,
            font=("Cascadia Mono", 9),
            exportselection=False,
        )
        image_scroll = ttk.Scrollbar(image_list_frame, orient=tk.VERTICAL, command=self.image_list.yview)
        self.image_list.configure(yscrollcommand=image_scroll.set)
        self.image_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        image_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        center = ttk.Frame(body, style="App.TFrame")
        center.grid(row=0, column=1, sticky="nsew")
        center.grid_rowconfigure(0, weight=1)
        center.grid_columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(
            center,
            background="#0b0f14",
            highlightthickness=0,
            cursor="crosshair",
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")

        right = ttk.Frame(body, style="Panel.TFrame", width=400, padding=(9, 10))
        right.grid(row=0, column=2, sticky="nsew", padx=(1, 0))
        right.grid_propagate(False)
        header = ttk.Frame(right, style="Panel.TFrame")
        header.pack(fill=tk.X, padx=4, pady=(0, 8))
        ttk.Label(header, text="Anchors", style="Title.TLabel").pack(side=tk.LEFT)
        ttk.Label(header, textvariable=self.count_var, style="Muted.TLabel").pack(side=tk.RIGHT)

        arrow_list_frame = ttk.Frame(right, style="Panel.TFrame")
        arrow_list_frame.pack(fill=tk.BOTH, expand=True)
        self.arrow_list = tk.Listbox(
            arrow_list_frame,
            activestyle="none",
            background="#141920",
            foreground="#cfd5dd",
            selectbackground="#1f8094",
            selectforeground="#ffffff",
            highlightthickness=1,
            highlightbackground="#29313c",
            borderwidth=0,
            font=("Cascadia Mono", 8),
            exportselection=False,
        )
        arrow_scroll = ttk.Scrollbar(arrow_list_frame, orient=tk.VERTICAL, command=self.arrow_list.yview)
        self.arrow_list.configure(yscrollcommand=arrow_scroll.set)
        self.arrow_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        arrow_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        confidence_panel = ttk.Frame(right, style="Panel.TFrame", padding=(4, 12, 4, 3))
        confidence_panel.pack(fill=tk.X)
        confidence_header = ttk.Frame(confidence_panel, style="Panel.TFrame")
        confidence_header.pack(fill=tk.X)
        ttk.Label(confidence_header, text="Confidence", style="Muted.TLabel").pack(side=tk.LEFT)
        self.confidence_value = ttk.Label(confidence_header, text="1.00", style="Muted.TLabel")
        self.confidence_value.pack(side=tk.RIGHT)
        self.confidence_scale = ttk.Scale(
            confidence_panel,
            from_=0.1,
            to=1.0,
            variable=self.confidence_var,
            command=self._on_confidence_change,
        )
        self.confidence_scale.pack(fill=tk.X, pady=(7, 0))
        ToolTip(self.confidence_scale, "Confidence stored with the selected root-to-tip anchor")

        status = ttk.Frame(self.root, style="Status.TFrame", padding=(10, 5))
        status.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(status, textvariable=self.status_var, style="Status.TLabel").pack(side=tk.LEFT)
        self.zoom_label = ttk.Label(status, text="100%", style="Status.TLabel")
        self.zoom_label.pack(side=tk.RIGHT)

    def _toolbar_button(
        self,
        parent: ttk.Frame,
        text: str,
        command,
        tooltip: str,
        *,
        accent: bool = False,
    ) -> ttk.Button:
        button = ttk.Button(
            parent,
            text=text,
            command=command,
            style="Accent.TButton" if accent else "Toolbar.TButton",
        )
        button.pack(side=tk.LEFT, padx=2)
        ToolTip(button, tooltip)
        return button

    def _bind_events(self) -> None:
        self.image_list.bind("<<ListboxSelect>>", self._on_image_list_select)
        self.arrow_list.bind("<<ListboxSelect>>", self._on_arrow_list_select)
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.bind("<ButtonPress-1>", self._on_draw_press)
        self.canvas.bind("<B1-Motion>", self._on_draw_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_draw_release)
        self.canvas.bind("<ButtonPress-2>", self._on_pan_press)
        self.canvas.bind("<B2-Motion>", self._on_pan_motion)
        self.canvas.bind("<ButtonRelease-2>", self._on_pan_release)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", lambda event: self._zoom_at(1.15, event.x, event.y))
        self.canvas.bind("<Button-5>", lambda event: self._zoom_at(1.0 / 1.15, event.x, event.y))

        self.root.bind("<Control-s>", lambda _event: self.save_current())
        self.root.bind("<Control-z>", lambda _event: self.undo())
        self.root.bind("<Control-y>", lambda _event: self.redo())
        self.root.bind("<Delete>", lambda _event: self.delete_selected())
        self.root.bind("<BackSpace>", lambda _event: self.delete_selected())
        self.root.bind("<Left>", lambda _event: self.navigate(-1))
        self.root.bind("<Right>", lambda _event: self.navigate(1))
        self.root.bind("<KeyPress-space>", self._on_space_press)
        self.root.bind("<KeyRelease-space>", self._on_space_release)
        self.root.bind("<KeyPress-a>", lambda _event: self.navigate(-1))
        self.root.bind("<KeyPress-d>", lambda _event: self.navigate(1))
        self.root.bind("<KeyPress-f>", lambda _event: self.fit_image())

    def choose_input_folder(self) -> None:
        initial = str(self.input_dir or Path.cwd())
        value = filedialog.askdirectory(parent=self.root, title="Open image folder", initialdir=initial)
        if value:
            self.open_folder(Path(value))

    def choose_output_folder(self) -> None:
        initial = str(self.output_dir or self.input_dir or Path.cwd())
        value = filedialog.askdirectory(parent=self.root, title="Choose annotation folder", initialdir=initial)
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
        images = sorted(
            path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            messagebox.showerror("No images", f"No supported images found in {directory}", parent=self.root)
            return
        self._save_current(silent=True)
        self.input_dir = directory
        self.image_paths = images
        self.current_index = 0
        self.output_dir = (
            output_dir.resolve()
            if output_dir is not None
            else directory.parent / f"{directory.name}_flow_guidance"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.annotations.clear()
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
                loaded = load_flow_annotations(
                    annotation_path,
                    image_path=image_path,
                    verify_image=True,
                )
                self.annotations[image_path.name] = [
                    CanvasArrow(
                        id=arrow.id,
                        start_px=tuple(float(v) for v in arrow.start_px),
                        end_px=tuple(float(v) for v in arrow.end_px),
                        confidence=float(arrow.confidence),
                    )
                    for arrow in loaded.arrows
                ]
            except Exception as exc:  # surfaced in the list and status bar
                self.failed_annotations[image_path.name] = str(exc)

    def _annotation_path(self, image_path: Path) -> Path:
        if self.output_dir is None:
            raise RuntimeError("annotation output directory is not set")
        return self.output_dir / f"{image_path.stem}.flow.json"

    def _current_path(self) -> Path | None:
        if 0 <= self.current_index < len(self.image_paths):
            return self.image_paths[self.current_index]
        return None

    def _current_arrows(self) -> list[CanvasArrow]:
        path = self._current_path()
        return self.annotations.setdefault(path.name, []) if path else []

    def _refresh_image_list(self) -> None:
        selected = self.current_index
        self.image_list.delete(0, tk.END)
        for path in self.image_paths:
            count = len(self.annotations.get(path.name, []))
            marker = "!" if path.name in self.failed_annotations else ("*" if path.name in self.dirty_images else " ")
            self.image_list.insert(tk.END, f"{marker} {path.name:<18} {count:>3}")
        self._select_image_list_row(selected)

    def _select_image_list_row(self, index: int) -> None:
        if not (0 <= index < self.image_list.size()):
            return
        self.image_list.selection_clear(0, tk.END)
        self.image_list.selection_set(index)
        self.image_list.activate(index)
        self.image_list.see(index)

    def _on_image_list_select(self, _event: tk.Event | None = None) -> None:
        selection = self.image_list.curselection()
        if not selection:
            return
        index = int(selection[0])
        if index == self.current_index:
            return
        self._save_current(silent=True)
        self.current_index = index
        self._load_current_image(force=True)

    def navigate(self, delta: int) -> None:
        if not self.image_paths:
            return
        index = max(0, min(len(self.image_paths) - 1, self.current_index + int(delta)))
        if index == self.current_index:
            return
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
        self.selected_arrow_id = None
        self.fit_mode = True
        self._fit_transform()
        self._render_scene(rebuild_image=True)
        self._refresh_arrow_list()
        self.path_var.set(f"{path.name}    ->    {self.output_dir}")
        if path.name in self.failed_annotations:
            self.status_var.set(f"Invalid existing JSON: {self.failed_annotations[path.name]}")
        else:
            self.status_var.set(f"Image {self.current_index + 1} of {len(self.image_paths)}")
        self._update_title()
        self.root.after(80, self.fit_image)

    def _fit_transform(self) -> None:
        if self.current_image is None:
            return
        canvas_width = max(self.canvas.winfo_width(), 2)
        canvas_height = max(self.canvas.winfo_height(), 2)
        image_width, image_height = self.current_image.size
        margin = 24
        self.scale = max(
            0.02,
            min((canvas_width - margin * 2) / image_width, (canvas_height - margin * 2) / image_height),
        )
        self.offset_x = (canvas_width - image_width * self.scale) * 0.5
        self.offset_y = (canvas_height - image_height * self.scale) * 0.5
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
            resized = self.current_image.resize((display_width, display_height), Image.Resampling.LANCZOS)
            self.current_photo = ImageTk.PhotoImage(resized)
            self.canvas.delete("all")
            self.current_image_item = self.canvas.create_image(
                self.offset_x,
                self.offset_y,
                anchor=tk.NW,
                image=self.current_photo,
                tags=("source_image",),
            )
        else:
            self.canvas.delete("arrow")
            self.canvas.delete("preview")
            if self.current_image_item is not None:
                self.canvas.coords(self.current_image_item, self.offset_x, self.offset_y)
        self._draw_arrows()
        self._draw_preview()
        self.canvas.tag_lower("source_image")
        self.zoom_label.configure(text=f"{self.scale * 100:.0f}%")

    def _draw_arrows(self) -> None:
        for arrow in self._current_arrows():
            x0, y0 = self.image_to_canvas(*arrow.start_px)
            x1, y1 = self.image_to_canvas(*arrow.end_px)
            selected = arrow.id == self.selected_arrow_id
            color = "#34d8ef" if selected else "#ff2f85"
            self.canvas.create_line(
                x0,
                y0,
                x1,
                y1,
                fill="#080a0d",
                width=7 if selected else 6,
                arrow=tk.LAST,
                arrowshape=(15, 18, 7),
                tags=("arrow", f"arrow:{arrow.id}"),
            )
            self.canvas.create_line(
                x0,
                y0,
                x1,
                y1,
                fill=color,
                width=3 if selected else 2,
                arrow=tk.LAST,
                arrowshape=(13, 16, 6),
                tags=("arrow", f"arrow:{arrow.id}"),
            )
            radius = 4 if selected else 3
            self.canvas.create_oval(
                x0 - radius,
                y0 - radius,
                x0 + radius,
                y0 + radius,
                fill=color,
                outline="#080a0d",
                width=1,
                tags=("arrow", f"arrow:{arrow.id}"),
            )

    def _draw_preview(self) -> None:
        if self.preview_start is None or self.preview_end is None:
            return
        x0, y0 = self.image_to_canvas(*self.preview_start)
        x1, y1 = self.image_to_canvas(*self.preview_end)
        self.canvas.create_line(
            x0,
            y0,
            x1,
            y1,
            fill="#0a0d11",
            width=7,
            arrow=tk.LAST,
            arrowshape=(15, 18, 7),
            tags=("preview",),
        )
        self.canvas.create_line(
            x0,
            y0,
            x1,
            y1,
            fill="#ffd166",
            width=3,
            arrow=tk.LAST,
            arrowshape=(13, 16, 6),
            tags=("preview",),
        )

    def image_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        return self.offset_x + float(x) * self.scale, self.offset_y + float(y) * self.scale

    def canvas_to_image(self, x: float, y: float, *, clamp: bool = False) -> tuple[float, float] | None:
        if self.current_image is None or self.scale <= 0:
            return None
        image_x = (float(x) - self.offset_x) / self.scale
        image_y = (float(y) - self.offset_y) / self.scale
        width, height = self.current_image.size
        if clamp:
            return (
                max(0.0, min(float(width - 1), image_x)),
                max(0.0, min(float(height - 1), image_y)),
            )
        if not (0.0 <= image_x <= width - 1 and 0.0 <= image_y <= height - 1):
            return None
        return image_x, image_y

    def _on_draw_press(self, event: tk.Event) -> None:
        self.canvas.focus_set()
        if self.space_down:
            self._start_pan(event.x, event.y)
            return
        point = self.canvas_to_image(event.x, event.y)
        if point is None:
            self.preview_start = None
            return
        self.preview_start = point
        self.preview_end = point
        self._render_scene(rebuild_image=False)

    def _on_draw_motion(self, event: tk.Event) -> None:
        if self.pan_start is not None:
            self._move_pan(event.x, event.y)
            return
        if self.preview_start is None:
            return
        point = self.canvas_to_image(event.x, event.y, clamp=True)
        if point is None:
            return
        self.preview_end = point
        self._render_scene(rebuild_image=False)

    def _on_draw_release(self, event: tk.Event) -> None:
        if self.pan_start is not None:
            self._finish_pan()
            return
        if self.preview_start is None:
            return
        point = self.canvas_to_image(event.x, event.y, clamp=True)
        start = self.preview_start
        self.preview_start = None
        self.preview_end = None
        if point is None or math.dist(start, point) < 2.0:
            self._render_scene(rebuild_image=False)
            return
        before = tuple(self._current_arrows())
        arrow = CanvasArrow(
            id=uuid.uuid4().hex[:16],
            start_px=start,
            end_px=point,
            confidence=float(self.confidence_var.get()),
        )
        self._current_arrows().append(arrow)
        self.selected_arrow_id = arrow.id
        self._record_history(before, tuple(self._current_arrows()))
        self._mark_dirty()
        self._refresh_arrow_list()
        self._render_scene(rebuild_image=False)

    def _on_pan_press(self, event: tk.Event) -> None:
        self._start_pan(event.x, event.y)

    def _on_pan_motion(self, event: tk.Event) -> None:
        self._move_pan(event.x, event.y)

    def _on_pan_release(self, _event: tk.Event) -> None:
        self._finish_pan()

    def _start_pan(self, x: float, y: float) -> None:
        self.pan_start = (float(x), float(y))
        self.pan_origin = (self.offset_x, self.offset_y)
        self.fit_mode = False
        self.canvas.configure(cursor="fleur")

    def _move_pan(self, x: float, y: float) -> None:
        if self.pan_start is None or self.pan_origin is None:
            return
        self.offset_x = self.pan_origin[0] + float(x) - self.pan_start[0]
        self.offset_y = self.pan_origin[1] + float(y) - self.pan_start[1]
        self._render_scene(rebuild_image=False)

    def _finish_pan(self) -> None:
        self.pan_start = None
        self.pan_origin = None
        self.canvas.configure(cursor="crosshair")

    def _on_space_press(self, _event: tk.Event) -> None:
        self.space_down = True
        self.canvas.configure(cursor="fleur")

    def _on_space_release(self, _event: tk.Event) -> None:
        self.space_down = False
        if self.pan_start is None:
            self.canvas.configure(cursor="crosshair")

    def _on_mousewheel(self, event: tk.Event) -> None:
        factor = 1.15 if event.delta > 0 else 1.0 / 1.15
        self._zoom_at(factor, event.x, event.y)

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

    def _on_canvas_resize(self, _event: tk.Event | None = None) -> None:
        if self.current_image is not None and self.fit_mode:
            self.root.after_idle(self.fit_image)

    def _on_right_click(self, event: tk.Event) -> None:
        arrow = self._nearest_arrow(event.x, event.y, max_distance=12.0)
        if arrow is None:
            self.selected_arrow_id = None
            self._refresh_arrow_list()
            self._render_scene(rebuild_image=False)
            return
        self.selected_arrow_id = arrow.id
        self._refresh_arrow_list()
        self._render_scene(rebuild_image=False)

    def _nearest_arrow(self, x: float, y: float, *, max_distance: float) -> CanvasArrow | None:
        best: tuple[float, CanvasArrow] | None = None
        for arrow in self._current_arrows():
            start = self.image_to_canvas(*arrow.start_px)
            end = self.image_to_canvas(*arrow.end_px)
            distance = self._point_segment_distance((float(x), float(y)), start, end)
            if distance <= max_distance and (best is None or distance < best[0]):
                best = distance, arrow
        return best[1] if best else None

    @staticmethod
    def _point_segment_distance(
        point: tuple[float, float],
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> float:
        px, py = point
        x0, y0 = start
        x1, y1 = end
        dx, dy = x1 - x0, y1 - y0
        length_sq = dx * dx + dy * dy
        if length_sq <= 1.0e-12:
            return math.hypot(px - x0, py - y0)
        t = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / length_sq))
        return math.hypot(px - (x0 + t * dx), py - (y0 + t * dy))

    def _refresh_arrow_list(self) -> None:
        arrows = self._current_arrows()
        self.arrow_list.delete(0, tk.END)
        selected_index = None
        for index, arrow in enumerate(arrows):
            self.arrow_list.insert(
                tk.END,
                f"{index + 1:02d}  {arrow.start_px[0]:5.0f},{arrow.start_px[1]:4.0f}  ->  "
                f"{arrow.end_px[0]:5.0f},{arrow.end_px[1]:4.0f}   {arrow.confidence:.2f}",
            )
            if arrow.id == self.selected_arrow_id:
                selected_index = index
        if selected_index is not None:
            self.arrow_list.selection_set(selected_index)
            self.arrow_list.see(selected_index)
            arrow = arrows[selected_index]
            self._updating_confidence = True
            self.confidence_var.set(arrow.confidence)
            self.confidence_value.configure(text=f"{arrow.confidence:.2f}")
            self._updating_confidence = False
        self.count_var.set(f"{len(arrows)} arrow{'s' if len(arrows) != 1 else ''}")

    def _on_arrow_list_select(self, _event: tk.Event | None = None) -> None:
        selection = self.arrow_list.curselection()
        arrows = self._current_arrows()
        if not selection or int(selection[0]) >= len(arrows):
            return
        arrow = arrows[int(selection[0])]
        self.selected_arrow_id = arrow.id
        self._updating_confidence = True
        self.confidence_var.set(arrow.confidence)
        self.confidence_value.configure(text=f"{arrow.confidence:.2f}")
        self._updating_confidence = False
        self._render_scene(rebuild_image=False)

    def _on_confidence_change(self, value: str) -> None:
        confidence = max(0.1, min(1.0, float(value)))
        self.confidence_value.configure(text=f"{confidence:.2f}")
        if self._updating_confidence or self.selected_arrow_id is None:
            return
        arrows = self._current_arrows()
        index = next((i for i, arrow in enumerate(arrows) if arrow.id == self.selected_arrow_id), None)
        if index is None or abs(arrows[index].confidence - confidence) < 0.005:
            return
        before = tuple(arrows)
        old = arrows[index]
        arrows[index] = CanvasArrow(old.id, old.start_px, old.end_px, confidence)
        self._record_history(before, tuple(arrows))
        self._mark_dirty()
        self._refresh_arrow_list()

    def delete_selected(self) -> None:
        if self.selected_arrow_id is None:
            return
        arrows = self._current_arrows()
        before = tuple(arrows)
        remaining = [arrow for arrow in arrows if arrow.id != self.selected_arrow_id]
        if len(remaining) == len(arrows):
            return
        arrows[:] = remaining
        self.selected_arrow_id = None
        self._record_history(before, tuple(arrows))
        self._mark_dirty()
        self._refresh_arrow_list()
        self._render_scene(rebuild_image=False)

    def clear_current(self) -> None:
        arrows = self._current_arrows()
        if not arrows:
            return
        if not messagebox.askyesno("Clear arrows", "Remove every arrow from this image?", parent=self.root):
            return
        before = tuple(arrows)
        arrows.clear()
        self.selected_arrow_id = None
        self._record_history(before, tuple())
        self._mark_dirty()
        self._refresh_arrow_list()
        self._render_scene(rebuild_image=False)

    def _record_history(self, before: tuple[CanvasArrow, ...], after: tuple[CanvasArrow, ...]) -> None:
        path = self._current_path()
        if path is None or before == after:
            return
        self.undo_stack.append(HistoryEntry(path.name, before, after))
        self.redo_stack.clear()
        self._update_history_buttons()

    def undo(self) -> None:
        if not self.undo_stack:
            return
        entry = self.undo_stack.pop()
        self.redo_stack.append(entry)
        self._apply_history(entry.image_name, entry.before)

    def redo(self) -> None:
        if not self.redo_stack:
            return
        entry = self.redo_stack.pop()
        self.undo_stack.append(entry)
        self._apply_history(entry.image_name, entry.after)

    def _apply_history(self, image_name: str, arrows: tuple[CanvasArrow, ...]) -> None:
        self.annotations[image_name] = list(arrows)
        self.dirty_images.add(image_name)
        if self._current_path() is None or self._current_path().name != image_name:
            index = next(i for i, path in enumerate(self.image_paths) if path.name == image_name)
            self.current_index = index
            self._select_image_list_row(index)
            self._load_current_image(force=True)
        else:
            self.selected_arrow_id = None
            self._refresh_arrow_list()
            self._render_scene(rebuild_image=False)
        self._refresh_image_list()
        self._update_history_buttons()
        self._update_title()

    def _update_history_buttons(self) -> None:
        self.undo_button.state(["!disabled"] if self.undo_stack else ["disabled"])
        self.redo_button.state(["!disabled"] if self.redo_stack else ["disabled"])

    def _mark_dirty(self) -> None:
        path = self._current_path()
        if path is None:
            return
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
            width, height = self.current_image.size
            flow_arrows: list[FlowArrow] = [
                make_flow_arrow(
                    arrow_id=arrow.id,
                    start_px=arrow.start_px,
                    end_px=arrow.end_px,
                    width=width,
                    height=height,
                    confidence=arrow.confidence,
                )
                for arrow in self._current_arrows()
            ]
            image_hash = self.image_hash_cache.setdefault(path.name, sha256_file(path))
            annotations = ImageFlowAnnotations(
                image_filename=path.name,
                width=width,
                height=height,
                sha256=image_hash,
                arrows=tuple(flow_arrows),
            )
            save_flow_annotations(self._annotation_path(path), annotations)
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
            "arrow_semantics": "root_to_tip",
            "images": [
                {
                    "filename": path.name,
                    "annotation": f"{path.stem}.flow.json",
                    "arrow_count": len(self.annotations.get(path.name, [])),
                }
                for path in self.image_paths
            ],
        }
        destination = self.output_dir / "project.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(destination)

    def _update_title(self) -> None:
        path = self._current_path()
        dirty = " *" if path is not None and path.name in self.dirty_images else ""
        suffix = f" - {path.name}{dirty}" if path else ""
        self.root.title(f"AniGroom Flow Anchors{suffix}")

    def _on_close(self) -> None:
        path = self._current_path()
        if path is not None and path.name in self.dirty_images:
            if not self._save_current(silent=False):
                if not messagebox.askyesno("Unsaved annotations", "Close without saving?", parent=self.root):
                    return
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate sparse root-to-tip fur-flow arrows on image folders.")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def configure_windows_dpi() -> None:
    if sys.platform != "win32":
        return
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
