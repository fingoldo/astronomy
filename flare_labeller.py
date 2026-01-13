"""
Flare Labeller UI for Active Learning Pipeline

A simple GUI app to quickly label stellar flare candidates from the active learning output.
Displays images and allows rapid labeling with keyboard shortcuts.

Usage:
    python flare_labeller.py [folder_path]

    If no folder is provided, a folder picker dialog will open.
"""

import json
import logging
import os
import re
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Optional

from PIL import Image, ImageTk

# Configure logging
logger = logging.getLogger(__name__)

# Save file location - configurable via environment variable
# Default: ~/.flare_labeller_state.json
SAVE_FILE = Path(os.environ.get("FLARE_LABELLER_STATE_FILE", Path.home() / ".flare_labeller_state.json"))

# Required keys for valid saved state (schema validation)
_REQUIRED_STATE_KEYS = {"folder", "pos_indices", "neg_indices"}

# =============================================================================
# UI Constants
# =============================================================================

# Image display limits (pixels)
MAX_IMAGE_WIDTH = 1200
MAX_IMAGE_HEIGHT = 800

# Window defaults
DEFAULT_WINDOW_WIDTH = 1300
DEFAULT_WINDOW_HEIGHT = 950

# File rescan interval (milliseconds)
FILE_RESCAN_INTERVAL_MS = 20_000

# Probability threshold for including removed_SEED images
REMOVED_SEED_PROBABILITY_THRESHOLD = 50

# Pre-compiled regex patterns for performance
_ITER_PATTERN = re.compile(r"iter(\d+)")
_PROBABILITY_PATTERN = re.compile(r"_P(\d+\.?\d*)pct_")
_SOURCE_ID_PATTERN = re.compile(r"(?:added_|removed_)?([A-Z0-9]+_MJD[\d.]+)_row")
_ROW_INDEX_PATTERN = re.compile(r"_row(\d+)_")

# Search patterns for image discovery
_SEARCH_PATTERNS = (
    "*/pseudo_pos/*.png",
    "*/pseudo_neg/*.png",
    "*/top_candidates/*.png",
    "*/top_pos_candidates/*.png",
    "*/uncertain/*.png",
    "*/bootstrap_discarded_consensus/*.png",
    "*/bootstrap_discarded_variance/*.png",
    "*/bootstrap_discarded_both/*.png",
    "*/removed_PSEUDO_POS/*.png",
)


def _extract_from_pattern(
    pattern: re.Pattern,
    text: str,
    group: int = 1,
    converter: type | None = None,
) -> Optional[str | int | float]:
    """
    Generic pattern extraction from text.

    Parameters
    ----------
    pattern : re.Pattern
        Pre-compiled regex pattern
    text : str
        Text to search
    group : int
        Regex group to extract (default: 1)
    converter : type, optional
        Type to convert result to (e.g., int, float)

    Returns
    -------
    Optional[str | int | float]
        Extracted value or None if no match
    """
    match = pattern.search(text)
    if match is None:
        return None
    value = match.group(group)
    if converter is not None:
        try:
            return converter(value)
        except (ValueError, TypeError):
            return None
    return value


def _validate_state_schema(state: dict) -> bool:
    """
    Validate saved state has required keys and proper types.

    Parameters
    ----------
    state : dict
        Loaded state dictionary

    Returns
    -------
    bool
        True if state is valid
    """
    if not isinstance(state, dict):
        return False
    if not _REQUIRED_STATE_KEYS.issubset(state.keys()):
        return False
    # Validate types
    if not isinstance(state.get("folder"), str):
        return False
    if not isinstance(state.get("pos_indices"), list):
        return False
    if not isinstance(state.get("neg_indices"), list):
        return False
    return True


def _discover_image_files(
    folder: Path,
    seen_source_ids: set[str],
    extract_probability_fn: Callable[[Path], Optional[float]],
    extract_source_id_fn: Callable[[Path], Optional[str]],
) -> tuple[list[Path], set[str]]:
    """
    Discover labellable image files in a folder.

    Parameters
    ----------
    folder : Path
        Base folder to search
    seen_source_ids : set[str]
        IDs already seen (for deduplication)
    extract_probability_fn : Callable
        Function to extract probability from filename
    extract_source_id_fn : Callable
        Function to extract source ID from filename

    Returns
    -------
    tuple[list[Path], set[str]]
        (discovered_files, updated_seen_ids)
    """
    sample_plots = folder / "sample_plots"
    updated_seen_ids = seen_source_ids.copy()

    if sample_plots.exists():
        all_files = []
        for pattern in _SEARCH_PATTERNS:
            all_files.extend(sample_plots.glob(pattern))

        # Add removed_SEED only if probability > 50%
        for f in sample_plots.glob("*/removed_SEED/*.png"):
            prob = extract_probability_fn(f)
            if prob is not None and prob > REMOVED_SEED_PROBABILITY_THRESHOLD:
                all_files.append(f)
    else:
        # Direct folder mode
        all_files = list(folder.glob("*.png"))

    # Sort by iteration number, then by filename
    def sort_key(p: Path) -> tuple:
        match = _ITER_PATTERN.search(str(p))
        iter_num = int(match.group(1)) if match else 0
        return (iter_num, p.name)

    sorted_files = sorted(all_files, key=sort_key)

    # Deduplicate by source ID
    unique_files = []
    for f in sorted_files:
        source_id = extract_source_id_fn(f)
        if source_id is None or source_id not in updated_seen_ids:
            unique_files.append(f)
            if source_id:
                updated_seen_ids.add(source_id)

    return unique_files, updated_seen_ids


class FlareLabeller:
    """Simple UI for labeling flare candidates."""

    def __init__(
        self,
        root: tk.Tk,
        folder: Path,
        initial_seen_ids: Optional[set[str]] = None,
        batch_mode: bool = False,
        output_file: Optional[Path] = None,
        previous_labels: Optional[dict[str, dict]] = None,
    ):
        self.root = root
        self.folder = folder
        self.batch_mode = batch_mode
        self.output_file = output_file
        self.previous_labels = previous_labels or {}  # {idx_str: {"label": 0/1, "source": "seed"}}
        self.root.title(f"Flare Labeller - {folder.name}" + (" [BATCH MODE]" if batch_mode else ""))

        # State
        self.image_files: list[Path] = []
        self.current_index: int = 0
        self.pos_indices: list[int] = []
        self.neg_indices: list[int] = []
        self.current_photo: Optional[ImageTk.PhotoImage] = None
        self.seen_source_ids: set[str] = initial_seen_ids or set()  # Track all IDs we've seen
        self.labeled_indices: set[int] = set()  # Track which images have been labeled (for batch mode)
        self.finish_requested: bool = False  # Set when user clicks "Finish Labelling"

        # Find all images
        self._find_images()

        # Build UI
        self._build_ui()

        # Bind keyboard shortcuts
        self.root.bind("<Left>", lambda e: self._prev_image())
        self.root.bind("<Right>", lambda e: self._next_image())
        self.root.bind("<Up>", lambda e: self._prev_image())
        self.root.bind("<Down>", lambda e: self._next_image())
        self.root.bind("f", lambda e: self._label_flare())
        self.root.bind("F", lambda e: self._label_flare())
        self.root.bind("n", lambda e: self._label_not_flare())
        self.root.bind("N", lambda e: self._label_not_flare())
        self.root.bind("v", lambda e: self._label_flare())
        self.root.bind("V", lambda e: self._label_flare())
        self.root.bind("x", lambda e: self._label_not_flare())
        self.root.bind("X", lambda e: self._label_not_flare())
        self.root.bind("<space>", lambda e: self._next_image())

        # Save window geometry on close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Show first image
        if self.image_files:
            self._show_current_image()
        else:
            messagebox.showwarning("No Images", f"No images found in {folder}")

        # Start periodic rescan for new files (every 20 seconds)
        self._schedule_rescan()

    def _extract_probability(self, filepath: Path) -> Optional[float]:
        """Extract probability from filename like '..._P52.67pct_...'."""
        return _extract_from_pattern(_PROBABILITY_PATTERN, filepath.name, converter=float)

    def _extract_source_id(self, filepath: Path) -> Optional[str]:
        """Extract source ID from filename.

        Example: 'removed_ZTFDR567208300020011_MJD58795.44066_row58961985_P60.69pct_cleaned.png'
        Returns: 'ZTFDR567208300020011_MJD58795.44066'
        """
        return _extract_from_pattern(_SOURCE_ID_PATTERN, filepath.name)

    def _find_images(self) -> None:
        """Find all labellable images in the folder."""
        self.image_files, self.seen_source_ids = _discover_image_files(
            self.folder,
            self.seen_source_ids,
            self._extract_probability,
            self._extract_source_id,
        )

    def _rescan_for_new_files(self) -> None:
        """Periodically check for new files and add them to the list."""
        current_paths = set(self.image_files)

        new_files, self.seen_source_ids = _discover_image_files(
            self.folder,
            self.seen_source_ids,
            self._extract_probability,
            self._extract_source_id,
        )

        # Filter to only truly new files (not already in current list)
        new_files = [f for f in new_files if f not in current_paths]

        if new_files:
            self.image_files.extend(new_files)
            self._show_current_image()

        self._schedule_rescan()

    def _schedule_rescan(self) -> None:
        """Schedule the next rescan in 20 seconds."""
        self.root.after(FILE_RESCAN_INTERVAL_MS, self._rescan_for_new_files)

    def _extract_row_index(self, filepath: Path) -> Optional[int]:
        """Extract row index from filename like 'added_..._row47833_...'."""
        return _extract_from_pattern(_ROW_INDEX_PATTERN, filepath.name, converter=int)

    def _build_ui(self) -> None:
        """Build the main UI."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # Top info bar
        info_frame = ttk.Frame(main_frame)
        info_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        self.progress_label = ttk.Label(info_frame, text="0 / 0", font=("Arial", 12))
        self.progress_label.pack(side="left")

        # Folder name and open button (right side)
        self.folder_label = ttk.Label(info_frame, text=self.folder.name, font=("Arial", 11, "bold"))
        self.folder_label.pack(side="left", padx=(20, 5))
        ttk.Button(info_frame, text="Open Folder...", command=self._open_new_folder).pack(side="left", padx=5)

        # Filename entry (readonly but selectable/copyable)
        self.filename_var = tk.StringVar()
        self.filename_label = tk.Entry(
            info_frame,
            textvariable=self.filename_var,
            font=("Arial", 10),
            state="readonly",
            readonlybackground="SystemButtonFace",
            relief="flat",
            width=60,
        )
        self.filename_label.pack(side="right")

        # Warning label for samples with previous labels (orange background)
        self.warning_label = tk.Label(
            info_frame,
            text="",
            font=("Arial", 10, "bold"),
            bg="#ff9800",
            fg="white",
            padx=5,
            pady=2,
        )
        # Pack but hide initially - will be shown when needed
        self.warning_label.pack(side="right", padx=(0, 10))
        self.warning_label.pack_forget()

        # Image display
        self.image_frame = ttk.Frame(main_frame, relief="sunken", borderwidth=2)
        self.image_frame.grid(row=1, column=0, sticky="nsew", pady=5)
        self.image_frame.columnconfigure(0, weight=1)
        self.image_frame.rowconfigure(0, weight=1)

        self.image_label = ttk.Label(self.image_frame)
        self.image_label.grid(row=0, column=0)

        # Navigation buttons (aligned with image frame edges)
        nav_frame = ttk.Frame(main_frame)
        nav_frame.grid(row=2, column=0, sticky="ew", pady=10)

        self.prev_btn = ttk.Button(nav_frame, text="< Prev", command=self._prev_image)
        self.prev_btn.pack(side="left", padx=(0, 5))  # Left edge aligned with image frame
        self.next_btn = ttk.Button(nav_frame, text="Next >", command=self._next_image)
        self.next_btn.pack(side="left", padx=5)

        # Clear All button (right edge aligned with image frame)
        ttk.Button(nav_frame, text="Clear All", command=self._clear_labels).pack(side="right", padx=(5, 0))

        # Labeling buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=3, column=0, sticky="ew", pady=10)

        # Not Flare button (red)
        self.not_flare_btn = tk.Button(
            btn_frame,
            text="✖  Not Flare (N/X)",
            command=self._label_not_flare,
            bg="#ff6b6b",
            fg="white",
            font=("Arial", 14, "bold"),
            width=20,
            height=2
        )
        self.not_flare_btn.pack(side="left", padx=20, expand=True)

        # Flare button (green)
        self.flare_btn = tk.Button(
            btn_frame,
            text="✔  Flare (F/V)",
            command=self._label_flare,
            bg="#51cf66",
            fg="white",
            font=("Arial", 14, "bold"),
            width=20,
            height=2
        )
        self.flare_btn.pack(side="right", padx=20, expand=True)

        # Results section
        results_frame = ttk.LabelFrame(main_frame, text="Labeled Indices (comma-separated)", padding="10")
        results_frame.grid(row=4, column=0, sticky="ew", pady=10)
        results_frame.columnconfigure(1, weight=1)

        # Positive indices
        ttk.Label(results_frame, text="Flares (pos):", foreground="green").grid(row=0, column=0, sticky="w", padx=5)
        self.pos_text = tk.Text(results_frame, height=2, width=60, wrap="word")
        self.pos_text.grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Button(results_frame, text="Copy", command=lambda: self._copy_to_clipboard(self.pos_text)).grid(row=0, column=2, padx=5)

        # Negative indices
        ttk.Label(results_frame, text="Not Flares (neg):", foreground="red").grid(row=1, column=0, sticky="w", padx=5, pady=(5, 0))
        self.neg_text = tk.Text(results_frame, height=2, width=60, wrap="word")
        self.neg_text.grid(row=1, column=1, sticky="ew", padx=5, pady=(5, 0))
        ttk.Button(results_frame, text="Copy", command=lambda: self._copy_to_clipboard(self.neg_text)).grid(row=1, column=2, padx=5, pady=(5, 0))

        # Stats
        self.stats_label = ttk.Label(results_frame, text="Labeled: 0 flares, 0 not flares")
        self.stats_label.grid(row=2, column=0, columnspan=3, pady=(10, 0), sticky="w")

        # Keyboard shortcuts help
        help_text = "Shortcuts: F/V=Flare, N/X=NotFlare, Space/→=Next, ←=Prev"
        ttk.Label(main_frame, text=help_text, font=("Arial", 9), foreground="gray").grid(row=5, column=0, pady=(5, 0))

        # Finish Labelling button (only in batch mode)
        if self.batch_mode:
            finish_btn = tk.Button(
                main_frame,
                text="🏁 Finish Labelling",
                command=self._finish_labelling,
                bg="#ff9800",
                fg="white",
                font=("Arial", 12, "bold"),
                width=20,
                height=1,
            )
            finish_btn.grid(row=6, column=0, pady=(15, 5))

    def _show_current_image(self) -> None:
        """Display the current image."""
        if not self.image_files or self.current_index >= len(self.image_files):
            return

        filepath = self.image_files[self.current_index]

        # Update progress
        self.progress_label.config(text=f"{self.current_index + 1} / {len(self.image_files)}")

        # Update navigation button states
        self.prev_btn.config(state="normal" if self.current_index > 0 else "disabled")
        self.next_btn.config(state="normal" if self.current_index < len(self.image_files) - 1 else "disabled")

        # Update filename (show relative path from sample_plots)
        try:
            rel_path = filepath.relative_to(self.folder / "sample_plots")
            self.filename_var.set(str(rel_path))
        except ValueError:
            self.filename_var.set(filepath.name)

        # Check for previous label and show warning
        row_idx = self._extract_row_index(filepath)
        idx_str = str(row_idx) if row_idx is not None else None
        if idx_str and idx_str in self.previous_labels:
            prev_info = self.previous_labels[idx_str]
            prev_label = "FLARE" if prev_info.get("label") == 1 else "NOT-FLARE"
            prev_source = prev_info.get("source", "unknown")
            self.warning_label.config(text=f"⚠ Previously: {prev_label} ({prev_source})")
            self.warning_label.pack(side="right", padx=(0, 10))
        else:
            self.warning_label.pack_forget()

        # Load and display image
        try:
            img = Image.open(filepath)

            # Resize if too large
            if img.width > MAX_IMAGE_WIDTH or img.height > MAX_IMAGE_HEIGHT:
                ratio = min(MAX_IMAGE_WIDTH / img.width, MAX_IMAGE_HEIGHT / img.height)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            self.current_photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.current_photo)

            # Highlight if already labeled and focus appropriate button
            row_idx = self._extract_row_index(filepath)
            if row_idx in self.pos_indices:
                self.image_frame.config(style="Pos.TFrame")
                self.flare_btn.focus_set()
            elif row_idx in self.neg_indices:
                self.image_frame.config(style="Neg.TFrame")
                self.not_flare_btn.focus_set()
            else:
                self.image_frame.config(style="TFrame")

        except (FileNotFoundError, OSError, ValueError) as e:
            logger.warning(f"Error loading image {filepath}: {e}")
            self.image_label.config(text=f"Error loading image:\n{e}")

    def _next_image(self) -> None:
        """Go to next image."""
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self._show_current_image()

    def _prev_image(self) -> None:
        """Go to previous image."""
        if self.current_index > 0:
            self.current_index -= 1
            self._show_current_image()

    def _label_flare(self) -> None:
        """Label current image as flare."""
        if not self.image_files:
            return

        filepath = self.image_files[self.current_index]
        row_idx = self._extract_row_index(filepath)

        if row_idx is None:
            messagebox.showwarning("Error", f"Could not extract row index from: {filepath.name}")
            return

        # Check for conflict with previous label
        idx_str = str(row_idx)
        if idx_str in self.previous_labels:
            prev_info = self.previous_labels[idx_str]
            if prev_info.get("label") != 1:  # Was NOT-FLARE, now FLARE
                prev_source = prev_info.get("source", "unknown")
                logger.warning(f"Expert conflict: idx={row_idx}, was NOT-FLARE ({prev_source}), now FLARE")
                messagebox.showinfo(
                    "Label Conflict",
                    f"⚠ You changed idx={row_idx} from NOT-FLARE ({prev_source}) to FLARE.\n"
                    "This will override the previous label."
                )

        # Remove from neg if present
        if row_idx in self.neg_indices:
            self.neg_indices.remove(row_idx)

        # Add to pos if not present
        if row_idx not in self.pos_indices:
            self.pos_indices.append(row_idx)

        # Track labeled index for batch mode
        self.labeled_indices.add(self.current_index)

        self._update_text_boxes()
        self._next_image()

        # Check batch completion
        if self.batch_mode:
            self._check_batch_complete()

    def _label_not_flare(self) -> None:
        """Label current image as not flare."""
        if not self.image_files:
            return

        filepath = self.image_files[self.current_index]
        row_idx = self._extract_row_index(filepath)

        if row_idx is None:
            messagebox.showwarning("Error", f"Could not extract row index from: {filepath.name}")
            return

        # Check for conflict with previous label
        idx_str = str(row_idx)
        if idx_str in self.previous_labels:
            prev_info = self.previous_labels[idx_str]
            if prev_info.get("label") != 0:  # Was FLARE, now NOT-FLARE
                prev_source = prev_info.get("source", "unknown")
                logger.warning(f"Expert conflict: idx={row_idx}, was FLARE ({prev_source}), now NOT-FLARE")
                messagebox.showinfo(
                    "Label Conflict",
                    f"⚠ You changed idx={row_idx} from FLARE ({prev_source}) to NOT-FLARE.\n"
                    "This will override the previous label."
                )

        # Remove from pos if present
        if row_idx in self.pos_indices:
            self.pos_indices.remove(row_idx)

        # Add to neg if not present
        if row_idx not in self.neg_indices:
            self.neg_indices.append(row_idx)

        # Track labeled index for batch mode
        self.labeled_indices.add(self.current_index)

        self._update_text_boxes()
        self._next_image()

        # Check batch completion
        if self.batch_mode:
            self._check_batch_complete()

    def _update_text_boxes(self) -> None:
        """Update the text boxes with current indices."""
        # Update pos text
        self.pos_text.delete("1.0", tk.END)
        self.pos_text.insert("1.0", ", ".join(str(i) for i in self.pos_indices))

        # Update neg text
        self.neg_text.delete("1.0", tk.END)
        self.neg_text.insert("1.0", ", ".join(str(i) for i in self.neg_indices))

        # Update stats
        self.stats_label.config(text=f"Labeled: {len(self.pos_indices)} flares, {len(self.neg_indices)} not flares")

        # Auto-save state
        self._save_state()

    def _copy_to_clipboard(self, text_widget: tk.Text) -> None:
        """Copy text widget content to clipboard."""
        content = text_widget.get("1.0", tk.END).strip()
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        self.root.update()

    def _save_state(self) -> None:
        """Save current state to file for recovery."""
        # Get window geometry
        geometry = self.root.geometry()
        is_maximized = self.root.state() == "zoomed"

        state = {
            "folder": str(self.folder),
            "pos_indices": self.pos_indices,
            "neg_indices": self.neg_indices,
            "current_index": self.current_index,
            "seen_source_ids": list(self.seen_source_ids),
            "window_geometry": geometry,
            "window_maximized": is_maximized,
        }
        try:
            with open(SAVE_FILE, "w") as f:
                json.dump(state, f, indent=2)
        except (OSError, IOError) as e:
            logger.warning(f"Could not save state: {e}")

    def _on_close(self) -> None:
        """Handle window close - save state and exit."""
        self._save_state()
        # Write output file in batch mode
        if self.batch_mode and self.output_file:
            self._write_batch_output()
        self.root.destroy()

    def _check_batch_complete(self) -> None:
        """Check if all images are labeled in batch mode and auto-close."""
        if len(self.labeled_indices) >= len(self.image_files):
            logger.info(f"Batch complete: {len(self.pos_indices)} flares, {len(self.neg_indices)} not flares")
            self._on_close()

    def _finish_labelling(self) -> None:
        """
        Handle Finish Labelling button click.

        Asks for confirmation, then sets finish_requested flag and closes.
        This signals the pipeline to train one final model and finalize.
        """
        n_labeled = len(self.pos_indices) + len(self.neg_indices)
        n_total = len(self.image_files)
        n_unlabeled = n_total - len(self.labeled_indices)

        msg = (
            f"Finish labelling and finalize the pipeline?\n\n"
            f"Labeled: {n_labeled} samples ({len(self.pos_indices)} flares, {len(self.neg_indices)} not flares)\n"
            f"Unlabeled in current batch: {n_unlabeled}\n\n"
            f"This will:\n"
            f"• Submit current labels\n"
            f"• Train final model\n"
            f"• Save all outputs and finalize\n\n"
            f"Continue?"
        )

        if messagebox.askyesno("Finish Labelling", msg):
            self.finish_requested = True
            logger.info(f"Finish requested: {len(self.pos_indices)} flares, {len(self.neg_indices)} not flares")
            self._on_close()

    def _write_batch_output(self) -> None:
        """Write batch labeling results to output file."""
        if not self.output_file:
            return
        result = {
            "pos": self.pos_indices,
            "neg": self.neg_indices,
        }
        if self.finish_requested:
            result["finish"] = True
        try:
            with open(self.output_file, "w") as f:
                json.dump(result, f)
            logger.info(f"Results written to {self.output_file}")
        except (OSError, IOError) as e:
            logger.error(f"Error writing output file: {e}")

    def _clear_state_file(self) -> None:
        """Delete the save file."""
        try:
            if SAVE_FILE.exists():
                SAVE_FILE.unlink()
        except OSError as e:
            logger.warning(f"Could not delete save file: {e}")

    def _has_labels(self) -> bool:
        """Check if there are any labels."""
        return bool(self.pos_indices or self.neg_indices)

    def _confirm_clear(self) -> bool:
        """Ask user to confirm clearing labels. Returns True if confirmed."""
        if not self._has_labels():
            return True
        return messagebox.askyesno(
            "Confirm Clear",
            f"You have {len(self.pos_indices)} flares and {len(self.neg_indices)} not-flares labeled.\n\n"
            "Are you sure you want to clear all labels?",
            default=messagebox.NO
        )

    def _clear_labels(self) -> None:
        """Clear all labels after confirmation."""
        if not self._confirm_clear():
            return
        self.pos_indices.clear()
        self.neg_indices.clear()
        self._update_text_boxes()
        self._clear_state_file()
        self._show_current_image()  # Refresh to remove highlight

    def _open_new_folder(self) -> None:
        """Open a new folder and reload images."""
        # Ask confirmation if there are labels
        if not self._confirm_clear():
            return

        folder = filedialog.askdirectory(title="Select Active Learning Output Folder")
        if not folder:
            return

        new_folder = Path(folder)
        if not new_folder.exists():
            messagebox.showwarning("Error", f"Folder does not exist: {new_folder}")
            return

        # Update state
        self.folder = new_folder
        self.root.title(f"Flare Labeller - {new_folder.name}")
        self.folder_label.config(text=new_folder.name)

        # Clear labels and seen IDs
        self.pos_indices.clear()
        self.neg_indices.clear()
        self.seen_source_ids.clear()
        self._update_text_boxes()
        self._clear_state_file()

        # Find new images
        self.current_index = 0
        self._find_images()

        if self.image_files:
            self._show_current_image()
            self._save_state()  # Save new folder
        else:
            self.image_label.config(image="", text="No images found")
            self.progress_label.config(text="0 / 0")
            self.filename_var.set("")
            messagebox.showwarning("No Images", f"No images found in {new_folder}")


def load_saved_state() -> Optional[dict]:
    """
    Load saved state from file if it exists.

    Performs schema validation to ensure state has required keys
    and proper types before returning.
    """
    try:
        if SAVE_FILE.exists():
            with open(SAVE_FILE) as f:
                state = json.load(f)
            # Validate schema before returning
            if not _validate_state_schema(state):
                logger.warning("Saved state failed schema validation, ignoring")
                return None
            return state
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"Could not load saved state: {e}")
    return None


def select_folder() -> Optional[Path]:
    """Open folder picker dialog."""
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select Active Learning Output Folder")
    root.destroy()
    return Path(folder) if folder else None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Flare Labeller UI for Active Learning Pipeline")
    parser.add_argument("folder", nargs="?", help="Path to active learning output folder")
    parser.add_argument("--batch-mode", action="store_true", help="Auto-close after all samples labeled")
    parser.add_argument("--output-file", type=str, help="Write results to this JSON file on completion")
    parser.add_argument("--previous-labels", type=str, help="JSON file with previous labels for conflict warnings")
    args = parser.parse_args()

    batch_mode = args.batch_mode
    output_file = Path(args.output_file) if args.output_file else None

    # Load previous labels if provided
    previous_labels: dict[str, dict] = {}
    if args.previous_labels:
        try:
            with open(args.previous_labels) as f:
                previous_labels = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load previous labels: {e}")

    # Try to load saved state first (only in non-batch mode)
    saved_state = None if batch_mode else load_saved_state()

    # Get folder from command line, saved state, or picker
    if args.folder:
        folder = Path(args.folder)
        if not folder.exists():
            logger.error(f"Folder does not exist: {folder}")
            sys.exit(1)
        saved_state = None  # Don't restore labels if folder specified via CLI
    elif saved_state and Path(saved_state["folder"]).exists():
        folder = Path(saved_state["folder"])
        logger.info(f"Restoring session: {folder.name}")
    else:
        folder = select_folder()
        if not folder:
            logger.info("No folder selected.")
            sys.exit(0)
        saved_state = None

    # Create main window
    root = tk.Tk()

    # Restore window geometry from saved state, or use default
    if saved_state and "window_geometry" in saved_state:
        root.geometry(saved_state["window_geometry"])
        if saved_state.get("window_maximized"):
            root.state("zoomed")
    else:
        root.geometry(f"{DEFAULT_WINDOW_WIDTH}x{DEFAULT_WINDOW_HEIGHT}")

    # Create app (pass saved seen_source_ids so _find_images respects them)
    initial_seen_ids = set(saved_state.get("seen_source_ids", [])) if saved_state else None
    app = FlareLabeller(
        root,
        folder,
        initial_seen_ids,
        batch_mode=batch_mode,
        output_file=output_file,
        previous_labels=previous_labels,
    )

    # Restore labels from saved state
    if saved_state:
        app.pos_indices = saved_state.get("pos_indices", [])
        app.neg_indices = saved_state.get("neg_indices", [])
        app.current_index = min(saved_state.get("current_index", 0), len(app.image_files) - 1) if app.image_files else 0
        app._update_text_boxes()
        app._show_current_image()

    # Bring window to focus (especially important when launched from subprocess)
    root.lift()
    root.attributes('-topmost', True)
    root.after(100, lambda: root.attributes('-topmost', False))
    root.focus_force()

    # Run
    root.mainloop()


if __name__ == "__main__":
    main()
