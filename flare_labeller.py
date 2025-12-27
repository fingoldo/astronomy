"""
Flare Labeller UI for Active Learning Pipeline

A simple GUI app to quickly label stellar flare candidates from the active learning output.
Displays images and allows rapid labeling with keyboard shortcuts.

Usage:
    python flare_labeller.py [folder_path]

    If no folder is provided, a folder picker dialog will open.
"""

import json
import re
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

from PIL import Image, ImageTk

# Save file location
SAVE_FILE = Path.home() / ".flare_labeller_state.json"


class FlareLabeller:
    """Simple UI for labeling flare candidates."""

    def __init__(
        self,
        root: tk.Tk,
        folder: Path,
        initial_seen_ids: Optional[set[str]] = None,
        batch_mode: bool = False,
        output_file: Optional[Path] = None,
    ):
        self.root = root
        self.folder = folder
        self.batch_mode = batch_mode
        self.output_file = output_file
        self.root.title(f"Flare Labeller - {folder.name}" + (" [BATCH MODE]" if batch_mode else ""))

        # State
        self.image_files: list[Path] = []
        self.current_index: int = 0
        self.pos_indices: list[int] = []
        self.neg_indices: list[int] = []
        self.current_photo: Optional[ImageTk.PhotoImage] = None
        self.seen_source_ids: set[str] = initial_seen_ids or set()  # Track all IDs we've seen
        self.labeled_indices: set[int] = set()  # Track which images have been labeled (for batch mode)

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
        match = re.search(r"_P(\d+\.?\d*)pct_", filepath.name)
        if match:
            return float(match.group(1))
        return None

    def _extract_source_id(self, filepath: Path) -> Optional[str]:
        """Extract source ID from filename.

        Example: 'removed_ZTFDR567208300020011_MJD58795.44066_row58961985_P60.69pct_cleaned.png'
        Returns: 'ZTFDR567208300020011_MJD58795.44066'
        """
        # Pattern: optional prefix (added_, removed_, etc.) + ID + _row...
        match = re.search(r"(?:added_|removed_)?([A-Z0-9]+_MJD[\d.]+)_row", filepath.name)
        if match:
            return match.group(1)
        return None

    def _find_images(self) -> None:
        """Find all labellable images in the folder."""
        sample_plots = self.folder / "sample_plots"
        if not sample_plots.exists():
            return

        # Find all PNG files in sample_plots subdirectories
        patterns = [
            "*/pseudo_pos/*.png",
            "*/pseudo_neg/*.png",
            "*/top_candidates/*.png",
            "*/top_pos_candidates/*.png",
            "*/uncertain/*.png",  # Expert mode uncertain samples
            "*/bootstrap_discarded_consensus/*.png",
            "*/bootstrap_discarded_variance/*.png",
            "*/bootstrap_discarded_both/*.png",
            "*/removed_PSEUDO_POS/*.png",  # Always include removed pseudo-positives
        ]

        all_files = []
        for pattern in patterns:
            all_files.extend(sample_plots.glob(pattern))

        # Add removed_SEED only if probability > 50%
        for f in sample_plots.glob("*/removed_SEED/*.png"):
            prob = self._extract_probability(f)
            if prob is not None and prob > 50:
                all_files.append(f)

        # Sort by iteration number, then by filename
        def sort_key(p: Path) -> tuple:
            # Extract iteration number from path like iter030
            iter_match = re.search(r"iter(\d+)", str(p))
            iter_num = int(iter_match.group(1)) if iter_match else 0
            return (iter_num, p.name)

        sorted_files = sorted(all_files, key=sort_key)

        # Deduplicate by source ID - keep only the first occurrence of each ID
        # (same source may appear in multiple iterations as it gets added/removed)
        unique_files = []
        for f in sorted_files:
            source_id = self._extract_source_id(f)
            if source_id is None or source_id not in self.seen_source_ids:
                unique_files.append(f)
                if source_id:
                    self.seen_source_ids.add(source_id)

        self.image_files = unique_files

    def _rescan_for_new_files(self) -> None:
        """Periodically check for new files and add them to the list."""
        sample_plots = self.folder / "sample_plots"
        if not sample_plots.exists():
            self._schedule_rescan()
            return

        # Collect all candidate files (same logic as _find_images)
        patterns = [
            "*/pseudo_pos/*.png",
            "*/pseudo_neg/*.png",
            "*/top_candidates/*.png",
            "*/top_pos_candidates/*.png",
            "*/uncertain/*.png",  # Expert mode uncertain samples
            "*/bootstrap_discarded_consensus/*.png",
            "*/bootstrap_discarded_variance/*.png",
            "*/bootstrap_discarded_both/*.png",
            "*/removed_PSEUDO_POS/*.png",
        ]

        all_files = []
        for pattern in patterns:
            all_files.extend(sample_plots.glob(pattern))

        for f in sample_plots.glob("*/removed_SEED/*.png"):
            prob = self._extract_probability(f)
            if prob is not None and prob > 50:
                all_files.append(f)

        # Sort by iteration number, then by filename
        def sort_key(p: Path) -> tuple:
            iter_match = re.search(r"iter(\d+)", str(p))
            iter_num = int(iter_match.group(1)) if iter_match else 0
            return (iter_num, p.name)

        sorted_files = sorted(all_files, key=sort_key)

        # Find only new files (IDs we haven't seen)
        new_files = []
        for f in sorted_files:
            source_id = self._extract_source_id(f)
            if source_id is None or source_id not in self.seen_source_ids:
                new_files.append(f)
                if source_id:
                    self.seen_source_ids.add(source_id)

        # Append new files to the list
        if new_files:
            self.image_files.extend(new_files)
            # Update progress display to show new total
            self._show_current_image()

        # Schedule next rescan
        self._schedule_rescan()

    def _schedule_rescan(self) -> None:
        """Schedule the next rescan in 20 seconds."""
        self.root.after(20000, self._rescan_for_new_files)

    def _extract_row_index(self, filepath: Path) -> Optional[int]:
        """Extract row index from filename like 'added_..._row47833_...'."""
        match = re.search(r"_row(\d+)_", filepath.name)
        if match:
            return int(match.group(1))
        return None

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

        self.filename_label = ttk.Label(info_frame, text="", font=("Arial", 10))
        self.filename_label.pack(side="right")

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
            self.filename_label.config(text=str(rel_path))
        except ValueError:
            self.filename_label.config(text=filepath.name)

        # Load and display image
        try:
            img = Image.open(filepath)

            # Resize if too large (max 1200x800)
            max_width, max_height = 1200, 800
            if img.width > max_width or img.height > max_height:
                ratio = min(max_width / img.width, max_height / img.height)
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

        except Exception as e:
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
        except Exception as e:
            print(f"Warning: Could not save state: {e}")

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
            print(f"Batch complete: {len(self.pos_indices)} flares, {len(self.neg_indices)} not flares")
            self._on_close()

    def _write_batch_output(self) -> None:
        """Write batch labeling results to output file."""
        if not self.output_file:
            return
        result = {
            "pos": self.pos_indices,
            "neg": self.neg_indices,
        }
        try:
            with open(self.output_file, "w") as f:
                json.dump(result, f)
            print(f"Results written to {self.output_file}")
        except Exception as e:
            print(f"Error writing output file: {e}")

    def _clear_state_file(self) -> None:
        """Delete the save file."""
        try:
            if SAVE_FILE.exists():
                SAVE_FILE.unlink()
        except Exception as e:
            print(f"Warning: Could not delete save file: {e}")

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
            self.filename_label.config(text="")
            messagebox.showwarning("No Images", f"No images found in {new_folder}")


def load_saved_state() -> Optional[dict]:
    """Load saved state from file if it exists."""
    try:
        if SAVE_FILE.exists():
            with open(SAVE_FILE) as f:
                return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load saved state: {e}")
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
    args = parser.parse_args()

    batch_mode = args.batch_mode
    output_file = Path(args.output_file) if args.output_file else None

    # Try to load saved state first (only in non-batch mode)
    saved_state = None if batch_mode else load_saved_state()

    # Get folder from command line, saved state, or picker
    if args.folder:
        folder = Path(args.folder)
        if not folder.exists():
            print(f"Error: Folder does not exist: {folder}")
            sys.exit(1)
        saved_state = None  # Don't restore labels if folder specified via CLI
    elif saved_state and Path(saved_state["folder"]).exists():
        folder = Path(saved_state["folder"])
        print(f"Restoring session: {folder.name}")
    else:
        folder = select_folder()
        if not folder:
            print("No folder selected.")
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
        root.geometry("1300x950")

    # Create app (pass saved seen_source_ids so _find_images respects them)
    initial_seen_ids = set(saved_state.get("seen_source_ids", [])) if saved_state else None
    app = FlareLabeller(
        root,
        folder,
        initial_seen_ids,
        batch_mode=batch_mode,
        output_file=output_file,
    )

    # Restore labels from saved state
    if saved_state:
        app.pos_indices = saved_state.get("pos_indices", [])
        app.neg_indices = saved_state.get("neg_indices", [])
        app.current_index = min(saved_state.get("current_index", 0), len(app.image_files) - 1) if app.image_files else 0
        app._update_text_boxes()
        app._show_current_image()

    # Run
    root.mainloop()


if __name__ == "__main__":
    main()
