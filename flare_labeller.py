"""
Flare Labeller UI for Active Learning Pipeline

A simple GUI app to quickly label stellar flare candidates from the active learning output.
Displays images and allows rapid labeling with keyboard shortcuts.

Usage:
    python flare_labeller.py [folder_path]

    If no folder is provided, a folder picker dialog will open.
"""

import re
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

from PIL import Image, ImageTk


class FlareLabeller:
    """Simple UI for labeling flare candidates."""

    def __init__(self, root: tk.Tk, folder: Path):
        self.root = root
        self.folder = folder
        self.root.title(f"Flare Labeller - {folder.name}")

        # State
        self.image_files: list[Path] = []
        self.current_index: int = 0
        self.pos_indices: list[int] = []
        self.neg_indices: list[int] = []
        self.current_photo: Optional[ImageTk.PhotoImage] = None

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
        self.root.bind("u", lambda e: self._undo_last())
        self.root.bind("U", lambda e: self._undo_last())

        # Show first image
        if self.image_files:
            self._show_current_image()
        else:
            messagebox.showwarning("No Images", f"No images found in {folder}")

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
            "*/bootstrap_discarded_consensus/*.png",
            "*/bootstrap_discarded_variance/*.png",
            "*/bootstrap_discarded_both/*.png",
        ]

        all_files = []
        for pattern in patterns:
            all_files.extend(sample_plots.glob(pattern))

        # Sort by iteration number, then by filename
        def sort_key(p: Path) -> tuple:
            # Extract iteration number from path like iter030
            iter_match = re.search(r"iter(\d+)", str(p))
            iter_num = int(iter_match.group(1)) if iter_match else 0
            return (iter_num, p.name)

        self.image_files = sorted(all_files, key=sort_key)

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

        self.filename_label = ttk.Label(info_frame, text="", font=("Arial", 10))
        self.filename_label.pack(side="right")

        # Image display
        self.image_frame = ttk.Frame(main_frame, relief="sunken", borderwidth=2)
        self.image_frame.grid(row=1, column=0, sticky="nsew", pady=5)
        self.image_frame.columnconfigure(0, weight=1)
        self.image_frame.rowconfigure(0, weight=1)

        self.image_label = ttk.Label(self.image_frame)
        self.image_label.grid(row=0, column=0)

        # Navigation buttons
        nav_frame = ttk.Frame(main_frame)
        nav_frame.grid(row=2, column=0, sticky="ew", pady=10)

        ttk.Button(nav_frame, text="< Prev", command=self._prev_image).pack(side="left", padx=5)
        ttk.Button(nav_frame, text="Next >", command=self._next_image).pack(side="left", padx=5)

        # Labeling buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=3, column=0, sticky="ew", pady=10)

        # Not Flare button (red)
        self.not_flare_btn = tk.Button(
            btn_frame,
            text="X  Not Flare (N/X)",
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
            text="V  Flare (F/V)",
            command=self._label_flare,
            bg="#51cf66",
            fg="white",
            font=("Arial", 14, "bold"),
            width=20,
            height=2
        )
        self.flare_btn.pack(side="right", padx=20, expand=True)

        # Undo button
        ttk.Button(nav_frame, text="Undo (U)", command=self._undo_last).pack(side="right", padx=5)

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
        self.stats_label.grid(row=2, column=0, columnspan=3, pady=(10, 0))

        # Keyboard shortcuts help
        help_text = "Shortcuts: F/V=Flare, N/X=NotFlare, Space/→=Next, ←=Prev, U=Undo"
        ttk.Label(main_frame, text=help_text, font=("Arial", 9), foreground="gray").grid(row=5, column=0, pady=(5, 0))

    def _show_current_image(self) -> None:
        """Display the current image."""
        if not self.image_files or self.current_index >= len(self.image_files):
            return

        filepath = self.image_files[self.current_index]

        # Update progress
        self.progress_label.config(text=f"{self.current_index + 1} / {len(self.image_files)}")

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

            # Highlight if already labeled
            row_idx = self._extract_row_index(filepath)
            if row_idx in self.pos_indices:
                self.image_frame.config(style="Pos.TFrame")
            elif row_idx in self.neg_indices:
                self.image_frame.config(style="Neg.TFrame")
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

        self._update_text_boxes()
        self._next_image()

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

        self._update_text_boxes()
        self._next_image()

    def _undo_last(self) -> None:
        """Undo the last labeling action."""
        # Try to undo from pos first (most recent)
        if self.pos_indices:
            last_pos = self.pos_indices[-1]
            # Check if current image matches
            if self.current_index > 0:
                prev_filepath = self.image_files[self.current_index - 1]
                prev_row = self._extract_row_index(prev_filepath)
                if prev_row == last_pos:
                    self.pos_indices.pop()
                    self._update_text_boxes()
                    self._prev_image()
                    return

        if self.neg_indices:
            last_neg = self.neg_indices[-1]
            if self.current_index > 0:
                prev_filepath = self.image_files[self.current_index - 1]
                prev_row = self._extract_row_index(prev_filepath)
                if prev_row == last_neg:
                    self.neg_indices.pop()
                    self._update_text_boxes()
                    self._prev_image()
                    return

        # If no match, just go back
        self._prev_image()

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

    def _copy_to_clipboard(self, text_widget: tk.Text) -> None:
        """Copy text widget content to clipboard."""
        content = text_widget.get("1.0", tk.END).strip()
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        self.root.update()


def select_folder() -> Optional[Path]:
    """Open folder picker dialog."""
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select Active Learning Output Folder")
    root.destroy()
    return Path(folder) if folder else None


def main():
    # Get folder from command line or picker
    if len(sys.argv) > 1:
        folder = Path(sys.argv[1])
        if not folder.exists():
            print(f"Error: Folder does not exist: {folder}")
            sys.exit(1)
    else:
        folder = select_folder()
        if not folder:
            print("No folder selected.")
            sys.exit(0)

    # Create main window
    root = tk.Tk()
    root.geometry("1300x950")

    # Create app
    app = FlareLabeller(root, folder)

    # Run
    root.mainloop()


if __name__ == "__main__":
    main()
