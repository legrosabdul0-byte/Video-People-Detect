# -*- coding: utf-8 -*-
"""Tkinter UI for the video people-counter.

Two modes share one window:
- Single video: pick a file, get a count + confidence + preview.
- Folder (batch): pick a folder, every video is scanned with a high-recall
  preset and the results are shown in a table.

Detection runs on a background thread and talks to the GUI thread through a
thread-safe queue, so the window never freezes.
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Optional

from .batch import BatchItem, scan_folder
from .config import DetectionConfig
from .detector import DetectionResult, PeopleDetector


def open_in_default_viewer(path: str) -> None:
    """Open a file with the OS default application (cross-platform)."""
    if sys.platform.startswith("win"):
        os.startfile(path)  # type: ignore[attr-defined]  # Windows only
    elif sys.platform == "darwin":
        subprocess.run(["open", path], check=False)
    else:
        subprocess.run(["xdg-open", path], check=False)


class PeopleCounterApp:
    """Main application window."""

    def __init__(self, config: Optional[DetectionConfig] = None):
        # Single-video detector keeps the original (precision-leaning) config.
        self.config = config or DetectionConfig()
        self.detector = PeopleDetector(self.config)
        # Folder mode favours recall ("don't miss anyone").
        self.batch_detector = PeopleDetector(DetectionConfig.high_recall())

        self.ui_queue: "queue.Queue[tuple]" = queue.Queue()
        self.running = False
        self.last_preview_path = ""
        self._row_ids: list[str] = []

        self._build_ui()
        self._poll_queue()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        self.root = tk.Tk()
        self.root.title("People Counter v5")
        self.root.geometry("940x720")

        tk.Label(
            self.root, text="People Counter v5", font=("Arial", 20, "bold")
        ).pack(pady=8)

        button_row = tk.Frame(self.root)
        button_row.pack(pady=4)

        self.video_btn = tk.Button(
            button_row, text="Select Video", font=("Arial", 14), command=self.choose_video
        )
        self.video_btn.grid(row=0, column=0, padx=6)

        self.folder_btn = tk.Button(
            button_row,
            text="Select Folder (Batch)",
            font=("Arial", 14),
            command=self.choose_folder,
        )
        self.folder_btn.grid(row=0, column=1, padx=6)

        self.preview_btn = tk.Button(
            button_row,
            text="Open Preview",
            font=("Arial", 12),
            command=self.open_preview,
            state=tk.DISABLED,
        )
        self.preview_btn.grid(row=0, column=2, padx=6)

        self.progress = ttk.Progressbar(
            self.root, orient="horizontal", length=820, mode="determinate"
        )
        self.progress.pack(pady=8)

        self.result_label = tk.Label(self.root, text="", font=("Arial", 15, "bold"))
        self.result_label.pack(pady=4)

        # Results table for batch mode.
        table_frame = tk.Frame(self.root)
        table_frame.pack(fill="both", expand=True, padx=12, pady=(4, 6))

        columns = ("people", "confidence", "status")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="tree headings", height=8)
        self.tree.heading("#0", text="Video")
        self.tree.heading("people", text="People")
        self.tree.heading("confidence", text="Confidence")
        self.tree.heading("status", text="Status")
        self.tree.column("#0", width=420, anchor="w")
        self.tree.column("people", width=90, anchor="center")
        self.tree.column("confidence", width=110, anchor="center")
        self.tree.column("status", width=140, anchor="center")

        scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        # Detailed log.
        self.text = tk.Text(self.root, font=("Consolas", 10), height=10)
        self.text.pack(fill="both", expand=True, padx=12, pady=(0, 12))

    # ------------------------------------------------------------------ #
    # Actions
    # ------------------------------------------------------------------ #
    def _busy(self) -> bool:
        if self.running:
            messagebox.showwarning("Running", "A scan is already running.")
            return True
        return False

    def _set_controls(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        self.video_btn.config(state=state)
        self.folder_btn.config(state=state)

    def choose_video(self) -> None:
        if self._busy():
            return
        path = filedialog.askopenfilename(
            filetypes=[
                ("Video Files", "*.mp4 *.mov *.avi *.mkv *.m4v *.wmv *.flv *.webm"),
                ("All Files", "*.*"),
            ]
        )
        if not path:
            return
        self._reset_views()
        self._set_controls(False)
        self._log(f"Selected: {path}")
        threading.Thread(target=self._run_single, args=(path,), daemon=True).start()

    def choose_folder(self) -> None:
        if self._busy():
            return
        folder = filedialog.askdirectory()
        if not folder:
            return
        self._reset_views()
        self._set_controls(False)
        self._log(f"Selected folder: {folder}")
        threading.Thread(target=self._run_batch, args=(folder,), daemon=True).start()

    def open_preview(self) -> None:
        if not self.last_preview_path or not os.path.exists(self.last_preview_path):
            messagebox.showwarning("No Preview", "No preview image found.")
            return
        open_in_default_viewer(self.last_preview_path)

    def _reset_views(self) -> None:
        self.text.delete("1.0", tk.END)
        self.progress["value"] = 0
        self.result_label.config(text="")
        self.preview_btn.config(state=tk.DISABLED)
        for row in self.tree.get_children():
            self.tree.delete(row)
        self._row_ids = []

    # ------------------------------------------------------------------ #
    # Background workers
    # ------------------------------------------------------------------ #
    def _run_single(self, path: str) -> None:
        self.running = True
        try:
            result = self.detector.detect(
                path,
                progress=lambda v: self.ui_queue.put(("progress", v)),
                log=self._log,
            )
            self.ui_queue.put(("done", result))
        except Exception:
            self.ui_queue.put(("error", traceback.format_exc()))
        finally:
            self.running = False
            self.ui_queue.put(("controls", True))

    def _run_batch(self, folder: str) -> None:
        self.running = True
        try:
            scan_folder(
                folder,
                self.batch_detector,
                save_preview=False,
                log=self._log,
                on_found=lambda items: self.ui_queue.put(("batch_found", items)),
                on_item_start=lambda i, it: self.ui_queue.put(("batch_start", i)),
                on_item_done=lambda i, it: self.ui_queue.put(("batch_item", i, it)),
                on_progress=lambda v: self.ui_queue.put(("progress", v)),
            )
            self.ui_queue.put(("batch_done",))
        except Exception:
            self.ui_queue.put(("error", traceback.format_exc()))
        finally:
            self.running = False
            self.ui_queue.put(("controls", True))

    # ------------------------------------------------------------------ #
    # Thread-safe log helper
    # ------------------------------------------------------------------ #
    def _log(self, msg: str) -> None:
        self.ui_queue.put(("log", msg))

    # ------------------------------------------------------------------ #
    # Queue pump (runs on the GUI thread)
    # ------------------------------------------------------------------ #
    def _poll_queue(self) -> None:
        try:
            while True:
                item = self.ui_queue.get_nowait()
                kind = item[0]
                if kind == "log":
                    self.text.insert(tk.END, item[1] + "\n")
                    self.text.see(tk.END)
                elif kind == "progress":
                    self.progress["value"] = item[1]
                elif kind == "controls":
                    self._set_controls(item[1])
                elif kind == "done":
                    self._show_single_result(item[1])
                elif kind == "batch_found":
                    self._populate_table(item[1])
                elif kind == "batch_start":
                    self._mark_scanning(item[1])
                elif kind == "batch_item":
                    self._update_row(item[1], item[2])
                elif kind == "batch_done":
                    self._finish_batch()
                elif kind == "error":
                    self._show_error(item[1])
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    # ------------------------------------------------------------------ #
    # Single-video rendering
    # ------------------------------------------------------------------ #
    def _show_single_result(self, result: DetectionResult) -> None:
        self.last_preview_path = result.preview_path

        self.text.insert(tk.END, "-" * 40 + "\n")
        self.text.insert(tk.END, f"Raw Counts: {result.raw_counts}\n")
        self.text.insert(tk.END, f"Used Counts: {result.used_counts}\n")
        self.text.insert(tk.END, f"Final Percentile: {result.final_percentile}%\n")
        self.text.insert(tk.END, f"Final Count: {result.final_count}\n")
        self.text.insert(tk.END, f"Confidence: {result.confidence}%\n")
        if result.preview_path:
            self.text.insert(tk.END, f"Preview saved: {result.preview_path}\n")
            self.preview_btn.config(state=tk.NORMAL)
        self.text.see(tk.END)

        self.result_label.config(
            text=f"Final Count: {result.final_count}    Confidence: {result.confidence}%"
        )
        messagebox.showinfo(
            "Result",
            f"Final People Count: {result.final_count}\nConfidence: {result.confidence}%",
        )

    # ------------------------------------------------------------------ #
    # Batch rendering
    # ------------------------------------------------------------------ #
    def _populate_table(self, items: list[BatchItem]) -> None:
        self._row_ids = []
        for it in items:
            row_id = self.tree.insert("", tk.END, text=it.name, values=("", "", "Pending"))
            self._row_ids.append(row_id)
        self.result_label.config(text=f"Scanning {len(items)} video(s)... (high-recall mode)")
        if not items:
            messagebox.showinfo("No Videos", "No video files found in that folder.")

    def _mark_scanning(self, index: int) -> None:
        if 0 <= index < len(self._row_ids):
            self.tree.item(self._row_ids[index], values=("", "", "Scanning..."))
            self.tree.see(self._row_ids[index])

    def _update_row(self, index: int, item: BatchItem) -> None:
        if not (0 <= index < len(self._row_ids)):
            return
        row_id = self._row_ids[index]
        if item.ok and item.result is not None:
            self.tree.item(
                row_id,
                values=(item.result.final_count, f"{item.result.confidence}%", "Done"),
            )
        else:
            self.tree.item(row_id, values=("-", "-", "Error"))

    def _finish_batch(self) -> None:
        done = sum(
            1
            for r in self._row_ids
            if self.tree.item(r, "values") and self.tree.item(r, "values")[2] == "Done"
        )
        total = len(self._row_ids)
        self.result_label.config(text=f"Scanned {done}/{total} video(s).")
        if total:
            messagebox.showinfo("Batch Complete", f"Scanned {done}/{total} video(s).")

    def _show_error(self, err: str) -> None:
        self.text.insert(tk.END, "\nERROR:\n" + err + "\n")
        self.text.see(tk.END)
        messagebox.showerror("Error", err[:1000])

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    PeopleCounterApp().run()


if __name__ == "__main__":
    main()
