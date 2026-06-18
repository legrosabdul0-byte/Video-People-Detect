# -*- coding: utf-8 -*-
"""Tkinter UI for the video people-counter.

The UI runs detection on a background thread and communicates with the main
(GUI) thread through a thread-safe queue, so the window never freezes.
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
        self.config = config or DetectionConfig()
        self.detector = PeopleDetector(self.config)

        self.ui_queue: "queue.Queue[tuple]" = queue.Queue()
        self.running = False
        self.last_preview_path = ""

        self._build_ui()
        self._poll_queue()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        self.root = tk.Tk()
        self.root.title("People Counter v5")
        self.root.geometry("900x660")

        tk.Label(
            self.root, text="People Counter v5", font=("Arial", 20, "bold")
        ).pack(pady=10)

        tk.Button(
            self.root, text="Select Video", font=("Arial", 15), command=self.choose_video
        ).pack(pady=6)

        self.preview_btn = tk.Button(
            self.root,
            text="Open Preview",
            font=("Arial", 13),
            command=self.open_preview,
            state=tk.DISABLED,
        )
        self.preview_btn.pack(pady=4)

        self.progress = ttk.Progressbar(
            self.root, orient="horizontal", length=780, mode="determinate"
        )
        self.progress.pack(pady=8)

        self.result_label = tk.Label(self.root, text="", font=("Arial", 16, "bold"))
        self.result_label.pack(pady=8)

        self.text = tk.Text(self.root, font=("Consolas", 10))
        self.text.pack(fill="both", expand=True, padx=12, pady=12)

    # ------------------------------------------------------------------ #
    # Detection thread wiring
    # ------------------------------------------------------------------ #
    def choose_video(self) -> None:
        if self.running:
            messagebox.showwarning("Running", "Detection is already running.")
            return

        path = filedialog.askopenfilename(
            filetypes=[
                ("Video Files", "*.mp4 *.mov *.avi *.mkv"),
                ("All Files", "*.*"),
            ]
        )
        if not path:
            return

        self.text.delete("1.0", tk.END)
        self.progress["value"] = 0
        self.result_label.config(text="")
        self.preview_btn.config(state=tk.DISABLED)
        self._log(f"Selected: {path}")

        threading.Thread(target=self._run_detection, args=(path,), daemon=True).start()

    def _run_detection(self, path: str) -> None:
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

    # ------------------------------------------------------------------ #
    # Thread-safe callbacks
    # ------------------------------------------------------------------ #
    def _log(self, msg: str) -> None:
        self.ui_queue.put(("log", msg))

    def open_preview(self) -> None:
        if not self.last_preview_path or not os.path.exists(self.last_preview_path):
            messagebox.showwarning("No Preview", "No preview image found.")
            return
        open_in_default_viewer(self.last_preview_path)

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
                elif kind == "done":
                    self._show_result(item[1])
                elif kind == "error":
                    self._show_error(item[1])
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _show_result(self, result: DetectionResult) -> None:
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
