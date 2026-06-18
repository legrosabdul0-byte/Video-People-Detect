# -*- coding: utf-8 -*-
"""Batch (folder) scanning: run the detector over every video in a folder.

The same loaded YOLO model is reused across every video, so scanning a folder
is much cheaper than running the single-video flow once per file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, List, Optional

from .detector import DetectionResult, PeopleDetector

# Common video container extensions.
VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv", ".flv", ".webm", ".mpeg", ".mpg",
}


@dataclass
class BatchItem:
    """One video in a batch scan and its outcome."""

    path: str
    name: str
    result: Optional[DetectionResult] = None
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.result is not None and not self.error


def find_videos(folder: str, recursive: bool = True) -> List[str]:
    """Return sorted paths of all video files in ``folder``."""
    found: List[str] = []
    if recursive:
        for root, _dirs, files in os.walk(folder):
            for name in files:
                if os.path.splitext(name)[1].lower() in VIDEO_EXTENSIONS:
                    found.append(os.path.join(root, name))
    else:
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if os.path.isfile(path) and os.path.splitext(name)[1].lower() in VIDEO_EXTENSIONS:
                found.append(path)
    return sorted(found)


def scan_folder(
    folder: str,
    detector: PeopleDetector,
    recursive: bool = True,
    save_preview: bool = False,
    log: Optional[Callable[[str], None]] = None,
    on_found: Optional[Callable[[List[BatchItem]], None]] = None,
    on_item_start: Optional[Callable[[int, BatchItem], None]] = None,
    on_item_done: Optional[Callable[[int, BatchItem], None]] = None,
    on_progress: Optional[Callable[[int], None]] = None,
) -> List[BatchItem]:
    """Scan every video in ``folder`` and return per-video results.

    Args:
        folder: Directory to scan.
        detector: A (possibly pre-configured) detector; its model is loaded
            once and reused for every video.
        recursive: Whether to descend into subfolders.
        save_preview: Save a per-video annotated preview next to each video.
        log: Optional log callback.
        on_found: Called once with the full item list as soon as videos are
            discovered (before any are processed).
        on_item_start / on_item_done: Per-video callbacks (index, item).
        on_progress: Overall progress callback (0-100, by videos completed).
    """
    videos = find_videos(folder, recursive=recursive)
    items = [BatchItem(path=p, name=os.path.relpath(p, folder)) for p in videos]

    if on_found:
        on_found(items)

    total = len(items)
    if total == 0:
        if log:
            log("No video files found in the selected folder.")
        return items

    if log:
        log(f"Found {total} video(s). Starting scan...")

    for index, item in enumerate(items):
        if on_item_start:
            on_item_start(index, item)
        if log:
            log(f"[{index + 1}/{total}] {item.name}")

        try:
            preview_path = None
            if save_preview:
                preview_path = os.path.splitext(item.path)[0] + "_people_preview.jpg"
            item.result = detector.detect(
                item.path,
                progress=None,
                log=log,
                save_preview=save_preview,
                preview_path=preview_path,
            )
        except Exception as exc:  # noqa: BLE001 - record and continue with the rest
            item.error = str(exc)
            if log:
                log(f"  ERROR: {exc}")

        if on_item_done:
            on_item_done(index, item)
        if on_progress:
            on_progress(int((index + 1) / total * 100))

    if log:
        ok = sum(1 for it in items if it.ok)
        log(f"Done. {ok}/{total} scanned successfully.")
    return items
