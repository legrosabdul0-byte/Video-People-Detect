# -*- coding: utf-8 -*-
"""Core people-detection logic, fully decoupled from any UI.

The detector samples frames from a video, runs YOLO person detection on them,
and aggregates the per-frame counts into a single robust estimate plus a
confidence score. Progress and log messages are delivered through optional
callbacks so the same code can drive a GUI, a CLI, or tests.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import DetectionConfig

LogCallback = Callable[[str], None]
ProgressCallback = Callable[[int], None]


@dataclass
class FrameSample:
    """A single analysed frame."""

    percent: int
    count: int
    annotated: Optional[np.ndarray] = None


@dataclass
class DetectionResult:
    """Aggregated detection outcome."""

    final_count: int
    confidence: int
    raw_counts: List[int]
    used_counts: List[int]
    final_percentile: int
    preview_path: str = ""
    samples: List[FrameSample] = field(default_factory=list)


class PeopleDetector:
    """Detects and counts people in a video using YOLO.

    The model is loaded lazily on first use and cached for subsequent runs.
    """

    def __init__(self, config: Optional[DetectionConfig] = None):
        self.config = config or DetectionConfig()
        self._model = None
        self._resolved_device: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Model handling
    # ------------------------------------------------------------------ #
    def _resolve_device(self) -> str:
        """Pick CUDA when available unless the user pinned a device."""
        if self.config.device:
            return self.config.device
        try:
            import torch  # imported lazily; only needed for auto-detection

            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _resolve_model_path(self) -> str:
        """Locate the model weights.

        When packaged as a PyInstaller one-file exe, bundled data is unpacked
        into ``sys._MEIPASS``. We prefer the bundled weights so the app runs
        fully offline; otherwise we fall back to the configured name and let
        ultralytics find or download it.
        """
        name = self.config.model_name
        if os.path.isabs(name) and os.path.exists(name):
            return name

        bundle_dir = getattr(sys, "_MEIPASS", None)
        if bundle_dir:
            bundled = os.path.join(bundle_dir, name)
            if os.path.exists(bundled):
                return bundled

        return name

    def load_model(self, log: Optional[LogCallback] = None) -> None:
        """Load the YOLO model once and reuse it."""
        if self._model is not None:
            return

        # Imported here so importing this module never forces ultralytics/torch.
        from ultralytics import YOLO

        self._resolved_device = self._resolve_device()
        model_path = self._resolve_model_path()
        if log:
            log(f"Loading model: {model_path} (device={self._resolved_device})")
        self._model = YOLO(model_path)

    # ------------------------------------------------------------------ #
    # Frame sampling
    # ------------------------------------------------------------------ #
    def _read_samples(
        self, video_path: str, log: Optional[LogCallback]
    ) -> Tuple[List[Tuple[int, np.ndarray]], int]:
        """Read the configured sample frames from the video.

        Returns a list of ``(percent, frame)`` tuples and the frame area.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Failed to open video.")

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                raise RuntimeError("Invalid video frame count.")

            frames: List[Tuple[int, np.ndarray]] = []
            frame_area = 0
            for point in self.config.sample_points:
                frame_id = min(int(total_frames * point), total_frames - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ok, frame = cap.read()
                percent = int(point * 100)
                if not ok or frame is None:
                    if log:
                        log(f"Skip {percent}%: failed to read frame")
                    continue
                if frame_area == 0:
                    h, w = frame.shape[:2]
                    frame_area = h * w
                frames.append((percent, frame))
            return frames, frame_area
        finally:
            cap.release()

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def _count_in_result(self, result, min_area: float) -> int:
        """Count person boxes above the minimum area in one YOLO result."""
        count = 0
        for box in result.boxes:
            if int(box.cls[0]) != self.config.person_class_id:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            if (x2 - x1) * (y2 - y1) < min_area:
                continue
            count += 1
        return count

    def _predict(self, frames: Sequence[np.ndarray]):
        """Run YOLO on frames, batched in a single call when enabled."""
        images = list(frames)
        common = dict(
            conf=self.config.conf,
            iou=self.config.iou,
            max_det=self.config.max_det,
            imgsz=self.config.imgsz,
            verbose=False,
        )
        if self._resolved_device:
            common["device"] = self._resolved_device

        if self.config.batch_inference:
            return self._model.predict(images, **common)
        return [self._model.predict(img, **common)[0] for img in images]

    # ------------------------------------------------------------------ #
    # Aggregation
    # ------------------------------------------------------------------ #
    def _aggregate(self, counts: Sequence[int]) -> Tuple[int, int, List[int]]:
        """Turn per-frame counts into a final estimate + confidence."""
        arr = sorted(counts)

        # Low counts are most likely under-counts (occlusion), so trim the
        # lowest samples before taking the percentile.
        if len(arr) >= 9:
            used = arr[2:]
        elif len(arr) >= 6:
            used = arr[1:]
        else:
            used = arr

        final_count = int(round(np.percentile(used, self.config.final_percentile)))

        mean = float(np.mean(used)) if used else 0.0
        if mean <= 0:
            confidence = 0
        else:
            relative_std = float(np.std(used)) / mean
            confidence = 100.0 - relative_std * self.config.confidence_spread_weight
            confidence = int(round(max(0.0, min(100.0, confidence))))

        return final_count, confidence, list(used)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def detect(
        self,
        video_path: str,
        progress: Optional[ProgressCallback] = None,
        log: Optional[LogCallback] = None,
        save_preview: bool = True,
        preview_path: Optional[str] = None,
    ) -> DetectionResult:
        """Detect and count people in ``video_path``.

        Args:
            video_path: Path to the input video.
            progress: Optional callback receiving an int percentage (0-100).
            log: Optional callback receiving human-readable status strings.
            save_preview: Whether to write an annotated preview image.
            preview_path: Optional explicit path for the preview image. When
                omitted, it defaults to ``config.preview_filename`` next to the
                input video (useful to give each video its own preview in batch
                mode and avoid overwriting).

        Returns:
            A :class:`DetectionResult`.
        """
        self.load_model(log)

        if log:
            log("Opening video...")
        frames, frame_area = self._read_samples(video_path, log)
        if not frames:
            raise RuntimeError("No valid frames found.")

        min_area = frame_area * self.config.min_area_ratio
        if log:
            log(f"Analyzing {len(frames)} frames...")

        results = self._predict([f for _, f in frames])

        samples: List[FrameSample] = []
        counts: List[int] = []
        for i, ((percent, _frame), result) in enumerate(zip(frames, results)):
            count = self._count_in_result(result, min_area)
            counts.append(count)
            samples.append(FrameSample(percent=percent, count=count, annotated=result.plot()))
            if log:
                log(f"{percent}% : {count} people")
            if progress:
                progress(int((i + 1) / len(frames) * 100))

        final_count, confidence, used_counts = self._aggregate(counts)

        saved_preview = ""
        if save_preview and samples:
            saved_preview = self._save_preview(
                video_path, samples, final_count, log, preview_path
            )

        return DetectionResult(
            final_count=final_count,
            confidence=confidence,
            raw_counts=counts,
            used_counts=used_counts,
            final_percentile=self.config.final_percentile,
            preview_path=saved_preview,
            samples=samples,
        )

    def _save_preview(
        self,
        video_path: str,
        samples: Sequence[FrameSample],
        final_count: int,
        log: Optional[LogCallback],
        preview_path: Optional[str] = None,
    ) -> str:
        """Write the annotated frame whose count is closest to ``final_count``."""
        best = min(samples, key=lambda s: abs(s.count - final_count))
        if best.annotated is None:
            return ""

        if preview_path is None:
            base_dir = os.path.dirname(os.path.abspath(video_path))
            preview_path = os.path.join(base_dir, self.config.preview_filename)
        cv2.imwrite(preview_path, best.annotated)
        if log:
            log(f"Preview frame: {best.percent}%, {best.count} people")
        return preview_path
