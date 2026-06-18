# -*- coding: utf-8 -*-
"""Configuration for the video people-counter.

All tunable parameters live here so the detection logic and the UI stay clean.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class DetectionConfig:
    """Tunable parameters for people detection.

    The defaults are tuned for classroom-style scenes where occlusion is common
    (people partially hidden behind desks / each other). In such scenes a missed
    detection is more likely than a false positive, so the final estimate leans
    toward the higher percentile of the sampled frame counts.
    """

    # --- Model ---
    model_name: str = "yolov8s.pt"
    conf: float = 0.20
    imgsz: int = 1280
    person_class_id: int = 0  # COCO class id for "person"
    # NMS IoU threshold. A box is dropped when it overlaps a higher-scoring box
    # by more than this. RAISING it keeps more heavily-overlapping people
    # instead of merging them into one (key for crowded / occluded scenes).
    iou: float = 0.70
    # Max detections kept per frame (raise it for very crowded scenes).
    max_det: int = 300

    # --- Frame sampling ---
    # Sample several frames across the middle of the clip so a single unlucky
    # moment (someone looking down / briefly occluded) does not skew the result.
    sample_points: List[float] = field(
        default_factory=lambda: [
            0.30, 0.35, 0.40, 0.45, 0.48, 0.50,
            0.52, 0.55, 0.60, 0.65, 0.70,
        ]
    )

    # Minimum box area as a fraction of the frame area. Scales with resolution
    # instead of being hard-coded in pixels.
    min_area_ratio: float = 0.00008

    # --- Aggregation ---
    # Occlusion biases counts low, so we favour a high percentile of the
    # sampled counts after trimming the lowest (likely under-counted) frames.
    final_percentile: int = 70

    # Confidence scoring: how strongly relative spread reduces confidence.
    confidence_spread_weight: float = 180.0

    # --- Performance ---
    # Run YOLO on all sampled frames in a single batched call when possible.
    batch_inference: bool = True
    # "" => auto-detect (CUDA if available, else CPU). Or set "cpu" / "cuda:0".
    device: str = ""

    # --- Output ---
    preview_filename: str = "people_detection_preview.jpg"

    @classmethod
    def high_recall(cls) -> "DetectionConfig":
        """Preset tuned to avoid missing people (favours recall over precision).

        Intended for batch / folder scanning where the priority is "don't miss
        anyone", even at the cost of the occasional false positive:

        - A larger model (``yolov8m``) handles occlusion / partial bodies much
          better than the small model.
        - A higher NMS ``iou`` (0.85) stops heavily-overlapping people from
          being merged into a single detection — the main cause of
          under-counting in crowded scenes.
        - Lower ``conf`` and smaller ``min_area_ratio`` catch farther away,
          smaller, or partially occluded people.
        - More ``sample_points`` make a single unlucky frame matter less, which
          also stabilises the estimate and tends to raise the reported
          confidence.
        - A higher ``final_percentile`` leans the aggregate against
          under-counting, and ``max_det`` is raised for very crowded frames.

        Trade-off: this is noticeably slower than the single-video default
        (bigger model + more frames), especially on CPU-only machines.
        """
        return cls(
            model_name="yolov8m.pt",
            conf=0.15,
            iou=0.85,
            max_det=1000,
            min_area_ratio=0.00005,
            sample_points=[round(0.20 + 0.04 * i, 2) for i in range(16)],  # 0.20..0.80
            final_percentile=80,
        )
