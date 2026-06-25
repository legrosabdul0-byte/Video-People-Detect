# -*- coding: utf-8 -*-
"""Video people-detection package."""

from .batch import BatchItem, find_videos, scan_folder
from .config import DetectionConfig
from .detector import DetectionResult, FrameSample, PeopleDetector

__all__ = [
    "DetectionConfig",
    "DetectionResult",
    "FrameSample",
    "PeopleDetector",
    "BatchItem",
    "find_videos",
    "scan_folder",
]

__version__ = "5.0.0"
