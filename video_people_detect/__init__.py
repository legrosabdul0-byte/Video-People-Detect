# -*- coding: utf-8 -*-
"""Video people-detection package."""

from .config import DetectionConfig
from .detector import DetectionResult, FrameSample, PeopleDetector

__all__ = [
    "DetectionConfig",
    "DetectionResult",
    "FrameSample",
    "PeopleDetector",
]

__version__ = "5.0.0"
