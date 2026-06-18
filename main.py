# -*- coding: utf-8 -*-
"""Entry point for the video people-counter.

Usage:
    python main.py                      # launch the GUI
    python main.py path/to/video.mp4    # run headless and print the result
"""

from __future__ import annotations

import argparse
import sys

from video_people_detect import DetectionConfig, PeopleDetector


def run_cli(video_path: str, no_preview: bool) -> int:
    detector = PeopleDetector(DetectionConfig())
    try:
        result = detector.detect(
            video_path,
            progress=None,
            log=lambda m: print(m),
            save_preview=not no_preview,
        )
    except Exception as exc:  # noqa: BLE001 - surface any failure to the user
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("-" * 40)
    print(f"Raw Counts:       {result.raw_counts}")
    print(f"Used Counts:      {result.used_counts}")
    print(f"Final Percentile: {result.final_percentile}%")
    print(f"Final Count:      {result.final_count}")
    print(f"Confidence:       {result.confidence}%")
    if result.preview_path:
        print(f"Preview saved:    {result.preview_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Count people in a video using YOLO.")
    parser.add_argument(
        "video", nargs="?", help="Video file. If omitted, the GUI is launched."
    )
    parser.add_argument(
        "--no-preview", action="store_true", help="Do not save the annotated preview image."
    )
    args = parser.parse_args()

    if args.video:
        return run_cli(args.video, args.no_preview)

    # No video argument -> launch the GUI.
    from video_people_detect.app import main as gui_main

    gui_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
