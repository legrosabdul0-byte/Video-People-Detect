# -*- coding: utf-8 -*-
"""Entry point for the video people-counter.

Usage:
    python main.py                      # launch the GUI
    python main.py path/to/video.mp4    # scan one video, print the result
    python main.py path/to/folder       # scan every video in a folder (batch)
"""

from __future__ import annotations

import argparse
import os
import sys

from video_people_detect import DetectionConfig, PeopleDetector, scan_folder


def run_single(video_path: str, no_preview: bool) -> int:
    detector = PeopleDetector(DetectionConfig())
    try:
        result = detector.detect(
            video_path,
            progress=None,
            log=lambda m: print(m),
            save_preview=not no_preview,
            keep_preview=False,
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


def run_batch(folder: str, save_preview: bool) -> int:
    # Folder scans favour recall ("don't miss anyone").
    detector = PeopleDetector(DetectionConfig.high_recall())
    items = scan_folder(
        folder,
        detector,
        save_preview=save_preview,
        log=lambda m: print(m),
    )
    if not items:
        return 1

    print("\n" + "=" * 60)
    print(f"{'Video':<40}{'People':>8}{'Conf':>8}")
    print("-" * 60)
    for it in items:
        if it.ok and it.result is not None:
            print(f"{it.name[:39]:<40}{it.result.final_count:>8}{it.result.confidence:>7}%")
        else:
            print(f"{it.name[:39]:<40}{'ERR':>8}{'-':>8}")
    print("=" * 60)
    ok = sum(1 for it in items if it.ok)
    print(f"Scanned {ok}/{len(items)} video(s) successfully.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Count people in a video using YOLO.")
    parser.add_argument(
        "path",
        nargs="?",
        help="Video file or a folder of videos. If omitted, the GUI is launched.",
    )
    parser.add_argument(
        "--no-preview", action="store_true", help="Do not save annotated preview image(s)."
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="In folder mode, save a per-video preview (off by default).",
    )
    args = parser.parse_args()

    if args.path:
        if os.path.isdir(args.path):
            return run_batch(args.path, save_preview=args.preview)
        return run_single(args.path, no_preview=args.no_preview)

    # No path argument -> launch the GUI.
    from video_people_detect.app import main as gui_main

    gui_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
