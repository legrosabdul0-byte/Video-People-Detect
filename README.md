<p align="right"><b>English</b> | <a href="README.zh-CN.md">中文</a></p>

# Video People Detect

Count the number of people in a video using a YOLOv8 detector. Designed for
classroom-style scenes where occlusion is common, so the estimate is made
robust by sampling several frames and aggregating the counts.

## Download (Windows, no Python needed)

Grab the latest `PeopleCounter-vX.Y.Z.exe` from the
[**Releases**](https://github.com/legrosabdul0-byte/Video-People-Detect/releases/latest)
page and double-click to run. Everything (Python, the libraries, and the YOLO
weights) is bundled into the single `.exe`, so it works even on machines that
have never had Python installed.

> Every push to `main` automatically builds a fresh `.exe` via GitHub Actions
> and publishes it to Releases, starting at **v1.0.0**.

## How it works

1. Sample a spread of frames from the middle of the clip (configurable).
2. Run YOLO person detection on the sampled frames (batched in one call).
3. Filter out boxes that are too small (area scales with resolution).
4. Aggregate the per-frame counts: trim the lowest (likely under-counted)
   frames, then take a high percentile to compensate for occlusion.
5. Produce a confidence score from the relative spread of the counts, and save
   an annotated preview of the frame closest to the final estimate.

## Install

```bash
pip install -r requirements.txt
```

The YOLO weights (`yolov8s.pt`) download automatically on first run.

## Usage

Launch the desktop GUI:

```bash
python main.py
```

Run headless on a single video (prints the result, no window required):

```bash
python main.py path/to/video.mp4
python main.py path/to/video.mp4 --no-preview
```

### Batch mode (scan a whole folder)

Point it at a **folder** and it scans every video inside (recursively) and
reports an approximate head count per file. In the GUI, use **Select Folder
(Batch)** to get a results table (Video / People / Confidence / Status).

```bash
python main.py path/to/folder            # prints a table, one row per video
python main.py path/to/folder --preview  # also save a preview next to each video
```

Batch mode uses a **high-recall preset** (`DetectionConfig.high_recall()`)
tuned for crowded / overlapping scenes: a larger model (`yolov8m`), a higher
NMS IoU threshold so heavily-overlapping people aren't merged into one, a lower
confidence threshold, a smaller minimum box size, and more sampled frames. It
is biased toward *not missing people* and gives a steadier confidence score —
at the cost of the occasional false positive and slower processing (especially
on CPU).

Use the detector programmatically:

```python
from video_people_detect import PeopleDetector, DetectionConfig

detector = PeopleDetector(DetectionConfig(final_percentile=70))
result = detector.detect("classroom.mp4", log=print)
print(result.final_count, result.confidence)
```

## Configuration

All tunable parameters live in [`video_people_detect/config.py`](video_people_detect/config.py),
including the model name, confidence threshold, inference image size, sample
points, minimum box area ratio, aggregation percentile, batching, and device.

## Project layout

```
main.py                       # GUI + CLI entry point
video_people_detect/
    config.py                 # DetectionConfig (all tunables)
    detector.py               # PeopleDetector (UI-independent logic)
    batch.py                  # scan_folder (batch / multi-video scanning)
    app.py                    # Tkinter GUI (single + folder modes)
.github/workflows/build-exe.yml   # CI: build Windows exe + publish Release
requirements.txt
```

## What changed vs. the original single-file script

- **Modular structure** — detection logic is fully separated from the Tkinter
  UI, so it can be reused from a CLI or tests.
- **Cross-platform preview** — replaces the Windows-only `os.startfile` with a
  helper that also works on macOS and Linux.
- **Batched inference** — all sampled frames are sent to YOLO in a single
  `predict` call for better throughput, especially on GPU.
- **Automatic device selection** — uses CUDA when available, falls back to CPU.
- **Headless CLI mode** — run on a video without opening a window.
- **Safer frame seeking, type hints, and docstrings** throughout.

## License

This project is licensed under the **GNU Affero General Public License v3.0
(AGPL-3.0)** — see [`LICENSE`](LICENSE).

AGPL-3.0 was chosen to stay consistent with [Ultralytics YOLO](https://github.com/ultralytics/ultralytics),
which this project depends on and is itself AGPL-3.0. If you distribute this
program (including the packaged `.exe`) or run a modified version as a network
service, you must make the corresponding source code available under the same
license.
