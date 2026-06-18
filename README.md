# Video People Detect

Count the number of people in a video using a YOLOv8 detector. Designed for
classroom-style scenes where occlusion is common, so the estimate is made
robust by sampling several frames and aggregating the counts.

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
    app.py                    # Tkinter GUI
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
