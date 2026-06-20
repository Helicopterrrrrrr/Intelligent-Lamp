# SmartLED Pose Demo

This repository contains a runnable YOLO pose demo for the SmartLED posture-detection workflow with **shoulder-width normalization and angle-based detection**.

## Features

- YOLO pose inference with Ultralytics `yolo11n-pose`
- Single-user target selection inside a desk ROI
- **Shoulder-width normalized distance detection**
- **Angle-based posture analysis with personalized calibration**
- **Neck angle detection (medical standard: 30°)**
- Rule-based posture and distance analysis with dual-mode (calibrated/fallback)
- Temporal smoothing, warning cooldown, and event logging
- Webcam/video demo overlay
- Reference calibration utility for personalized thresholds
- Minimal tests for geometry, selection, and rule behavior

## Project Layout

- `smartled_pose/`: core pipeline package
- `configs/default.yaml`: runtime thresholds and ROI
- `scripts/run_demo.py`: webcam/video posture demo
- `scripts/calibrate_reference.py`: collect baseline shoulder width and bbox ratio
- `scripts/init_event_labels.py`: create event-label templates for dataset validation
- `tests/`: unit tests for rule logic

## Quick Start

1. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

If the current Anaconda environment fails with a `pyexpat` or `xml.parsers.expat` import error while running `pip`, create a clean environment first:

```powershell
conda create -n smartled python=3.11 -y
conda activate smartled
python -m pip install -r requirements.txt
```

2. Calibrate normal sitting posture (recommended):

```powershell
python scripts/calibrate_reference.py --source 0 --output calibration/reference.json
```

Sit in a normal posture inside the ROI. Press `c` to capture samples (10-20 recommended), `w` to save and quit.

3. Run the demo with calibrated reference:

```powershell
python scripts/run_demo.py --source 0 --reference calibration/reference.json
```

Or run without calibration (uses fallback thresholds):

```powershell
python scripts/run_demo.py --source 0
```

## Runtime Controls

- `q`: quit
- `s`: save current event log to JSON

The calibration script also supports:

- `c`: capture the current frame's seated reference features
- `w`: write the averaged reference file and quit

## What's New (2026-05-26)

**Shoulder-Width Normalization & Angle-Based Detection**:
- Personalized calibration system for each user
- Normalized features eliminate height/body-type variations
- Angle-based detection relative to user's normal posture
- Dual-mode: calibrated (normalized) or fallback (fixed thresholds)
- Neck angle detection based on medical standards

See [docs/肩宽归一化功能说明.md](docs/肩宽归一化功能说明.md) for details.

## Testing

Run tests with plugin autoload disabled:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest -v
```

Or on Unix:

```bash
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -m pytest -v
```

## Dataset Preparation Helpers

- Put validation clips under `data/videos/`
- Generate event-label templates:

```powershell
python scripts/init_event_labels.py --video-dir data/videos --output data/event_labels/template.jsonl
```

## Notes

- The default model name is `yolo11n-pose.pt`. Ultralytics will download it on first run if needed.
- The rule layer is intentionally transparent so the same logic can move to Android after model export.
- Thresholds and ROI are externalized in `configs/default.yaml`.
