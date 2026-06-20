from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import cv2

from _bootstrap import ensure_repo_root_on_path


ensure_repo_root_on_path()

from smartled_pose import PosturePipeline, load_config
from smartled_pose.calibration import ReferenceCalibrator
from smartled_pose.config import load_reference, merge_reference
from smartled_pose.types import PostureOutput, ReferenceFeatures


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}

LABEL_EXPECTATIONS = {
    "calibration_normal": {
        "expected_user_state": "computer_use_or_study",
        "expected_posture_state": "normal",
        "expected_brightness_action": "keep_or_reference",
    },
    "computer_normal": {
        "expected_user_state": "computer_use",
        "expected_posture_state": "normal",
        "expected_brightness_action": "dim",
    },
    "reading_normal": {
        "expected_user_state": "reading",
        "expected_posture_state": "normal",
        "expected_brightness_action": "brighten",
    },
    "reading_abnormal": {
        "expected_user_state": "reading",
        "expected_posture_state": "abnormal",
        "expected_brightness_action": "brighten_and_warn",
    },
    "computer_abnormal": {
        "expected_user_state": "computer_use",
        "expected_posture_state": "abnormal",
        "expected_brightness_action": "dim_and_warn",
    },
    "absent": {
        "expected_user_state": "absent",
        "expected_posture_state": "none",
        "expected_brightness_action": "off_or_low",
    },
}


def output_to_row(output: PostureOutput, frame_index: int, timestamp: float) -> dict[str, Any]:
    features = output.features
    row = {
        "frame_index": frame_index,
        "timestamp": timestamp,
        "presence_state": output.presence_state,
        "distance_level": output.distance_level,
        "raw_distance_level": output.raw_distance_level,
        "posture_label": output.posture_label,
        "raw_posture_label": output.raw_posture_label,
        "event_state": output.event_state,
        "pose_conf": output.metrics.get("pose_conf", 0.0),
        "bbox_area_ratio": output.metrics.get("bbox_area_ratio", 0.0),
        "kp_valid_ratio": output.metrics.get("kp_valid_ratio", 0.0),
        "head_tilt_deg": output.metrics.get("head_tilt_deg", 0.0),
        "torso_tilt_deg": output.metrics.get("torso_tilt_deg", 0.0),
    }
    if features:
        row.update(
            {
                "shoulder_width_px": features.shoulder_width_px,
                "head_shoulder_distance_ratio": features.head_shoulder_distance_ratio,
                "neck_angle_deg": features.neck_angle_deg,
                "roi_overlap": features.roi_overlap,
            }
        )
    else:
        row.update(
            {
                "shoulder_width_px": 0.0,
                "head_shoulder_distance_ratio": 0.0,
                "neck_angle_deg": 0.0,
                "roi_overlap": 0.0,
            }
        )
    return row


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def counts(key: str) -> dict[str, int]:
        return dict(Counter(str(row[key]) for row in rows))

    def avg(key: str) -> float:
        values = [float(row.get(key, 0.0)) for row in rows]
        return mean(values) if values else 0.0

    frames = len(rows)
    seated = sum(1 for row in rows if row["presence_state"] == "seated")
    absent = sum(1 for row in rows if row["presence_state"] == "absent")
    raw_abnormal = sum(
        1
        for row in rows
        if row["raw_posture_label"] != "normal" or row["raw_distance_level"] == "too_close"
    )
    warning = sum(1 for row in rows if row["event_state"] == "warning_active")

    return {
        "frames": frames,
        "seated_rate": seated / frames if frames else 0.0,
        "absent_rate": absent / frames if frames else 0.0,
        "raw_abnormal_rate": raw_abnormal / frames if frames else 0.0,
        "warning_rate": warning / frames if frames else 0.0,
        "presence_counts": counts("presence_state"),
        "raw_distance_counts": counts("raw_distance_level"),
        "raw_posture_counts": counts("raw_posture_label"),
        "event_counts": counts("event_state"),
        "avg_pose_conf": avg("pose_conf"),
        "avg_kp_valid_ratio": avg("kp_valid_ratio"),
        "avg_bbox_area_ratio": avg("bbox_area_ratio"),
        "avg_shoulder_width_px": avg("shoulder_width_px"),
        "avg_head_shoulder_distance_ratio": avg("head_shoulder_distance_ratio"),
        "avg_neck_angle_deg": avg("neck_angle_deg"),
        "avg_torso_tilt_deg": avg("torso_tilt_deg"),
        "avg_head_tilt_deg": avg("head_tilt_deg"),
    }


def iter_media(label_dir: Path, suffixes: set[str]) -> list[Path]:
    if not label_dir.exists():
        return []
    return sorted(path for path in label_dir.iterdir() if path.suffix.lower() in suffixes)


def add_image_to_calibrator(path: Path, pipeline: PosturePipeline, calibrator: ReferenceCalibrator) -> int:
    image = cv2.imread(str(path))
    if image is None:
        return 0
    snapshot = pipeline.process_frame(image)
    if snapshot.output.features and snapshot.output.detection:
        calibrator.add(snapshot.output.features)
        return 1
    return 0


def add_video_to_calibrator(
    path: Path,
    pipeline: PosturePipeline,
    calibrator: ReferenceCalibrator,
    sample_every: int,
    max_samples: int,
) -> int:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return 0

    added = 0
    frame_index = 0
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    while added < max_samples:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % sample_every == 0:
            snapshot = pipeline.process_frame(frame, timestamp=frame_index / fps)
            if snapshot.output.features and snapshot.output.detection:
                calibrator.add(snapshot.output.features)
                added += 1
        frame_index += 1

    capture.release()
    return added


def build_reference(
    dataset_root: Path,
    config: dict,
    output_path: Path,
    sample_every: int,
    max_video_samples: int,
) -> tuple[ReferenceFeatures, int]:
    pipeline = PosturePipeline(config)
    calibrator = ReferenceCalibrator()
    samples = 0

    image_dir = dataset_root / "images" / "calibration_normal"
    for path in iter_media(image_dir, IMAGE_SUFFIXES):
        samples += add_image_to_calibrator(path, pipeline, calibrator)

    video_dir = dataset_root / "videos" / "calibration_normal"
    for path in iter_media(video_dir, VIDEO_SUFFIXES):
        samples += add_video_to_calibrator(path, pipeline, calibrator, sample_every, max_video_samples)

    reference = calibrator.save(output_path)
    return reference, samples


def reference_to_dict(reference: ReferenceFeatures | dict[str, Any]) -> dict[str, Any]:
    if isinstance(reference, dict):
        return dict(reference)
    return {
        "shoulder_width_px": reference.shoulder_width_px,
        "bbox_area_ratio": reference.bbox_area_ratio,
        "head_shoulder_distance_ratio": reference.head_shoulder_distance_ratio,
        "torso_angle_deg": reference.torso_angle_deg,
        "head_tilt_deg": reference.head_tilt_deg,
    }


def evaluate_image(path: Path, pipeline: PosturePipeline) -> list[dict[str, Any]]:
    image = cv2.imread(str(path))
    if image is None:
        return []
    snapshot = pipeline.process_frame(image, timestamp=0.0)
    return [output_to_row(snapshot.output, frame_index=0, timestamp=0.0)]


def evaluate_video(path: Path, pipeline: PosturePipeline, sample_every: int) -> list[dict[str, Any]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return []

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    rows: list[dict[str, Any]] = []
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % sample_every == 0:
            timestamp = frame_index / fps
            snapshot = pipeline.process_frame(frame, timestamp=timestamp)
            rows.append(output_to_row(snapshot.output, frame_index=frame_index, timestamp=timestamp))
        frame_index += 1

    capture.release()
    return rows


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "type",
        "file",
        "expected_user_state",
        "expected_posture_state",
        "expected_brightness_action",
        "frames",
        "seated_rate",
        "absent_rate",
        "raw_abnormal_rate",
        "warning_rate",
        "presence_counts",
        "raw_distance_counts",
        "raw_posture_counts",
        "event_counts",
        "avg_pose_conf",
        "avg_kp_valid_ratio",
        "avg_bbox_area_ratio",
        "avg_shoulder_width_px",
        "avg_head_shoulder_distance_ratio",
        "avg_neck_angle_deg",
        "avg_torso_tilt_deg",
        "avg_head_tilt_deg",
        "path",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            encoded = dict(row)
            for key in ("presence_counts", "raw_distance_counts", "raw_posture_counts", "event_counts"):
                encoded[key] = json.dumps(encoded[key], ensure_ascii=False)
            writer.writerow({key: encoded.get(key, "") for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Process a SmartLED lab dataset with image/video labels.")
    parser.add_argument("--dataset-root", default="data/lab_20260610", help="Collected dataset root.")
    parser.add_argument("--config", default="configs/default.yaml", help="Runtime config.")
    parser.add_argument("--output-dir", default="output/lab_20260610/processed", help="Output directory.")
    parser.add_argument("--reference", default=None, help="Existing reference JSON. If omitted, build from calibration_normal.")
    parser.add_argument("--video-sample-every", type=int, default=5, help="Evaluate one frame every N video frames.")
    parser.add_argument("--calibration-sample-every", type=int, default=15, help="Use one calibration frame every N frames.")
    parser.add_argument("--max-calibration-video-samples", type=int, default=40, help="Max samples per calibration video.")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config = load_config(args.config)
    reference_path = Path(args.reference) if args.reference else output_dir / "reference.json"
    if args.reference:
        reference = load_reference(reference_path)
        calibration_samples = 0
    else:
        reference, calibration_samples = build_reference(
            dataset_root=dataset_root,
            config=base_config,
            output_path=reference_path,
            sample_every=max(1, args.calibration_sample_every),
            max_video_samples=max(1, args.max_calibration_video_samples),
        )

    reference_payload = reference_to_dict(reference)
    config = merge_reference(base_config, reference_payload)
    item_summaries: list[dict[str, Any]] = []

    labels = sorted(
        {
            path.name
            for media_root in (dataset_root / "images", dataset_root / "videos")
            if media_root.exists()
            for path in media_root.iterdir()
            if path.is_dir()
        }
    )
    for label in labels:
        expectations = LABEL_EXPECTATIONS.get(
            label,
            {
                "expected_user_state": "",
                "expected_posture_state": "",
                "expected_brightness_action": "",
            },
        )

        for media_type, suffixes in (("image", IMAGE_SUFFIXES), ("video", VIDEO_SUFFIXES)):
            media_dir = dataset_root / f"{media_type}s" / label
            for path in iter_media(media_dir, suffixes):
                pipeline = PosturePipeline(config)
                if media_type == "image":
                    rows = evaluate_image(path, pipeline)
                else:
                    rows = evaluate_video(path, pipeline, sample_every=max(1, args.video_sample_every))
                summary = summarize_rows(rows)
                summary.update(
                    {
                        "label": label,
                        "type": media_type,
                        "file": path.name,
                        "path": str(path),
                        **expectations,
                    }
                )
                item_summaries.append(summary)

    payload = {
        "dataset_root": str(dataset_root),
        "config": args.config,
        "reference_path": str(reference_path),
        "calibration_samples": calibration_samples,
        "reference": reference_payload,
        "note": "The current posture pipeline evaluates presence/posture only. computer_use vs reading is kept as expected metadata, not predicted yet.",
        "items": item_summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_csv(output_dir / "summary.csv", item_summaries)

    print(f"Reference: {reference_path}")
    print(f"Calibration samples: {calibration_samples}")
    print(f"Items processed: {len(item_summaries)}")
    print(f"Summary: {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
