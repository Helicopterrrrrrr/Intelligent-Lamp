from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from smartled_pose import PosturePipeline, load_config
from smartled_pose.config import load_reference, merge_reference


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


def count_values(rows: list[dict], key: str) -> dict[str, int]:
    return dict(Counter(row[key] for row in rows))


def mean_metric(rows: list[dict], key: str) -> float:
    values = [float(row["metrics"].get(key, 0.0)) for row in rows]
    return sum(values) / len(values) if values else 0.0


def evaluate_video(video_path: Path, config: dict) -> dict:
    pipeline = PosturePipeline(config)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    rows: list[dict] = []
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        timestamp = frame_index / fps
        snapshot = pipeline.process_frame(frame, timestamp=timestamp)
        rows.append(
            {
                "frame_index": frame_index,
                "timestamp": timestamp,
                "presence_state": snapshot.output.presence_state,
                "distance_level": snapshot.output.distance_level,
                "raw_distance_level": snapshot.output.raw_distance_level,
                "posture_label": snapshot.output.posture_label,
                "raw_posture_label": snapshot.output.raw_posture_label,
                "event_state": snapshot.output.event_state,
                "metrics": snapshot.output.metrics,
            }
        )
        frame_index += 1

    capture.release()

    summary = {
        "video": video_path.name,
        "path": str(video_path.as_posix()),
        "frames": len(rows),
        "presence_counts": count_values(rows, "presence_state"),
        "distance_counts": count_values(rows, "distance_level"),
        "raw_distance_counts": count_values(rows, "raw_distance_level"),
        "posture_counts": count_values(rows, "posture_label"),
        "raw_posture_counts": count_values(rows, "raw_posture_label"),
        "event_counts": count_values(rows, "event_state"),
        "avg_pose_conf": mean_metric(rows, "pose_conf"),
        "avg_bbox_area_ratio": mean_metric(rows, "bbox_area_ratio"),
        "avg_head_tilt_deg": mean_metric(rows, "head_tilt_deg"),
        "avg_torso_tilt_deg": mean_metric(rows, "torso_tilt_deg"),
        "avg_kp_valid_ratio": mean_metric(rows, "kp_valid_ratio"),
        "avg_roi_overlap": mean_metric(rows, "roi_overlap"),
    }
    return summary


def load_manifest(manifest_path: Path | None) -> dict[str, dict]:
    if not manifest_path or not manifest_path.exists():
        return {}
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {record["file"]: record for record in records}


def write_csv(output_path: Path, rows: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "file",
                "scenario",
                "camera_angle",
                "lighting",
                "people",
                "expected_fit",
                "scenario_group",
                "expected_anomalies",
                "frames",
                "presence_counts",
                "distance_counts",
                "raw_distance_counts",
                "posture_counts",
                "raw_posture_counts",
                "event_counts",
                "avg_pose_conf",
                "avg_bbox_area_ratio",
                "avg_head_tilt_deg",
                "avg_torso_tilt_deg",
                "avg_kp_valid_ratio",
                "avg_roi_overlap",
                "source_url",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["video"],
                    row.get("scenario", ""),
                    row.get("camera_angle", ""),
                    row.get("lighting", ""),
                    row.get("people", ""),
                    row.get("expected_fit", ""),
                    row.get("scenario_group", ""),
                    json.dumps(row.get("expected_anomalies", []), ensure_ascii=False),
                    row["frames"],
                    json.dumps(row["presence_counts"], ensure_ascii=False),
                    json.dumps(row["distance_counts"], ensure_ascii=False),
                    json.dumps(row["raw_distance_counts"], ensure_ascii=False),
                    json.dumps(row["posture_counts"], ensure_ascii=False),
                    json.dumps(row["raw_posture_counts"], ensure_ascii=False),
                    json.dumps(row["event_counts"], ensure_ascii=False),
                    f"{row['avg_pose_conf']:.4f}",
                    f"{row['avg_bbox_area_ratio']:.4f}",
                    f"{row['avg_head_tilt_deg']:.4f}",
                    f"{row['avg_torso_tilt_deg']:.4f}",
                    f"{row['avg_kp_valid_ratio']:.4f}",
                    f"{row['avg_roi_overlap']:.4f}",
                    row.get("source_url", ""),
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-evaluate SmartLED posture rules on a directory of videos.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--reference", default=None, help="Optional JSON calibration reference.")
    parser.add_argument("--input-dir", default="data/public_videos", help="Directory containing evaluation videos.")
    parser.add_argument(
        "--manifest",
        default="data/public_videos/manifest.json",
        help="Optional JSON manifest with camera-angle metadata.",
    )
    parser.add_argument("--output-dir", default="output/batch_eval", help="Directory for summary outputs.")
    args = parser.parse_args()

    config = load_config(args.config)
    config = merge_reference(config, load_reference(args.reference))
    input_dir = Path(args.input_dir)
    manifest = load_manifest(Path(args.manifest))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for video_path in sorted(input_dir.iterdir()):
        if video_path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        print(f"Evaluating {video_path.name} ...")
        summary = evaluate_video(video_path, config)
        meta = manifest.get(video_path.name, {})
        summary.update(meta)
        summaries.append(summary)

    (output_dir / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(output_dir / "summary.csv", summaries)
    print(f"Wrote {len(summaries)} summaries to {output_dir}")


if __name__ == "__main__":
    main()
