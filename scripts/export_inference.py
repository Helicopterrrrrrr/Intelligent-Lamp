from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from smartled_pose import PosturePipeline, load_config
from smartled_pose.config import load_reference, merge_reference
from smartled_pose.visualize import draw_output


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


def infer_mode(input_path: Path) -> str:
    if input_path.is_dir():
        return "image_dir"
    suffix = input_path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    raise ValueError(f"Unsupported input path: {input_path}")


def export_image_dir(pipeline: PosturePipeline, input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for path in sorted(input_dir.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        image = cv2.imread(str(path))
        if image is None:
            continue
        snapshot = pipeline.process_frame(image, timestamp=time.time())
        rendered = draw_output(image, snapshot.output, pipeline.roi)
        out_path = output_dir / path.name
        cv2.imwrite(str(out_path), rendered)
        summary.append(
            {
                "file": path.name,
                "presence_state": snapshot.output.presence_state,
                "distance_level": snapshot.output.distance_level,
                "raw_distance_level": snapshot.output.raw_distance_level,
                "posture_label": snapshot.output.posture_label,
                "raw_posture_label": snapshot.output.raw_posture_label,
                "event_state": snapshot.output.event_state,
                "metrics": snapshot.output.metrics,
            }
        )
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Exported {len(summary)} images to {output_dir}")


def export_single_image(pipeline: PosturePipeline, input_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(input_path))
    if image is None:
        raise RuntimeError(f"Unable to read image: {input_path}")
    snapshot = pipeline.process_frame(image, timestamp=time.time())
    rendered = draw_output(image, snapshot.output, pipeline.roi)
    out_path = output_dir / input_path.name
    cv2.imwrite(str(out_path), rendered)
    payload = {
        "file": input_path.name,
        "presence_state": snapshot.output.presence_state,
        "distance_level": snapshot.output.distance_level,
        "raw_distance_level": snapshot.output.raw_distance_level,
        "posture_label": snapshot.output.posture_label,
        "raw_posture_label": snapshot.output.raw_posture_label,
        "event_state": snapshot.output.event_state,
        "metrics": snapshot.output.metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps([payload], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Exported image to {out_path}")


def export_video(pipeline: PosturePipeline, input_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = output_dir / f"{input_path.stem}_overlay.mp4"
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    events = []
    frame_idx = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        ts = frame_idx / fps
        snapshot = pipeline.process_frame(frame, timestamp=ts)
        rendered = draw_output(frame, snapshot.output, pipeline.roi)
        writer.write(rendered)
        events.append(
            {
                "frame_index": frame_idx,
                "timestamp": ts,
                "presence_state": snapshot.output.presence_state,
                "distance_level": snapshot.output.distance_level,
                "raw_distance_level": snapshot.output.raw_distance_level,
                "posture_label": snapshot.output.posture_label,
                "raw_posture_label": snapshot.output.raw_posture_label,
                "event_state": snapshot.output.event_state,
                "metrics": snapshot.output.metrics,
            }
        )
        frame_idx += 1

    writer.release()
    capture.release()
    (output_dir / f"{input_path.stem}_events.json").write_text(
        json.dumps(events, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Exported video overlay to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SmartLED pose inference overlays for an image, image folder, or video.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--reference", default=None, help="Optional JSON calibration reference.")
    parser.add_argument("--input", required=True, help="Input image, directory, or video.")
    parser.add_argument("--output-dir", default="output/inference", help="Directory for exported overlays.")
    args = parser.parse_args()

    config = load_config(args.config)
    config = merge_reference(config, load_reference(args.reference))
    pipeline = PosturePipeline(config)

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    mode = infer_mode(input_path)
    if mode == "image_dir":
        export_image_dir(pipeline, input_path, output_dir)
    elif mode == "image":
        export_single_image(pipeline, input_path, output_dir)
    else:
        export_video(pipeline, input_path, output_dir)


if __name__ == "__main__":
    main()
