from __future__ import annotations

import argparse
import time

import cv2

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from smartled_pose import PosturePipeline, load_config
from smartled_pose.calibration import ReferenceCalibrator
from smartled_pose.types import PostureOutput
from smartled_pose.visualize import draw_output


def parse_source(source: str):
    if source.isdigit():
        return int(source)
    return source


def can_capture_reference(output: PostureOutput) -> bool:
    return output.presence_state == "seated" and output.features is not None


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect baseline reference features for distance rules.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--source", default="0", help="Camera index or video path.")
    parser.add_argument("--output", default="calibration/reference.json", help="Where to save the reference JSON.")
    args = parser.parse_args()

    pipeline = PosturePipeline(load_config(args.config))
    calibrator = ReferenceCalibrator()

    capture = cv2.VideoCapture(parse_source(str(args.source)))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, pipeline.config["camera"]["width"])
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, pipeline.config["camera"]["height"])
    capture.set(cv2.CAP_PROP_FPS, pipeline.config["camera"]["fps"])

    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video source: {args.source}")

    print("Sit in a normal posture inside the ROI. Press 'c' to capture a sample, 'w' to write the reference, or 'q' to quit.")
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        snapshot = pipeline.process_frame(frame, timestamp=time.time())
        vis = draw_output(frame, snapshot.output, pipeline.roi)
        cv2.putText(
            vis,
            f"samples: {len(calibrator.shoulder_width_samples)}",
            (20, vis.shape[0] - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.imshow("SmartLED Reference Calibration", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            if can_capture_reference(snapshot.output):
                calibrator.add(snapshot.output.features)
                print("Captured reference sample.")
            else:
                print("Current frame is not a stable seated posture. Move into the ROI and try again.")
        if key == ord("w"):
            reference = calibrator.save(args.output)
            print(f"Saved reference to {args.output}:")
            print(f"  shoulder_width_px={reference.shoulder_width_px}")
            print(f"  bbox_area_ratio={reference.bbox_area_ratio}")
            print(f"  head_shoulder_distance_ratio={reference.head_shoulder_distance_ratio}")
            print(f"  torso_angle_deg={reference.torso_angle_deg}")
            print(f"  head_tilt_deg={reference.head_tilt_deg}")
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
