from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - optional display helper
    Image = None
    ImageDraw = None
    ImageFont = None


LABELS = [
    "calibration_normal",
    "computer_normal",
    "computer_abnormal",
    "reading_normal",
    "reading_abnormal",
    "absent",
    "ignore",
]

LABEL_TEXT = {
    "calibration_normal": "校准正常坐姿",
    "computer_normal": "正常用电脑",
    "computer_abnormal": "用电脑异常",
    "reading_normal": "正常看书/写字",
    "reading_abnormal": "看书异常",
    "absent": "离开座位",
    "ignore": "忽略/过渡",
}

LABEL_COLORS = {
    "calibration_normal": (0, 200, 255),
    "computer_normal": (0, 220, 0),
    "computer_abnormal": (0, 0, 255),
    "reading_normal": (255, 180, 0),
    "reading_abnormal": (255, 0, 255),
    "absent": (160, 160, 160),
    "ignore": (80, 80, 80),
}


@dataclass
class Segment:
    label: str
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    def as_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["duration"] = self.duration
        return payload


def load_font(size: int):
    if ImageFont is None:
        return None
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/msyhbd.ttc"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def draw_text_lines(frame, lines: list[tuple[str, tuple[int, int, int]]], x: int, y: int, line_gap: int = 28):
    if Image is None or ImageDraw is None:
        for idx, (text, color) in enumerate(lines):
            ascii_text = text.encode("ascii", errors="ignore").decode("ascii")
            cv2.putText(frame, ascii_text, (x, y + idx * line_gap), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        return frame

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font = load_font(22)
    for idx, (text, color) in enumerate(lines):
        rgb_color = (color[2], color[1], color[0])
        draw.text((x, y + idx * line_gap - 21), text, font=font, fill=rgb_color)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def default_output_path(video_path: Path) -> Path:
    return Path("data") / "annotations" / f"{video_path.stem}_segments.json"


def frame_time(frame_index: int, fps: float) -> float:
    return frame_index / fps if fps > 0 else 0.0


def save_annotations(path: Path, video_path: Path, fps: float, frame_count: int, segments: list[Segment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "video": str(video_path),
        "fps": fps,
        "frame_count": frame_count,
        "duration": frame_time(frame_count, fps),
        "labels": LABELS,
        "segments": [segment.as_payload() for segment in segments if segment.duration > 0.0],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = path.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["label", "start_time", "end_time", "duration", "start_frame", "end_frame"],
        )
        writer.writeheader()
        for segment in payload["segments"]:
            writer.writerow(segment)


def load_existing(path: Path) -> list[Segment]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    segments = []
    for item in payload.get("segments", []):
        segments.append(
            Segment(
                label=item["label"],
                start_time=float(item["start_time"]),
                end_time=float(item["end_time"]),
                start_frame=int(item["start_frame"]),
                end_frame=int(item["end_frame"]),
            )
        )
    return segments


def draw_timeline(frame, segments: list[Segment], current_label: str | None, current_start_frame: int | None, frame_index: int, frame_count: int):
    h, w = frame.shape[:2]
    y = h - 28
    x0, x1 = 20, w - 20
    cv2.rectangle(frame, (x0, y - 8), (x1, y + 8), (35, 35, 35), -1)
    if frame_count <= 0:
        return frame

    def xpos(idx: int) -> int:
        return int(x0 + (x1 - x0) * max(0, min(idx, frame_count)) / frame_count)

    for segment in segments:
        color = LABEL_COLORS.get(segment.label, (255, 255, 255))
        cv2.rectangle(frame, (xpos(segment.start_frame), y - 8), (xpos(segment.end_frame), y + 8), color, -1)
    if current_label and current_start_frame is not None:
        color = LABEL_COLORS.get(current_label, (255, 255, 255))
        cv2.rectangle(frame, (xpos(current_start_frame), y - 8), (xpos(frame_index), y + 8), color, -1)
    cv2.line(frame, (xpos(frame_index), y - 14), (xpos(frame_index), y + 14), (255, 255, 255), 2)
    return frame


def draw_overlay(
    frame,
    video_path: Path,
    frame_index: int,
    frame_count: int,
    fps: float,
    paused: bool,
    speed: float,
    current_label: str | None,
    current_start_frame: int | None,
    segments: list[Segment],
):
    vis = frame.copy()
    h, w = vis.shape[:2]
    overlay = vis.copy()
    cv2.rectangle(overlay, (10, 10), (min(w - 10, 820), 250), (0, 0, 0), -1)
    vis = cv2.addWeighted(overlay, 0.55, vis, 0.45, 0)

    current_time = frame_time(frame_index, fps)
    duration = frame_time(frame_count, fps)
    status = "暂停" if paused else f"播放 {speed:.1f}x"
    label_desc = "未开始" if not current_label else f"{LABEL_TEXT.get(current_label, current_label)} ({current_label})"
    lines: list[tuple[str, tuple[int, int, int]]] = [
        ("SmartLED 视频标注", (255, 255, 255)),
        (f"视频: {video_path.name}", (255, 255, 255)),
        (f"时间: {current_time:.2f}s / {duration:.2f}s    帧: {frame_index}/{frame_count}    状态: {status}", (255, 255, 255)),
        (f"当前段: {label_desc}", LABEL_COLORS.get(current_label or "ignore", (0, 255, 255))),
        ("按键: 1校准 2电脑正常 3电脑异常 4看书正常 5看书异常 6离开 7忽略", (255, 255, 255)),
        ("按键: 空格暂停 | a/d后退/前进1秒 | ,/.逐帧 | u撤销 | s保存 | q保存退出", (255, 255, 255)),
        (f"已完成段数: {len(segments)}", (255, 255, 255)),
    ]
    vis = draw_text_lines(vis, lines, 25, 40)
    vis = draw_timeline(vis, segments, current_label, current_start_frame, frame_index, frame_count)
    return vis


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate a full SmartLED video with time segments.")
    parser.add_argument("--video", required=True, help="Video to annotate.")
    parser.add_argument("--output", default=None, help="Output JSON path. Default: data/annotations/<video>_segments.json")
    parser.add_argument("--resume", action="store_true", help="Load existing annotations from output path.")
    args = parser.parse_args()

    video_path = Path(args.video)
    output_path = Path(args.output) if args.output else default_output_path(video_path)
    segments = load_existing(output_path) if args.resume else []

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = int(segments[-1].end_frame) if segments else 0
    current_label: str | None = None
    current_start_frame: int | None = None
    if segments:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    paused = True
    speed = 1.0
    window_name = "SmartLED Video Annotator"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            break

        preview = draw_overlay(
            frame,
            video_path=video_path,
            frame_index=frame_index,
            frame_count=frame_count,
            fps=fps,
            paused=paused,
            speed=speed,
            current_label=current_label,
            current_start_frame=current_start_frame,
            segments=segments,
        )
        cv2.imshow(window_name, preview)
        delay = 0 if paused else max(1, int(1000 / max(1.0, fps * speed)))
        key = cv2.waitKey(delay) & 0xFF

        if key == ord("q"):
            if current_label and current_start_frame is not None and frame_index > current_start_frame:
                segments.append(
                    Segment(
                        label=current_label,
                        start_time=frame_time(current_start_frame, fps),
                        end_time=frame_time(frame_index, fps),
                        start_frame=current_start_frame,
                        end_frame=frame_index,
                    )
                )
            save_annotations(output_path, video_path, fps, frame_count, segments)
            print(f"Saved annotations to {output_path}")
            break
        if key == ord("s"):
            save_annotations(output_path, video_path, fps, frame_count, segments)
            print(f"Saved annotations to {output_path}")
        elif key == ord(" "):
            paused = not paused
        elif key in (ord("["), ord("-")):
            speed = max(0.25, speed / 2.0)
        elif key in (ord("]"), ord("=")):
            speed = min(4.0, speed * 2.0)
        elif key == ord("a"):
            frame_index = max(0, frame_index - int(fps))
            paused = True
        elif key == ord("d"):
            frame_index = min(max(0, frame_count - 1), frame_index + int(fps))
            paused = True
        elif key == ord(","):
            frame_index = max(0, frame_index - 1)
            paused = True
        elif key == ord("."):
            frame_index = min(max(0, frame_count - 1), frame_index + 1)
            paused = True
        elif key == ord("u"):
            if current_label is not None:
                current_label = None
                current_start_frame = None
            elif segments:
                last = segments.pop()
                frame_index = last.start_frame
                print(f"Undid segment: {last.label} {last.start_time:.2f}-{last.end_time:.2f}s")
            paused = True
        elif ord("1") <= key <= ord("7"):
            label_index = key - ord("1")
            label = LABELS[label_index]
            if current_label and current_start_frame is not None and frame_index > current_start_frame:
                segments.append(
                    Segment(
                        label=current_label,
                        start_time=frame_time(current_start_frame, fps),
                        end_time=frame_time(frame_index, fps),
                        start_frame=current_start_frame,
                        end_frame=frame_index,
                    )
                )
                print(f"Segment: {current_label} {frame_time(current_start_frame, fps):.2f}-{frame_time(frame_index, fps):.2f}s")
            current_label = label
            current_start_frame = frame_index
            print(f"Started label: {label} at {frame_time(frame_index, fps):.2f}s")
        elif not paused:
            frame_index += 1

        if frame_index >= frame_count - 1:
            paused = True
            frame_index = max(0, frame_count - 1)

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
