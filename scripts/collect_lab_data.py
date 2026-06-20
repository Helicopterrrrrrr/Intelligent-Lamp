from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from _bootstrap import ensure_repo_root_on_path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - optional display helper
    Image = None
    ImageDraw = None
    ImageFont = None


ensure_repo_root_on_path()


DEFAULT_LABELS = [
    "calibration_normal",
    "computer_normal",
    "reading_normal",
    "reading_abnormal",
    "computer_abnormal",
    "absent",
]

BACKENDS = {
    "default": None,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
}

LABEL_TEXT = {
    "calibration_normal": "校准正常坐姿",
    "computer_normal": "正常用电脑",
    "reading_normal": "正常看书/写字",
    "reading_abnormal": "看书异常",
    "computer_abnormal": "用电脑异常",
    "absent": "离开座位",
    # Backward-compatible labels for older collections.
    "normal": "正常坐姿",
    "head_down": "低头",
    "too_close": "距离过近",
    "leaning_left": "左歪",
    "leaning_right": "右歪",
    "hand_support": "托脸/疲劳",
    "normal_borderline": "轻微姿态",
}


def parse_source(source: str):
    if source.isdigit():
        return int(source)
    return source


def make_capture(source: str, backend: str) -> cv2.VideoCapture:
    parsed = parse_source(source)
    backend_id = BACKENDS[backend]
    if backend_id is None:
        return cv2.VideoCapture(parsed)
    return cv2.VideoCapture(parsed, backend_id)


def now_stem() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def ensure_notes(output_root: Path, labels: list[str]) -> None:
    notes_path = output_root / "notes.md"
    if notes_path.exists():
        return
    lines = [
        "# Lab Data Collection Notes",
        "",
        "## Setup",
        "",
        "- Camera position:",
        "- Light condition:",
        "- Subject distance:",
        "- Phone/camera app:",
        "",
        "## Labels",
        "",
    ]
    lines.extend(f"- `{label}`:" for label in labels)
    notes_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_manifest(output_root: Path, payload: dict) -> None:
    manifest_path = output_root / "manifest.jsonl"
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _load_font(size: int):
    if ImageFont is None:
        return None
    font_candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/msyhbd.ttc"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]
    for candidate in font_candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def draw_text_lines(frame, lines: list[tuple[str, tuple[int, int, int]]], x: int, y: int, line_gap: int = 28):
    if Image is None or ImageDraw is None:
        for idx, (text, color) in enumerate(lines):
            cv2.putText(
                frame,
                text.encode("ascii", errors="ignore").decode("ascii"),
                (x, y + idx * line_gap),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                color,
                2,
                cv2.LINE_AA,
            )
        return frame

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font = _load_font(22)
    for idx, (text, color) in enumerate(lines):
        rgb_color = (color[2], color[1], color[0])
        draw.text((x, y + idx * line_gap - 20), text, font=font, fill=rgb_color)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def draw_text(frame, text: str, x: int, y: int, color: tuple[int, int, int], size: int = 48):
    if Image is None or ImageDraw is None:
        cv2.putText(
            frame,
            text.encode("ascii", errors="ignore").decode("ascii"),
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.6,
            color,
            4,
            cv2.LINE_AA,
        )
        return frame

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font = _load_font(size)
    rgb_color = (color[2], color[1], color[0])
    draw.text((x, y - size), text, font=font, fill=rgb_color)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def draw_overlay(
    frame,
    labels: list[str],
    label_index: int,
    subject: str,
    condition: str,
    output_root: Path,
    recording: bool,
    record_started_at: float | None,
    frame_count: int,
    countdown_action: str | None,
    countdown_remaining: float | None,
):
    vis = frame.copy()
    h, w = vis.shape[:2]
    panel_w = min(720, w - 20)
    panel_h = 245
    overlay = vis.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
    vis = cv2.addWeighted(overlay, 0.55, vis, 0.45, 0)

    current = labels[label_index]
    current_text = LABEL_TEXT.get(current, current)
    lines: list[tuple[str, tuple[int, int, int]]] = [
        ("SmartLED 姿态数据采集", (255, 255, 255)),
        (f"被试: {subject}    场景: {condition}", (255, 255, 255)),
        (f"当前姿态 [{label_index + 1}/{len(labels)}]: {current_text} ({current})", (0, 255, 255)),
        ("按键: 1-9 选择姿态 | [ / ] 上一个/下一个", (255, 255, 255)),
        ("按键: s 倒计时拍照 | r 倒计时录制/停止录制 | q 退出", (255, 255, 255)),
        (f"保存目录: {output_root}", (255, 255, 255)),
        (f"画面: {w}x{h}", (255, 255, 255)),
    ]
    if recording:
        elapsed = time.time() - (record_started_at or time.time())
        lines.append((f"录制中 {elapsed:05.1f} 秒    帧数: {frame_count}", (0, 0, 255)))

    vis = draw_text_lines(vis, lines, 25, 40)

    if recording:
        cv2.circle(vis, (w - 42, 42), 13, (0, 0, 255), -1)
    if countdown_action and countdown_remaining is not None:
        seconds = max(1, int(countdown_remaining + 0.999))
        action_text = "拍照" if countdown_action == "snapshot" else "开始录制"
        center_text = f"{action_text}倒计时: {seconds}"
        cv2.rectangle(vis, (0, 0), (w, h), (0, 0, 0), 8)
        vis = draw_text(vis, center_text, max(30, w // 2 - 220), h // 2, (0, 255, 255), size=54)
    return vis


def create_video_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, size)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect labeled posture images and videos from a camera.")
    parser.add_argument("--source", default="0", help="Camera index or video source. Default: 0.")
    parser.add_argument("--backend", choices=sorted(BACKENDS), default="default", help="OpenCV camera backend.")
    parser.add_argument("--output-root", default=None, help="Output directory. Default: data/lab_YYYYMMDD.")
    parser.add_argument("--subject", default="subject01", help="Subject id used in file names.")
    parser.add_argument("--condition", default="lab", help="Condition tag, e.g. light_on or lamp_only.")
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS), help="Comma-separated posture labels.")
    parser.add_argument("--width", type=int, default=1280, help="Requested camera width.")
    parser.add_argument("--height", type=int, default=720, help="Requested camera height.")
    parser.add_argument("--fps", type=float, default=30.0, help="Requested camera FPS.")
    parser.add_argument("--native-resolution", action="store_true", help="Do not force camera width/height/fps.")
    parser.add_argument("--warmup-frames", type=int, default=30, help="Frames to discard after opening camera.")
    parser.add_argument("--countdown", type=float, default=3.0, help="Seconds to wait before snapshot or recording starts.")
    args = parser.parse_args()

    labels = [item.strip() for item in args.labels.split(",") if item.strip()]
    if not labels:
        raise ValueError("At least one label is required.")

    output_root = Path(args.output_root) if args.output_root else Path("data") / f"lab_{datetime.now():%Y%m%d}"
    for label in labels:
        (output_root / "images" / label).mkdir(parents=True, exist_ok=True)
        (output_root / "videos" / label).mkdir(parents=True, exist_ok=True)
    ensure_notes(output_root, labels)

    cap = make_capture(args.source, args.backend)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {args.source} with backend={args.backend}")

    if not args.native_resolution:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)

    for _ in range(max(0, args.warmup_frames)):
        cap.read()
        time.sleep(0.005)

    label_index = 0
    writer: cv2.VideoWriter | None = None
    record_path: Path | None = None
    record_label: str | None = None
    record_started_at: float | None = None
    record_frame_count = 0
    record_size: tuple[int, int] | None = None
    record_fps = args.fps
    pending_action: str | None = None
    pending_label: str | None = None
    pending_started_at: float | None = None

    cv2.namedWindow("SmartLED Lab Collector", cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Camera frame read failed.")
                break

            h, w = frame.shape[:2]
            if writer is not None:
                if record_size and (w, h) != record_size:
                    frame_to_write = cv2.resize(frame, record_size)
                else:
                    frame_to_write = frame
                writer.write(frame_to_write)
                record_frame_count += 1

            preview = draw_overlay(
                frame,
                labels=labels,
                label_index=label_index,
                subject=args.subject,
                condition=args.condition,
                output_root=output_root,
                recording=writer is not None,
                record_started_at=record_started_at,
                frame_count=record_frame_count,
                countdown_action=pending_action,
                countdown_remaining=(
                    args.countdown - (time.time() - pending_started_at)
                    if pending_action and pending_started_at is not None
                    else None
                ),
            )
            cv2.imshow("SmartLED Lab Collector", preview)
            key = cv2.waitKey(1) & 0xFF

            if pending_action and pending_started_at is not None:
                elapsed = time.time() - pending_started_at
                if elapsed >= args.countdown:
                    label = pending_label or labels[label_index]
                    if pending_action == "snapshot":
                        path = output_root / "images" / label / f"{args.subject}_{label}_{args.condition}_{now_stem()}.jpg"
                        cv2.imwrite(str(path), frame)
                        append_manifest(
                            output_root,
                            {
                                "type": "image",
                                "label": label,
                                "subject": args.subject,
                                "condition": args.condition,
                                "path": str(path),
                                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                                "source": args.source,
                                "backend": args.backend,
                                "width": w,
                                "height": h,
                                "countdown_seconds": args.countdown,
                            },
                        )
                        print(f"Saved image: {path}")
                    elif pending_action == "record" and writer is None:
                        path = output_root / "videos" / label / f"{args.subject}_{label}_{args.condition}_{now_stem()}.mp4"
                        actual_fps = cap.get(cv2.CAP_PROP_FPS)
                        if actual_fps and actual_fps > 1:
                            record_fps = float(actual_fps)
                        record_size = (w, h)
                        writer = create_video_writer(path, record_fps, record_size)
                        if not writer.isOpened():
                            writer.release()
                            writer = None
                            raise RuntimeError(f"Unable to create video writer: {path}")
                        record_path = path
                        record_label = label
                        record_started_at = time.time()
                        record_frame_count = 0
                        print(f"Recording started: {path}")
                    pending_action = None
                    pending_label = None
                    pending_started_at = None
                if key != ord("q"):
                    continue

            if key == ord("q"):
                break
            if ord("1") <= key <= ord("9"):
                requested = key - ord("1")
                if requested < len(labels):
                    label_index = requested
            elif key in (ord("["), ord(",")):
                label_index = (label_index - 1) % len(labels)
            elif key in (ord("]"), ord(".")):
                label_index = (label_index + 1) % len(labels)
            elif key == ord("s"):
                pending_action = "snapshot"
                pending_label = labels[label_index]
                pending_started_at = time.time()
                print(f"Snapshot countdown started: {pending_label}")
            elif key == ord("r"):
                if writer is None:
                    pending_action = "record"
                    pending_label = labels[label_index]
                    pending_started_at = time.time()
                    print(f"Recording countdown started: {pending_label}")
                else:
                    writer.release()
                    duration = time.time() - (record_started_at or time.time())
                    append_manifest(
                        output_root,
                        {
                            "type": "video",
                            "label": record_label,
                            "subject": args.subject,
                            "condition": args.condition,
                            "path": str(record_path),
                            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                            "source": args.source,
                            "backend": args.backend,
                            "width": record_size[0] if record_size else w,
                            "height": record_size[1] if record_size else h,
                            "fps": record_fps,
                            "duration_seconds": duration,
                            "frames": record_frame_count,
                        },
                    )
                    print(f"Recording stopped: {record_path} ({duration:.1f}s)")
                    writer = None
                    record_path = None
                    record_label = None
                    record_started_at = None
                    record_frame_count = 0
                    record_size = None
    finally:
        if writer is not None:
            writer.release()
            print(f"Recording closed: {record_path}")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
