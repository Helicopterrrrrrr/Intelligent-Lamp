from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from smartled_pose import PosturePipeline, load_config
from smartled_pose.config import load_reference, merge_reference
from smartled_pose.posture_classifier import (
    ClassifierDecision,
    ClassifierDecisionSmoother,
    STATE_TEXT,
    StateClassifierDecision,
    apply_side_view_overrides,
    classifier_for_mode,
    extract_side_features,
    state_classifier,
)
from smartled_pose.visualize import POSTURE_TEXT, _cn, _draw_chinese_lines, draw_output


FEEDBACK_LABELS = [
    "calibration_normal",
    "computer_normal",
    "computer_abnormal",
    "reading_normal",
    "reading_abnormal",
    "absent",
    "ignore",
]

FEEDBACK_TEXT = {
    "calibration_normal": "校准正常坐姿",
    "computer_normal": "正常用电脑",
    "computer_abnormal": "用电脑异常",
    "reading_normal": "正常看书/写字",
    "reading_abnormal": "看书异常",
    "absent": "离开座位",
    "ignore": "忽略/过渡",
}


def parse_source(source: str):
    if source.isdigit():
        return int(source)
    return source


def draw_current_prediction(
    frame,
    output,
    decision: ClassifierDecision | None,
    state_decision: StateClassifierDecision | None,
    mode: str,
    feedback_message: str | None = None,
):
    if output.presence_state == "absent":
        state = STATE_TEXT["absent"]
        color = (220, 220, 220)
    elif state_decision is not None:
        state = state_decision.text
        color = (255, 80, 80) if state_decision.label.endswith("abnormal") else (80, 255, 80)
        if state_decision.label == "ignore":
            color = (180, 180, 180)
    elif output.presence_state == "seated_candidate":
        state = "候选在位"
        color = (255, 220, 80)
    elif decision is not None:
        state = "姿态异常" if decision.stable_label == "abnormal" else "姿态正常"
        color = (255, 80, 80) if decision.stable_label == "abnormal" else (80, 255, 80)
    else:
        state = _cn(POSTURE_TEXT, output.raw_posture_label)
        color = (255, 80, 80) if output.raw_posture_label != "normal" else (80, 255, 80)

    mode_text = {
        "computer": "电脑模式",
        "reading": "看书模式",
        "rules": "规则模式",
        "state7": "七状态识别",
    }.get(mode, mode)
    lines = [
        f"当前预测: {state}",
        f"当前模式: {mode_text}",
        "反馈: c=预测正确 | 1校准 2电脑正 3电脑异 4看书正 5看书异 6离开 7忽略",
    ]
    if state_decision is not None:
        lines.insert(2, f"置信度: {state_decision.confidence:.2f}")
    if feedback_message:
        lines.append(feedback_message)
    return _draw_chinese_lines(frame, lines, start_x=24, start_y=28, line_gap=42, fill=color)


def predicted_label_for_feedback(
    output,
    mode: str,
    decision: ClassifierDecision | None,
    state_decision: StateClassifierDecision | None,
) -> str:
    if output.presence_state == "absent":
        return "absent"
    if state_decision is not None:
        return state_decision.label
    if decision is not None and mode in {"computer", "reading"}:
        return f"{mode}_abnormal" if decision.stable_label == "abnormal" else f"{mode}_normal"
    if output.raw_posture_label == "normal":
        return "ignore"
    return output.raw_posture_label


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    try:
        if hasattr(value, "item"):
            return value.item()
    except Exception:
        pass
    return value


def save_feedback(
    output_path: str | Path,
    frame,
    save_frame: bool,
    mode: str,
    predicted_label: str,
    correct_label: str,
    snapshot,
    feature_row: dict[str, float] | None,
    decision: ClassifierDecision | None,
    state_decision: StateClassifierDecision | None,
) -> None:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    frame_path = None
    if save_frame:
        frame_dir = target.with_suffix("").parent / f"{target.with_suffix('').name}_frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        stem = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        frame_path = frame_dir / f"{stem}_{correct_label}.jpg"
        cv2.imwrite(str(frame_path), frame)

    record = {
        "created_at": datetime.now().isoformat(timespec="milliseconds"),
        "timestamp": snapshot.output.timestamp,
        "mode": mode,
        "predicted_label": predicted_label,
        "correct_label": correct_label,
        "is_correct": predicted_label == correct_label,
        "presence_state": snapshot.output.presence_state,
        "raw_distance_level": snapshot.output.raw_distance_level,
        "raw_posture_label": snapshot.output.raw_posture_label,
        "event_state": snapshot.output.event_state,
        "state_confidence": state_decision.confidence if state_decision else None,
        "state_model": state_decision.model_name if state_decision else None,
        "binary_probability": decision.probability if decision else None,
        "binary_threshold": decision.threshold if decision else None,
        "features": json_ready(feature_row or {}),
        "frame_path": str(frame_path) if frame_path else None,
    }
    with target.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Feedback saved: predicted={predicted_label}, correct={correct_label}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SmartLED YOLO pose demo.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--reference", default=None, help="Optional JSON calibration reference.")
    parser.add_argument("--source", default="0", help="Camera index or video path.")
    parser.add_argument("--save-events", default="output/events/latest_events.json", help="Where to save the event log on demand.")
    parser.add_argument("--mode", choices=["rules", "computer", "reading", "state7"], default="rules", help="Decision mode. Use state7 to output the seven annotated states.")
    parser.add_argument("--classifier-samples-dir", default="output/annotated_classifier_random_1s", help="Directory containing *_annotated_samples.jsonl classifier samples.")
    parser.add_argument("--state-samples-dir", default="output/state_classifier_1s", help="Directory containing all_state_samples.jsonl for seven-state prediction.")
    parser.add_argument("--classifier-hold-seconds", type=float, default=2.0, help="Seconds above threshold before classifier stable abnormal.")
    parser.add_argument("--classifier-clear-seconds", type=float, default=2.0, help="Seconds below threshold before classifier returns to normal.")
    parser.add_argument("--feedback-output", default="data/feedback/live_feedback.jsonl", help="Where to append interactive correction feedback.")
    parser.add_argument("--no-feedback-frames", action="store_true", help="Do not save image frames with feedback records.")
    args = parser.parse_args()

    config = load_config(args.config)
    config = merge_reference(config, load_reference(args.reference))
    pipeline = PosturePipeline(config)
    classifier = None
    state_clf = None
    smoother = None
    if args.mode in {"computer", "reading"}:
        classifier = classifier_for_mode(args.mode, args.classifier_samples_dir)
        smoother = ClassifierDecisionSmoother(
            hold_seconds=args.classifier_hold_seconds,
            clear_seconds=args.classifier_clear_seconds,
        )
        print(
            f"Loaded classifier mode={args.mode}, model={classifier.model_name}, "
            f"threshold={classifier.threshold}, samples={classifier.samples_path}"
        )
    elif args.mode == "state7":
        state_clf = state_classifier(args.state_samples_dir)
        print(f"Loaded seven-state classifier model={state_clf.model_name}, samples={state_clf.samples_path}")

    capture = cv2.VideoCapture(parse_source(str(args.source)))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, config["camera"]["width"])
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config["camera"]["height"])
    capture.set(cv2.CAP_PROP_FPS, config["camera"]["fps"])

    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video source: {args.source}")

    last_feedback_message: str | None = None
    last_feedback_at = 0.0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        snapshot = pipeline.process_frame(frame, timestamp=time.time())
        classifier_decision = None
        state_decision = None
        feature_row = None
        if classifier and smoother:
            feature_row = extract_side_features(
                snapshot.output,
                reference=pipeline.reference,
                frame_width=frame.shape[1],
                frame_height=frame.shape[0],
            )
            probability = classifier.predict_probability(feature_row) if feature_row else 0.0
            raw_label = "abnormal" if probability >= classifier.threshold else "normal"
            stable_label, classifier_state = smoother.update(
                timestamp=snapshot.output.timestamp,
                is_abnormal=raw_label == "abnormal",
                presence_state=snapshot.output.presence_state,
            )
            classifier_decision = ClassifierDecision(
                mode=args.mode,
                probability=probability,
                threshold=classifier.threshold,
                raw_label=raw_label,
                stable_label=stable_label,
                event_state=classifier_state,
                model_name=classifier.model_name,
            )
            snapshot.output.metrics["classifier_probability"] = probability
            snapshot.output.metrics["classifier_threshold"] = classifier.threshold
            snapshot.output.metrics["classifier_raw_abnormal"] = 1.0 if raw_label == "abnormal" else 0.0
            snapshot.output.metrics["classifier_stable_abnormal"] = 1.0 if stable_label == "abnormal" else 0.0
        elif state_clf:
            if snapshot.output.presence_state == "absent":
                state_decision = StateClassifierDecision(
                    label="absent",
                    text=STATE_TEXT["absent"],
                    confidence=1.0,
                    model_name=state_clf.model_name,
                )
            else:
                feature_row = extract_side_features(
                    snapshot.output,
                    reference=pipeline.reference,
                    frame_width=frame.shape[1],
                    frame_height=frame.shape[0],
                )
                if feature_row:
                    state_decision = apply_side_view_overrides(state_clf.predict(feature_row), feature_row)
                else:
                    state_decision = StateClassifierDecision(
                        label="ignore",
                        text=STATE_TEXT["ignore"],
                        confidence=0.0,
                        model_name=state_clf.model_name,
                    )
            snapshot.output.metrics["state_confidence"] = state_decision.confidence if state_decision else 0.0

        predicted_label = predicted_label_for_feedback(snapshot.output, args.mode, classifier_decision, state_decision)
        vis = draw_output(frame, snapshot.output, pipeline.roi, show_overlay=False)
        feedback_message = last_feedback_message if time.time() - last_feedback_at <= 1.5 else None
        vis = draw_current_prediction(
            vis,
            snapshot.output,
            classifier_decision,
            state_decision,
            args.mode,
            feedback_message=feedback_message,
        )
        cv2.imshow("SmartLED Pose Demo", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            pipeline.event_manager.save(args.save_events)
            print(f"Saved events to {args.save_events}")
        elif key == ord("c"):
            save_feedback(
                args.feedback_output,
                frame,
                not args.no_feedback_frames,
                args.mode,
                predicted_label,
                predicted_label,
                snapshot,
                feature_row,
                classifier_decision,
                state_decision,
            )
            last_feedback_message = f"反馈已记录: 预测正确 ({FEEDBACK_TEXT.get(predicted_label, predicted_label)})"
            last_feedback_at = time.time()
        elif ord("1") <= key <= ord("7"):
            correct_label = FEEDBACK_LABELS[key - ord("1")]
            save_feedback(
                args.feedback_output,
                frame,
                not args.no_feedback_frames,
                args.mode,
                predicted_label,
                correct_label,
                snapshot,
                feature_row,
                classifier_decision,
                state_decision,
            )
            last_feedback_message = f"反馈已记录: 正确状态={FEEDBACK_TEXT.get(correct_label, correct_label)}"
            last_feedback_at = time.time()

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
