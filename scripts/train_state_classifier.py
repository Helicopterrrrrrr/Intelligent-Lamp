from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from _bootstrap import ensure_repo_root_on_path


ensure_repo_root_on_path()

from smartled_pose import PosturePipeline, load_config
from smartled_pose.config import load_reference, merge_reference
from scripts.train_from_annotations import load_annotation, sample_times, sanitize_segments
from scripts.train_posture_classifier import SIDE_FEATURES, attach_reference_to_metrics, features_from_output


STATE_LABELS = [
    "calibration_normal",
    "computer_normal",
    "computer_abnormal",
    "reading_normal",
    "reading_abnormal",
    "ignore",
]

STATE_TEXT = {
    "calibration_normal": "校准正常坐姿",
    "computer_normal": "正常用电脑",
    "computer_abnormal": "用电脑异常",
    "reading_normal": "正常看书/写字",
    "reading_abnormal": "看书异常",
    "absent": "离开座位",
    "ignore": "忽略/过渡",
}


def extract_state_samples(
    annotation: dict[str, Any],
    config: dict,
    reference: dict[str, Any],
    sample_seconds: float,
) -> list[dict[str, Any]]:
    video_path = Path(annotation["video"])
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or float(annotation.get("fps", 25.0) or 25.0)
    pipeline = PosturePipeline(config)
    samples: list[dict[str, Any]] = []
    for segment in sanitize_segments(annotation.get("segments", [])):
        label = segment["label"]
        if label not in STATE_LABELS:
            continue
        for ts in sample_times(float(segment["start_time"]), float(segment["end_time"]), sample_seconds):
            frame_index = int(round(ts * fps))
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            snapshot = pipeline.process_frame(frame, timestamp=ts)
            attach_reference_to_metrics(snapshot.output, reference)
            sample = features_from_output(snapshot.output, label, STATE_LABELS.index(label), video_path, frame_index)
            if sample:
                sample["timestamp"] = ts
                sample["annotation"] = annotation["_annotation_path"]
                samples.append(sample)

    capture.release()
    return samples


class MultiClassKNN:
    def __init__(self, k: int) -> None:
        self.k = k

    def fit(self, samples: list[dict[str, Any]]) -> None:
        x = np.array([[float(sample.get(name, 0.0)) for name in SIDE_FEATURES] for sample in samples], dtype=np.float64)
        y = np.array([int(sample["target"]) for sample in samples], dtype=np.int32)
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        self.std[self.std < 1e-6] = 1.0
        self.x_train = (x - self.mean) / self.std
        self.y_train = y

    def predict(self, sample: dict[str, Any]) -> tuple[int, float]:
        x = np.array([float(sample.get(name, 0.0)) for name in SIDE_FEATURES], dtype=np.float64)
        x = (x - self.mean) / self.std
        distances = np.linalg.norm(self.x_train - x, axis=1)
        nearest = np.argsort(distances)[: self.k]
        counts = Counter(int(self.y_train[idx]) for idx in nearest)
        label_id, count = counts.most_common(1)[0]
        return label_id, count / float(self.k)


def split_samples(samples: list[dict[str, Any]], train_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    rows = list(samples)
    rng.shuffle(rows)
    split_at = max(1, min(len(rows) - 1, int(len(rows) * train_ratio)))
    return rows[:split_at], rows[split_at:]


def evaluate_knn(samples: list[dict[str, Any]], k: int, train_ratio: float, seed: int) -> dict[str, Any]:
    train, test = split_samples(samples, train_ratio=train_ratio, seed=seed)
    model = MultiClassKNN(k)
    model.fit(train)
    correct = 0
    confusion: dict[str, Counter] = {}
    for sample in test:
        pred, conf = model.predict(sample)
        target = int(sample["target"])
        correct += int(pred == target)
        true_label = STATE_LABELS[target]
        pred_label = STATE_LABELS[pred]
        confusion.setdefault(true_label, Counter())[pred_label] += 1
    return {
        "k": k,
        "train_count": len(train),
        "test_count": len(test),
        "accuracy": correct / len(test) if test else 0.0,
        "confusion": {label: dict(counter) for label, counter in confusion.items()},
    }


def write_report(output_dir: Path, results: dict[str, Any]) -> None:
    lines = ["# Seven-State Classifier Report", ""]
    lines.append(f"Sample interval: {results['sample_seconds']} seconds")
    lines.append(f"Labels: `{STATE_LABELS + ['absent']}`")
    lines.append("")
    lines.append(f"Samples: {results['sample_count']}")
    lines.append(f"Label counts: `{results['label_counts']}`")
    lines.append("")
    lines.append("| k | accuracy | train | test |")
    lines.append("|---:|---:|---:|---:|")
    for item in results["evaluations"]:
        lines.append(f"| {item['k']} | {item['accuracy']:.3f} | {item['train_count']} | {item['test_count']} |")
    lines.append("")
    best = results["best"]
    lines.append(f"Best k: `{best['k']}`, accuracy: `{best['accuracy']:.3f}`")
    lines.append("")
    lines.append("Best confusion:")
    for label, counter in best["confusion"].items():
        lines.append(f"- `{label}`: {counter}")
    (output_dir / "state_classifier_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a seven-state posture/context classifier from annotations.")
    parser.add_argument("--annotations-dir", default="data/annotations")
    parser.add_argument("--config", default="configs/relaxed.yaml")
    parser.add_argument("--reference", default="output/lab_20260610/processed_relaxed/reference.json")
    parser.add_argument("--output-dir", default="output/state_classifier_1s")
    parser.add_argument("--feedback", default=None, help="Optional live feedback JSONL to append as extra state samples.")
    parser.add_argument("--sample-seconds", type=float, default=1.0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    annotations = [load_annotation(path) for path in sorted(Path(args.annotations_dir).glob("*_segments.json"))]
    reference = load_reference(args.reference)
    config = merge_reference(load_config(args.config), reference)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples: list[dict[str, Any]] = []
    for annotation in annotations:
        samples.extend(
            extract_state_samples(
                annotation=annotation,
                config=config,
                reference=reference,
                sample_seconds=args.sample_seconds,
            )
        )

    if args.feedback:
        feedback_path = Path(args.feedback)
        if feedback_path.exists():
            for line in feedback_path.read_text(encoding="utf-8-sig").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                label = record.get("correct_label")
                features = record.get("features") or {}
                if label not in STATE_LABELS or not features:
                    continue
                sample = dict(features)
                sample.update(
                    {
                        "label": label,
                        "target": STATE_LABELS.index(label),
                        "source": record.get("frame_path") or "feedback",
                        "frame_index": -1,
                        "timestamp": record.get("timestamp"),
                        "annotation": str(feedback_path),
                    }
                )
                samples.append(sample)

    (output_dir / "all_state_samples.jsonl").write_text(
        "\n".join(json.dumps(sample, ensure_ascii=False) for sample in samples) + "\n",
        encoding="utf-8",
    )

    evaluations = [evaluate_knn(samples, k=k, train_ratio=args.train_ratio, seed=args.seed) for k in (3, 5, 7, 9, 15)]
    best = max(evaluations, key=lambda item: item["accuracy"])
    results = {
        "sample_seconds": args.sample_seconds,
        "sample_count": len(samples),
        "label_counts": dict(Counter(sample["label"] for sample in samples)),
        "labels": STATE_LABELS,
        "absent_handled_by_presence_rule": True,
        "evaluations": evaluations,
        "best": best,
    }
    (output_dir / "state_classifier_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output_dir, results)
    print(f"Wrote state classifier results to {output_dir}")


if __name__ == "__main__":
    main()
