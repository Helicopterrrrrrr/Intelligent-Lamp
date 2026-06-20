from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import cv2

from _bootstrap import ensure_repo_root_on_path


ensure_repo_root_on_path()

from smartled_pose import PosturePipeline, load_config
from smartled_pose.config import load_reference, merge_reference
from scripts.train_posture_classifier import GaussianNB, SIDE_FEATURES, attach_reference_to_metrics, features_from_output, metrics_for


MODE_LABELS = {
    "computer": {"computer_normal": 0, "computer_abnormal": 1},
    "reading": {"reading_normal": 0, "reading_abnormal": 1},
}


def load_annotation(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["_annotation_path"] = str(path)
    return payload


def sanitize_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    prev_end = 0.0
    for segment in sorted(segments, key=lambda item: (float(item["start_time"]), float(item["end_time"]))):
        start = max(float(segment["start_time"]), prev_end)
        end = float(segment["end_time"])
        if end <= start:
            continue
        cleaned = dict(segment)
        cleaned["start_time"] = start
        cleaned["end_time"] = end
        cleaned["duration"] = end - start
        sanitized.append(cleaned)
        prev_end = end
    return sanitized


def sample_times(start: float, end: float, sample_seconds: float) -> list[float]:
    if end <= start:
        return []
    # Skip a small boundary margin so labels near transitions do not poison training.
    margin = min(1.0, max(0.0, (end - start) * 0.1))
    current = start + margin
    last = end - margin
    if current > last:
        current = (start + end) / 2.0
        last = current
    times = []
    while current <= last + 1e-6:
        times.append(current)
        current += sample_seconds
    return times


def extract_samples_from_annotation(
    annotation: dict[str, Any],
    config: dict,
    reference: dict[str, Any],
    mode: str,
    sample_seconds: float,
) -> list[dict[str, Any]]:
    video_path = Path(annotation["video"])
    labels = MODE_LABELS[mode]
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or float(annotation.get("fps", 25.0) or 25.0)
    pipeline = PosturePipeline(config)
    samples: list[dict[str, Any]] = []
    for segment in sanitize_segments(annotation.get("segments", [])):
        label = segment["label"]
        if label not in labels:
            continue
        target = labels[label]
        for ts in sample_times(float(segment["start_time"]), float(segment["end_time"]), sample_seconds):
            frame_index = int(round(ts * fps))
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            snapshot = pipeline.process_frame(frame, timestamp=ts)
            attach_reference_to_metrics(snapshot.output, reference)
            sample = features_from_output(snapshot.output, label, target, video_path, frame_index)
            if sample:
                sample["timestamp"] = ts
                sample["annotation"] = annotation["_annotation_path"]
                samples.append(sample)

    capture.release()
    return samples


def evaluate_leave_one_video_out(samples: list[dict[str, Any]]) -> dict[str, Any]:
    by_video: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        by_video.setdefault(sample["source"], []).append(sample)

    folds = []
    all_predictions: list[dict[str, Any]] = []
    for held_out_video, test in sorted(by_video.items()):
        train = [sample for source, rows in by_video.items() if source != held_out_video for sample in rows]
        train_targets = {int(sample["target"]) for sample in train}
        test_targets = {int(sample["target"]) for sample in test}
        if len(train_targets) < 2 or not test:
            folds.append(
                {
                    "held_out_video": held_out_video,
                    "train_count": len(train),
                    "test_count": len(test),
                    "skipped": True,
                    "reason": "Need both normal and abnormal samples in training.",
                }
            )
            continue

        model = GaussianNB.fit(train, SIDE_FEATURES)
        metrics = metrics_for(model, test)
        fold = {
            "held_out_video": held_out_video,
            "train_count": len(train),
            "test_count": len(test),
            "test_targets": dict(Counter(str(sample["target"]) for sample in test)),
            "metrics": metrics,
        }
        folds.append(fold)
        for sample in test:
            probability = model.predict_proba_one(sample)
            all_predictions.append(
                {
                    "source": sample["source"],
                    "timestamp": sample.get("timestamp"),
                    "frame_index": sample["frame_index"],
                    "label": sample["label"],
                    "target": sample["target"],
                    "probability": probability,
                    "prediction": int(probability >= 0.5),
                }
            )

    aggregate = metrics_from_predictions(all_predictions)
    return {"folds": folds, "aggregate": aggregate, "predictions": all_predictions}


def evaluate_random_frame_split(
    samples: list[dict[str, Any]],
    train_ratio: float,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    split_at = max(1, min(len(shuffled) - 1, int(len(shuffled) * train_ratio))) if len(shuffled) > 1 else len(shuffled)
    train = shuffled[:split_at]
    test = shuffled[split_at:]
    model = GaussianNB.fit(train, SIDE_FEATURES)
    predictions = []
    for sample in test:
        probability = model.predict_proba_one(sample)
        predictions.append(
            {
                "source": sample["source"],
                "timestamp": sample.get("timestamp"),
                "frame_index": sample["frame_index"],
                "label": sample["label"],
                "target": sample["target"],
                "probability": probability,
                "prediction": int(probability >= 0.5),
            }
        )
    return {
        "train_count": len(train),
        "test_count": len(test),
        "train_label_counts": dict(Counter(sample["label"] for sample in train)),
        "test_label_counts": dict(Counter(sample["label"] for sample in test)),
        "aggregate": metrics_from_predictions(predictions),
        "predictions": predictions,
    }


def train_final_model(samples: list[dict[str, Any]]) -> dict[str, Any] | None:
    if len({int(sample["target"]) for sample in samples}) < 2:
        return None
    model = GaussianNB.fit(samples, SIDE_FEATURES)
    return {
        "features": SIDE_FEATURES,
        "model": model.to_dict(),
        "train_metrics": metrics_for(model, samples),
    }


def metrics_from_predictions(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if not predictions:
        return {}
    tp = tn = fp = fn = 0
    by_label: dict[str, Counter] = {}
    for item in predictions:
        expected = int(item["target"])
        predicted = int(item["prediction"])
        if expected == 1 and predicted == 1:
            tp += 1
        elif expected == 0 and predicted == 0:
            tn += 1
        elif expected == 0 and predicted == 1:
            fp += 1
        else:
            fn += 1
        counter = by_label.setdefault(item["label"], Counter())
        counter["total"] += 1
        counter[f"expected_{expected}"] += 1
        counter[f"predicted_{predicted}"] += 1

    total = tp + tn + fp + fn
    return {
        "total": total,
        "accuracy": (tp + tn) / total if total else 0.0,
        "abnormal_precision": tp / (tp + fp) if tp + fp else 0.0,
        "abnormal_recall": tp / (tp + fn) if tp + fn else 0.0,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "by_label": {label: dict(counter) for label, counter in by_label.items()},
    }


def write_report(output_dir: Path, results: dict[str, Any]) -> None:
    lines = ["# Annotated Video Classifier Report", ""]
    lines.append(f"Sample interval: {results['sample_seconds']} seconds")
    lines.append(f"Evaluation strategy: `{results['eval_strategy']}`")
    lines.append("")
    for mode, payload in results["modes"].items():
        lines.append(f"## {mode.title()} Mode")
        lines.append("")
        lines.append(f"- Samples: {payload['sample_count']}")
        lines.append(f"- Label counts: `{payload['label_counts']}`")
        agg = payload["evaluation"].get("aggregate", {})
        lines.append(
            f"- Aggregate: accuracy={agg.get('accuracy', 0.0):.3f}, "
            f"precision={agg.get('abnormal_precision', 0.0):.3f}, recall={agg.get('abnormal_recall', 0.0):.3f}, "
            f"confusion={agg.get('confusion', {})}"
        )
        if results["eval_strategy"] == "random_frame_split":
            lines.append(f"- Train/Test: {payload['evaluation']['train_count']} / {payload['evaluation']['test_count']}")
            lines.append(f"- Train labels: `{payload['evaluation']['train_label_counts']}`")
            lines.append(f"- Test labels: `{payload['evaluation']['test_label_counts']}`")
            lines.append("")
            continue
        lines.append("")
        lines.append("| held-out video | train | test | accuracy | precision | recall | confusion |")
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for fold in payload["evaluation"]["folds"]:
            if fold.get("skipped"):
                lines.append(f"| {Path(fold['held_out_video']).name} | {fold['train_count']} | {fold['test_count']} | skipped | skipped | skipped | {fold['reason']} |")
                continue
            metrics = fold["metrics"]
            lines.append(
                f"| {Path(fold['held_out_video']).name} | {fold['train_count']} | {fold['test_count']} | "
                f"{metrics.get('accuracy', 0.0):.3f} | {metrics.get('abnormal_precision', 0.0):.3f} | "
                f"{metrics.get('abnormal_recall', 0.0):.3f} | {metrics.get('confusion', {})} |"
            )
        lines.append("")
    (output_dir / "annotated_classifier_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train posture classifiers from full-video segment annotations.")
    parser.add_argument("--annotations-dir", default="data/annotations")
    parser.add_argument("--config", default="configs/relaxed.yaml")
    parser.add_argument("--reference", default="output/lab_20260610/processed_relaxed/reference.json")
    parser.add_argument("--output-dir", default="output/annotated_classifier")
    parser.add_argument("--sample-seconds", type=float, default=2.0)
    parser.add_argument(
        "--eval-strategy",
        choices=["leave_one_video_out", "random_frame_split"],
        default="leave_one_video_out",
        help="Use random_frame_split to merge all frames and randomly split train/test.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    annotations = [load_annotation(path) for path in sorted(Path(args.annotations_dir).glob("*_segments.json"))]
    reference = load_reference(args.reference)
    config = merge_reference(load_config(args.config), reference)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "annotations_dir": args.annotations_dir,
        "config": args.config,
        "reference": args.reference,
        "sample_seconds": args.sample_seconds,
        "eval_strategy": args.eval_strategy,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "modes": {},
    }
    for mode in ("computer", "reading"):
        samples: list[dict[str, Any]] = []
        for annotation in annotations:
            samples.extend(
                extract_samples_from_annotation(
                    annotation=annotation,
                    config=config,
                    reference=reference,
                    mode=mode,
                    sample_seconds=args.sample_seconds,
                )
            )
        if args.eval_strategy == "random_frame_split":
            evaluation = evaluate_random_frame_split(samples, train_ratio=args.train_ratio, seed=args.seed)
        else:
            evaluation = evaluate_leave_one_video_out(samples)
        final_model = train_final_model(samples)
        results["modes"][mode] = {
            "sample_count": len(samples),
            "label_counts": dict(Counter(sample["label"] for sample in samples)),
            "target_counts": dict(Counter(str(sample["target"]) for sample in samples)),
            "evaluation": evaluation,
            "final_model": final_model,
        }
        (output_dir / f"{mode}_annotated_samples.jsonl").write_text(
            "\n".join(json.dumps(sample, ensure_ascii=False) for sample in samples) + "\n",
            encoding="utf-8",
        )

    (output_dir / "annotated_classifier_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(output_dir, results)
    print(f"Wrote annotated classifier results to {output_dir}")


if __name__ == "__main__":
    main()
