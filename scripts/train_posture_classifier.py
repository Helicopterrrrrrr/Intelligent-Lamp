from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from _bootstrap import ensure_repo_root_on_path


ensure_repo_root_on_path()

from smartled_pose import PosturePipeline, load_config
from smartled_pose.config import load_reference, merge_reference
from smartled_pose.feature_extractor import LEFT_HIP, LEFT_SHOULDER, NOSE, RIGHT_HIP, RIGHT_SHOULDER
from smartled_pose.geometry import midpoint
from smartled_pose.types import Keypoint, PostureOutput


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}

MODE_LABELS = {
    "computer": {
        "normal": ["calibration_normal", "computer_normal"],
        "abnormal": ["computer_abnormal"],
    },
    "reading": {
        "normal": ["calibration_normal", "reading_normal"],
        "abnormal": ["reading_abnormal"],
    },
}

BASELINE_FEATURES = [
    "pose_conf",
    "kp_valid_ratio",
    "bbox_area_ratio",
    "shoulder_width_px",
    "head_shoulder_distance_ratio",
    "head_tilt_deg",
]

SIDE_FEATURES = BASELINE_FEATURES + [
    "bbox_width_ratio",
    "bbox_height_ratio",
    "bbox_aspect",
    "shoulder_ratio",
    "bbox_ratio",
    "head_distance_ratio_deviation",
    "nose_shoulder_dx_ratio",
    "abs_nose_shoulder_dx_ratio",
    "nose_shoulder_y_ratio",
    "nose_bbox_x_ratio",
    "nose_bbox_y_ratio",
    "shoulder_bbox_y_ratio",
    "left_shoulder_conf",
    "right_shoulder_conf",
    "nose_conf",
]


@dataclass
class GaussianNB:
    feature_names: list[str]
    classes: list[int]
    means: dict[int, list[float]]
    variances: dict[int, list[float]]
    priors: dict[int, float]

    @classmethod
    def fit(cls, samples: list[dict[str, Any]], feature_names: list[str]) -> "GaussianNB":
        classes = sorted({int(sample["target"]) for sample in samples})
        total = len(samples)
        means: dict[int, list[float]] = {}
        variances: dict[int, list[float]] = {}
        priors: dict[int, float] = {}
        for class_id in classes:
            rows = [sample for sample in samples if int(sample["target"]) == class_id]
            matrix = np.array([[float(row[name]) for name in feature_names] for row in rows], dtype=np.float64)
            means[class_id] = matrix.mean(axis=0).tolist()
            variances[class_id] = (matrix.var(axis=0) + 1e-6).tolist()
            priors[class_id] = len(rows) / float(total or 1)
        return cls(feature_names=feature_names, classes=classes, means=means, variances=variances, priors=priors)

    def predict_proba_one(self, sample: dict[str, Any]) -> float:
        x = np.array([float(sample[name]) for name in self.feature_names], dtype=np.float64)
        log_scores = {}
        for class_id in self.classes:
            mean = np.array(self.means[class_id], dtype=np.float64)
            var = np.array(self.variances[class_id], dtype=np.float64)
            log_likelihood = -0.5 * np.sum(np.log(2.0 * math.pi * var) + ((x - mean) ** 2) / var)
            log_scores[class_id] = math.log(self.priors[class_id] + 1e-12) + float(log_likelihood)

        max_score = max(log_scores.values())
        exp_scores = {class_id: math.exp(score - max_score) for class_id, score in log_scores.items()}
        denom = sum(exp_scores.values()) or 1.0
        return exp_scores.get(1, 0.0) / denom

    def predict_one(self, sample: dict[str, Any], threshold: float = 0.5) -> int:
        return int(self.predict_proba_one(sample) >= threshold)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "gaussian_nb",
            "feature_names": self.feature_names,
            "classes": self.classes,
            "means": self.means,
            "variances": self.variances,
            "priors": self.priors,
        }


def kp_at(output: PostureOutput, index: int) -> Keypoint | None:
    if not output.detection or index >= len(output.detection.keypoints):
        return None
    return output.detection.keypoints[index]


def valid(kp: Keypoint | None, min_conf: float = 0.2) -> bool:
    return kp is not None and kp.conf >= min_conf


def safe_ratio(value: float, denominator: float | None) -> float:
    if denominator is None or abs(denominator) < 1e-6:
        return 0.0
    return value / denominator


def features_from_output(output: PostureOutput, label: str, target: int, source: Path, frame_index: int) -> dict[str, Any] | None:
    if not output.detection or not output.features:
        return None

    features = output.features
    box = output.detection.bbox
    left_shoulder = kp_at(output, LEFT_SHOULDER)
    right_shoulder = kp_at(output, RIGHT_SHOULDER)
    nose = kp_at(output, NOSE)
    left_hip = kp_at(output, LEFT_HIP)
    right_hip = kp_at(output, RIGHT_HIP)

    shoulder_center = None
    if valid(left_shoulder) and valid(right_shoulder):
        shoulder_center = midpoint(left_shoulder, right_shoulder)

    hip_center = None
    if valid(left_hip) and valid(right_hip):
        hip_center = midpoint(left_hip, right_hip)

    bbox_w = box.width
    bbox_h = box.height
    shoulder_width = features.shoulder_width_px

    nose_shoulder_dx_ratio = 0.0
    nose_shoulder_y_ratio = 0.0
    nose_bbox_x_ratio = 0.0
    nose_bbox_y_ratio = 0.0
    if valid(nose):
        nose_bbox_x_ratio = safe_ratio(nose.x - box.x1, bbox_w)
        nose_bbox_y_ratio = safe_ratio(nose.y - box.y1, bbox_h)
        if shoulder_center:
            nose_shoulder_dx_ratio = safe_ratio(nose.x - shoulder_center.x, shoulder_width)
            nose_shoulder_y_ratio = safe_ratio(nose.y - shoulder_center.y, shoulder_width)

    shoulder_bbox_y_ratio = 0.0
    if shoulder_center:
        shoulder_bbox_y_ratio = safe_ratio(shoulder_center.y - box.y1, bbox_h)

    hip_shoulder_dx_ratio = 0.0
    if shoulder_center and hip_center:
        hip_shoulder_dx_ratio = safe_ratio(hip_center.x - shoulder_center.x, shoulder_width)

    reference = output.metrics.get("reference", {}) if output.metrics else {}
    shoulder_ref = reference.get("shoulder_width_px")
    bbox_ref = reference.get("bbox_area_ratio")
    head_ref = reference.get("head_shoulder_distance_ratio")

    sample = {
        "label": label,
        "target": target,
        "source": str(source),
        "frame_index": frame_index,
        "pose_conf": features.pose_conf,
        "kp_valid_ratio": features.kp_valid_ratio,
        "bbox_area_ratio": features.bbox_area_ratio,
        "shoulder_width_px": features.shoulder_width_px,
        "head_shoulder_distance_ratio": features.head_shoulder_distance_ratio,
        "head_tilt_deg": features.head_tilt_deg,
        "bbox_width_ratio": bbox_w / 1280.0,
        "bbox_height_ratio": bbox_h / 720.0,
        "bbox_aspect": safe_ratio(bbox_w, bbox_h),
        "shoulder_ratio": safe_ratio(features.shoulder_width_px, shoulder_ref),
        "bbox_ratio": safe_ratio(features.bbox_area_ratio, bbox_ref),
        "head_distance_ratio_deviation": safe_ratio(features.head_shoulder_distance_ratio, head_ref),
        "nose_shoulder_dx_ratio": nose_shoulder_dx_ratio,
        "abs_nose_shoulder_dx_ratio": abs(nose_shoulder_dx_ratio),
        "nose_shoulder_y_ratio": nose_shoulder_y_ratio,
        "nose_bbox_x_ratio": nose_bbox_x_ratio,
        "nose_bbox_y_ratio": nose_bbox_y_ratio,
        "shoulder_bbox_y_ratio": shoulder_bbox_y_ratio,
        "hip_shoulder_dx_ratio": hip_shoulder_dx_ratio,
        "left_shoulder_conf": left_shoulder.conf if left_shoulder else 0.0,
        "right_shoulder_conf": right_shoulder.conf if right_shoulder else 0.0,
        "nose_conf": nose.conf if nose else 0.0,
    }
    return sample


def attach_reference_to_metrics(output: PostureOutput, reference: dict[str, Any]) -> None:
    output.metrics["reference"] = reference


def iter_media(dataset_root: Path, labels: list[str]) -> list[tuple[str, Path]]:
    media: list[tuple[str, Path]] = []
    for label in labels:
        for root_name, suffixes in (("images", IMAGE_SUFFIXES), ("videos", VIDEO_SUFFIXES)):
            label_dir = dataset_root / root_name / label
            if not label_dir.exists():
                continue
            for path in sorted(label_dir.iterdir()):
                if path.suffix.lower() in suffixes:
                    media.append((label, path))
    return media


def extract_samples(
    dataset_root: Path,
    config: dict,
    reference: dict[str, Any],
    mode: str,
    video_sample_every: int,
) -> list[dict[str, Any]]:
    labels_for_mode = MODE_LABELS[mode]
    label_to_target = {label: 0 for label in labels_for_mode["normal"]}
    label_to_target.update({label: 1 for label in labels_for_mode["abnormal"]})

    samples: list[dict[str, Any]] = []
    for label, path in iter_media(dataset_root, list(label_to_target)):
        target = label_to_target[label]
        pipeline = PosturePipeline(config)
        if path.suffix.lower() in IMAGE_SUFFIXES:
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            snapshot = pipeline.process_frame(frame, timestamp=0.0)
            attach_reference_to_metrics(snapshot.output, reference)
            sample = features_from_output(snapshot.output, label, target, path, frame_index=0)
            if sample:
                samples.append(sample)
            continue

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            continue
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % video_sample_every == 0:
                snapshot = pipeline.process_frame(frame, timestamp=frame_index / fps)
                attach_reference_to_metrics(snapshot.output, reference)
                sample = features_from_output(snapshot.output, label, target, path, frame_index)
                if sample:
                    samples.append(sample)
            frame_index += 1
        capture.release()
    return samples


def split_samples(
    samples: list[dict[str, Any]],
    train_ratio: float,
    split_strategy: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if split_strategy == "image_train_video_test":
        train = [sample for sample in samples if f"{Path('images')}" in sample["source"]]
        test = [sample for sample in samples if f"{Path('videos')}" in sample["source"]]
        return train, test

    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    by_source: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        by_source.setdefault(sample["source"], []).append(sample)

    for rows in by_source.values():
        rows.sort(key=lambda row: int(row["frame_index"]))
        if len(rows) == 1:
            train.append(rows[0])
            continue
        split_at = max(1, min(len(rows) - 1, int(len(rows) * train_ratio)))
        train.extend(rows[:split_at])
        test.extend(rows[split_at:])
    return train, test


def metrics_for(model: GaussianNB, samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {}
    tp = tn = fp = fn = 0
    by_label: dict[str, Counter] = {}
    probs: list[float] = []
    for sample in samples:
        expected = int(sample["target"])
        predicted = model.predict_one(sample)
        prob = model.predict_proba_one(sample)
        probs.append(prob)
        if expected == 1 and predicted == 1:
            tp += 1
        elif expected == 0 and predicted == 0:
            tn += 1
        elif expected == 0 and predicted == 1:
            fp += 1
        else:
            fn += 1
        counter = by_label.setdefault(sample["label"], Counter())
        counter[f"expected_{expected}"] += 1
        counter[f"predicted_{predicted}"] += 1
        counter["total"] += 1

    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "total": total,
        "accuracy": (tp + tn) / total if total else 0.0,
        "abnormal_precision": precision,
        "abnormal_recall": recall,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "avg_abnormal_probability": sum(probs) / len(probs),
        "by_label": {label: dict(counter) for label, counter in by_label.items()},
    }


def write_report(output_dir: Path, results: dict[str, Any]) -> None:
    lines = ["# Posture Classifier Experiment", ""]
    lines.append("This experiment trains tiny Gaussian Naive Bayes classifiers from YOLO pose features.")
    lines.append(
        "Chronological splits are useful for separability checks but not a final generalization claim. "
        "The image_train_video_test split is stricter, but it may underfit when only a few snapshots are available."
    )
    lines.append("")
    for mode, mode_result in results["modes"].items():
        lines.append(f"## {mode.title()} Mode")
        lines.append("")
        lines.append(f"- Samples: {mode_result['sample_count']}")
        lines.append(f"- Train/Test: {mode_result['train_count']} / {mode_result['test_count']}")
        lines.append("")
        lines.append("| feature set | accuracy | abnormal precision | abnormal recall | confusion |")
        lines.append("|---|---:|---:|---:|---|")
        for feature_set in ("baseline", "side_view"):
            metrics = mode_result[feature_set]["test_metrics"]
            lines.append(
                f"| {feature_set} | {metrics.get('accuracy', 0.0):.3f} | "
                f"{metrics.get('abnormal_precision', 0.0):.3f} | "
                f"{metrics.get('abnormal_recall', 0.0):.3f} | "
                f"{metrics.get('confusion', {})} |"
            )
        lines.append("")
        lines.append("Per-label side-view test counts:")
        for label, counts in mode_result["side_view"]["test_metrics"].get("by_label", {}).items():
            lines.append(f"- `{label}`: {counts}")
        lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append("- If side-view features improve recall over baseline, the current camera needs side-specific geometry features.")
    lines.append("- If both classifiers are weak, collect more clips per mode and train/test by whole clips rather than frames.")
    (output_dir / "classifier_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train small posture classifiers from collected lab pose features.")
    parser.add_argument("--dataset-root", default="data/lab_20260610")
    parser.add_argument("--config", default="configs/relaxed.yaml")
    parser.add_argument("--reference", default="output/lab_20260610/processed_relaxed/reference.json")
    parser.add_argument("--output-dir", default="output/lab_20260610/classifier")
    parser.add_argument("--video-sample-every", type=int, default=5)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument(
        "--split-strategy",
        choices=["chronological", "image_train_video_test"],
        default="chronological",
        help="chronological splits within each media file; image_train_video_test trains on snapshots and tests videos.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference = load_reference(args.reference)
    config = merge_reference(load_config(args.config), reference)

    results: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "config": args.config,
        "reference": args.reference,
        "modes": {},
    }
    for mode in ("computer", "reading"):
        samples = extract_samples(
            dataset_root=dataset_root,
            config=config,
            reference=reference,
            mode=mode,
            video_sample_every=max(1, args.video_sample_every),
        )
        train, test = split_samples(samples, train_ratio=args.train_ratio, split_strategy=args.split_strategy)
        mode_result: dict[str, Any] = {
            "sample_count": len(samples),
            "train_count": len(train),
            "test_count": len(test),
            "split_strategy": args.split_strategy,
            "label_counts": dict(Counter(sample["label"] for sample in samples)),
            "target_counts": dict(Counter(str(sample["target"]) for sample in samples)),
        }
        for feature_set, feature_names in (("baseline", BASELINE_FEATURES), ("side_view", SIDE_FEATURES)):
            model = GaussianNB.fit(train, feature_names)
            mode_result[feature_set] = {
                "features": feature_names,
                "train_metrics": metrics_for(model, train),
                "test_metrics": metrics_for(model, test),
                "model": model.to_dict(),
            }
        results["modes"][mode] = mode_result

        (output_dir / f"{mode}_samples.jsonl").write_text(
            "\n".join(json.dumps(sample, ensure_ascii=False) for sample in samples) + "\n",
            encoding="utf-8",
        )

    (output_dir / "classifier_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output_dir, results)
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
