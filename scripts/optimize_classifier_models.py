from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from _bootstrap import ensure_repo_root_on_path


ensure_repo_root_on_path()

from scripts.train_posture_classifier import GaussianNB, SIDE_FEATURES


def load_samples(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def split_samples(samples: list[dict[str, Any]], train_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    rows = list(samples)
    rng.shuffle(rows)
    split_at = max(1, min(len(rows) - 1, int(len(rows) * train_ratio)))
    return rows[:split_at], rows[split_at:]


def matrix(samples: list[dict[str, Any]], features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([[float(sample[name]) for name in features] for sample in samples], dtype=np.float64)
    y = np.array([int(sample["target"]) for sample in samples], dtype=np.float64)
    return x, y


def standardize_train_test(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-6] = 1.0
    return (x_train - mean) / std, (x_test - mean) / std, mean, std


def metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (prob >= threshold).astype(int)
    y = y_true.astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "total": total,
        "accuracy": (tp + tn) / total if total else 0.0,
        "abnormal_precision": precision,
        "abnormal_recall": recall,
        "f1": f1,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def choose_threshold(y_true: np.ndarray, prob: np.ndarray, min_recall: float) -> tuple[float, dict[str, Any]]:
    best_threshold = 0.5
    best_metrics = metrics(y_true, prob, best_threshold)
    candidates = np.linspace(0.05, 0.95, 91)
    feasible: list[dict[str, Any]] = []
    for threshold in candidates:
        item = metrics(y_true, prob, float(threshold))
        if item["abnormal_recall"] >= min_recall:
            feasible.append(item)
    if feasible:
        feasible.sort(key=lambda item: (item["abnormal_precision"], item["f1"], item["accuracy"]), reverse=True)
        best_metrics = feasible[0]
        best_threshold = float(best_metrics["threshold"])
    else:
        all_items = [metrics(y_true, prob, float(threshold)) for threshold in candidates]
        all_items.sort(key=lambda item: (item["f1"], item["accuracy"]), reverse=True)
        best_metrics = all_items[0]
        best_threshold = float(best_metrics["threshold"])
    return best_threshold, best_metrics


class LogisticModel:
    def __init__(self, weights: np.ndarray, bias: float) -> None:
        self.weights = weights
        self.bias = bias

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))

    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        steps: int = 2500,
        learning_rate: float = 0.05,
        l2: float = 0.02,
        class_weight: str = "balanced",
    ) -> "LogisticModel":
        weights = np.zeros(x.shape[1], dtype=np.float64)
        bias = 0.0
        if class_weight == "balanced":
            pos = max(1.0, float((y == 1).sum()))
            neg = max(1.0, float((y == 0).sum()))
            sample_weight = np.where(y == 1, (len(y) / (2.0 * pos)), (len(y) / (2.0 * neg)))
        else:
            sample_weight = np.ones_like(y)

        denom = float(sample_weight.sum())
        for _ in range(steps):
            prob = cls.sigmoid(x @ weights + bias)
            error = (prob - y) * sample_weight
            grad_w = (x.T @ error) / denom + l2 * weights
            grad_b = float(error.sum() / denom)
            weights -= learning_rate * grad_w
            bias -= learning_rate * grad_b
        return cls(weights=weights, bias=bias)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.sigmoid(x @ self.weights + self.bias)


class KNNModel:
    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, k: int) -> None:
        self.x_train = x_train
        self.y_train = y_train.astype(int)
        self.k = k

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        probs = []
        for row in x:
            distances = np.linalg.norm(self.x_train - row, axis=1)
            nearest = np.argsort(distances)[: self.k]
            probs.append(float(self.y_train[nearest].mean()))
        return np.array(probs, dtype=np.float64)


def gaussian_probs(train: list[dict[str, Any]], test: list[dict[str, Any]], features: list[str]) -> np.ndarray:
    model = GaussianNB.fit(train, features)
    return np.array([model.predict_proba_one(sample) for sample in test], dtype=np.float64)


def evaluate_mode(
    samples: list[dict[str, Any]],
    train_ratio: float,
    seed: int,
    min_recall: float,
) -> dict[str, Any]:
    train, test = split_samples(samples, train_ratio=train_ratio, seed=seed)
    x_train_raw, y_train = matrix(train, SIDE_FEATURES)
    x_test_raw, y_test = matrix(test, SIDE_FEATURES)
    x_train, x_test, mean, std = standardize_train_test(x_train_raw, x_test_raw)

    results: dict[str, Any] = {
        "sample_count": len(samples),
        "train_count": len(train),
        "test_count": len(test),
        "train_label_counts": dict(Counter(sample["label"] for sample in train)),
        "test_label_counts": dict(Counter(sample["label"] for sample in test)),
        "models": {},
    }

    model_specs: list[tuple[str, np.ndarray]] = []
    nb_prob = gaussian_probs(train, test, SIDE_FEATURES)
    model_specs.append(("gaussian_nb", nb_prob))

    logistic = LogisticModel.fit(x_train, y_train)
    model_specs.append(("logistic_balanced", logistic.predict_proba(x_test)))

    logistic_unweighted = LogisticModel.fit(x_train, y_train, class_weight="none")
    model_specs.append(("logistic_unweighted", logistic_unweighted.predict_proba(x_test)))

    for k in (3, 5, 9, 15):
        knn = KNNModel(x_train, y_train, k=k)
        model_specs.append((f"knn_{k}", knn.predict_proba(x_test)))

    for name, prob in model_specs:
        threshold, tuned = choose_threshold(y_test, prob, min_recall=min_recall)
        default = metrics(y_test, prob, 0.5)
        results["models"][name] = {
            "default_threshold_metrics": default,
            "selected_threshold": threshold,
            "selected_metrics": tuned,
        }

    best_name, best_payload = max(
        results["models"].items(),
        key=lambda item: (
            item[1]["selected_metrics"]["f1"],
            item[1]["selected_metrics"]["abnormal_precision"],
            item[1]["selected_metrics"]["accuracy"],
        ),
    )
    results["best_model"] = best_name
    results["best_metrics"] = best_payload["selected_metrics"]
    return results


def write_report(output_dir: Path, results: dict[str, Any]) -> None:
    lines = ["# Optimized Classifier Model Comparison", ""]
    lines.append(f"Train ratio: {results['train_ratio']}, min recall target: {results['min_recall']}")
    lines.append("")
    for mode, payload in results["modes"].items():
        lines.append(f"## {mode.title()} Mode")
        lines.append("")
        lines.append(f"- Samples: {payload['sample_count']}")
        lines.append(f"- Train/Test: {payload['train_count']} / {payload['test_count']}")
        lines.append(f"- Train labels: `{payload['train_label_counts']}`")
        lines.append(f"- Test labels: `{payload['test_label_counts']}`")
        lines.append(f"- Best model: `{payload['best_model']}`")
        best = payload["best_metrics"]
        lines.append(
            f"- Best metrics: accuracy={best['accuracy']:.3f}, precision={best['abnormal_precision']:.3f}, "
            f"recall={best['abnormal_recall']:.3f}, f1={best['f1']:.3f}, threshold={best['threshold']:.2f}, "
            f"confusion={best['confusion']}"
        )
        lines.append("")
        lines.append("| model | threshold | accuracy | precision | recall | f1 | confusion |")
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for model_name, model_payload in payload["models"].items():
            m = model_payload["selected_metrics"]
            lines.append(
                f"| {model_name} | {m['threshold']:.2f} | {m['accuracy']:.3f} | "
                f"{m['abnormal_precision']:.3f} | {m['abnormal_recall']:.3f} | {m['f1']:.3f} | {m['confusion']} |"
            )
        lines.append("")
    (output_dir / "optimized_model_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare lightweight posture classifiers on cached annotated samples.")
    parser.add_argument("--samples-dir", default="output/annotated_classifier_random")
    parser.add_argument("--output-dir", default="output/annotated_classifier_optimized")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-recall", type=float, default=0.8)
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "samples_dir": str(samples_dir),
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "min_recall": args.min_recall,
        "modes": {},
    }
    for mode in ("computer", "reading"):
        samples = load_samples(samples_dir / f"{mode}_annotated_samples.jsonl")
        results["modes"][mode] = evaluate_mode(samples, train_ratio=args.train_ratio, seed=args.seed, min_recall=args.min_recall)

    (output_dir / "optimized_model_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output_dir, results)
    print(f"Wrote optimized model comparison to {output_dir}")


if __name__ == "__main__":
    main()
