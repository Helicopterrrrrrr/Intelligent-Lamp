from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .feature_extractor import LEFT_HIP, LEFT_SHOULDER, NOSE, RIGHT_HIP, RIGHT_SHOULDER
from .geometry import midpoint
from .types import Keypoint, PostureOutput, ReferenceFeatures


SIDE_FEATURES = [
    "pose_conf",
    "kp_valid_ratio",
    "bbox_area_ratio",
    "shoulder_width_px",
    "head_shoulder_distance_ratio",
    "head_tilt_deg",
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


@dataclass(slots=True)
class ClassifierDecision:
    mode: str
    probability: float
    threshold: float
    raw_label: str
    stable_label: str
    event_state: str
    model_name: str


@dataclass(slots=True)
class StateClassifierDecision:
    label: str
    text: str
    confidence: float
    model_name: str


STATE_TEXT = {
    "calibration_normal": "校准正常坐姿",
    "computer_normal": "正常用电脑",
    "computer_abnormal": "用电脑异常",
    "reading_normal": "正常看书/写字",
    "reading_abnormal": "看书异常",
    "absent": "离开座位",
    "ignore": "忽略/过渡",
}

STATE_LABELS = [
    "calibration_normal",
    "computer_normal",
    "computer_abnormal",
    "reading_normal",
    "reading_abnormal",
    "ignore",
]

def _kp(output: PostureOutput, index: int) -> Keypoint | None:
    if not output.detection or index >= len(output.detection.keypoints):
        return None
    return output.detection.keypoints[index]


def _valid(kp: Keypoint | None, min_conf: float = 0.2) -> bool:
    return kp is not None and kp.conf >= min_conf


def _safe_ratio(value: float, denominator: float | None) -> float:
    if denominator is None or abs(denominator) < 1e-6:
        return 0.0
    return value / denominator


def reference_to_dict(reference: ReferenceFeatures) -> dict[str, float | None]:
    return {
        "shoulder_width_px": reference.shoulder_width_px,
        "bbox_area_ratio": reference.bbox_area_ratio,
        "head_shoulder_distance_ratio": reference.head_shoulder_distance_ratio,
        "torso_angle_deg": reference.torso_angle_deg,
        "head_tilt_deg": reference.head_tilt_deg,
    }


def extract_side_features(
    output: PostureOutput,
    reference: ReferenceFeatures,
    frame_width: int = 1280,
    frame_height: int = 720,
) -> dict[str, float] | None:
    if not output.detection or not output.features:
        return None

    pose = output.features
    box = output.detection.bbox
    left_shoulder = _kp(output, LEFT_SHOULDER)
    right_shoulder = _kp(output, RIGHT_SHOULDER)
    nose = _kp(output, NOSE)
    left_hip = _kp(output, LEFT_HIP)
    right_hip = _kp(output, RIGHT_HIP)

    shoulder_center = None
    if _valid(left_shoulder) and _valid(right_shoulder):
        shoulder_center = midpoint(left_shoulder, right_shoulder)

    hip_center = None
    if _valid(left_hip) and _valid(right_hip):
        hip_center = midpoint(left_hip, right_hip)

    bbox_w = box.width
    bbox_h = box.height
    shoulder_width = pose.shoulder_width_px

    nose_shoulder_dx_ratio = 0.0
    nose_shoulder_y_ratio = 0.0
    nose_bbox_x_ratio = 0.0
    nose_bbox_y_ratio = 0.0
    if _valid(nose):
        nose_bbox_x_ratio = _safe_ratio(nose.x - box.x1, bbox_w)
        nose_bbox_y_ratio = _safe_ratio(nose.y - box.y1, bbox_h)
        if shoulder_center:
            nose_shoulder_dx_ratio = _safe_ratio(nose.x - shoulder_center.x, shoulder_width)
            nose_shoulder_y_ratio = _safe_ratio(nose.y - shoulder_center.y, shoulder_width)

    shoulder_bbox_y_ratio = 0.0
    if shoulder_center:
        shoulder_bbox_y_ratio = _safe_ratio(shoulder_center.y - box.y1, bbox_h)

    hip_shoulder_dx_ratio = 0.0
    if shoulder_center and hip_center:
        hip_shoulder_dx_ratio = _safe_ratio(hip_center.x - shoulder_center.x, shoulder_width)

    return {
        "pose_conf": pose.pose_conf,
        "kp_valid_ratio": pose.kp_valid_ratio,
        "bbox_area_ratio": pose.bbox_area_ratio,
        "shoulder_width_px": pose.shoulder_width_px,
        "head_shoulder_distance_ratio": pose.head_shoulder_distance_ratio,
        "head_tilt_deg": pose.head_tilt_deg,
        "bbox_width_ratio": _safe_ratio(bbox_w, float(frame_width)),
        "bbox_height_ratio": _safe_ratio(bbox_h, float(frame_height)),
        "bbox_aspect": _safe_ratio(bbox_w, bbox_h),
        "shoulder_ratio": _safe_ratio(pose.shoulder_width_px, reference.shoulder_width_px),
        "bbox_ratio": _safe_ratio(pose.bbox_area_ratio, reference.bbox_area_ratio),
        "head_distance_ratio_deviation": _safe_ratio(
            pose.head_shoulder_distance_ratio,
            reference.head_shoulder_distance_ratio,
        ),
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


class KNNPostureClassifier:
    def __init__(
        self,
        mode: str,
        samples_path: str | Path,
        k: int,
        threshold: float,
        feature_names: list[str] | None = None,
    ) -> None:
        self.mode = mode
        self.samples_path = Path(samples_path)
        self.k = k
        self.threshold = threshold
        self.feature_names = feature_names or SIDE_FEATURES
        self.model_name = f"knn_{k}"
        samples = self._load_samples(self.samples_path, mode)
        if not samples:
            raise RuntimeError(f"No classifier samples for mode={mode}: {self.samples_path}")
        self._fit(samples)

    @staticmethod
    def _load_samples(samples_path: Path, mode: str) -> list[dict[str, Any]]:
        if not samples_path.exists():
            return []
        mode_labels = {
            "computer": {"computer_normal", "computer_abnormal"},
            "reading": {"reading_normal", "reading_abnormal"},
        }[mode]
        rows = []
        for line in samples_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("label") in mode_labels:
                rows.append(row)
        return rows

    def _fit(self, samples: list[dict[str, Any]]) -> None:
        x = np.array([[float(sample.get(name, 0.0)) for name in self.feature_names] for sample in samples], dtype=np.float64)
        y = np.array([int(sample["target"]) for sample in samples], dtype=np.int32)
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        self.std[self.std < 1e-6] = 1.0
        self.x_train = (x - self.mean) / self.std
        self.y_train = y

    def predict_probability(self, feature_row: dict[str, float]) -> float:
        x = np.array([float(feature_row.get(name, 0.0)) for name in self.feature_names], dtype=np.float64)
        x = (x - self.mean) / self.std
        distances = np.linalg.norm(self.x_train - x, axis=1)
        nearest = np.argsort(distances)[: self.k]
        return float(self.y_train[nearest].mean())


class MultiStateKNNClassifier:
    def __init__(
        self,
        samples_path: str | Path,
        k: int = 3,
        feature_names: list[str] | None = None,
    ) -> None:
        self.samples_path = Path(samples_path)
        self.k = k
        self.feature_names = feature_names or SIDE_FEATURES
        self.model_name = f"state_knn_{k}"
        samples = self._load_samples(self.samples_path)
        if not samples:
            raise RuntimeError(f"No state classifier samples: {self.samples_path}")
        self._fit(samples)

    @staticmethod
    def _load_samples(samples_path: Path) -> list[dict[str, Any]]:
        if not samples_path.exists():
            return []
        rows = []
        for line in samples_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("label") in STATE_LABELS:
                rows.append(row)
        return rows

    def _fit(self, samples: list[dict[str, Any]]) -> None:
        x = np.array([[float(sample.get(name, 0.0)) for name in self.feature_names] for sample in samples], dtype=np.float64)
        y = np.array([STATE_LABELS.index(sample["label"]) for sample in samples], dtype=np.int32)
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        self.std[self.std < 1e-6] = 1.0
        self.x_train = (x - self.mean) / self.std
        self.y_train = y

    def predict(self, feature_row: dict[str, float]) -> StateClassifierDecision:
        x = np.array([float(feature_row.get(name, 0.0)) for name in self.feature_names], dtype=np.float64)
        x = (x - self.mean) / self.std
        distances = np.linalg.norm(self.x_train - x, axis=1)
        nearest = np.argsort(distances)[: self.k]
        counts: dict[int, int] = {}
        for idx in nearest:
            label_id = int(self.y_train[idx])
            counts[label_id] = counts.get(label_id, 0) + 1
        label_id, count = max(counts.items(), key=lambda item: item[1])
        label = STATE_LABELS[label_id]
        return StateClassifierDecision(
            label=label,
            text=STATE_TEXT[label],
            confidence=count / float(self.k),
            model_name=self.model_name,
        )


def apply_side_view_overrides(decision: StateClassifierDecision, feature_row: dict[str, float]) -> StateClassifierDecision:
    head_deviation = feature_row.get("head_distance_ratio_deviation", 0.0)
    forward_ratio = feature_row.get("abs_nose_shoulder_dx_ratio", 0.0)
    shoulder_ratio = feature_row.get("shoulder_ratio", 0.0)
    bbox_ratio = feature_row.get("bbox_ratio", 0.0)
    kp_ratio = feature_row.get("kp_valid_ratio", 0.0)
    nose_x = feature_row.get("nose_bbox_x_ratio", 1.0)
    nose_y = feature_row.get("nose_bbox_y_ratio", 1.0)
    normal_to_abnormal = {
        "computer_normal": "computer_abnormal",
        "reading_normal": "reading_abnormal",
    }

    # Current side-view feedback: computer use is best separated from reading
    # by nose position inside the person box. The user-facing classes should
    # remain mode-aware even when the KNN neighborhood is biased toward reading.
    if (
        decision.label in {"reading_normal", "reading_abnormal"}
        and kp_ratio >= 0.34
        and bbox_ratio >= 1.03
        and shoulder_ratio >= 1.55
        and nose_x <= 0.17
        and nose_y <= 0.49
    ):
        return StateClassifierDecision(
            label="computer_abnormal",
            text=STATE_TEXT["computer_abnormal"],
            confidence=max(decision.confidence, 0.72),
            model_name=f"{decision.model_name}+computer_abnormal_feedback",
        )

    if (
        decision.label in {"reading_normal", "reading_abnormal"}
        and kp_ratio >= 0.34
        and 0.80 <= bbox_ratio <= 1.10
        and head_deviation >= 0.58
        and forward_ratio <= 0.65
        and nose_x >= 0.18
        and nose_y <= 0.35
    ):
        return StateClassifierDecision(
            label="computer_normal",
            text=STATE_TEXT["computer_normal"],
            confidence=max(decision.confidence, 0.70),
            model_name=f"{decision.model_name}+computer_normal_feedback",
        )

    if (
        decision.label == "reading_abnormal"
        and kp_ratio >= 0.34
        and bbox_ratio <= 0.88
        and 0.35 <= nose_y <= 0.49
        and head_deviation >= 0.50
        and shoulder_ratio <= 1.57
        and nose_x >= 0.24
    ):
        return StateClassifierDecision(
            label="reading_normal",
            text=STATE_TEXT["reading_normal"],
            confidence=max(decision.confidence, 0.70),
            model_name=f"{decision.model_name}+reading_normal_feedback",
        )

    # Feedback showed near-computer posture often appears as computer_normal
    # while head/shoulder and nose/shoulder forward ratios are already high.
    # Keep the override narrow: when shoulder width collapses, the side-view
    # ratios become unstable and caused false abnormal feedback.
    if (
        decision.label in {"computer_normal", "reading_normal"}
        and kp_ratio >= 0.45
        and 0.88 <= shoulder_ratio <= 1.08
        and 1.24 <= head_deviation <= 1.70
        and 1.05 <= forward_ratio <= 1.35
    ):
        abnormal_label = normal_to_abnormal[decision.label]
        return StateClassifierDecision(
            label=abnormal_label,
            text=STATE_TEXT[abnormal_label],
            confidence=max(decision.confidence, 0.67),
            model_name=f"{decision.model_name}+side_override",
        )

    if (
        decision.label in normal_to_abnormal
        and kp_ratio >= 0.35
        and shoulder_ratio >= 1.40
        and bbox_ratio >= 0.98
        and forward_ratio >= 0.75
    ):
        abnormal_label = normal_to_abnormal[decision.label]
        return StateClassifierDecision(
            label=abnormal_label,
            text=STATE_TEXT[abnormal_label],
            confidence=max(decision.confidence, 0.67),
            model_name=f"{decision.model_name}+scale_override",
        )

    if (
        decision.label == "reading_normal"
        and kp_ratio >= 0.34
        and 0.52 <= forward_ratio <= 0.75
        and head_deviation <= 0.85
        and bbox_ratio >= 0.70
        and nose_y <= 0.34
    ):
        return StateClassifierDecision(
            label="computer_normal",
            text=STATE_TEXT["computer_normal"],
            confidence=max(decision.confidence, 0.67),
            model_name=f"{decision.model_name}+computer_pose_override",
        )

    if (
        decision.label == "reading_normal"
        and kp_ratio >= 0.40
        and forward_ratio >= 1.0
        and 1.12 <= shoulder_ratio <= 1.34
        and bbox_ratio >= 1.0
    ):
        return StateClassifierDecision(
            label="computer_abnormal",
            text=STATE_TEXT["computer_abnormal"],
            confidence=max(decision.confidence, 0.67),
            model_name=f"{decision.model_name}+computer_forward_override",
        )

    if (
        decision.label in {"reading_normal", "reading_abnormal"}
        and kp_ratio >= 0.45
        and forward_ratio >= 0.78
        and shoulder_ratio >= 1.32
        and bbox_ratio >= 1.05
        and nose_x <= 0.20
        and nose_y <= 0.42
    ):
        return StateClassifierDecision(
            label="computer_abnormal",
            text=STATE_TEXT["computer_abnormal"],
            confidence=max(decision.confidence, 0.68),
            model_name=f"{decision.model_name}+computer_screen_lean_override",
        )
    return decision


class ClassifierDecisionSmoother:
    def __init__(self, hold_seconds: float = 2.0, clear_seconds: float = 2.0) -> None:
        self.hold_seconds = hold_seconds
        self.clear_seconds = clear_seconds
        self.abnormal_started_at: float | None = None
        self.normal_started_at: float | None = None
        self.stable_label = "normal"

    def update(self, timestamp: float, is_abnormal: bool, presence_state: str) -> tuple[str, str]:
        if presence_state != "seated":
            self.abnormal_started_at = None
            self.normal_started_at = None
            self.stable_label = "normal"
            return self.stable_label, "inactive"

        if is_abnormal:
            self.normal_started_at = None
            if self.abnormal_started_at is None:
                self.abnormal_started_at = timestamp
            if timestamp - self.abnormal_started_at >= self.hold_seconds:
                self.stable_label = "abnormal"
                return self.stable_label, "warning_active"
            return self.stable_label, "pending_abnormal"

        self.abnormal_started_at = None
        if self.normal_started_at is None:
            self.normal_started_at = timestamp
        if timestamp - self.normal_started_at >= self.clear_seconds:
            self.stable_label = "normal"
            return self.stable_label, "normal"
        if self.stable_label == "abnormal":
            return self.stable_label, "pending_normal"
        return self.stable_label, "normal"


def classifier_for_mode(mode: str, samples_dir: str | Path) -> KNNPostureClassifier:
    samples_dir = Path(samples_dir)
    if mode == "computer":
        return KNNPostureClassifier(
            mode=mode,
            samples_path=samples_dir / "computer_annotated_samples.jsonl",
            k=3,
            threshold=0.34,
        )
    if mode == "reading":
        return KNNPostureClassifier(
            mode=mode,
            samples_path=samples_dir / "reading_annotated_samples.jsonl",
            k=9,
            threshold=0.45,
        )
    raise ValueError(f"Unsupported mode: {mode}")


def state_classifier(samples_dir: str | Path = "output/state_classifier_1s") -> MultiStateKNNClassifier:
    return MultiStateKNNClassifier(
        samples_path=Path(samples_dir) / "all_state_samples.jsonl",
        k=3,
    )
