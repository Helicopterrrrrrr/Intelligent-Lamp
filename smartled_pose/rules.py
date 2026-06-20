from __future__ import annotations

from dataclasses import dataclass

from .types import PoseFeatures, ReferenceFeatures


@dataclass(slots=True)
class RuleState:
    current_presence: str = "absent"
    candidate_started_at: float | None = None
    missing_started_at: float | None = None
    abnormal_started_at: float | None = None
    cooldown_until: float = 0.0


class RuleEngine:
    def __init__(self, config: dict, reference: ReferenceFeatures) -> None:
        self.config = config
        self.reference = reference
        self.state = RuleState()

    def evaluate(
        self,
        timestamp: float,
        features: PoseFeatures | None,
        detection_found: bool,
    ) -> tuple[str, str, str, str]:
        presence = self._evaluate_presence(timestamp, features, detection_found)
        distance = self._evaluate_distance(features, presence)
        raw_label = self._evaluate_posture(features, presence, distance)
        event_state, posture_label = self._evaluate_event(timestamp, presence, distance, raw_label)
        return presence, distance, posture_label, event_state

    def _evaluate_presence(self, timestamp: float, features: PoseFeatures | None, detection_found: bool) -> str:
        rules = self.config["rules"]
        selection = self.config["selection"]
        use_roi = selection.get("use_roi", True)
        in_roi = bool(
            features
            and (not use_roi or features.roi_overlap >= selection["min_roi_overlap"])
            and (not use_roi or features.center_in_roi)
        )
        valid = bool(
            detection_found
            and (features is None or in_roi)
        )
        if valid:
            self.state.missing_started_at = None
            if self.state.current_presence == "absent":
                if self.state.candidate_started_at is None:
                    self.state.candidate_started_at = timestamp
                elapsed = timestamp - self.state.candidate_started_at
                if elapsed >= rules["seated_candidate_seconds"]:
                    self.state.current_presence = "seated"
                    return "seated"
                return "seated_candidate"
            self.state.current_presence = "seated"
            return "seated"

        self.state.candidate_started_at = None
        if self.state.current_presence == "seated":
            if self.state.missing_started_at is None:
                self.state.missing_started_at = timestamp
            elapsed = timestamp - self.state.missing_started_at
            if elapsed >= rules["absent_seconds"]:
                self.state.current_presence = "absent"
                self.state.abnormal_started_at = None
                return "absent"
            return "seated"

        self.state.current_presence = "absent"
        return "absent"

    def _evaluate_distance(self, features: PoseFeatures | None, presence: str) -> str:
        if presence != "seated" or not features:
            return "normal"

        rules = self.config["rules"]
        shoulder_reference = self.reference.shoulder_width_px
        bbox_reference = self.reference.bbox_area_ratio

        shoulder_ratio = None
        if shoulder_reference and shoulder_reference > 0 and features.shoulder_width_px > 0:
            shoulder_ratio = features.shoulder_width_px / shoulder_reference
        bbox_ratio = None
        if bbox_reference and bbox_reference > 0 and features.bbox_area_ratio > 0:
            bbox_ratio = features.bbox_area_ratio / bbox_reference

        ratio = shoulder_ratio or bbox_ratio
        if ratio:
            if ratio >= rules["too_close_ratio"]:
                return "too_close"
            if ratio >= rules["near_ratio"]:
                return "near"
            return "normal"

        if features.bbox_area_ratio >= rules["fallback_too_close_bbox_area_ratio"]:
            return "too_close"
        if features.bbox_area_ratio >= rules["fallback_near_bbox_area_ratio"]:
            return "near"
        return "normal"

    def _evaluate_posture(self, features: PoseFeatures | None, presence: str, distance: str) -> str:
        if presence != "seated" or not features:
            return "normal"

        rules = self.config["rules"]

        if distance == "too_close":
            return "head_down"

        head_ratio_ref = self.reference.head_shoulder_distance_ratio
        if head_ratio_ref is not None:
            if features.head_shoulder_distance_ratio > 0 and head_ratio_ref > 0:
                head_ratio_deviation = features.head_shoulder_distance_ratio / head_ratio_ref
                if head_ratio_deviation >= rules["head_distance_ratio_threshold"]:
                    return "head_down"

        if features.neck_angle_deg > 0 and features.neck_angle_deg > rules["neck_angle_threshold_deg"]:
            return "head_down"

        head_tilt_ref = self.reference.head_tilt_deg
        if head_tilt_ref is not None and features.head_tilt_deg != 0.0:
            head_tilt_deviation = abs(features.head_tilt_deg - head_tilt_ref)
            if head_tilt_deviation >= rules["head_tilt_deviation_deg"]:
                return "head_down"

        torso_angle_ref = self.reference.torso_angle_deg
        if torso_angle_ref is not None:
            torso_deviation = abs(features.torso_tilt_deg - torso_angle_ref)
            if torso_deviation >= rules["torso_angle_deviation_deg"]:
                if features.torso_tilt_deg < torso_angle_ref:
                    return "leaning_left"
                else:
                    return "leaning_right"
        else:
            normalized_drop = 0.0
            if features.shoulder_width_px > 0:
                normalized_drop = features.nose_to_shoulder_center_y / features.shoulder_width_px

            if normalized_drop >= rules["head_down_drop_shoulder_ratio"] or abs(features.head_tilt_deg) >= rules["head_down_angle_deg"]:
                return "head_down"

            if features.torso_tilt_deg <= -rules["lean_angle_deg"]:
                return "leaning_left"
            if features.torso_tilt_deg >= rules["lean_angle_deg"]:
                return "leaning_right"

        return "normal"

    def _evaluate_event(self, timestamp: float, presence: str, distance: str, raw_label: str) -> tuple[str, str]:
        rules = self.config["rules"]
        if presence != "seated":
            self.state.abnormal_started_at = None
            return "none", "normal"

        if timestamp < self.state.cooldown_until:
            return "cooldown", "normal"

        abnormal = distance == "too_close" or raw_label != "normal"
        if not abnormal:
            self.state.abnormal_started_at = None
            return "none", "normal"

        if self.state.abnormal_started_at is None:
            self.state.abnormal_started_at = timestamp
            return "none", "normal"

        elapsed = timestamp - self.state.abnormal_started_at
        if elapsed >= rules["warning_hold_seconds"]:
            self.state.cooldown_until = timestamp + rules["cooldown_seconds"]
            self.state.abnormal_started_at = None
            return "warning_active", raw_label
        return "none", "normal"
