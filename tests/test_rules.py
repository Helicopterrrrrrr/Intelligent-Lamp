from smartled_pose.rules import RuleEngine
from smartled_pose.target_selector import TargetSelector
from smartled_pose.types import BoundingBox, Keypoint, PoseDetection, PoseFeatures, ROI, ReferenceFeatures


def make_config():
    return {
        "camera": {"width": 1280, "height": 720, "fps": 30},
        "selection": {
            "hold_last_seconds": 1.0,
            "min_confidence": 0.5,
            "min_roi_overlap": 0.3,
            "kp_confidence": 0.3,
        },
        "rules": {
            "kp_valid_ratio_min": 0.5,
            "seated_candidate_seconds": 1.5,
            "absent_seconds": 2.0,
            "warning_hold_seconds": 2.0,
            "cooldown_seconds": 3.0,
            "label_window_seconds": 0.5,
            "near_ratio": 1.15,
            "too_close_ratio": 1.25,
            "fallback_near_bbox_area_ratio": 0.18,
            "fallback_too_close_bbox_area_ratio": 0.24,
            "lean_angle_deg": 12.0,
            "head_down_drop_shoulder_ratio": 0.28,
            "head_down_angle_deg": 18.0,
            "head_distance_ratio_threshold": 1.3,
            "torso_angle_deviation_deg": 15.0,
            "head_tilt_deviation_deg": 15.0,
            "neck_angle_threshold_deg": 30.0,
        },
    }


def make_features(**overrides):
    base = {
        "bbox_area_ratio": 0.14,
        "shoulder_width_px": 120.0,
        "nose_to_shoulder_center_y": 4.0,
        "torso_tilt_deg": 0.0,
        "head_tilt_deg": 0.0,
        "kp_valid_ratio": 0.8,
        "roi_overlap": 0.7,
        "center_in_roi": True,
        "shoulders_in_roi": True,
        "pose_conf": 0.9,
    }
    base.update(overrides)
    return PoseFeatures(**base)


def make_detection(x1, y1, x2, y2, score=0.8):
    keypoints = [Keypoint(float(x1), float(y1), 0.9) for _ in range(17)]
    return PoseDetection(bbox=BoundingBox(x1, y1, x2, y2, score), keypoints=keypoints, score=score)


def test_presence_promotes_to_seated_after_candidate_window():
    engine = RuleEngine(make_config(), ReferenceFeatures())
    features = make_features()
    assert engine.evaluate(0.0, features, True)[0] == "seated_candidate"
    assert engine.evaluate(0.8, features, True)[0] == "seated_candidate"
    assert engine.evaluate(1.6, features, True)[0] == "seated"


def test_distance_uses_reference_ratio_when_available():
    reference = ReferenceFeatures(shoulder_width_px=100.0, bbox_area_ratio=0.10)
    engine = RuleEngine(make_config(), reference)
    features = make_features(shoulder_width_px=130.0)
    engine.state.current_presence = "seated"
    presence, distance, _, _ = engine.evaluate(10.0, features, True)
    assert presence == "seated"
    assert distance == "too_close"


def test_leaning_right_detected_from_torso_tilt():
    engine = RuleEngine(make_config(), ReferenceFeatures())
    engine.state.current_presence = "seated"
    features = make_features(torso_tilt_deg=15.0)
    _, _, label, event = engine.evaluate(10.0, features, True)
    assert label == "normal"
    assert event == "none"
    _, _, label, event = engine.evaluate(12.2, features, True)
    assert label == "leaning_right"
    assert event == "warning_active"


def test_warning_activates_after_hold_duration():
    engine = RuleEngine(make_config(), ReferenceFeatures())
    engine.state.current_presence = "seated"
    features = make_features(shoulder_width_px=140.0, bbox_area_ratio=0.26)
    _, _, label, event = engine.evaluate(5.0, features, True)
    assert label == "normal"
    assert event == "none"
    _, _, label, event = engine.evaluate(7.2, features, True)
    assert label == "head_down"
    assert event == "warning_active"


def test_head_ratio_reference_works_without_torso_reference():
    reference = ReferenceFeatures(head_shoulder_distance_ratio=1.0, torso_angle_deg=None)
    engine = RuleEngine(make_config(), reference)
    engine.state.current_presence = "seated"
    features = make_features(head_shoulder_distance_ratio=1.4)
    _, _, label, event = engine.evaluate(10.0, features, True)
    assert label == "normal"
    assert event == "none"
    _, _, label, event = engine.evaluate(12.2, features, True)
    assert label == "head_down"
    assert event == "warning_active"


def test_selector_prefers_roi_overlap():
    roi = ROI(100, 100, 400, 400)
    selector = TargetSelector(roi=roi, hold_last_seconds=1.0, kp_confidence=0.3)
    far = make_detection(10, 10, 90, 90, score=0.95)
    near = make_detection(150, 150, 300, 350, score=0.70)
    selected = selector.select([far, near], frame_width=1280, frame_height=720, timestamp=1.0)
    assert selected is near
