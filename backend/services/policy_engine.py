import time

from backend.services import lamp_service
from backend.services.db import fetch_all, fetch_one


COMPUTER_POSE_STATES = frozenset({"computer_normal", "computer_abnormal"})
READING_POSE_STATES = frozenset({"reading_normal", "reading_abnormal"})


def evaluate(derived_state, telemetry=None, policy=None):
    policy = policy or lamp_service.get_policy()
    config = policy["config"]
    if not policy["enabled"]:
        return {"type": "none", "reason": "policy_disabled"}

    presence = derived_state.get("presence_state")
    study_state = derived_state.get("study_state")
    env_labels = derived_state.get("env_labels") or []
    lux = (telemetry or {}).get("lux")
    now = int(time.time())

    if presence == "away":
        away_seconds = _estimate_away_seconds(derived_state.get("device_id"), now)
        if away_seconds >= int(config.get("away_off_seconds", 180)):
            return {"type": "set_light", "power": False, "reason": "away_timeout_off"}
        if away_seconds >= int(config.get("away_dim_seconds", 60)):
            return {
                "type": "set_light",
                "power": True,
                "brightness": int(config.get("dim_brightness", 20)),
                "color_temperature": int(config.get("rest_color_temperature", 3000)),
                "reason": "away_timeout_dim",
            }
        return {"type": "none", "reason": "away_waiting_grace"}

    if presence != "present":
        return {"type": "none", "reason": "presence_unknown"}

    color_temperature = int(config.get("day_study_color_temperature", 4300))
    if study_state == "idle":
        color_temperature = int(config.get("rest_color_temperature", 3000))

    light_action = _evaluate_present_lighting(lux, env_labels, config)
    if light_action.get("power") is False:
        return {
            "type": "set_light",
            "power": False,
            "reason": light_action["reason"],
        }

    activity_mode = _stable_activity_mode(derived_state, now, config)
    if activity_mode:
        light_action = _adjust_lighting_for_activity(light_action, activity_mode, config)

    return {
        "type": "set_light",
        "power": True,
        "brightness": light_action["brightness"],
        "color_temperature": color_temperature,
        "reason": light_action["reason"],
    }


def maybe_execute(derived_state, telemetry=None):
    policy = lamp_service.get_policy()
    binding_state = lamp_service.get_cached_state()
    if not binding_state["bound"]:
        action = evaluate(derived_state, telemetry, policy)
        return {"executed": False, "status": "skipped", "reason": "lamp_not_bound", "action": action}

    state = binding_state.get("state") or {}
    mode = state.get("mode")
    now = int(time.time())
    manual_until = state.get("manual_override_until") or 0
    if mode in ("off", "manual"):
        return {"executed": False, "status": "skipped", "reason": f"mode_{mode}"}
    if mode == "manual_override" and manual_until > now:
        return {"executed": False, "status": "skipped", "reason": "manual_override_active"}

    action = evaluate(derived_state, telemetry, policy)
    if action.get("type") != "set_light":
        return {"executed": False, "status": "skipped", "reason": action.get("reason"), "action": action}
    if _same_as_current(action, state):
        return {"executed": False, "status": "skipped", "reason": "state_already_matches", "action": action}
    age = lamp_service.latest_auto_command_age(action.get("reason"))
    debounce = int(policy["config"].get("debounce_seconds", 10))
    if age is not None and age < debounce:
        return {"executed": False, "status": "skipped", "reason": "debounced", "action": action}

    result = lamp_service.execute_lamp_action(
        {k: action[k] for k in ("power", "brightness", "color_temperature") if k in action},
        requested_by="auto",
        reason=action.get("reason"),
        mode="auto",
    )
    return {"executed": result.get("status") == "success", "action": action, **result}


def _stable_activity_mode(derived_state, now, config):
    if not config.get("activity_adjustment_enabled", True):
        return None

    current_mode = _pose_activity_mode(derived_state.get("pose_state"))
    if current_mode is None:
        return None

    device_id = derived_state.get("device_id")
    if not device_id:
        return current_mode

    stable_seconds = int(config.get("activity_stable_seconds", 4))
    min_samples = int(config.get("activity_stable_min_samples", 3))
    min_ratio = float(config.get("activity_stable_ratio", 0.7))
    start_at = now - max(1, stable_seconds) + 1

    rows = fetch_all(
        """
        SELECT pose_state FROM derived_states
        WHERE device_id = ?
          AND timestamp >= ?
          AND timestamp <= ?
          AND presence_state = 'present'
          AND pose_state IN (
            'computer_normal',
            'computer_abnormal',
            'reading_normal',
            'reading_abnormal'
          )
        ORDER BY timestamp DESC
        """,
        (device_id, start_at, now),
    )
    samples = [current_mode]
    samples.extend(
        mode for mode in (_pose_activity_mode(row.get("pose_state")) for row in rows) if mode is not None
    )

    if len(samples) < min_samples:
        return None

    candidate_count = sum(1 for mode in samples if mode == current_mode)
    if candidate_count / len(samples) < min_ratio:
        return None
    return current_mode


def _pose_activity_mode(pose_state):
    if pose_state in COMPUTER_POSE_STATES:
        return "computer"
    if pose_state in READING_POSE_STATES:
        return "reading"
    return None


def _adjust_lighting_for_activity(light_action, activity_mode, config):
    adjusted = dict(light_action)
    brightness = int(adjusted["brightness"])

    if activity_mode == "computer":
        brightness += int(config.get("computer_brightness_delta", -15))
        brightness = min(brightness, int(config.get("computer_max_brightness", 45)))
    elif activity_mode == "reading":
        brightness += int(config.get("reading_brightness_delta", 15))
        brightness = max(brightness, int(config.get("reading_min_brightness", 55)))

    min_brightness = int(config.get("min_auto_brightness", config.get("dim_brightness", 20)))
    max_brightness = int(config.get("max_auto_brightness", config.get("too_dark_brightness", 80)))
    adjusted["brightness"] = int(max(min_brightness, min(max_brightness, brightness)))
    adjusted["reason"] = f"{adjusted['reason']}_{activity_mode}"
    return adjusted


def _same_as_current(action, state):
    for key in ("power", "brightness", "color_temperature"):
        if key in action and action[key] != state.get(key):
            return False
    return True


def _evaluate_present_lighting(lux, env_labels, config):
    if lux is None:
        brightness = int(config.get("normal_brightness", 55))
        if "too_dark" in env_labels:
            brightness = int(config.get("too_dark_brightness", 80))
        return {"power": True, "brightness": brightness, "reason": "present_study_lighting"}

    lux_value = float(lux)
    on_lux = float(config.get("light_on_lux", 200))
    off_lux = float(config.get("light_off_lux", 1000))
    min_brightness = int(config.get("min_auto_brightness", config.get("dim_brightness", 20)))
    max_brightness = int(config.get("max_auto_brightness", config.get("too_dark_brightness", 80)))

    if off_lux <= on_lux:
        off_lux = on_lux + 1

    if lux_value > off_lux:
        return {"power": False, "reason": "ambient_bright_off"}

    if lux_value <= on_lux:
        return {"power": True, "brightness": max_brightness, "reason": "ambient_dark_full"}

    ratio = (lux_value - on_lux) / (off_lux - on_lux)
    brightness = round(max_brightness - ratio * (max_brightness - min_brightness))
    return {
        "power": True,
        "brightness": int(max(min_brightness, min(max_brightness, brightness))),
        "reason": "ambient_lux_curve",
    }


def _estimate_away_seconds(device_id, now):
    if not device_id:
        return 0
    last_present = fetch_one(
        """
        SELECT timestamp FROM derived_states
        WHERE device_id = ? AND presence_state = 'present'
        ORDER BY id DESC LIMIT 1
        """,
        (device_id,),
    )
    if not last_present:
        return 0
    return max(0, now - int(last_present["timestamp"]))
