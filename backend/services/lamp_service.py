import json
import io
import threading
import time
import uuid
from pathlib import Path

from backend.config import Config
from backend.services.db import execute, fetch_all, fetch_one

try:
    from mijiaAPI import get_device_info, mijiaAPI
    from qrcode import QRCode
except Exception:  # pragma: no cover - lets the app boot without optional deps.
    get_device_info = None
    mijiaAPI = None
    QRCode = None


DEFAULT_POLICY = {
    "manual_override_minutes": 30,
    "presence_on": True,
    "away_dim_seconds": 60,
    "away_off_seconds": 180,
    "light_on_lux": 180,
    "light_off_lux": 480,
    "too_dark_lux": 150,
    "normal_lux": 350,
    "min_auto_brightness": 5,
    "max_auto_brightness": 80,
    "too_dark_brightness": 80,
    "normal_brightness": 55,
    "dim_brightness": 20,
    "day_study_color_temperature": 4300,
    "evening_study_color_temperature": 3700,
    "rest_color_temperature": 3000,
    "activity_adjustment_enabled": True,
    "activity_stable_seconds": 4,
    "activity_stable_min_samples": 3,
    "activity_stable_ratio": 0.7,
    "computer_brightness_delta": -15,
    "computer_max_brightness": 45,
    "reading_brightness_delta": 15,
    "reading_min_brightness": 55,
    "debounce_seconds": 10,
}

PROPERTY_ALIASES = {
    "power": ("power", "switch", "on"),
    "brightness": ("brightness", "bright", "light-brightness"),
    "color_temperature": (
        "color-temperature",
        "color_temperature",
        "colour-temperature",
        "ct",
    ),
}

_qr_sessions = {}
_qr_lock = threading.Lock()


def _now():
    return int(time.time())


def _json(data):
    return json.dumps(data or {}, ensure_ascii=False)


def _api():
    if mijiaAPI is None:
        raise RuntimeError("mijiaAPI is not installed")
    Path(Config.MIJIA_AUTH_PATH).parent.mkdir(parents=True, exist_ok=True)
    return mijiaAPI(Config.MIJIA_AUTH_PATH)


def _record_account(auth_data):
    user_id = str(auth_data.get("userId") or auth_data.get("cUserId") or "")
    existing = fetch_one("SELECT * FROM mijia_accounts WHERE auth_path = ?", (Config.MIJIA_AUTH_PATH,))
    if existing:
        execute(
            """
            UPDATE mijia_accounts
            SET user_id = ?, login_status = 'active', updated_at = CURRENT_TIMESTAMP, last_verified_at = ?
            WHERE id = ?
            """,
            (user_id, _now(), existing["id"]),
        )
        return fetch_one("SELECT * FROM mijia_accounts WHERE id = ?", (existing["id"],))
    account_id = execute(
        """
        INSERT INTO mijia_accounts (user_id, auth_path, login_status, last_verified_at)
        VALUES (?, ?, 'active', ?)
        """,
        (user_id, Config.MIJIA_AUTH_PATH, _now()),
    )
    return fetch_one("SELECT * FROM mijia_accounts WHERE id = ?", (account_id,))


def get_account():
    return fetch_one("SELECT * FROM mijia_accounts WHERE login_status = 'active' ORDER BY id DESC LIMIT 1")


def get_login_status():
    account = get_account()
    auth_exists = Path(Config.MIJIA_AUTH_PATH).is_file()
    available = False
    message = None
    if auth_exists and mijiaAPI is not None:
        try:
            available = bool(_api().available)
        except Exception as exc:
            message = str(exc)
    return {
        "logged_in": bool(account and auth_exists),
        "available": available,
        "account": _serialize_account(account),
        "message": message,
    }


def _serialize_account(row):
    if not row:
        return None
    return {
        "id": row["id"],
        "user_id": row.get("user_id"),
        "login_status": row.get("login_status"),
        "last_verified_at": row.get("last_verified_at"),
    }


def start_qr_login():
    session_id = f"qr_{uuid.uuid4().hex[:12]}"
    with _qr_lock:
        _qr_sessions[session_id] = {
            "status": "pending",
            "created_at": _now(),
            "expires_at": _now() + 180,
            "account": None,
            "error": None,
        }

    thread = threading.Thread(target=_run_qr_login, args=(session_id,), daemon=True)
    thread.start()
    deadline = time.time() + 5
    while time.time() < deadline:
        with _qr_lock:
            if _qr_sessions[session_id].get("login_url") or _qr_sessions[session_id].get("status") == "failed":
                break
        time.sleep(0.1)
    return {
        "session_id": session_id,
        "status": get_qr_status(session_id)["status"],
        "expires_in": 180,
        "qr_image_url": f"/api/lamp/mijia/login/qr/{session_id}/image",
    }


def _run_qr_login(session_id):
    try:
        api = _api()

        def capture_qr(login_url, box_size=10):
            with _qr_lock:
                _qr_sessions[session_id]["login_url"] = login_url
                _qr_sessions[session_id]["status"] = "pending"

        api._print_qr = capture_qr
        auth_data = api.QRlogin()
        account = _record_account(auth_data)
        with _qr_lock:
            _qr_sessions[session_id].update(
                {
                    "status": "confirmed",
                    "account": _serialize_account(account),
                    "error": None,
                }
            )
    except Exception as exc:
        with _qr_lock:
            _qr_sessions[session_id].update({"status": "failed", "error": str(exc)})


def get_qr_status(session_id):
    with _qr_lock:
        session = _qr_sessions.get(session_id)
        if not session:
            return None
        if session["status"] == "pending" and _now() > session["expires_at"]:
            session["status"] = "expired"
        return dict(session)


def get_qr_png(session_id):
    session = get_qr_status(session_id)
    if not session or not session.get("login_url") or QRCode is None:
        return None
    qr = QRCode(border=1, box_size=8)
    qr.add_data(session["login_url"])
    qr.make(fit=True)
    image = qr.make_image(fill_color="black", back_color="white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def list_devices():
    devices = _api().get_devices_list()
    shared = []
    try:
        shared = _api().get_shared_devices_list()
    except Exception:
        shared = []
    items = []
    seen = set()
    for device in [*devices, *shared]:
        did = str(device.get("did") or "")
        if not did or did in seen:
            continue
        seen.add(did)
        model = device.get("model") or ""
        name = device.get("name") or model or did
        bindable = _looks_like_lamp(device)
        items.append(
            {
                "did": did,
                "name": name,
                "model": model,
                "room_name": device.get("room_name") or device.get("roomName"),
                "online": _device_online(device),
                "bindable": bindable,
                "home_id": device.get("home_id"),
            }
        )
    return items


def _looks_like_lamp(device):
    model = (device.get("model") or "").lower()
    name = (device.get("name") or "").lower()
    return any(token in model or token in name for token in ("light", "lamp", "\u53f0\u706f", "\u706f"))


def _device_online(device):
    if "isOnline" in device:
        return bool(device.get("isOnline"))
    if "is_online" in device:
        return bool(device.get("is_online"))
    if "online" in device:
        return bool(device.get("online"))
    return None


def bind_device(payload):
    account = get_account()
    if not account:
        raise RuntimeError("Mi Home account is not logged in")

    did = str(payload.get("did") or "")
    if not did:
        raise ValueError("did is required")

    devices = list_devices()
    device = next((item for item in devices if item["did"] == did), None)
    if device is None:
        device = {
            "did": did,
            "name": payload.get("name") or did,
            "model": payload.get("model") or "",
            "room_name": payload.get("room_name"),
            "online": None,
        }

    capability = build_capability(device.get("model") or payload.get("model") or "")
    execute("UPDATE lamp_bindings SET is_active = 0")
    binding_id = execute(
        """
        INSERT INTO lamp_bindings (account_id, did, name, model, room_name, is_active, capability_json)
        VALUES (?, ?, ?, ?, ?, 1, ?)
        """,
        (
            account["id"],
            did,
            payload.get("name") or device.get("name"),
            payload.get("model") or device.get("model"),
            payload.get("room_name") or device.get("room_name"),
            _json(capability),
        ),
    )
    ensure_lamp_state(binding_id)
    return get_current_binding()


def build_capability(model):
    capability = {}
    if not model or get_device_info is None:
        return capability
    try:
        info = get_device_info(model, cache_path=Config.MIJIA_AUTH_DIR)
    except Exception:
        return capability
    props = info.get("properties") or []
    for canonical, aliases in PROPERTY_ALIASES.items():
        match = _find_property(props, aliases)
        if match:
            capability[canonical] = {
                "name": match.get("name"),
                "method": match.get("method"),
                "type": match.get("type"),
                "range": match.get("range"),
                "rw": match.get("rw"),
                "description": match.get("description"),
            }
    return capability


def _find_property(props, aliases):
    for prop in props:
        name = (prop.get("name") or "").lower()
        desc = (prop.get("description") or "").lower()
        if any(alias.lower() == name or alias.lower() in name or alias.lower() in desc for alias in aliases):
            if "w" in (prop.get("rw") or ""):
                return prop
    return None


def get_current_binding():
    row = fetch_one("SELECT * FROM lamp_bindings WHERE is_active = 1 ORDER BY id DESC LIMIT 1")
    if not row:
        return None
    row["capability"] = json.loads(row["capability_json"]) if row.get("capability_json") else {}
    row.pop("capability_json", None)
    row["is_active"] = bool(row.get("is_active"))
    return row


def unbind_current():
    binding = get_current_binding()
    if not binding:
        return False
    execute("UPDATE lamp_bindings SET is_active = 0, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (binding["id"],))
    return True


def ensure_lamp_state(binding_id):
    row = fetch_one("SELECT * FROM lamp_states WHERE binding_id = ? ORDER BY id DESC LIMIT 1", (binding_id,))
    if row:
        return row
    state_id = execute(
        """
        INSERT INTO lamp_states (binding_id, mode, source, raw_prop_json, manual_override_until)
        VALUES (?, 'auto', 'system', '{}', 0)
        """,
        (binding_id,),
    )
    return fetch_one("SELECT * FROM lamp_states WHERE id = ?", (state_id,))


def get_cached_state():
    binding = get_current_binding()
    if not binding:
        return {"bound": False, "binding": None, "state": None}
    state = ensure_lamp_state(binding["id"])
    return {"bound": True, "binding": binding, "state": _serialize_state(state)}


def refresh_state():
    binding = get_current_binding()
    if not binding:
        return {"bound": False, "binding": None, "state": None}
    values = _read_bound_props(binding)
    state = ensure_lamp_state(binding["id"])
    merged = {
        "power": values.get("power", state.get("power")),
        "brightness": values.get("brightness", state.get("brightness")),
        "color_temperature": values.get("color_temperature", state.get("color_temperature")),
        "online": True,
        "source": "device",
        "raw_prop_json": _json(values),
        "id": state["id"],
    }
    execute(
        """
        UPDATE lamp_states
        SET power = ?, brightness = ?, color_temperature = ?, online = ?, source = ?,
            raw_prop_json = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (
            _bool_to_int(merged["power"]),
            merged["brightness"],
            merged["color_temperature"],
            1,
            "device",
            merged["raw_prop_json"],
            state["id"],
        ),
    )
    return get_cached_state()


def _read_bound_props(binding):
    capability = binding.get("capability") or {}
    params = []
    names = []
    for name in ("power", "brightness", "color_temperature"):
        prop = capability.get(name)
        method = (prop or {}).get("method")
        if method:
            params.append({"did": binding["did"], **method})
            names.append(name)
    if not params:
        return {}
    result = _api().get_devices_prop(params)
    if isinstance(result, dict):
        result = [result]
    values = {}
    for name, item in zip(names, result):
        if item.get("code") == 0:
            values[name] = item.get("value")
    return values


def apply_manual_control(payload):
    override_minutes = int(payload.get("override_minutes") or DEFAULT_POLICY["manual_override_minutes"])
    action = {
        key: payload[key]
        for key in ("power", "brightness", "color_temperature")
        if key in payload and payload[key] is not None
    }
    return execute_lamp_action(
        action,
        requested_by="manual",
        reason="manual_control",
        mode="manual_override",
        manual_override_until=_now() + override_minutes * 60,
    )


def set_mode(mode):
    if mode not in ("auto", "manual_override", "manual", "off"):
        raise ValueError("unsupported mode")
    binding = get_current_binding()
    if not binding:
        raise RuntimeError("lamp is not bound")
    state = ensure_lamp_state(binding["id"])
    manual_until = state.get("manual_override_until") or 0
    if mode == "auto":
        manual_until = 0
    elif mode == "manual_override" and manual_until <= _now():
        manual_until = _now() + int(DEFAULT_POLICY["manual_override_minutes"]) * 60
    execute(
        """
        UPDATE lamp_states
        SET mode = ?, manual_override_until = ?, source = 'manual', updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (mode, manual_until, state["id"]),
    )
    return get_cached_state()


def execute_lamp_action(action, requested_by="auto", reason=None, mode=None, manual_override_until=None):
    binding = get_current_binding()
    if not binding:
        command_id = _log_command(None, "set_light", requested_by, action, "skipped", reason, "lamp is not bound")
        return {"command_id": command_id, "status": "skipped", "reason": "lamp is not bound"}

    state = ensure_lamp_state(binding["id"])
    capability = binding.get("capability") or {}
    command_id = _log_command(binding["id"], "set_light", requested_by, action, "pending", reason, None)

    params = []
    for key in ("power", "brightness", "color_temperature"):
        if key not in action:
            continue
        prop = capability.get(key)
        method = (prop or {}).get("method")
        if not method:
            continue
        params.append({"did": binding["did"], **method, "value": _normalize_value(key, action[key], prop)})

    if not params:
        _finish_command(command_id, "skipped", "no writable lamp capability matched")
        return {"command_id": command_id, "status": "skipped", "reason": "no writable lamp capability matched"}

    try:
        result = _api().set_devices_prop(params)
        if isinstance(result, dict):
            result = [result]
        failed = [item for item in result if item.get("code", 0) not in (0, 1)]
        if failed:
            raise RuntimeError(_json(failed))
        _update_state_from_action(state["id"], action, mode, manual_override_until, requested_by, result)
        _finish_command(command_id, "success", None)
        return {"command_id": command_id, "status": "success", "result": result}
    except Exception as exc:
        _finish_command(command_id, "failed", str(exc))
        return {"command_id": command_id, "status": "failed", "error_message": str(exc)}


def _normalize_value(key, value, prop):
    if key == "power":
        return bool(value)
    value = int(value)
    value_range = (prop or {}).get("range")
    if value_range and len(value_range) >= 2:
        value = max(int(value_range[0]), min(int(value_range[1]), value))
    return value


def _update_state_from_action(state_id, action, mode, manual_override_until, source, raw_result):
    current = fetch_one("SELECT * FROM lamp_states WHERE id = ?", (state_id,))
    next_mode = mode or current.get("mode") or "auto"
    override_until = manual_override_until
    if override_until is None:
        override_until = current.get("manual_override_until") or 0
    execute(
        """
        UPDATE lamp_states
        SET power = ?, brightness = ?, color_temperature = ?, mode = ?, online = 1,
            source = ?, raw_prop_json = ?, manual_override_until = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (
            _bool_to_int(action.get("power", current.get("power"))),
            action.get("brightness", current.get("brightness")),
            action.get("color_temperature", current.get("color_temperature")),
            next_mode,
            source,
            _json(raw_result),
            override_until,
            state_id,
        ),
    )


def _bool_to_int(value):
    if value is None:
        return None
    return 1 if bool(value) else 0


def _log_command(binding_id, command_type, requested_by, payload, status, reason, error_message):
    return execute(
        """
        INSERT INTO lamp_commands (binding_id, command_type, requested_by, payload_json, status, reason, error_message, executed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (binding_id, command_type, requested_by, _json(payload), status, reason, error_message, _now() if status != "pending" else None),
    )


def _finish_command(command_id, status, error_message):
    execute(
        """
        UPDATE lamp_commands
        SET status = ?, error_message = ?, executed_at = ?
        WHERE id = ?
        """,
        (status, error_message, _now(), command_id),
    )


def get_commands(limit=30):
    rows = fetch_all(
        """
        SELECT * FROM lamp_commands
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )
    for row in rows:
        row["payload"] = json.loads(row["payload_json"]) if row.get("payload_json") else {}
        row.pop("payload_json", None)
    return rows


def latest_auto_command_age(reason=None):
    row = fetch_one(
        """
        SELECT executed_at FROM lamp_commands
        WHERE requested_by = 'auto' AND status = 'success'
          AND (? IS NULL OR reason = ?)
        ORDER BY id DESC LIMIT 1
        """,
        (reason, reason),
    )
    if not row or not row.get("executed_at"):
        return None
    return max(0, _now() - int(row["executed_at"]))


def get_policy():
    row = fetch_one("SELECT * FROM lamp_policies WHERE name = 'default' LIMIT 1")
    if not row:
        policy_id = execute(
            """
            INSERT INTO lamp_policies (name, enabled, config_json)
            VALUES ('default', 1, ?)
            """,
            (_json(DEFAULT_POLICY),),
        )
        row = fetch_one("SELECT * FROM lamp_policies WHERE id = ?", (policy_id,))
    return {
        "id": row["id"],
        "name": row["name"],
        "enabled": bool(row["enabled"]),
        "config": {**DEFAULT_POLICY, **(json.loads(row["config_json"]) if row.get("config_json") else {})},
        "updated_at": row.get("updated_at"),
    }


def update_policy(payload):
    current = get_policy()
    config = {**current["config"], **(payload.get("config") or payload)}
    enabled = payload.get("enabled", current["enabled"])
    execute(
        """
        UPDATE lamp_policies
        SET enabled = ?, config_json = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (1 if enabled else 0, _json(config), current["id"]),
    )
    return get_policy()


def _serialize_state(row):
    if not row:
        return None
    manual_until = row.get("manual_override_until") or 0
    mode = row.get("mode") or "auto"
    if mode == "manual_override" and manual_until and manual_until <= _now():
        mode = "auto"
    return {
        "power": None if row.get("power") is None else bool(row.get("power")),
        "brightness": row.get("brightness"),
        "color_temperature": row.get("color_temperature"),
        "mode": mode,
        "online": None if row.get("online") is None else bool(row.get("online")),
        "source": row.get("source"),
        "manual_override_until": manual_until,
        "updated_at": row.get("updated_at"),
    }
