from flask import Blueprint, Response, request

from backend.services import lamp_service, policy_engine
from backend.utils.response import error, ok


lamp_api = Blueprint("lamp_api", __name__)


@lamp_api.get("/mijia/status")
def mijia_status():
    return ok(lamp_service.get_login_status())


@lamp_api.post("/mijia/login/qr")
def start_qr_login():
    try:
        return ok(lamp_service.start_qr_login(), 201)
    except Exception as exc:
        return error(str(exc), 500)


@lamp_api.get("/mijia/login/qr/<session_id>/status")
def qr_login_status(session_id):
    session = lamp_service.get_qr_status(session_id)
    if not session:
        return error("login session not found", 404)
    return ok({"session": session})


@lamp_api.get("/mijia/login/qr/<session_id>/image")
def qr_login_image(session_id):
    image = lamp_service.get_qr_png(session_id)
    if not image:
        return error("QR image is not ready", 404)
    return Response(image, mimetype="image/png")


@lamp_api.get("/mijia/devices")
def mijia_devices():
    try:
        return ok({"items": lamp_service.list_devices()})
    except Exception as exc:
        return error(str(exc), 502)


@lamp_api.get("/binding/current")
def current_binding():
    return ok({"binding": lamp_service.get_current_binding()})


@lamp_api.post("/binding")
def bind_device():
    try:
        payload = request.get_json(silent=True) or {}
        return ok({"binding": lamp_service.bind_device(payload)}, 201)
    except ValueError as exc:
        return error(str(exc), 400)
    except Exception as exc:
        return error(str(exc), 502)


@lamp_api.delete("/binding/current")
def unbind_device():
    return ok({"unbound": lamp_service.unbind_current()})


@lamp_api.get("/state")
def lamp_state():
    refresh = request.args.get("refresh", default="0")
    try:
        data = lamp_service.refresh_state() if refresh in ("1", "true") else lamp_service.get_cached_state()
        return ok(data)
    except Exception as exc:
        data = lamp_service.get_cached_state()
        data["message"] = str(exc)
        return ok(data)


@lamp_api.post("/control")
def control_lamp():
    payload = request.get_json(silent=True) or {}
    try:
        result = lamp_service.apply_manual_control(payload)
        return ok({"command": result, **lamp_service.get_cached_state()})
    except ValueError as exc:
        return error(str(exc), 400)
    except Exception as exc:
        return error(str(exc), 502)


@lamp_api.post("/mode")
def set_mode():
    payload = request.get_json(silent=True) or {}
    try:
        return ok(lamp_service.set_mode(payload.get("mode")))
    except ValueError as exc:
        return error(str(exc), 400)
    except Exception as exc:
        return error(str(exc), 502)


@lamp_api.get("/commands")
def commands():
    limit = request.args.get("limit", default=30, type=int)
    return ok({"items": lamp_service.get_commands(limit)})


@lamp_api.get("/policy")
def get_policy():
    return ok({"policy": lamp_service.get_policy()})


@lamp_api.put("/policy")
def update_policy():
    payload = request.get_json(silent=True) or {}
    return ok({"policy": lamp_service.update_policy(payload)})


@lamp_api.post("/policy/test")
def test_policy():
    payload = request.get_json(silent=True) or {}
    derived = {
        "device_id": payload.get("device_id", "preview"),
        "presence_state": payload.get("presence_state"),
        "distance_level": payload.get("distance_level"),
        "env_labels": payload.get("env_label") or payload.get("env_labels") or [],
        "pose_state": payload.get("pose_state"),
        "study_state": payload.get("study_state"),
    }
    return ok({"decision": policy_engine.evaluate(derived, payload)})
