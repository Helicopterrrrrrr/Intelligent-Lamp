import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app import app


if __name__ == "__main__":
    host = os.getenv("STUDYPILOT_BACKEND_HOST", "0.0.0.0")
    port = int(os.getenv("STUDYPILOT_BACKEND_PORT", "5000"))
    app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)
