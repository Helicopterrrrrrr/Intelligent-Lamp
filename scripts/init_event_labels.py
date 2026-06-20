from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create JSONL templates for event-level validation labels.")
    parser.add_argument("--video-dir", default="data/videos", help="Directory containing validation clips.")
    parser.add_argument("--output", default="data/event_labels/template.jsonl", help="Path to output JSONL.")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for path in sorted(video_dir.glob("*")):
        if path.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv"}:
            continue
        lines.append(
            json.dumps(
                {
                    "clip": str(path.as_posix()),
                    "presence": "seated",
                    "distance_level": "normal",
                    "posture_label": "normal",
                    "start_time": 0.0,
                    "end_time": None,
                    "notes": "",
                },
                ensure_ascii=False,
            )
        )

    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(lines)} template records to {output}")


if __name__ == "__main__":
    main()
