"""Run metadata helpers for reproducible paper jobs."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def save_run_metadata(path: str, args, extra: dict | None = None) -> None:
    """Write args, git commit, and optional extra config beside an output."""
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "args": vars(args) if hasattr(args, "__dict__") else dict(args),
    }
    if extra:
        payload.update(extra)

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
