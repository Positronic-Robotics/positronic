"""Append-only model transcripts, separate from the robot's signal recordings."""

import json
import threading
from pathlib import Path
from typing import Any


class Transcript:
    """A session's API bodies and execution decisions. A missing directory disables recording."""

    def __init__(self, directory: Path | None):
        self.path = directory / 'transcript.jsonl' if directory is not None else None
        self._lock = threading.Lock()

    def write(self, event: str, **data: Any) -> None:
        if self.path is None:
            return
        line = json.dumps({'event': event, **data}, allow_nan=False)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open('a') as output:
                output.write(line + '\n')
