"""In-memory model transcripts for episode static data."""

import threading
from copy import deepcopy
from typing import Any


class Transcript:
    """Thread-safe events with independent snapshots for the episode recorder."""

    def __init__(self):
        self._events: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def write(self, event: str, **data: Any) -> None:
        entry = deepcopy({'event': event, **data})
        with self._lock:
            self._events.append(entry)

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return deepcopy(self._events)
