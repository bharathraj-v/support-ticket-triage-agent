"""Crash-safe record of what we decided and posted, keyed by ticket id.

Flow per ticket: save decision as "pending" -> POST to helpdesk -> mark "posted".
If we crash between the last two steps, the next run re-posts the stored decision (a 409 means it
already went through). That is what guarantees no customer gets two answers.
"""
import json
import os
import tempfile
from pathlib import Path


class State:
    def __init__(self, path: Path):
        self.path = path
        self._data: dict[str, dict] = {}
        if path.exists():
            self._data = json.loads(path.read_text())

    def get(self, ticket_id: str) -> dict | None:
        return self._data.get(ticket_id)

    def is_posted(self, ticket_id: str) -> bool:
        return self._data.get(ticket_id, {}).get("status") == "posted"

    def save_pending(self, ticket_id: str, action: str, payload: dict) -> None:
        self._data[ticket_id] = {"status": "pending", "action": action, "payload": payload}
        self._flush()

    def mark_posted(self, ticket_id: str) -> None:
        self._data[ticket_id]["status"] = "posted"
        self._flush()

    def clear(self) -> None:
        self._data = {}
        self._flush()

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent)
        with os.fdopen(fd, "w") as f:
            json.dump(self._data, f, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)  # atomic: a crash never leaves half a file
