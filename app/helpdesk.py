"""Helpdesk API client: rate limiting, retries, and safe handling of 409."""
import random
import threading
import time
from typing import Any, Iterator

import httpx

MAX_RPS = 4.0  # stay under the 5 req/s limit
MAX_ATTEMPTS = 6


class AlreadyHandled(Exception):
    """409: the ticket already has an answer. Never retry."""


class HelpdeskError(Exception):
    pass


class Helpdesk:
    def __init__(self, base_url: str, key: str, email: str, client: httpx.Client | None = None,
                 sleep=time.sleep):
        self._client = client or httpx.Client(timeout=20)
        self._base = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {key}", "X-Candidate-Email": email}
        self._sleep = sleep
        self._lock = threading.Lock()
        self._last_call = 0.0

    def _throttle(self) -> None:
        with self._lock:
            wait = self._last_call + 1.0 / MAX_RPS - time.monotonic()
            if wait > 0:
                self._sleep(wait)
            self._last_call = time.monotonic()

    def _request(self, method: str, path: str, **kwargs) -> Any:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self._throttle()
            try:
                resp = self._client.request(method, self._base + path, headers=self._headers, **kwargs)
            except httpx.TransportError as exc:
                if attempt == MAX_ATTEMPTS:
                    raise HelpdeskError(f"network error: {exc}") from exc
                self._backoff(attempt)
                continue
            if resp.status_code == 429:
                self._sleep(float(resp.headers.get("Retry-After", "1")))
                continue
            if resp.status_code == 503:
                if attempt == MAX_ATTEMPTS:
                    raise HelpdeskError("service unavailable after retries")
                self._backoff(attempt)
                continue
            if resp.status_code == 409:
                raise AlreadyHandled(path)
            if resp.status_code >= 400:
                raise HelpdeskError(f"{resp.status_code} on {method} {path}: {resp.text[:300]}")
            return resp.json()
        raise HelpdeskError(f"gave up on {method} {path}")

    def _backoff(self, attempt: int) -> None:
        self._sleep(min(8.0, 0.5 * 2 ** (attempt - 1)) + random.uniform(0, 0.25))

    # --- reads ---
    def status(self) -> dict:
        return self._request("GET", "/status")

    def examples(self) -> list[dict]:
        data = self._request("GET", "/examples")
        return data["examples"] if isinstance(data, dict) and "examples" in data else data

    def iter_tickets(self, status: str = "open") -> Iterator[dict]:
        cursor = None
        while True:
            params: dict = {"limit": 10, "status": status}
            if cursor:
                params["cursor"] = cursor
            page = self._request("GET", "/tickets", params=params)
            yield from page["tickets"]
            cursor = page.get("next_cursor")
            if not cursor:
                return

    def get_ticket(self, ticket_id: str) -> dict:
        return self._request("GET", f"/tickets/{ticket_id}")

    # --- actions (one per ticket; 409 raises AlreadyHandled) ---
    def reply(self, ticket_id: str, body: dict) -> dict:
        return self._request("POST", f"/tickets/{ticket_id}/reply", json=body)

    def handoff(self, ticket_id: str, body: dict) -> dict:
        return self._request("POST", f"/tickets/{ticket_id}/handoff", json=body)

    def close(self, ticket_id: str, body: dict) -> dict:
        return self._request("POST", f"/tickets/{ticket_id}/close", json=body)

    def reset(self) -> dict:
        return self._request("POST", "/practice/reset")
