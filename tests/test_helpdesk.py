import httpx
import pytest
import respx

from app.helpdesk import AlreadyHandled, Helpdesk, HelpdeskError

BASE = "https://hd.test/api"


def make(sleeps):
    return Helpdesk(BASE, "k", "me@example.com", sleep=sleeps.append)


@respx.mock
def test_retries_503_then_succeeds():
    route = respx.post(f"{BASE}/tickets/T1/reply").mock(
        side_effect=[httpx.Response(503), httpx.Response(503), httpx.Response(200, json={"ok": True})])
    sleeps = []
    assert make(sleeps).reply("T1", {"x": 1}) == {"ok": True}
    assert route.call_count == 3


@respx.mock
def test_429_honours_retry_after():
    respx.get(f"{BASE}/status").mock(
        side_effect=[httpx.Response(429, headers={"Retry-After": "2"}), httpx.Response(200, json={})])
    sleeps = []
    make(sleeps).status()
    assert 2.0 in sleeps


@respx.mock
def test_409_is_never_retried():
    route = respx.post(f"{BASE}/tickets/T1/reply").mock(return_value=httpx.Response(409, json={}))
    with pytest.raises(AlreadyHandled):
        make([]).reply("T1", {})
    assert route.call_count == 1


@respx.mock
def test_gives_up_after_repeated_503():
    respx.get(f"{BASE}/status").mock(return_value=httpx.Response(503))
    with pytest.raises(HelpdeskError):
        make([]).status()


@respx.mock
def test_pagination_follows_cursor():
    respx.get(f"{BASE}/tickets").mock(side_effect=[
        httpx.Response(200, json={"tickets": [{"id": "T1"}], "next_cursor": "T1"}),
        httpx.Response(200, json={"tickets": [{"id": "T2"}], "next_cursor": None}),
    ])
    assert [t["id"] for t in make([]).iter_tickets()] == ["T1", "T2"]
