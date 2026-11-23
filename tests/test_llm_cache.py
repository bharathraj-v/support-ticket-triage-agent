from types import SimpleNamespace

from app.config import ROOT
from app.faq import load_helpcentre
from app.llm import Llm
from app.models import Decision

HC = load_helpcentre(ROOT / "data" / "helpcentre.md")
D = Decision(language="nl", category="question", priority="P4", action="handoff", reason="r",
             handoff_note="n", confidence=0.9)


class FakeClient:
    def __init__(self):
        self.calls = 0
        self.models = self

    def generate_content(self, model, contents, config):
        self.calls += 1
        meta = SimpleNamespace(prompt_token_count=10, candidates_token_count=5, thoughts_token_count=0)
        return SimpleNamespace(parsed=D, usage_metadata=meta)


T = {"channel": "email", "customer_plan": "Basis"}


def test_second_process_reuses_saved_calls(tmp_path):
    cache = tmp_path / "c.jsonl"
    c1 = FakeClient()
    Llm("k", "m", HC, cache_path=cache, _client=c1).decide(T, "s", "b")
    c2 = FakeClient()
    llm2 = Llm("k", "m", HC, cache_path=cache, _client=c2)
    assert llm2.decide(T, "s", "b") == D
    assert c1.calls == 1 and c2.calls == 0 and llm2.usage.cache_hits == 1


def test_changed_input_is_not_served_from_cache(tmp_path):
    cache = tmp_path / "c.jsonl"
    c = FakeClient()
    llm = Llm("k", "m", HC, cache_path=cache, _client=c)
    llm.decide(T, "s", "b1")
    llm.decide(T, "s", "b2")
    assert c.calls == 2


def test_corrupt_last_line_is_ignored(tmp_path):
    cache = tmp_path / "c.jsonl"
    c = FakeClient()
    Llm("k", "m", HC, cache_path=cache, _client=c).decide(T, "s", "b")
    with cache.open("a") as f:
        f.write('{"key": "abc", "pars')
    c2 = FakeClient()
    Llm("k", "m", HC, cache_path=cache, _client=c2).decide(T, "s", "b")
    assert c2.calls == 0


def test_rate_limit_waits_for_the_delay_google_asks_for(tmp_path):
    class Flaky(FakeClient):
        def generate_content(self, model, contents, config):
            if self.calls == 0:
                self.calls += 1
                raise RuntimeError("429 RESOURCE_EXHAUSTED ... Please retry in 39.9s.")
            return super().generate_content(model, contents, config)
    sleeps = []
    llm = Llm("k", "m", HC, _client=Flaky(), _sleep=sleeps.append)
    assert llm.decide(T, "s", "b") == D
    assert any(s >= 40 for s in sleeps)


def test_min_interval_spaces_real_calls_but_not_cache_hits(tmp_path):
    sleeps = []
    llm = Llm("k", "m", HC, cache_path=tmp_path / "c.jsonl", min_interval=5, _client=FakeClient(),
              _sleep=sleeps.append)
    llm.decide(T, "s", "b1")
    llm.decide(T, "s", "b2")
    n = len(sleeps)
    llm.decide(T, "s", "b2")  # cached: no API call, no wait
    assert n >= 1 and len(sleeps) == n
