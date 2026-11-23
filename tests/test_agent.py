import json

from app.agent import Agent
from app.config import ROOT
from app.faq import load_helpcentre
from app.helpdesk import AlreadyHandled
from app.models import Decision, ReplyCheck
from app.state import State

HC = load_helpcentre(ROOT / "data" / "helpcentre.md")


def dec(**kw):
    base = dict(language="nl", category="question", priority="P4", action="reply", reason="FAQ F02",
                faq_ids=["F02"], draft_reply="Hoi, ga naar Team.", handoff_note=None, confidence=0.9)
    return Decision(**{**base, **kw})


class FakeLlm:
    def __init__(self, decision, ok=True):
        self.decision, self.ok, self.calls = decision, ok, 0

    def decide(self, ticket, subject, body):
        self.calls += 1
        if isinstance(self.decision, Exception):
            raise self.decision
        return self.decision

    def check_reply(self, language, faq_ids, reply, ticket_text=""):
        return ReplyCheck(written_in_expected_language=self.ok, english_back_translation="x",
                          supported_by_faq=self.ok)


class FakeHd:
    def __init__(self, tickets, crash_on_first_post=False, conflict=False):
        self.tickets, self.posts, self.conflict = tickets, [], conflict
        self.crash = crash_on_first_post

    def iter_tickets(self, status="open"):
        return iter(self.tickets)

    def _act(self, name):
        def f(tid, body):
            if self.crash:
                self.crash = False
                raise KeyboardInterrupt  # simulate the process dying mid-post
            if self.conflict:
                raise AlreadyHandled(tid)
            self.posts.append((name, tid, body))
        return f

    def __getattr__(self, name):
        if name in ("reply", "handoff", "close"):
            return self._act(name)
        raise AttributeError(name)


def agent(tmp_path, hd, llm, verify=True):
    return Agent(hd, llm, HC, State(tmp_path / "s.json"), tmp_path / "log.jsonl", 0.7, verify)


T = {"id": "T1", "subject": "Nieuwe medewerker", "body": "Hoe voeg ik een medewerker toe?"}


def test_good_reply_is_sent(tmp_path):
    hd = FakeHd([T])
    agent(tmp_path, hd, FakeLlm(dec())).run()
    assert [p[0] for p in hd.posts] == ["reply"]


def test_rule_forces_handoff_even_if_model_replies(tmp_path):
    t = {"id": "T2", "subject": "Terugbetaling", "body": "Wij willen een terugbetaling"}
    hd = FakeHd([t])
    agent(tmp_path, hd, FakeLlm(dec())).run()
    name, _, body = hd.posts[0]
    assert name == "handoff" and body["priority"] == "P2" and body["handoff_note"]


def test_reply_without_valid_faq_becomes_handoff(tmp_path):
    hd = FakeHd([T])
    agent(tmp_path, hd, FakeLlm(dec(faq_ids=["F99"]))).run()
    assert hd.posts[0][0] == "handoff"


def test_failed_dutch_check_becomes_handoff(tmp_path):
    hd = FakeHd([T])
    agent(tmp_path, hd, FakeLlm(dec(), ok=False)).run()
    assert hd.posts[0][0] == "handoff"


def test_low_confidence_becomes_handoff(tmp_path):
    hd = FakeHd([T])
    agent(tmp_path, hd, FakeLlm(dec(confidence=0.3))).run()
    assert hd.posts[0][0] == "handoff"


def test_payment_details_in_reply_never_sent(tmp_path):
    hd = FakeHd([T])
    agent(tmp_path, hd, FakeLlm(dec(draft_reply="Uw IBAN NL91 ABNA 0417 1643 00 is ok"))).run()
    assert hd.posts[0][0] == "handoff"
    assert "NL91" not in json.dumps(hd.posts[0][2])


def test_llm_failure_escalates_to_human(tmp_path):
    hd = FakeHd([T])
    agent(tmp_path, hd, FakeLlm(RuntimeError("boom"))).run()
    assert hd.posts[0][0] == "handoff"


def test_empty_ticket_closed_without_llm(tmp_path):
    llm = FakeLlm(dec())
    hd = FakeHd([{"id": "T3", "subject": "", "body": " "}])
    agent(tmp_path, hd, llm).run()
    assert hd.posts[0][0] == "close" and llm.calls == 0


def test_rerun_after_crash_posts_same_decision_once(tmp_path):
    hd = FakeHd([T], crash_on_first_post=True)
    llm = FakeLlm(dec())
    try:
        agent(tmp_path, hd, llm).run()
    except KeyboardInterrupt:
        pass
    assert hd.posts == []
    agent(tmp_path, hd, llm).run()  # new process, same state file
    assert len(hd.posts) == 1 and llm.calls == 1  # decision reused, not recomputed
    agent(tmp_path, hd, llm).run()
    assert len(hd.posts) == 1  # third run does nothing


def test_409_marks_done_and_is_not_retried(tmp_path):
    hd = FakeHd([T], conflict=True)
    a = agent(tmp_path, hd, FakeLlm(dec()))
    s = a.run()
    assert a.state.is_posted("T1") and not s.failed


def test_log_has_one_line_per_ticket_without_ticket_text(tmp_path):
    hd = FakeHd([T])
    agent(tmp_path, hd, FakeLlm(dec())).run()
    lines = (tmp_path / "log.jsonl").read_text().splitlines()
    assert len(lines) == 1 and "medewerker" not in lines[0].lower()
