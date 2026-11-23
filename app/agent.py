"""Orchestration: fetch open tickets, decide, post exactly one action each, log one line each."""
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .faq import HelpCentre
from .helpdesk import AlreadyHandled, Helpdesk
from .redact import redact
from .rules import evaluate
from .state import State
from .validate import Outcome, fallback_note, finalize


@dataclass
class RunSummary:
    total: int = 0
    posted: dict = field(default_factory=lambda: {"reply": 0, "handoff": 0, "close": 0})
    skipped_already_done: int = 0
    failed: list = field(default_factory=list)


class Agent:
    def __init__(self, helpdesk: Helpdesk, llm, helpcentre: HelpCentre, state: State, log_path: Path,
                 threshold: float, verify_replies: bool = True):
        self.hd, self.llm, self.hc, self.state = helpdesk, llm, helpcentre, state
        self.log_path, self.threshold, self.verify = log_path, threshold, verify_replies

    # --- one ticket ---
    def decide(self, ticket: dict) -> tuple[Outcome, dict]:
        subject, body = ticket.get("subject") or "", ticket.get("body") or ""
        rules = evaluate(subject, body)
        meta = {"rule_flags": list(rules.flags)}
        if rules.is_empty:
            payload = {"language": "nl", "category": "other", "priority": "P4",
                       "reason": "Ticket is empty or has no meaningful content.", "confidence": 0.95}
            return Outcome("close", payload, ["rule:empty"], [], 0.95), meta
        try:
            d = self.llm.decide(ticket, redact(subject), redact(body))
        except Exception as exc:  # model unavailable or garbage: a human must handle it
            meta["llm_error"] = type(exc).__name__
            payload = {"language": "nl", "category": "other", "priority": rules.min_priority or "P3",
                       "reason": "Automatic triage failed; escalated to a human.", "confidence": 0.0,
                       "handoff_note": fallback_note(subject, rules.flags, "model error")}
            return Outcome("handoff", payload, ["llm_error"], [], 0.0), meta
        check = self.llm.check_reply if self.verify else None
        return finalize(subject, d, rules, self.hc, self.threshold, check), meta

    def _post(self, ticket_id: str, action: str, payload: dict) -> str:
        send = {"reply": self.hd.reply, "handoff": self.hd.handoff, "close": self.hd.close}[action]
        try:
            send(ticket_id, payload)
            return "posted"
        except AlreadyHandled:
            return "already_handled"

    def handle(self, ticket: dict) -> str:
        tid = ticket["id"]
        started = time.monotonic()
        stored = self.state.get(tid)
        if stored and stored["status"] == "posted":
            return "skipped"
        meta: dict = {}
        if stored:  # crashed after deciding: reuse the exact same decision
            action, payload, downgrades, faq_ids, conf = (
                stored["action"], stored["payload"], [], [], stored["payload"].get("confidence", 0))
            meta["resumed"] = True
        else:
            outcome, meta = self.decide(ticket)
            action, payload, downgrades, faq_ids, conf = (
                outcome.action, outcome.payload, outcome.downgrades, outcome.faq_ids, outcome.confidence)
            self.state.save_pending(tid, action, payload)
        result = self._post(tid, action, payload)
        self.state.mark_posted(tid)
        self._log({
            "ticket": tid, "action": action, "result": result, "language": payload["language"],
            "category": payload["category"], "priority": payload["priority"], "confidence": conf,
            "faq_ids": faq_ids, "downgrades": downgrades, "latency_ms": int((time.monotonic() - started) * 1000),
            **meta})
        return action if result == "posted" else "already_handled"

    # --- whole run ---
    def run(self) -> RunSummary:
        summary = RunSummary()
        for ticket in list(self.hd.iter_tickets("open")):
            summary.total += 1
            try:
                outcome = self.handle(ticket)
            except Exception as exc:
                summary.failed.append({"ticket": ticket.get("id"), "error": type(exc).__name__})
                self._log({"ticket": ticket.get("id"), "result": "error", "error": type(exc).__name__})
                continue
            if outcome == "skipped":
                summary.skipped_already_done += 1
            elif outcome in summary.posted:
                summary.posted[outcome] += 1
        return summary

    def _log(self, record: dict) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
