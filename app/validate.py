"""Turn a raw model Decision into something safe to send. Anything doubtful becomes a handoff."""
from dataclasses import dataclass, field

from .faq import HelpCentre
from .models import Decision
from .redact import contains_payment_details, redact
from .rules import RuleResult, raise_priority


@dataclass
class Outcome:
    action: str                      # reply | handoff | close
    payload: dict                    # body for the helpdesk endpoint
    downgrades: list[str] = field(default_factory=list)  # why the model's choice was overridden
    faq_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    notes: list[str] = field(default_factory=list)  # reviewer details, for debugging
    check_problems: list[str] = field(default_factory=list)  # reviewer's problems, used for the repair step


def fallback_note(subject: str, flags: tuple[str, ...], why: str, model_reason: str = "") -> str:
    topic = ", ".join(f.replace("_", " ") for f in flags) or "needs human review"
    return (f"Automatic triage could not safely answer this ticket ({why}). "
            f"Subject: \"{redact(subject)}\". Topic flags: {topic}. "
            f"Model's read of the ticket: {redact(model_reason)} Please read the original ticket and respond.")


def finalize(subject: str, d: Decision, rules: RuleResult, hc: HelpCentre, threshold: float,
             reply_check=None, ticket_text: str = "") -> Outcome:
    """`reply_check` is a callable (language, faq_ids, reply, ticket_text) -> ReplyCheck, run only for replies."""
    downgrades: list[str] = []
    priority = raise_priority(d.priority, rules.min_priority)
    action = d.action
    note = d.handoff_note
    reply = d.draft_reply
    faq_ids = [i for i in d.faq_ids if i in hc.ids]
    notes: list[str] = []
    problems: list[str] = []

    if rules.force_handoff and action != "handoff":
        downgrades.append("rule:" + "+".join(rules.flags))
        action = "handoff"
    if action == "close" and not rules.is_empty and d.confidence < threshold:
        downgrades.append("low_confidence_close")
        action = "handoff"
    if action == "reply":
        if not reply or not reply.strip():
            downgrades.append("empty_reply")
        elif not faq_ids:
            downgrades.append("no_valid_faq_id")
        elif d.confidence < threshold:
            downgrades.append("low_confidence")
        elif contains_payment_details(reply):
            downgrades.append("payment_details_in_reply")
        elif reply_check is not None:
            check = reply_check(d.language, faq_ids, reply, ticket_text)
            problems = list(check.problems)
            notes = list(check.problems) + ["back-translation: " + check.english_back_translation, "draft: " + reply]
            if not check.written_in_expected_language:
                downgrades.append("wrong_language")
            if not check.supported_by_faq:
                downgrades.append("reply_not_supported_by_faq")
        if downgrades:
            action = "handoff"

    base = {"language": d.language, "category": d.category, "priority": priority,
            "reason": redact(d.reason), "confidence": d.confidence}
    if action == "reply":
        payload = {**base, "draft_reply": reply}
    elif action == "close":
        payload = base
    else:
        if not note or not note.strip() or d.action != "handoff":
            note = fallback_note(subject, rules.flags,
                                 ", ".join(downgrades) or "model chose handoff without a note", d.reason)
        payload = {**base, "handoff_note": redact(note)}
    return Outcome(action=action, payload=payload, downgrades=downgrades, faq_ids=faq_ids, confidence=d.confidence, notes=notes,
                   check_problems=problems)
