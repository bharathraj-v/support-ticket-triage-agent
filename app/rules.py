"""Deterministic safety net, independent of the LLM.

The client's rules say some topics always go to a human. We detect them with keyword patterns
(Dutch, English, German) and force a handoff whatever the model says. Over-matching only costs a
handoff, which is never punished; under-matching would reach a customer, which is.
"""
import re
from dataclasses import dataclass

# flag -> (minimum priority, patterns)
_FLAGS: dict[str, tuple[str, list[str]]] = {
    "security_incident": ("P1", [
        r"gehackt", r"\bhack", r"datalek", r"data ?breach", r"onbevoegd", r"unauthori[sz]ed",
        r"phishing", r"account (?:is )?(?:overgenomen|compromitt?eerd|compromised)",
        r"iemand anders (?:heeft )?(?:toegang|ingelogd)", r"verdachte (?:login|activiteit|inlog)",
        r"suspicious (?:login|activity)", r"sicherheitsvorfall", r"gehackt",
    ]),
    "legal_threat": ("P1", [
        r"advocaat", r"juridisch", r"\blegal\b", r"lawyer", r"\bsue\b", r"rechtszaak", r"rechtbank",
        r"aansprakelijk", r"\bdagvaard", r"ingebrekestelling", r"anwalt", r"klage",
        r"autoriteit persoonsgegevens", r"consumentenbond",
    ]),
    "refund_or_dispute": ("P2", [
        r"terugbetal", r"terugstort", r"restitutie", r"geld terug", r"refund", r"creditnota",
        r"onterecht (?:afgeschreven|gefactureerd|in rekening)", r"betwist", r"dispute", r"chargeback",
        r"storneren", r"dubbel (?:afgeschreven|betaald|gefactureerd)", r"twice charged", r"erstattung",
        r"ten onrechte (?:afgeschreven|gefactureerd|in rekening)", r"verkeerd bedrag",
    ]),
    "cancellation_or_contract": ("P2", [
        r"opzeg", r"opgezegd", r"abonnement (?:beëindig|stopzetten|stoppen)", r"beëindig",
        r"\bcancel", r"terminate (?:my|our|the)", r"contract(?:wijziging| wijzig| aanpass| verlengen| opzeggen)",
        r"kündig", r"annuleer (?:ons|mijn|het) abonnement", r"stopzetten",
    ]),
    "personal_data_request": ("P2", [
        r"\bavg\b", r"\bgdpr\b", r"\bdsgvo\b", r"persoonsgegevens", r"inzageverzoek",
        r"recht op vergetelheid", r"right to be forgotten", r"verwijder (?:al )?(?:mijn|onze) (?:gegevens|data)",
        r"delete (?:all )?(?:my|our) (?:data|personal)", r"data subject", r"datenauskunft",
        r"kopie van (?:mijn|onze) (?:gegevens|data)",
    ]),
    "churn_threat": ("P2", [
        r"overstappen", r"stap(?:pen)? over", r"naar (?:een )?(?:andere|concurrent)", r"concurrent",
        r"switch(?:ing)? to", r"look(?:ing)? for (?:an )?alternative", r"alternatief zoeken",
        r"zoeken we (?:een )?(?:ander|alternatief)", r"we (?:gaan|zijn) weg", r"nicht mehr nutzen",
        r"wechseln zu", r"verlaten", r"leave (?:you|roosterly)",
    ]),
}
_COMPILED = {k: (prio, [re.compile(p, re.I) for p in pats]) for k, (prio, pats) in _FLAGS.items()}

_PRIORITY_ORDER = ["P1", "P2", "P3", "P4"]


@dataclass(frozen=True)
class RuleResult:
    flags: tuple[str, ...]
    min_priority: str | None
    is_empty: bool

    @property
    def force_handoff(self) -> bool:
        return bool(self.flags)


def evaluate(subject: str, body: str) -> RuleResult:
    text = f"{subject}\n{body}"
    flags = [name for name, (_, pats) in _COMPILED.items() if any(p.search(text) for p in pats)]
    min_prio = None
    for f in flags:
        prio = _COMPILED[f][0]
        if min_prio is None or _PRIORITY_ORDER.index(prio) < _PRIORITY_ORDER.index(min_prio):
            min_prio = prio
    is_empty = len(re.sub(r"\W+", "", text)) < 3
    return RuleResult(tuple(flags), min_prio, is_empty)


def raise_priority(current: str, minimum: str | None) -> str:
    """Return the more urgent of the two priorities (P1 is most urgent)."""
    if minimum is None:
        return current
    return min(current, minimum, key=_PRIORITY_ORDER.index)
