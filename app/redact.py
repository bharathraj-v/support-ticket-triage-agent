"""Mask payment details (card numbers, IBANs) before text reaches the LLM, logs or replies."""
import re

_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,4})?\b")
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")


def _luhn(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _mask_card(m: re.Match) -> str:
    digits = re.sub(r"\D", "", m.group(0))
    return "[CARD REDACTED]" if 13 <= len(digits) <= 19 and _luhn(digits) else m.group(0)


def redact(text: str) -> str:
    text = _IBAN.sub("[IBAN REDACTED]", text)
    return _CARD.sub(_mask_card, text)


def contains_payment_details(text: str) -> bool:
    return redact(text) != text
