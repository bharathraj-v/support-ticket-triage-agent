from app.faq import load_helpcentre
from app.config import ROOT
from app.redact import contains_payment_details, redact


def test_iban_redacted():
    out = redact("Mijn rekening is NL91 ABNA 0417 1643 00 graag")
    assert "NL91" not in out and "[IBAN REDACTED]" in out


def test_iban_without_spaces():
    assert "[IBAN REDACTED]" in redact("NL91ABNA0417164300")


def test_valid_card_redacted_invalid_kept():
    assert "[CARD REDACTED]" in redact("kaart 4111 1111 1111 1111")
    assert redact("order 1234 5678 1234 5678") == "order 1234 5678 1234 5678"


def test_plain_text_untouched():
    t = "Hoe voeg ik 3 medewerkers toe? Zie F02, 60 minuten."
    assert redact(t) == t and not contains_payment_details(t)


def test_faq_parses_all_entries_and_rules():
    hc = load_helpcentre(ROOT / "data" / "helpcentre.md")
    assert hc.ids == {f"F{i:02d}" for i in range(1, 13)}
    assert "refunds" in hc.rules
    assert "Wachtwoord vergeten" in hc.entries["F01"].text
