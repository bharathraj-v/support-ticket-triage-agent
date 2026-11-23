import pytest

from app.rules import evaluate, raise_priority


@pytest.mark.parametrize("subject,body,flag", [
    ("Terugbetaling", "Kunnen we die maand terugkrijgen? Terugbetaling graag", "refund_or_dispute"),
    ("Opzeggen", "Wij willen ons abonnement opzeggen", "cancellation_or_contract"),
    ("AVG verzoek", "Ik wil inzage in mijn persoonsgegevens", "personal_data_request"),
    ("Hulp", "Ik denk dat ons account gehackt is", "security_incident"),
    ("Dreigement", "Mijn advocaat neemt contact op", "legal_threat"),
    ("Overstappen", "Anders gaan we overstappen naar een concurrent", "churn_threat"),
    ("Refund", "Please refund last month", "refund_or_dispute"),
    ("Derde keer", "Als niemand mij belt, zeggen we op.", "churn_threat"),
])
def test_flags_detected(subject, body, flag):
    assert flag in evaluate(subject, body).flags


def test_ordinary_question_not_flagged():
    r = evaluate("Nieuwe medewerker", "Hoe voeg ik een nieuwe medewerker toe aan ons rooster?")
    assert not r.force_handoff and r.min_priority is None


def test_min_priority_is_most_urgent():
    r = evaluate("x", "advocaat en terugbetaling")
    assert r.min_priority == "P1"


def test_empty_detected():
    assert evaluate("", "  ").is_empty
    assert not evaluate("Hallo", "vraag over rooster").is_empty


def test_raise_priority():
    assert raise_priority("P4", "P2") == "P2"
    assert raise_priority("P1", "P2") == "P1"
    assert raise_priority("P3", None) == "P3"
