import pytest

from app.lang import detect_language


@pytest.mark.parametrize("text,expected", [
    ("Nobody in our three restaurants can open the app since 08:00. Staff cannot clock in and the lunch shift "
     "starts in an hour.", "en"),
    ("Hoi, ik ben mijn wachtwoord vergeten en het manageraccount staat op mijn naam. Hoe kom ik er weer in?", "nl"),
    ("Wie können meine Mitarbeiter untereinander Schichten tauschen? Wir haben das noch nicht gefunden.", "de"),
    ("Typo", None),
    ("", None),
])
def test_detects_language_or_abstains(text, expected):
    assert detect_language(text) == expected
