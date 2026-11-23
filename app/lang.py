"""Cheap, model-independent language check (Dutch / English / German) using common function words.

The LLM labels the ticket's language; this is an independent second opinion that needs no API call and
that I can unit-test without reading Dutch. It stays silent (None) when the text is too short to judge.
"""
import re

_WORDS = {
    "nl": "de het een en ik we wij jullie niet van voor op te dat ons naar hoe kunnen kan mijn graag dit als bij met "
          "zijn er maar ook nog wat waar onze heb hebben staat uw zijn wordt worden geen nodig sinds hebben dan "
          "zou zouden willen willen kunt moet moeten".split(),
    "en": "the and are we our you your can not to of for it that this with have from on how my when since but was "
          "were will would could should does do an at be been has had their there what where which".split(),
    "de": "der die das und ist nicht wir ihr ich mit für von zu den dem ein eine auf wie können unsere haben auch "
          "sind wird werden kein bitte seit aber oder wenn dass über nach bei".split(),
}
_SETS = {k: set(v) for k, v in _WORDS.items()}


def detect_language(text: str) -> str | None:
    tokens = re.findall(r"[a-zA-ZäöüßÄÖÜ']+", text.lower())
    scores = {lang: sum(t in words for t in tokens) for lang, words in _SETS.items()}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    (best, top), (_, second) = ranked[0], ranked[1]
    if top >= 3 and top >= 2 * second + 1:
        return best
    return None
