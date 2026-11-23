"""Parse the help centre file into entries (F01..F12) and the client's rules."""
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FaqEntry:
    id: str
    title: str
    text: str


@dataclass(frozen=True)
class HelpCentre:
    entries: dict[str, FaqEntry]
    rules: str
    raw: str

    @property
    def ids(self) -> set[str]:
        return set(self.entries)


def load_helpcentre(path: Path) -> HelpCentre:
    raw = path.read_text(encoding="utf-8")
    body, _, rules = raw.partition("## The client's rules for the support team")
    entries: dict[str, FaqEntry] = {}
    for m in re.finditer(r"^## (F\d{2}) · (.+?)\n(.*?)(?=^## |\Z)", body, re.S | re.M):
        entries[m.group(1)] = FaqEntry(m.group(1), m.group(2).strip(), m.group(3).strip().strip("-").strip())
    if not entries:
        raise ValueError(f"No FAQ entries found in {path}")
    return HelpCentre(entries=entries, rules=rules.strip(), raw=raw)
