"""Broad lexical audit independent of remover grammar, lexicon and acceptance guards.

Hits mean 'review', not 'delete'. This deliberately finds body prose and quotations.
"""
import re

PATTERNS = {
    "grade_word": re.compile(r"\b(?:top\s+secret|secret|confidential|unclassified|unclas|SBU|FOUO|limited\s+official\s+use)\b", re.I),
    "handling_word": re.compile(r"\b(?:NOFORN|NODIS|EXDIS|LIMDIS|SIPDIS|SPECAT|ORCON|NOCONTRACT|WNINTEL|NO\s+FOREIGN\s+DISSEM|EYES\s+ONLY)\b", re.I),
    "bracket_abbreviation": re.compile(r"[\[(]\s*(?:TS|S|C|U)\s*(?:[/—–-][^\])\n]{0,100})?[\])]", re.I),
    "administrative": re.compile(r"\b(?:classified\s+by|declassif(?:y|ied|ication)|E\.?\s*O\.?\s*\d{4,6}|classification\s*:)\b", re.I),
    "scope_notice": re.compile(r"\b(?:entire|full)\s+(?:text|message|contents|document|cable)\b", re.I),
    "spaced_grade": re.compile(r"\b(?:C\s+O\s+N\s+F\s+I\s+D\s+E\s+N\s+T\s+I\s+A\s+L|S\s+E\s+C\s+R\s+E\s+T|T\s+O\s+P\s+S\s+E\s+C\s+R\s+E\s+T)\b", re.I),
}


def scan(text):
    found = []
    for kind, pattern in PATTERNS.items():
        for m in pattern.finditer(text):
            found.append(dict(start=m.start(), end=m.end(), text=m.group(), kind=kind,
                              status="review", context=text[max(0,m.start()-140):m.end()+180]))
    return sorted(found, key=lambda x: (x["start"], x["end"], x["kind"]))
