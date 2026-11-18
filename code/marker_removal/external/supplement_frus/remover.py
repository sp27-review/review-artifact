"""Conservative, label-blind span grammar. Does not import the existing pipeline."""
import json
import re
from pathlib import Path
from typing import NamedTuple


class Token(NamedTuple):
    value: str
    start: int
    end: int


LEXICON = json.loads(Path(__file__).with_name("lexicon.json").read_text())
OPEN = {"(", "["}
CLOSE = {")", "]"}
SEPARATORS = {"-", "–", "—", ":", ".", ",", ";"}
TOKEN = re.compile(r"[^\W\d_]+|\d+|_|[^\w\s]", re.UNICODE)
GRADES = sorted((tuple(x.split()) for x in LEXICON["grades"] + LEXICON["grade_abbreviations"]), key=len, reverse=True)
HANDLING = sorted((tuple(x.split()) for x in LEXICON["handling"]), key=len, reverse=True)
START_WORDS = {p[0] for p in GRADES + HANDLING} | set(LEXICON["scope_heads"]) | OPEN


def tokens(text):
    return [Token(m.group().upper(), m.start(), m.end()) for m in TOKEN.finditer(text)]


def phrase_at(ts, i, phrases):
    for phrase in phrases:
        if tuple(t.value for t in ts[i:i+len(phrase)]) == phrase:
            return i+len(phrase)
    return None


def grade_at(ts, i):
    # Letter-spaced banners have the same lexical identity, without rewriting text.
    for phrase in ("TOPSECRET", "CONFIDENTIAL", "UNCLASSIFIED", "SECRET"):
        segment = ts[i:i+len(phrase)]
        if len(segment) == len(phrase) and all(len(t.value) == 1 for t in segment):
            if "".join(t.value for t in segment) == phrase:
                return i+len(phrase)
    return phrase_at(ts, i, GRADES)


def scope_at(ts, i):
    if i+1 < len(ts) and ts[i].value in LEXICON["scope_heads"] and ts[i+1].value in LEXICON["scope_objects"]:
        return i+2
    return None


def line_bounds(text, a, b):
    return text.rfind("\n", 0, a)+1, text.find("\n", b) if "\n" in text[b:] else len(text)


def whole_notice_context(text, a, b):
    """Only a complete, standalone declaration, with optional paragraph number."""
    ls, le = line_bounds(text, a, b)
    prefix, suffix = text[ls:a], text[b:le]
    if not re.fullmatch(r"\s*(?:\d{1,4}[.)]\s*)?", prefix):
        return False
    if suffix.strip(" \t\r.;:"):
        return False
    nearby = text[max(0, ls-180):ls]
    if re.search(r"(?:QUOTE|QUOTED|EXCERPT|HE WROTE|SHE WROTE)\s*[:.\r\n ]*$", nearby, re.I):
        return False
    body = text[a:b]
    return len(body) <= 180 and not re.search(r"\n\s*\n", body)


def notice_end(ts, i):
    """[grade] [handling] [scope], or scope [is] [grade]; punctuation is structural."""
    j = i
    if ts[j].value in OPEN:
        j += 1
    grade = grade_at(ts, j)
    if grade is not None:
        j = grade
        if j < len(ts) and ts[j].value in CLOSE:
            j += 1
        while j < len(ts) and ts[j].value == "/":
            end = phrase_at(ts, j+1, HANDLING + [("SENSITIVE",)])
            if end is None:
                return None
            j = end
        while j < len(ts) and ts[j].value in SEPARATORS:
            j += 1
        if j < len(ts) and ts[j].value in OPEN:
            j += 1
        end = scope_at(ts, j)
        if end is None:
            return None
        j = end
    else:
        scope = scope_at(ts, j)
        if scope is None:
            return None
        j = scope
        while j < len(ts) and ts[j].value in SEPARATORS | {"IS"}:
            j += 1
        if j < len(ts) and ts[j].value in OPEN:
            j += 1
        end = grade_at(ts, j)
        if end is None:
            return None
        j = end
    if j < len(ts) and ts[j].value in CLOSE:
        j += 1
    if j < len(ts) and ts[j].value == ".":
        j += 1
    return j


def header_context(text, a, b):
    """Current-document routing preamble, before body/blank-line boundary."""
    if a > 1800:
        return False
    line_end = text.find("\n", b)
    after = text[b:line_end if line_end >= 0 else len(text)].strip()
    boundary = (not after or after[0] in ";,/" or
                bool(re.match(r"\.?(?:\s*)(?:E\.?O\.?\s*\d|SUBJECT\s*:|TAGS\s*:|FM\s|TO\s|INFO\s)", after, re.I)) or
                after == ".")
    if not boundary:
        return False
    before = text[:a]
    if re.search(r"[\"“”‘’]|\b(?:QUOTE|QUOTED|UNQUOTE|SUBJECT|SUMMARY|COMMENT)\b", before, re.I):
        return False
    route = re.search(r"\bFM\s+[A-Z]", before) and re.search(r"\bTO\s+[A-Z]", before)
    transmission = route and re.match(r"\s*(?:TELEGRAM\b|[RP]\s+\d{6}Z\b)", before, re.I)
    # Archival TXT may insert blank lines between transmission fields.
    # Allow them only in an explicit telegram/FM/TO preamble, before SUBJECT.
    if re.search(r"\n[ \t\r]*\n", before.strip()) and not transmission:
        return False
    # A bare leading cable number is an explicit metadata slot.
    if re.fullmatch(r"\s*(?:\d{1,8}[.]\s*)?", before):
        return True
    # Require multiple routing fields, or an initial cable-number/handling field.
    numbered = re.match(r"\s*\d{1,8}\.\s*(?:EXDIS|NODIS|LIMDIS|NOFORN|SIPDIS)\b", before, re.I)
    return bool(route or numbered)


def detect(text):
    ts = tokens(text)
    edits = []
    for i, t in enumerate(ts):
        if t.value not in START_WORDS and t.value not in {"T", "S", "C", "U"}:
            continue
        j = notice_end(ts, i)
        if j is not None and whole_notice_context(text, t.start, ts[j-1].end):
            edits.append(dict(start=t.start, end=ts[j-1].end, rules=["whole_document_declaration_grammar"]))
            continue
        j = phrase_at(ts, i, HANDLING)
        if j is not None and header_context(text, t.start, ts[j-1].end):
            # Match the code, never the routing destinations or surrounding prose.
            if ts[j-1].end-t.start <= 60:
                start = t.start
                instruction = re.search(r"\bmilitary\s+addressees\s+handle\s+as\s*$", text[:start], re.I)
                if instruction:
                    start = instruction.start()
                edits.append(dict(start=start, end=ts[j-1].end, rules=["current_document_routing_code"]))
        j = grade_at(ts, i)
        if j is not None and not text[:t.start].strip() and whole_notice_context(text, t.start, ts[j-1].end):
            if ts[j-1].end-t.start <= 60 and ts[j-1].end-t.start > 2:
                edits.append(dict(start=t.start, end=ts[j-1].end, rules=["initial_grade_banner"]))
    merged = []
    for edit in sorted(edits, key=lambda x: (x["start"], x["end"])):
        if merged and edit["start"] < merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], edit["end"])
            merged[-1]["rules"] = sorted(set(merged[-1]["rules"]+edit["rules"]))
        else:
            merged.append(dict(edit))
    for edit in merged:
        edit["text"] = text[edit["start"]:edit["end"]]
    return merged


def apply(text, edits):
    parts, last = [], 0
    for edit in edits:
        a, b = edit["start"], edit["end"]
        if not (last <= a < b <= len(text)) or text[a:b] != edit["text"]:
            raise ValueError("Invalid or overlapping deletion coordinates")
        parts.append(text[last:a]); last = b
    parts.append(text[last:])
    return "".join(parts)
