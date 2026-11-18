#!/usr/bin/env python3
"""(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
from __future__ import annotations
import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path("/path/to/workspace")
sys.path.insert(0, str(ROOT / "src"))
_R4 = ROOT / "experiments/dataset_construction/pipeline/r4/p5_r4_build_marker_dataset.py"
_spec = importlib.util.spec_from_file_location("p5r4", _R4)
R4 = importlib.util.module_from_spec(_spec)
sys.modules["p5r4"] = R4
_spec.loader.exec_module(R4)


GRADE_TOKEN = r"(?:TS|S|C|U)"
SEP = r"(?P<sep>\s*[/\\|·-]{1,2}\s*|\s+)"
TAIL = r"(?P<tail>[^)\n]{0,20})"
PORTION_CLOSED = re.compile(r"\(\s*(?P<grade>" + GRADE_TOKEN + r")" + SEP + TAIL + r"\)")
PORTION_OPEN = re.compile(r"\(\s*(?P<grade>" + GRADE_TOKEN + r")(?P<sep>\s*[/\\|·-]{1,2}\s*)" + TAIL + r"(?=\n|$)")

_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
_PARA_NUM = re.compile(r"[0-9]{1,3}\s*[.,)]|[a-z]\s*[.,)]|[IVXivx]{1,4}\s*[.,)]")
_BULLET = re.compile(r"[-—•*_]+")
_SENT_END = re.compile(r"[.!?:;\"”]\s*$")
_ONLY_NUMS = re.compile(r"(?:\d+\.\s*)+$")


_TAIL_DIGITS = re.compile(r"^[\d\s.\-]+$")
_SERIAL = re.compile(r"^(?:N\b|N\s|No\.?\b|/N\b)", re.I)
_ORG_CHAIN = re.compile(r"^[A-Z][A-Z&]{1,}(?:\s*/\s*[A-Z][A-Z&]{1,}){1,}$")
_KNOWN_NON_MARK = re.compile(
    r"^(?:Bahn|2[A-Z]?\b|235\b|130\b|141\b|5[AB]\b|Clo\b|Cl\d|and\b|or\b|Bank\b|Corps\b)", re.I)


def _line_bounds(text: str, i: int) -> tuple[int, int]:
    ls = text.rfind("\n", 0, i) + 1
    le = text.find("\n", i)
    return ls, (len(text) if le < 0 else le)


def at_leading(text: str, start: int) -> bool:
    ls, _ = _line_bounds(text, start)
    before = text[ls:start].strip()
    if before == "":
        return True
    if _PARA_NUM.fullmatch(before) or _BULLET.fullmatch(before) or _ONLY_NUMS.fullmatch(before):
        return True
    return bool(_SENT_END.search(before))


def at_trailing(text: str, start: int, end: int) -> bool:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    ls, le = _line_bounds(text, start)
    if text[end:le].strip():
        return False
    head = text[ls:start].strip()
    words = _WORD.findall(head)
    return 0 < len(words) <= 12 and not re.search(r"[.!?]$", head)


_HAS_LOWER = re.compile(r"[a-z]")
_TWO_LETTERS = re.compile(r"[A-Za-z]{2,}")
_ABBREV_DOT = re.compile(r"^[A-Za-z]{1,3}\.$")


def tail_rejected(tail: str, sep: str = "", closed: bool = True, grade: str = "") -> bool:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    t = tail.strip()
    if not t:
        return False
    if _TAIL_DIGITS.match(t):
        digits = re.sub(r"\D", "", t)
        if "-" in sep or len(digits) >= 3:
            return True


    if not sep.strip() and _HAS_LOWER.search(t):
        return True

    if _ABBREV_DOT.match(t):
        return True




    if not sep.strip():
        joined = (grade + re.sub(r"[^A-Za-z]", "", t)).lower()
        if len(joined) >= 5 and (joined in R4._ENGLISH_WORDS or _near_real_word(joined)):
            return True

    if not sep.strip() and re.search(r"\d", t):
        return True


    if "-" in sep and t.isalpha() and t.islower():
        return True


    if re.search(r"\d{3,}", t):
        return True


    if not closed and (_TWO_LETTERS.search(t) or re.search(r"\d", t)):
        return True
    return bool(_SERIAL.match(t) or "&" in t
                or _ORG_CHAIN.match(t) or _KNOWN_NON_MARK.match(t))


_CHIEF_CTX = re.compile(r"\bChief\b[^.\n]{0,40}$", re.I)


def portion_spans(text: str) -> list[tuple[int, int]]:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    spans: list[tuple[int, int]] = []
    for rx, closed in ((PORTION_CLOSED, True), (PORTION_OPEN, False)):
        for m in rx.finditer(text):
            if tail_rejected(m.group("tail"), m.group("sep") or "", closed, m.group("grade")):
                continue


            if m.group("grade") == "C" and _CHIEF_CTX.search(text[max(0, m.start() - 60):m.start()]):
                continue
            if at_leading(text, m.start()) or at_trailing(text, m.start(), m.end()):
                spans.append((m.start(), m.end()))
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    merged: list[tuple[int, int]] = []
    for s, e in spans:
        if merged and s < merged[-1][1]:
            if e > merged[-1][1]:
                merged[-1] = (merged[-1][0], e)
            continue
        merged.append((s, e))
    return merged



_ALPHA = re.compile(r"[A-Za-z]{3,}")


_DIGIT_RUN = re.compile(r"\d{3,}")
_ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def _near_real_word(token: str) -> bool:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    a = re.sub(r"[^A-Za-z]", "", token).lower()
    if len(a) < 4:
        return False
    if a in R4._ENGLISH_WORDS:
        return True
    for i in range(len(a)):
        if a[:i] + a[i + 1:] in R4._ENGLISH_WORDS:
            return True
        for c in _ALPHABET:
            if c != a[i] and a[:i] + c + a[i + 1:] in R4._ENGLISH_WORDS:
                return True
    for i in range(len(a) + 1):
        for c in _ALPHABET:
            if a[:i] + c + a[i:] in R4._ENGLISH_WORDS:
                return True
    return False


FUZZY_CLASS_FORMS_STRICT = R4.FUZZY_CLASS_FORMS





_CLASS_SCREEN = tuple((tgt, mx, len(tgt), frozenset(tgt)) for tgt, mx in FUZZY_CLASS_FORMS_STRICT)
_CLASS_LEN_MIN = min(l - m for _, m, l, _ in _CLASS_SCREEN)
_CLASS_LEN_MAX = max(l + m for _, m, l, _ in _CLASS_SCREEN)


def fuzzy_class_distance(tok: str):
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    lt = len(tok)
    if lt < _CLASS_LEN_MIN or lt > _CLASS_LEN_MAX:
        return None
    chars = frozenset(tok)
    best = None
    for tgt, mx, lg, tchars in _CLASS_SCREEN:
        if abs(lt - lg) > mx:
            continue
        if len(tchars) - len(chars & tchars) > mx:
            continue
        d = R4._edit_distance(tok, tgt)
        if d <= mx and (best is None or d < best):
            best = d
    return best


_DOC_ID = re.compile(r"\b[A-Z]{2,}[-/]?[A-Z]?\d+[-/]\w+")
_INSTRUCTION = re.compile(r"^\s*(?:Regrade|Downgrade|Upgrade|Classify|Declassify)\b", re.I)


def is_ocr_noise_banner(line: str) -> bool:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    if _DIGIT_RUN.search(line):
        return False
    if _DOC_ID.search(line):
        return False
    if _INSTRUCTION.match(line.strip()):
        return False
    tokens = [t for t in (R4._alpha_upper(w) for w in line.split()) if len(t) >= 5]
    hits = [(t, d) for t in tokens if (d := fuzzy_class_distance(t)) is not None]
    if len(hits) != 1:
        return False
    d_class = hits[0][1]
    if d_class > 0 and _near_real_word(hits[0][0]):
        return False

    for w in _ALPHA.findall(line):
        up = w.upper()
        if fuzzy_class_distance(up) is not None:
            continue
        if R4._is_real_word(w):
            return False
    return True


_CLASS_TOKEN = re.compile(r"[A-Za-z][A-Za-z@#$%&*]{3,}")


def banner_grade_spans(text: str) -> list[tuple[int, int]]:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    out = []
    pos = 0
    for line in text.split("\n"):
        ls, le = pos, pos + len(line)
        pos = le + 1
        s = line.strip()
        if not s or len(_WORD.findall(s)) > 8:
            continue
        if not any(_CLASS_LEN_MIN <= len(R4._alpha_upper(w)) <= _CLASS_LEN_MAX for w in s.split()):
            continue
        if R4.is_solo_classification_line(s):
            continue
        if not is_ocr_noise_banner(s):
            continue
        for m in _CLASS_TOKEN.finditer(line):
            up = R4._alpha_upper(m.group())
            if len(up) < 5:
                continue
            d = fuzzy_class_distance(up)
            if d is None:
                continue
            if d > 0 and _near_real_word(up):
                continue
            out.append((ls + m.start(), ls + m.end()))
    return out







_FUNCTION_WORDS = frozenset("""is are was were be been being of to by the that and but in for with
from as at it this which have has had not their its our will would than then when where who""".split())
_GRADE_WORD = re.compile(r"(?<![A-Za-z])(TOP\s{0,3}SECRET|SECRET|CONFIDENTIAL|UNCLASSIFIED)(?![A-Za-z])")
_PROPER_NOUN_LINE = re.compile(r"SECRET\s+SERVICE", re.I)


def _function_word_count(line: str) -> int:
    return sum(1 for w in _WORD.findall(line) if w.lower() in _FUNCTION_WORDS)


def banner_grade_token_spans(text: str) -> list[tuple[int, int]]:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    out = []
    for m in _GRADE_WORD.finditer(text):
        ls = text.rfind("\n", 0, m.start()) + 1
        le = text.find("\n", m.end())
        le = len(text) if le < 0 else le
        line = text[ls:le]
        stripped = line.strip()
        nw = len(_WORD.findall(stripped))
        if nw > 8:
            continue
        if _PROPER_NOUN_LINE.search(stripped) or _DIGIT_RUN.search(stripped):
            continue



        head = stripped[:max(0, m.start() - ls - (len(line) - len(line.lstrip())))]
        at_head = len(_WORD.findall(head)) == 0
        if _function_word_count(stripped) >= (2 if at_head else 1):
            continue
        out.append((m.start(), m.end()))
    return out



def scrub_r5(text: str) -> tuple[str, dict]:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    spans = [(s, e, "portion") for s, e in portion_spans(text)]
    spans += [(s, e, "banner") for s, e in banner_grade_spans(text)]
    spans += [(s, e, "banner") for s, e in banner_grade_token_spans(text)]
    spans.sort()
    kept, out, last = [], [], 0
    stats = {"portion": 0, "banner": 0, "chars": 0}
    spans = sorted(set(spans))
    for s, e, kind in spans:
        if s < last:
            continue
        out.append(text[last:s]); last = e
        stats[kind] += 1; stats["chars"] += e - s
        kept.append((s, e, kind))
    out.append(text[last:])
    new = "".join(out)
    new = re.sub(r"[ \t]{2,}", " ", new)
    new = re.sub(r"\n{3,}", "\n\n", new)
    stats["removed_spans"] = kept
    return new, stats
