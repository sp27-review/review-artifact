#!/usr/bin/env python3
"""(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from corelib.textmatch import edit_distance as _edit_distance

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # main2_IA_CIA/ (config.py)
from config import MASKING



LABEL_ORDER = {
    "UNCLASSIFIED": 0,
    "CONFIDENTIAL": 1,
    "SECRET": 2,
    "TOP SECRET": 3,
}

LABEL_PATTERNS = {
    "TOP SECRET": re.compile(r"\bTOP\s+SECRET\b|^\s*\(?TS\)?\s*$", re.I),
    "SECRET": re.compile(r"\bSECRET\b|^\s*\(?S\)?\s*$", re.I),
    "CONFIDENTIAL": re.compile(r"\bCONFIDENTIAL\b|^\s*\(?C\)?\s*$", re.I),
    "UNCLASSIFIED": re.compile(r"\bUNCLASSIFIED\b|^\s*\(?U\)?\s*$", re.I),
}

CONTROL_MARKING_PATTERN = re.compile(
    r"\b(NOFORN|ORCON|NOCONTRACT|CONTROL|DISSEM|EXDIS|NODIS|"
    r"U\.?S\.?\s+OFFICIALS\s+ONLY)\b",
    re.I,
)

RELEASE_BOILERPLATE_PATTERN = re.compile(
    r"APPROVED\s+FOR\s+RELEASE|DECLASSIFIED|SANITIZED|"
    r"RELEASE\s+IN\s+FULL|CIA[-\s]?RDP|IA[-\s]?RDP|RDP\d",
    re.I,
)

# Re-classification / downgrade / declassification-instruction stamps that carry
# classification history and must not leak into content-only text.
RECLASS_STAMP_PATTERN = re.compile(
    r"CHANGE\w*\s+IN\s+CLASS|CLASS\w*\.?\s+CHANGED|CHANGED\s+TO|DDA\s+MEMO|"
    r"REVIEW\s+COMPLETED|DECLASSIFY\s+ON|DERIVED\s+FROM",
    re.I,
)

# --- OCR-robust (fuzzy) banner/stamp detection ---------------------------------
# Precision-first: fuzzy matching is applied ONLY to short lines in the top/bottom
# banner zone by the caller, so ordinary body words (secure, release of hostages,
# classified material) in long body sentences are never touched.

# Target forms and their allowed edit distance (tighter for shorter words).
FUZZY_CLASS_FORMS = MASKING.fuzzy_class_forms
# Max words for a line to be considered a banner/stamp rather than body text.
BANNER_MAX_WORDS = MASKING.banner_max_words


def _alpha_upper(text: str) -> str:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    return re.sub(r"[^A-Z]", "", text.upper())






def _load_english_words() -> set[str]:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    try:
        return {w.strip().lower() for w in open("/usr/share/dict/words") if w.strip().isalpha()}
    except OSError:
        return set()

_ENGLISH_WORDS = _load_english_words()

def _is_real_word(tok: str) -> bool:
    a = re.sub(r"[^A-Za-z]", "", tok).lower()
    return len(a) >= 3 and a in _ENGLISH_WORDS

_CLASSWORDS_UPPER = ("TOP SECRET", "TOPSECRET", "SECRET", "CONFIDENTIAL", "UNCLASSIFIED")

def _is_protected_body_line(text: str) -> bool:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    s = text.strip()
    if not s:
        return False



    if re.search(r"RDP\s*\d", s.upper()):
        return False
    if _scrub_release_fragment_spans(s):
        return False

    if re.search(r"\bTWX\b", s, re.I) and re.search(r"\b(?:IN\s*\d{3,}|\d{1,2}\s+[A-Z][a-z]{2}\s+\d{2})", s):
        return True

    lower_words = re.findall(r"\b[a-z][a-z']{2,}\b", s)

    marking_hits = len(re.findall(r"(?i)secret|confidential|unclassified|noforn|classif|declassif|"
                                  r"downgrad|espionage|dissem|control|exempt|approved|release", s))
    if len(lower_words) >= 3 and marking_hits <= max(1, len(lower_words) // 2):
        return True
    return False


def _strip_banner_tokens_in_prose(text: str) -> tuple[str, list[str]]:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    removed: list[str] = []

    def _cut(pat, s):
        def repl(m):
            removed.append(m.group(0)); return " "
        return pat.sub(repl, s)
    text = _cut(re.compile(r"(?<![A-Za-z])TOP\s+SECRET(?![A-Za-z])"), text)
    text = _cut(re.compile(r"(?<![A-Za-z])(SECRET|CONFIDENTIAL|UNCLASSIFIED)(?![A-Za-z])"), text)

    if re.search(r"\bTWX\b", text, re.I):
        text = _cut(re.compile(r"(?i)\bSecret[.,]"), text)
        text = _cut(re.compile(r"(?<![A-Za-z])STAT[.,]?(?![A-Za-z])"), text)
    return re.sub(r"[ \t]{2,}", " ", text), removed



def fuzzy_classification_line(text: str) -> bool:
    """True if a short banner line contains an OCR-approximate classification word.
    Caller must first restrict to short top/bottom lines."""
    tokens = [t for t in (_alpha_upper(w) for w in text.split()) if t]
    candidates = list(tokens) + [tokens[i] + tokens[i + 1] for i in range(len(tokens) - 1)]
    for cand in candidates:
        if len(cand) < 5:
            continue
        for target, max_edits in FUZZY_CLASS_FORMS:
            if _edit_distance(cand, target) <= max_edits:
                return True
    return False


def fuzzy_release_line(text: str) -> bool:
    """True if a line is an OCR-approximate release/declassify stamp. The CIA-RDP
    number is the most reliable signature; else 'Approved'+'Release' both fuzzy-match."""
    if re.search(r"RDP\s*\d", text.upper()):
        return True
    tokens = [t for t in (_alpha_upper(w) for w in text.split()) if len(t) >= 5]
    has_appr = any(_edit_distance(t, "APPROVED") <= 2 for t in tokens)
    has_rel = any(_edit_distance(t, "RELEASE") <= 2 for t in tokens)
    if has_appr and has_rel:
        return True
    if any(_edit_distance(t, "DECLASSIFIED") <= 3 for t in tokens):
        return True
    if any(_edit_distance(t, "SANITIZED") <= 2 for t in tokens):
        return True
    return False


def is_solo_classification_line(text: str) -> bool:
    """True if the line is essentially a bare classification word (plus OCR noise),
    e.g. 'CONFIDENTIAL', 'y CONFIDENTIAL |'. Such lines are banners regardless of
    zone. A line with a second real content word (e.g. 'secret information',
    'SECRET SERVICE') is NOT solo, so body sentences are preserved."""
    tokens = [t for t in (_alpha_upper(w) for w in text.split()) if len(t) >= 5]
    if len(tokens) != 1:
        return False
    for target, max_edits in FUZZY_CLASS_FORMS:
        if _edit_distance(tokens[0], target) <= max_edits:
            return True
    return False


def is_fuzzy_banner_or_stamp(line: "Line") -> bool:
    """Precision-first removal test for OCR-robust mode. A re-classification stamp or
    a bare classification word is a banner anywhere; otherwise only short lines in the
    top/bottom banner zone qualify for fuzzy removal, so body sentences are preserved."""
    text = line.text
    if RECLASS_STAMP_PATTERN.search(text):
        return True
    if is_solo_classification_line(text):
        return True
    if line.zone in {"top", "bottom"} and len(text.split()) <= BANNER_MAX_WORDS:
        return fuzzy_classification_line(text) or fuzzy_release_line(text)
    return False


# --- Sub-line (in-line) scrubbing for OCR-robust mode ---------------------------
# Banners/stamps sometimes get OCR-merged into long lines, so line-level removal
# misses them. These rules remove ONLY fixed marking phrases inside a line; free
# body prose is never touched (precision-first, no content holes).

# Portion marks like "(Confidential)", "(SECRET No Foreign Dissem)". Case-sensitive
# capitalized/upper forms only, and the tail must be marking vocabulary (caps/dissem),
# so body parentheticals like "(secret talks between ...)" survive.
_CLASS_CAP = r"(?:Top[\s-]{0,3}Secret|TOP[\s-]{0,3}SECRET|Secret|SECRET|Confidential|CONFIDENTIAL|Unclassified|UNCLASSIFIED)"
PORTION_MARK_RE = re.compile(
    r"\(\s*" + _CLASS_CAP +
    r"(?:\s+[Nn]o\s+[Ff]oreign\s+[Dd]issem|\s+NOFORN|\s+[A-Z0-9 /-]{0,20})?\s*\)"
)



PORTION_ABBR_RE = re.compile(r"\(\s*(?:TS|[USC])\s*\)(?:\s*//\s*\w+)?")
# Checkbox rows on routing slips: two or more adjacent classification words
# ("UNCLASSIFIED CONFIDENTIAL SECRET TOP SECRET") never occur in prose.
# Separators include bracket/checkbox junk ("Ex] SECRET [] CONFIDENTIAL (-]").
CHECKBOX_RUN_RE = re.compile(
    r"(?:(?:TOP[\s-]{0,3}SECRET|SECRET|CONFIDENTIAL|UNCLASSIFIED)[\s|,.•·—\-\[\]()={}~:_]{0,6}){2,}"
)
# Checkbox-line fallback: a single line holding 3+ classification words (with other
# box labels like INTERNAL interleaved) is a routing-slip row, never prose. Strip
# just the classification words from that line.
CHECKBOX_LINE_WORD_RE = re.compile(r"TOP[\s-]{0,3}SECRET|SECRET|CONFIDENTIAL|UNCLASSIFIED")
# Formulaic admin phrases that state the classification.
ADMIN_PHRASE_RES = (




    re.compile(r"\bTALENT[-\s~]*(?:KEYHOLE)?\s*(?:CONTROL|CHANNELS?|CLEAR\w*|CERTIFIED|"
               r"SYSTEM|REGULATIONS?|OFFICERS?|MATERIALS?|COVER)\w*", re.I),




    re.compile(r"\bHANDLE\s+(?:IN|VIA)\s+TALENT[-\s~]*(?:KEYHOLE)?(?:\s*ONLY)?", re.I),
    re.compile(r"\bKEYHOLE\s+(?:CHANNELS?|CONTROL|CLEAR\w*|MATERIALS?|SIGINT|and\s+TALENT)\w*", re.I),
    re.compile(r"\bTALENT[-\s~]*KEYHOLE\b", re.I),


    re.compile(r"(?:in\s+)?accordance\s+with\s+KEYHOLE"),
    re.compile(r"\bTCS[-\s]?\d{3,}[-\s/]?\d*", re.I),
    re.compile(r"WARNING\s+NOTICE[^\n]{0,70}", re.I),
    re.compile(r"(?:INTELLIGENCE\s+)?SOURCES\s+(?:AND|OR)\s+METHODS\s+INVOLVED", re.I),
    re.compile(r"\bWNINTEL\b|\bNOCONTRACT\b|\bORCON\b|\bNOFORN\b|\bLIMDIS\b|\bEXDIS\b|\bNODIS\b", re.I),


    re.compile(r"NOT\s+RELEASABLE\s+TO\s+(?:[^\n]{0,15}?)?(?:FOREIGN|CONTRACT)\w*[^\n]{0,15}", re.I),




    re.compile(r"\b(?:TOP\s+SECRET|SECRET|CONFIDENTIAL)\s*//\s*[A-Z/]{0,12}"),

    re.compile(r"\b(?:TOP\s+SECRET|SECRET|CONFIDENTIAL)\s+SECTION\s+\d+\s+OF\s+\d+", re.I),

    re.compile(r"this\s+document\s+is\s+(?:TOP\s+SECRET|SECRET|CONFIDENTIAL)\S*", re.I),
    re.compile(r"(?:[Aa][Ll]{2}\s+)?[Pp]aragraphs?\s+(?:are|is)\s+classified\s+"
               r"(?:TOP[\s-]{0,3}SECRET|SECRET|CONFIDENTIAL)"),
    re.compile(r"[Ww]hen\s+[Ss]eparated\s+[Ff]rom\s+" + _CLASS_CAP + r"\s+[Aa]ttachment"),
    # e.g. cover/routing forms: "Top Secret — (Security classification)"
    re.compile(_CLASS_CAP + r"\s*[—–\-~:]{0,3}\s*[a-z .]{0,8}\(\s*[Ss]ecurity\b[^)]{0,25}\)?:?"),
    # Espionage-act warning boilerplate: reveals the document is classified

    re.compile(r"\b\w?his\s+document\s+contains\s+classified\s+information"),

    re.compile(r"END\s+[OU]F\s+MESSAGE[\s.:-]{0,4}"
               r"(?:TOP[\s-]{0,3}SECRET|SECRET|CONFIDENTIAL|UNCLASSIFIED)"),

    re.compile(r"(?:TOP[\s-]{0,3}SECRET|SECRET|CONFIDENTIAL)\s*[/,;-]?\s*"
               r"(?:NOFORN|ORCON|NOCONTRACT|WNINTEL|EYES\s+ONLY)\b"),


    re.compile(r"FOR\s+THE\W{0,3}\s*PRESIDENT\s+ONLY", re.I),

    re.compile(r"(?:kept\s+in\s+)?communications?\s+intelligence\s+channels", re.I),
    re.compile(r"especially\s+indoctrinated(?:\s+and\s+authorized)?", re.I),
    re.compile(r"warning\s+notice[\s:]+sensitive\s+intelligence\s+sources", re.I),
    re.compile(r"sensitive\s+intelligence\s+sources\s+and\s+methods", re.I),
    re.compile(r"handle\s+via\s+[\w\s-]{0,20}channels?\s+only", re.I),






    re.compile(r"within[\s:]+.{5,70}?espionage\s+(?:act|laws?)", re.I | re.S),
    re.compile(r"transmission\s+or\s+revelation\s+of\s+(?:its\s+contents|which)", re.I),
    re.compile(r"an\s+unauthorized\s+person\s+[iI]s\s+prohibited\s+b?y?\s+law", re.I),




    re.compile(r"espionage\w*[\s\S]{0,60}?(?:18\s*u\.?\s*s\.?\s*c|[bs]ections?\s*7\d)", re.I),
    re.compile(r"(?:meaning|weaning)\s+[o0-9]f?\s+(?:the\s+)?espionage\w*", re.I),



    re.compile(r"prosecut\w*\s+under\s+(?:the\s+)?espionage\s+(?:act|laws?)", re.I),

    re.compile(r"\(\s*(?:TOP\s+)?SECRET\s*[/-]\s*CONTROL(?:LED)?(?:\s+DISSEM\w*)?", re.I),

    re.compile(r"\bHANDLE\s+(?:IN|VIA|BY|VA|WIA|W\w{1,2})\s*TALENT[-\s~]*(?:KEYHOLE)?(?:\s*ONLY)?", re.I),




    re.compile(r"\b[Hh][Aa][Nn][Dd][Ll][Ee][\s\S]{0,15}?TALENT\b[^\n]{0,30}"),

    # "Secret Control Officers who receive and/or release the attached",


    re.compile(r"(?:TOP[\s-]{0,3})?SE[CG]RET\s+CONTROL\s+(?:OFFICERS?|PERSONNEL)", re.I),








    #    INFORMATION AFFECTING THE NATIONAL DEFENSE (OF THE UNITED STATES)"
    re.compile(r"contains?\s+information\s+affect\w*[\s\S]{0,15}?(?:the\s+)?"
               r"na[\s-]*tional\s+defense(?:[\s\S]{0,30}?united\s+states)?", re.I),
    re.compile(r"affect\w*[\s-]+the[\s-]+na[\s-]*tional[\s-]+defense"
               r"[\s\S]{0,30}?united\s+states", re.I),

    re.compile(r"(?:title\s*1?8|18\s*u\.?\s*s\.?\s*c)[\s\S]{0,30}?[bs]ections?\s*7\d{2}", re.I),

    re.compile(r"unauthorized\s+person\s+[il1]s\s+prohibited\s+b?y?\s+law", re.I),

    #    Control personnel and those individuals whose official duties relate to the matter"
    re.compile(r"limited\s+to\s+.{0,12}?control\s+personnel", re.I),
    re.compile(r"individuals\s+whose\s+official\s+duties\s+relate", re.I),


    re.compile(r"(?:next\s+\d*\s*)?page\(?s?\)?\s+in\s+document\s+(?:exempt|denied)", re.I),

    re.compile(r"downgrade\w*\s+to\s+(?:TOP[\s-]{0,3}SECRET|SECRET|CONFIDENTIAL|UNCLASSIFIED)", re.I),



    re.compile(r"\bBACKGROUND\s+USE\s+ONLY\b|\bNO[\s_]+FOREIG\w{0,2}[\s'’]*DISSEM\w*|"
               r"\bNO[\s_]+DISSEM(?!INAT)\w*(?:\s+ABROAD)?|\bXGDS[-\s]?\d?\b|"
               r"contains?\s+codew\s?ords?", re.I),




    re.compile(r"\bEYES\s+ONLY\b"),
    re.compile(r"\bNO\s+FOREIGN\s+DISSEM\w*\b"),


    re.compile(r"\(?\s*U?\s*\)?\s*REPORT\s+CLASS\w{0,4}[^\n]{0,45}"),


    re.compile(r"[Ii]t\s+is\s+to\s+be\s+seen\s+only\s+by\s+\S{0,4}\s*personnel[^\n]{0,90}"),
    re.compile(r"COMMUNICATIONS\s+INTELLIGENCE\s+REGULATIONS?"),
    re.compile(r"COMMUNICATIONS\s+INTELLIGENCE\s+CHANNELS?"),


    re.compile(r"(?:TOP[\s-]{0,3}SECRET|SECRET|CONFIDENTIAL)\s*/\s*"
               r"(?:NODIS|XGDS|EXDIS|LIMDIS|NOFORN|ORCON|NOCONTRACT)[\w/]*"),







    re.compile(r"(?:TOP[\s-]{0,3}SE[CG]RET|SE[CG]RET)\s*[/.\-]{0,3}\s*"
               r"[CG]ON[ITL]{1,2}RO[LI][A-Z]{0,2}\b"),






    re.compile(r"SE[CG]RET\s*[-/]{1,3}\s*U[.,][\s.,]{0,3}S\.?"),

    re.compile(r"\bTOPSEC\b"),

    re.compile(r"\b(?:TOP\s+)?SECRET\s+ATTACHMENT\b"),



    re.compile(r"\b\w{0,7}ved\s+[fF]or\s+Release[^\n]{0,30}"),

    # "APPROVED FOR ⏎ 23 5038/ ⏎ RELEASEU".
    re.compile(r"Approved\s+F\w*[Rr]ele\w*[^\n]{0,25}"),
    re.compile(r"APPROVED\s+FOR[\s\S]{0,18}RELEASE\w*"),



    re.compile(r"\b3\.\s*[0-9]\s*(?:\(\s*[a-z0-9]{1,2}\s*\)){1,2}"),


    re.compile(r"E\.?\s*O\.?\s*1[0-9]{4}\s*[:;,]?\s*(?:GDS|XGDS|RDS)?\b"),

    re.compile(r"\(?\s*(?:TOP\s+SECRET|SECRET|CONFIDENTIAL|UNCLASSIFIED)\s*[-=:]\s*"
               r"ENTIRE\s+TEXT\s*\)?", re.I),

    re.compile(r"[Aa]ll\s+paragraphs?(?:\s+and\s+sub-?paragraphs?)?\s+"
               r"(?:are\s+|is\s+)?classified(?:\s+(?:TOP\s+SECRET|SECRET|CONFIDENTIAL))?"),

    re.compile(r"(?:TOP\s+SECRET|SECRET|CONFIDENTIAL)\s+"
               r"(?:January|February|March|April|May|June|July|August|September|"
               r"October|November|December)\s+\d{1,2},?\s+\d{4}"),

    re.compile(r"(?:AMEMBASSY|SECSTATE|USMISSION|AMCONSUL)\s+[A-Z][A-Z ]{2,24}\s+"
               r"(?:TOP\s+SECRET|SECRET|CONFIDENTIAL)"),

    re.compile(r"(?:TOP\s+SECRET|SECRET|CONFIDENTIAL)\s+(?:[A-Z]{3,12}:?\s+){1,2}\d{3,5}\b"),



    re.compile(r"[CG]LASS[.,]{0,2}\s+\w{0,2}[CG]H[A-Za-z]{2,7}[.,]{0,2}\s*"
               r"(?:\([^)\n]{0,4}\)?)?\s*[:;]?\s*T[A-Z]{0,2}[^\n]{0,10}"),

    re.compile(r"\bUNCLAS\b\s*\w{0,4}(?:/\w{1,6}){0,3}"),

    re.compile(r"(?:TOP\s+SECRET|SECRET|CONFIDENTIAL)\s+NOFOR\w*[^\n]{0,25}"),



    re.compile(r"\bEyes\s+Only\b"),
    re.compile(r"\bNo\s+Foreign\s+Dissem\w*\b"),

    re.compile(r"\bNOFORN\b"),


    re.compile(r"(?:U\.?\s?S\.?,?\s+)?OFFICIALS\s+ONLY"),

    re.compile(r"[Ss]ecurity\s+of\s+this\s+document\s+must\s+be\s+main-?\s*\n?tained[^\n]{0,60}"),
    re.compile(r"No\s+action\s+is\s+to\s+be\s+take\w{0,2}\s*o?n?\s+any\s+"
               r"COMMUNICATIONS\s+INTELLIGENCE[^\n]{0,60}"),

    # "Secret Noforn", "Confidential Noforn", "Secret/NoForn", "This table is

    re.compile(r"(?:Top\s+Secret|Secret|Confidential|TOP\s+SECRET|SECRET|CONFIDENTIAL)"
               r"\s*/?\s*[Nn]o\s?[Ff]orn\b[^\n]{0,20}"),
    re.compile(r"\b[Nn]o[Ff]orn\b"),
)


# (_scrub_control_banner_spans / _scrub_fragmented_markings_spans /






LATE_ADMIN_PHRASE_RES = (


    re.compile(r"\bSEGRET\b|\bSEORET\b|\bSECRFT\b|\bSFCRET\b|\bSECBET\b|"
               r"\bTUP[\s-]{0,3}SE[CG]RET\b", re.I),





    re.compile(r"\(?(?:TOP[\s-]{0,3})?SECRET\s*[/~]\s*"
               r"(?:[A-Z(][A-Z0-9 .,()/~-]{0,60})"),
)





EXEMPTION_CODE_RE = re.compile(
    r"\b\d{1,2}(?:\s*X\s*|x)\d{1,2}(?:[-\s]*(?:HUM|GE|HD|WMD|[A-F]|\d))?\b")



EXEMPTION_CODE_LOOSE_RE = re.compile(
    r"\b(?:2\d|5\d|9\d)\s*[xX]\s*\n?\s*\d(?:\s*-\s*(?:HUM|GE|HD|WMD|E|\d))?\b")



EXEMPTION_CODE_OCR_RE = re.compile(
    r"\b[OD0-9]{0,3}[0-9]X\d[A-Za-z]?_?\b")


def _scrub_release_fragment_spans(text: str) -> list[tuple[int, int]]:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    tokens = list(re.finditer(r"\S+", text))
    spans = []
    prose_forms = {"APPROVED", "APPROVAL", "APPROVE", "RELEASE", "RELEASED", "RELEASES"}
    i = 0
    while i < len(tokens):
        raw = tokens[i].group()
        alpha = re.sub(r"[^A-Za-z]", "", raw).upper()

        if len(alpha) >= 10 and _edit_distance(alpha[-10:], "FORRELEASE", 2) <= 2 \
                and alpha not in prose_forms:
            spans.append((tokens[i].start(), tokens[i].end()))
            i += 1
            continue


        if (len(alpha) >= 5 and _edit_distance(alpha[:6], "APPROV", 1) <= 1
                and not alpha.startswith("APPROPRIAT")):
            for j in range(i + 1, min(i + 4, len(tokens))):
                a2 = re.sub(r"[^A-Za-z]", "", tokens[j].group()).upper()
                if len(a2) >= 4 and _edit_distance(a2[:5], "RELEA", 2) <= 2:


                    # ("appropriate remedial", "approval and release from").
                    both_prose = (raw[:1].islower()
                                  and tokens[j].group()[:1].islower())





                    both_real_words = _is_real_word(raw) and _is_real_word(tokens[j].group())
                    if not both_prose and not both_real_words:

                        end = tokens[j].end()
                        for k in range(j + 1, min(j + 3, len(tokens))):
                            if re.fullmatch(r"[\d/:.\-()]+", tokens[k].group()):
                                end = tokens[k].end()
                            else:
                                break
                        spans.append((tokens[i].start(), end))
                        i = j
                    break
        i += 1
    return spans


def _scrub_fragmented_markings_spans(text: str) -> list[tuple[int, int]]:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    tokens = list(re.finditer(r"\S+", text))
    alphas = [(re.sub(r"[^A-Za-z]", "", t.group()).upper(), i)
              for i, t in enumerate(tokens)]
    alphas = [(a, i) for a, i in alphas if a]
    spans = []
    used_until = -1
    for pos in range(len(alphas)):
        if alphas[pos][1] <= used_until:
            continue
        matched = 0
        for k in (3, 2, 1):
            if pos + k > len(alphas):
                continue
            cand = "".join(a for a, _ in alphas[pos:pos + k])

            if not (4 if k == 1 else 6) <= len(cand) <= 13:
                continue
            first_alpha = alphas[pos][0]
            first_raw = tokens[alphas[pos][1]].group()
            raw_alpha = re.sub(r"[^A-Za-z]", "", first_raw)


            top_like = (
                first_alpha not in {"TO", "TOO", "FOR", "NOT", "NON", "SO", "GO", "UP", "ON", "IN"}
                and (_edit_distance(first_alpha, "TOP", 1) <= 1
                     or (len(first_alpha) == 3
                         and first_alpha[0] in set("TFILJY")
                         and first_alpha[1] in set("OQCGDU")
                         and first_alpha[2] in set("PFRBND")))
            )
            if _edit_distance(cand, "TOPSECRET", 2) <= 2 and (k == 1 or top_like):




                raw_join = "".join(
                    re.sub(r"[^A-Za-z]", "", tokens[a_i].group())
                    for _, a_i in alphas[pos:pos + k])
                clean_verbatim = (_edit_distance(cand, "TOPSECRET", 2) == 0
                                  and not raw_join.isupper())
                if not clean_verbatim:
                    matched = k
            elif (k >= 2 and first_alpha.startswith("CON") and first_raw[:1].isupper()
                  and _edit_distance(cand, "CONFIDENTIAL", 2) <= 2):
                matched = k
            elif (k == 1 and raw_alpha.isupper() and len(first_alpha) >= 10
                  and _edit_distance(cand, "DECLASSIFIED", 2) <= 2):
                matched = k



            elif (k == 1 and raw_alpha.isupper() and 6 <= len(first_alpha) <= 8
                  and (first_alpha.endswith("NOFORN")
                       or _edit_distance(first_alpha, "NOFORN", 1) <= 1)):
                matched = k



            elif k == 1 and re.search(r"[-/]", first_raw):
                parts = [re.sub(r"[^A-Za-z]", "", p).upper()
                         for p in re.split(r"[-/]+", first_raw) if p]
                parts = [p for p in parts if p]
                if (len(parts) >= 2 and len(parts[0]) == 3
                        and parts[0][0] in set("TFILJY")
                        and parts[0][1] in set("OQCGDU")
                        and parts[0][2] in set("PFRBND")
                        and _edit_distance("".join(parts[1:]), "SECRET", 2) <= 2):
                    matched = k








            elif k >= 2 and top_like and first_raw[:1].isupper():
                rest = "".join(a for a, _ in alphas[pos + 1:pos + k])


                rest_common = (k == 2 and rest in {
                    "SET", "SEE", "SEAT", "SECT", "SORT", "SUIT", "SITE", "SIDE",
                    "SECURE", "SECRETARY", "SECTOR", "SECTION", "SOVIET", "SECURITY",
                    "SELECT", "STREET", "SERIES", "SEVERE", "SILENT", "SUBJECT"})
                tail_fragmented = (k == 2 or all(
                    len(a) <= 4 for a, _ in alphas[pos + 1:pos + k]))
                max_ed = 2 if len(rest) >= 6 else 3
                if (2 <= len(rest) <= 8 and not rest_common and tail_fragmented
                        and _edit_distance(rest, "SECRET", max_ed) <= max_ed):
                    matched = k




            elif (k == 2 and first_alpha in {"TO", "TOO", "ON", "IN", "SO"}
                  and 3 <= len(alphas[pos + 1][0]) <= 6):
                rest = alphas[pos + 1][0]
                if (rest != "SECRET"
                        and rest not in {"SECURE", "SECRETE", "CRETE", "EGRET", "CREST"}
                        and (_edit_distance(rest, "SECRET", 1) <= 1
                             or _edit_distance(rest, "CRET", 1) <= 1)):
                    matched = k


            elif (k == 1 and raw_alpha.isupper() and 4 <= len(first_alpha) <= 5
                  and first_alpha not in {"CREST", "CRETE", "EGRET", "CARET", "CREPT"}
                  and first_alpha != "SECRET"
                  and _edit_distance(first_alpha, "SECRET", 2) <= 2):
                matched = k



            elif (raw_alpha.isupper() and cand.startswith("UN")
                  and len(first_alpha) >= (9 if k == 1 else 6)
                  and 9 <= len(cand) <= 16
                  and cand not in {"UNCLASSIFIED", "UNDERSTATED", "UNSATISFIED",
                                   "UNJUSTIFIED", "UNSPECIFIED", "UNQUALIFIED",
                                   "UNIDENTIFIED", "UNMODIFIED", "UNVERIFIED",
                                   "UNDERSTAND", "UNDERSTOOD", "UNIVERSITIES"}
                  and _edit_distance(cand, "UNCLASSIFIED", 5) <= (4 if k == 1 else 5)):
                matched = k


            elif (k == 1 and raw_alpha.isupper() and 10 <= len(first_alpha) <= 13
                  and first_alpha.startswith("CON")
                  and first_alpha not in {"CONFIDENTIAL", "CONFIDENCE", "CONFIDENTLY",
                                          "CONTINENTAL", "CONVENTIONAL", "CONSTITUTION",
                                          "CONSIDERABLE", "CONSEQUENCE", "CONFEDERATE",
                                          "CONDITIONAL", "CONFISCATED", "CONSTITUENT"}
                  and _edit_distance(first_alpha, "CONFIDENTIAL", 4) <= 4):
                matched = k
            if matched:
                break
        if matched:
            spans.append((tokens[alphas[pos][1]].start(),
                          tokens[alphas[pos + matched - 1][1]].end()))
            used_until = alphas[pos + matched - 1][1]
    return spans


def _scrub_control_banner_spans(text: str) -> list[tuple[int, int]]:
    """Find OCR-merged control-banner spans like 'SECRET/CONTROL - U.S. OFFICIALS ONLY'
    (incl. heavy OCR damage: 'GECRBT/CONTRUL = U.S. ...', 'SECRET/GOSTAGL - U.S. ...').
    Requires an ALL-CAPS fuzzy classification token plus >=2 banner anchors nearby,
    so prose like 'Secret U.S. officials only knew' is never matched."""
    tokens = list(re.finditer(r"\S+", text))
    spans = []
    i = 0
    while i < len(tokens):
        raw = tokens[i].group()
        parts = [p for p in re.split(r"[/@\-]", raw) if p]
        alpha0 = re.sub(r"[^A-Za-z]", "", parts[0]) if parts else ""
        is_class = (
            len(alpha0) >= 5
            and alpha0.isupper()
            and min(_edit_distance(alpha0, t) for t in ("SECRET", "CONFIDENTIAL")) <= 2
        )
        if not is_class:
            i += 1
            continue
        anchors = 0
        last_end = tokens[i].end()
        # signature 1: CLASSWORD/XXXX compound or a trailing slash on the class token

        if "/" in raw or "@" in raw:
            anchors += 1
        window_hi = min(i + 6, len(tokens) - 1)
        window_text = raw + " " + text[tokens[i].end():tokens[window_hi].end()]
        # signature 2: a U.S.-like fragment ("U.S.", "U., S.", "~U. S")
        us_match = re.search(r"\bU\s*[.,]?\s*,?\s*S\b", window_text)
        if us_match:
            anchors += 1
        last_j = i
        for j in range(i + 1, window_hi + 1):
            a = re.sub(r"[^A-Za-z]", "", tokens[j].group()).upper()
            if not a:
                continue
            if (a == "US" or _edit_distance(a, "CONTROL") <= 2
                    or _edit_distance(a, "OFFICIALS") <= 3
                    or (len(a) >= 3 and _edit_distance(a, "ONLY") <= 1)):
                anchors += 1
                last_end = tokens[j].end()
                last_j = j
        if anchors >= 2:
            spans.append((tokens[i].start(), max(last_end, tokens[i].end())))
            i = last_j + 1
        else:
            i += 1
    return spans


def scrub_inline_markings(text: str) -> tuple[str, list[str]]:
    """Remove fixed marking phrases merged inside a surviving line. Returns the
    scrubbed text and the list of removed substrings (for auditing)."""
    removed = []

    def _cut(pattern: re.Pattern, s: str) -> str:
        def repl(m):
            removed.append(m.group(0))
            return " "
        return pattern.sub(repl, s)

    for pattern in ADMIN_PHRASE_RES:
        text = _cut(pattern, text)
    text = _cut(EXEMPTION_CODE_RE, text)
    text = _cut(EXEMPTION_CODE_LOOSE_RE, text)
    text = _cut(EXEMPTION_CODE_OCR_RE, text)
    text = _cut(PORTION_MARK_RE, text)
    text = _cut(PORTION_ABBR_RE, text)




    lines = text.split("\n")
    for index, line in enumerate(lines):
        if len(CHECKBOX_LINE_WORD_RE.findall(line)) >= 3:
            lines[index] = _cut(CHECKBOX_LINE_WORD_RE, line)
    text = "\n".join(lines)
    text = _cut(CHECKBOX_RUN_RE, text)
    for start, end in reversed(_scrub_control_banner_spans(text)):
        removed.append(text[start:end])
        text = text[:start] + " " + text[end:]
    for start, end in reversed(_scrub_fragmented_markings_spans(text)):
        removed.append(text[start:end])
        text = text[:start] + " " + text[end:]
    for start, end in reversed(_scrub_release_fragment_spans(text)):
        removed.append(text[start:end])
        text = text[:start] + " " + text[end:]


    for pattern in LATE_ADMIN_PHRASE_RES:
        text = _cut(pattern, text)
    return re.sub(r"[ \t]{2,}", " ", text), removed




@dataclass
class Line:
    page_index: int
    line_index: int
    page_width: int
    page_height: int
    text: str
    bbox: tuple[int, int, int, int]
    y_center_ratio: float
    zone: str
    avg_confidence: float


def main() -> None:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    records = []
    stats = {
        "input_docs": 0,
        "labeled_docs": 0,
        "ambiguous_docs": 0,
        "unlabeled_docs": 0,
        "masked_token_leak_docs": 0,
    }

    with args.eligible.open("r", encoding="utf-8") as manifest:
        for line in manifest:
            if not line.strip():
                continue
            stats["input_docs"] += 1
            source = json.loads(line)
            record = build_record(
                source,
                remove_leak_lines=args.remove_leak_lines,
                ocr_robust=args.ocr_robust_mask,
            )
            if record["status"] == "labeled":
                stats["labeled_docs"] += 1
            elif record["status"] == "ambiguous":
                stats["ambiguous_docs"] += 1
            else:
                stats["unlabeled_docs"] += 1
            if record["masked_leakage_counts"]:
                stats["masked_token_leak_docs"] += 1
            if args.include_all or record["status"] == "labeled":
                records.append(record)

    with args.out.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Wrote: {args.out}")


def parse_args() -> argparse.Namespace:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    parser = argparse.ArgumentParser(
        description="Build marker-visible/masked CIA OCR dataset JSONL."
    )
    parser.add_argument("--eligible", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="Include ambiguous/unlabeled records for auditing.",
    )
    parser.add_argument(
        "--remove-leak-lines",
        action="store_true",
        help="Remove any OCR line containing label/release tokens from marker_masked_text.",
    )
    parser.add_argument(
        "--ocr-robust-mask",
        action="store_true",
        help="Precision-first OCR-robust banner/stamp removal: only short top/bottom "
        "banner-zone lines matching a fuzzy classification/release form (or a "
        "re-classification stamp) are removed, so body text is preserved.",
    )
    return parser.parse_args()


def build_record(source: dict[str, Any], remove_leak_lines: bool = False, ocr_robust: bool = False) -> dict[str, Any]:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    identifier = source["identifier"]
    xml_path = Path(source["xml_path"])
    lines = parse_lines(xml_path)
    high_conf_label = source.get("label", "")
    high_conf_evidence = source.get("evidence", [])
    if high_conf_label:
        official_candidates = evidence_lines(lines, high_conf_evidence)
        label_votes = {high_conf_label: len(official_candidates) or 1}
        label = high_conf_label
        status = "labeled"
    else:
        official_candidates = [
            line for line in lines if is_official_label_line(line)
        ]
        label_votes = collect_label_votes(official_candidates)
        label = choose_label(label_votes)
        status = status_for_votes(label_votes, label)

    visible_pages = reconstruct_text(lines, removed=set())
    removable_indices = removable_line_indices(
        lines,
        official_candidates,
        remove_leak_lines=remove_leak_lines,
        ocr_robust=ocr_robust,
    )
    inline_scrub_log: list[str] = []
    masked_pages = reconstruct_text(
        lines,
        removed=removable_indices,
        scrub_log=inline_scrub_log if ocr_robust else None,
    )
    marker_visible_text = "\n\n".join(visible_pages)
    marker_masked_text = "\n\n".join(masked_pages)
    if ocr_robust:


        marker_masked_text, extra_bits = scrub_inline_markings(marker_masked_text)
        inline_scrub_log.extend(extra_bits)

    return {
        "identifier": identifier,
        "label": label,
        "status": status,
        "source": source,
        "source_label_reason": source.get("reason", ""),
        "label_votes": label_votes,
        "removed_line_count": len(removable_indices),
        "scrubbed_inline": inline_scrub_log,
        "removed_lines": [
            line_metadata(lines[index])
            for index in sorted(removable_indices)
        ],
        "candidate_lines": [
            line_metadata(line)
            for line in official_candidates
        ],
        "marker_visible_text": marker_visible_text,
        "marker_masked_text": marker_masked_text,
        "masked_leakage_counts": leakage_counts(marker_masked_text, fuzzy=True),
    }


def parse_lines(xml_path: Path) -> list[Line]:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return []

    parsed_lines = []
    for page_index, page in enumerate(root.findall(".//OBJECT")):
        page_width = parse_int(page.attrib.get("width"))
        page_height = parse_int(page.attrib.get("height"))
        if page_height <= 0:
            continue
        for line_index, line in enumerate(page.findall(".//LINE")):
            words = line.findall(".//WORD")
            text = " ".join((word.text or "").strip() for word in words).strip()
            if not text:
                continue
            bbox = line_bbox(words)
            if bbox is None:
                continue
            y_center_ratio = ((bbox[1] + bbox[3]) / 2) / page_height
            parsed_lines.append(
                Line(
                    page_index=page_index,
                    line_index=line_index,
                    page_width=page_width,
                    page_height=page_height,
                    text=text,
                    bbox=bbox,
                    y_center_ratio=y_center_ratio,
                    zone=zone_for_ratio(y_center_ratio),
                    avg_confidence=avg_confidence(words),
                )
            )
    return parsed_lines


def is_official_label_line(line: Line) -> bool:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    labels = matched_labels(line.text)
    if not labels:
        return False
    if RELEASE_BOILERPLATE_PATTERN.search(line.text):
        return False
    if line.zone in {"top", "bottom"}:
        return True
    folded = " ".join(line.text.upper().split())
    if any(folded.startswith(label) for label in LABEL_ORDER):
        return True
    if CONTROL_MARKING_PATTERN.search(line.text) and len(folded) <= 80:
        return True
    return False


def removable_line_indices(
    lines: list[Line],
    official_candidates: list[Line],
    remove_leak_lines: bool = False,
    ocr_robust: bool = False,
) -> set[int]:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    candidate_ids = {id(line) for line in official_candidates}
    removable = set()
    for index, line in enumerate(lines):




        if _is_protected_body_line(line.text):
            continue
        if id(line) in candidate_ids:
            removable.add(index)
        elif RELEASE_BOILERPLATE_PATTERN.search(line.text):
            removable.add(index)
        elif (line.zone in {"top", "bottom"}
              and len(line.text.split()) <= BANNER_MAX_WORDS
              and any(m.group(0).isupper()
                      for m in CONTROL_MARKING_PATTERN.finditer(line.text))):


            removable.add(index)
        elif ocr_robust and is_fuzzy_banner_or_stamp(line):
            # Precision-first OCR-robust removal: zone(top/bottom) + short line +
            # fuzzy marker only, so body sentences are preserved (no content holes).
            removable.add(index)
        elif remove_leak_lines and leakage_line_pattern(line.text):
            removable.add(index)
    return removable


def leakage_line_pattern(text: str) -> bool:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    if RELEASE_BOILERPLATE_PATTERN.search(text):
        return True
    if CONTROL_MARKING_PATTERN.search(text):
        return True
    return bool(matched_labels(text))


def evidence_lines(lines: list[Line], evidence: list[dict[str, Any]]) -> list[Line]:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    keys = {
        (
            parse_int(item.get("page", item.get("page_index"))),
            parse_int(item.get("line", item.get("line_index"))),
        )
        for item in evidence
        if isinstance(item, dict)
    }
    selected = [
        line for line in lines if (line.page_index, line.line_index) in keys
    ]
    if selected:
        return selected

    # Fallback for older manifests without line indices.
    evidence_texts = {
        " ".join(str(item.get("text", "")).split()).casefold()
        for item in evidence
        if isinstance(item, dict)
    }
    return [
        line
        for line in lines
        if " ".join(line.text.split()).casefold() in evidence_texts
    ]


def collect_label_votes(lines: list[Line]) -> dict[str, int]:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    votes = {label: 0 for label in LABEL_ORDER}
    for line in lines:
        for label in matched_labels(line.text):
            votes[label] += 1
    return {label: count for label, count in votes.items() if count}


def choose_label(votes: dict[str, int]) -> str:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    if not votes:
        return ""
    return max(votes, key=lambda label: LABEL_ORDER[label])


def status_for_votes(votes: dict[str, int], label: str) -> str:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    if not label:
        return "unlabeled"
    if len(votes) == 1:
        return "labeled"
    highest = LABEL_ORDER[label]
    lower_votes = sum(
        count for vote_label, count in votes.items() if LABEL_ORDER[vote_label] < highest
    )
    if lower_votes and votes[label] == 0:
        return "ambiguous"
    return "labeled"


def reconstruct_text(
    lines: list[Line],
    removed: set[int],
    scrub_log: list[str] | None = None,
) -> list[str]:
    """Render pages from surviving lines. When scrub_log is given (OCR-robust mode),
    fixed marking phrases merged inside surviving lines are scrubbed out and the
    removed substrings appended to scrub_log."""
    pages: dict[int, list[Line]] = {}
    for index, line in enumerate(lines):
        if index in removed:
            continue
        pages.setdefault(line.page_index, []).append(line)

    rendered_pages = []
    for page_index in sorted(pages):
        page_lines = sorted(pages[page_index], key=lambda line: (line.bbox[1], line.bbox[0]))
        page_text = "\n".join(line.text for line in page_lines)
        if scrub_log is not None:
            # Scrub on the page string (not per line): OCR often splits one marking
            # phrase across adjacent lines ("Top Secret —" / "(Security classification)"),
            # and \s in the patterns matches the intervening newline.
            page_text, removed_bits = scrub_inline_markings(page_text)
            scrub_log.extend(removed_bits)



            out_lines = []
            for ln in page_text.split("\n"):
                if _is_protected_body_line(ln):
                    ln, bits = _strip_banner_tokens_in_prose(ln)
                    scrub_log.extend(bits)
                out_lines.append(ln)
            page_text = "\n".join(out_lines)
        rendered_pages.append(page_text)
    return rendered_pages


def leakage_counts(text: str, fuzzy: bool = False) -> dict[str, int]:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    counts = {}
    for label, pattern in LABEL_PATTERNS.items():
        count = len(pattern.findall(text))
        if count:
            counts[label] = count
    release_count = len(RELEASE_BOILERPLATE_PATTERN.findall(text))
    if release_count:
        counts["RELEASE_BOILERPLATE"] = release_count
    if fuzzy:
        # Honest residual check: count short lines whose OCR-garbled text still
        # approximates a classification word or release stamp (what exact-regex misses).
        residual = 0
        for raw_line in text.split("\n"):
            if len(raw_line.split()) > BANNER_MAX_WORDS:
                continue
            if fuzzy_classification_line(raw_line) or fuzzy_release_line(raw_line):
                residual += 1
        if residual:
            counts["FUZZY_RESIDUAL"] = residual
    return counts


def matched_labels(text: str) -> list[str]:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    labels = [
        label for label, pattern in LABEL_PATTERNS.items() if pattern.search(text)
    ]
    if "TOP SECRET" in labels and "SECRET" in labels:
        labels.remove("SECRET")
    return labels


def line_metadata(line: Line) -> dict[str, Any]:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    return {
        "page_index": line.page_index,
        "line_index": line.line_index,
        "text": line.text,
        "labels": matched_labels(line.text),
        "bbox": line.bbox,
        "zone": line.zone,
        "y_center_ratio": round(line.y_center_ratio, 6),
        "avg_confidence": round(line.avg_confidence, 2),
    }


def line_bbox(words: list[ET.Element]) -> tuple[int, int, int, int] | None:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    boxes = []
    for word in words:
        coords = parse_coords(word.attrib.get("coords", ""))
        if coords:
            boxes.append(coords)
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def parse_coords(raw: str) -> tuple[int, int, int, int] | None:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    parts = [parse_int(part) for part in raw.split(",")]
    if len(parts) != 4:
        return None
    return min(parts[0], parts[2]), min(parts[1], parts[3]), max(parts[0], parts[2]), max(parts[1], parts[3])


def avg_confidence(words: list[ET.Element]) -> float:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    values = [
        parse_int(word.attrib.get("x-confidence"))
        for word in words
        if word.attrib.get("x-confidence") is not None
    ]
    return sum(values) / len(values) if values else -1.0


def zone_for_ratio(y_ratio: float) -> str:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    if y_ratio <= 0.15:
        return "top"
    if y_ratio >= 0.85:
        return "bottom"
    return "body"


def parse_int(value: Any) -> int:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":
    main()
