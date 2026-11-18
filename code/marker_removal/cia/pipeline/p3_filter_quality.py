#!/usr/bin/env python3
"""(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""

from __future__ import annotations

import argparse
import csv
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # main2_IA_CIA/ (config.py)
from config import QUALITY_GATE, WORDS_PATH
from corelib.ocrtext import normalize_linebreaks


WEIRD_CHAR_RE = re.compile(r"[^A-Za-z0-9\s.,;:()\'\"?!/&%$#@*+=–—-]")
TOKEN_RE = re.compile(r"[A-Za-z]+")


def load_dictionary(path: str = WORDS_PATH, extra_paths: list[str] | None = None) -> set[str]:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    with open(path, encoding="utf-8", errors="ignore") as f:
        words = {w.strip().lower() for w in f if w.strip().isalpha()}
    for extra in extra_paths or []:
        with open(extra, encoding="utf-8") as f:
            for line in f:
                token = line.split("#")[0].strip().lower()
                if token.isalpha():
                    words.add(token)
    return words


def is_known_word(t: str, dictionary: set[str]) -> bool:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    if t in dictionary:
        return True
    for suffix, restores in (("s", ("",)), ("es", ("",)), ("ed", ("", "e")),
                             ("ing", ("", "e")), ("ly", ("",)), ("er", ("", "e")),
                             ("ers", ("", "e")), ("est", ("", "e"))):
        if t.endswith(suffix) and len(t) - len(suffix) >= 3:
            stem = t[: -len(suffix)]
            if any(stem + r in dictionary for r in restores):
                return True
            if len(stem) >= 4 and stem[-1] == stem[-2] and stem[:-1] in dictionary:
                return True  # stopped -> stop
    for i in range(3, len(t) - 2):
        left, right = t[:i], t[i:]
        left_ok = left in dictionary or (left.endswith("s") and left[:-1] in dictionary)
        right_ok = right in dictionary or (right.endswith("s") and right[:-1] in dictionary)
        if left_ok and right_ok:
            return True
    return False


def text_quality(txt: str, dictionary: set[str]) -> dict:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    tokens = [t for t in TOKEN_RE.findall(txt) if len(t) >= 3]
    lower = [t for t in tokens if t[0].islower()]
    counts = Counter(t.lower() for t in tokens)
    known = {t: is_known_word(t, dictionary) for t in {t.lower() for t in tokens}}

    oov = (sum(1 for t in lower if not known[t.lower()]) / len(lower)) if lower else None
    singleton_oov = (
        sum(1 for t in tokens if not known[t.lower()] and counts[t.lower()] == 1) / len(tokens)
        if tokens else None
    )
    sym_per_kchar = 1000.0 * len(WEIRD_CHAR_RE.findall(txt)) / max(len(txt), 1)
    upper_frac = (len(tokens) - len(lower)) / len(tokens) if tokens else 0.0
    return {
        "n_tokens": len(tokens),
        "n_lower": len(lower),
        "upper_frac": round(upper_frac, 4),
        "oov": round(oov, 4) if oov is not None else None,
        "singleton_oov": round(singleton_oov, 4) if singleton_oov is not None else None,
        "sym_per_kchar": round(sym_per_kchar, 3),
    }


def layout_quality(xml_path: Path,
                   narrow_width_ratio: float = QUALITY_GATE.narrow_width_ratio,
                   cluster_gap_ratio: float = QUALITY_GATE.cluster_gap_ratio) -> dict:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return {"multi_col_pages": -1, "layout_pages": 0, "narrow_line_frac": None}

    multi = layout_pages = narrow = total_lines = 0
    for obj in root.iter("OBJECT"):
        width = int(obj.attrib.get("width", "0") or 0)
        if not width:
            continue
        lines = []
        for line in obj.iter("LINE"):
            xs1, xs2, word_count = [], [], 0
            for word in line.iter("WORD"):
                coords = word.attrib.get("coords")
                if not coords:
                    continue
                parts = [int(p) for p in re.split(r"[,\s]+", coords.strip()) if p]
                xs1.append(min(parts[0], parts[2]))
                xs2.append(max(parts[0], parts[2]))
                word_count += 1
            if word_count >= 2:
                lines.append((min(xs1), max(xs2)))
        if len(lines) < 10:
            continue
        layout_pages += 1
        total_lines += len(lines)
        page_narrow = [(x1, x2) for x1, x2 in lines if (x2 - x1) < narrow_width_ratio * width]
        narrow += len(page_narrow)
        if len(page_narrow) >= 0.6 * len(lines):
            starts = sorted(x1 for x1, _ in page_narrow)
            clusters, current = [], [starts[0]]
            for s in starts[1:]:
                if s - current[-1] <= cluster_gap_ratio * width:
                    current.append(s)
                else:
                    clusters.append(current)
                    current = [s]
            clusters.append(current)
            big = [c for c in clusters if len(c) >= max(4, 0.15 * len(page_narrow))]
            if len(big) >= 2:
                multi += 1
    return {
        "multi_col_pages": multi,
        "layout_pages": layout_pages,
        "narrow_line_frac": round(narrow / total_lines, 4) if total_lines else None,
    }


def decide(q: dict, layout: dict, args: argparse.Namespace) -> tuple[bool, str]:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    if (layout["narrow_line_frac"] is not None
            and layout["narrow_line_frac"] >= args.narrow_frac_max
            and layout["multi_col_pages"] >= 1):
        return False, (f"multi_column_layout:narrow={layout['narrow_line_frac']},"
                       f"pages={layout['multi_col_pages']}/{layout['layout_pages']}")

    all_caps = q["upper_frac"] > args.allcaps_upper_frac or q["n_lower"] < args.allcaps_min_lower
    if all_caps:
        if q["singleton_oov"] is None:
            return False, "allcaps_no_tokens"
        if q["singleton_oov"] > args.allcaps_singleton_oov_max:
            return False, f"allcaps_singleton_oov:{q['singleton_oov']}"
        if q["sym_per_kchar"] > args.allcaps_sym_max:
            return False, f"allcaps_symbol_noise:{q['sym_per_kchar']}"
        return True, f"pass_allcaps:s_oov={q['singleton_oov']},sym={q['sym_per_kchar']}"

    if q["oov"] is None:
        return False, "no_lowercase_tokens"
    if q["oov"] > args.oov_max:
        return False, f"oov:{q['oov']}"
    return True, f"pass:oov={q['oov']}"


def main() -> int:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="eligible manifest(jsonl, txt_path/xml_path x0040)")
    parser.add_argument("--out-jsonl", required=True, help="x000f x0099 manifest(x009a x009b x009c)")
    parser.add_argument("--out-csv", required=True, help="x009d x0099 x004d x009e CSV")
    parser.add_argument("--oov-max", type=float, default=QUALITY_GATE.oov_max)
    parser.add_argument("--allcaps-upper-frac", type=float, default=QUALITY_GATE.allcaps_upper_frac)
    parser.add_argument("--allcaps-min-lower", type=int, default=QUALITY_GATE.allcaps_min_lower)
    parser.add_argument("--allcaps-singleton-oov-max", type=float, default=QUALITY_GATE.allcaps_singleton_oov_max)
    parser.add_argument("--allcaps-sym-max", type=float, default=QUALITY_GATE.allcaps_sym_max)
    parser.add_argument("--narrow-frac-max", type=float, default=QUALITY_GATE.narrow_frac_max)
    parser.add_argument("--extra-dict", action="append", default=[],
                        help="x009f x004c x005d(x00a0 x00a1 x00a0 x00a2, # x00a3 x00a4). x0049 x004a x004b.")
    parser.add_argument("--normalize-linebreaks", action=argparse.BooleanOptionalAction, default=True,
                        help="OOV x004d x0098 x00a5 x00a6 x00a7+x00a8 x00a9(M5). x004d x00aa "
                             "x00ab x00ac x00ad(x00ae x00af) raw OCR x00b0. "
                             "--no-normalize-linebreaks x00b1 raw x00b2(v3r4) x00b3.")
    args = parser.parse_args()

    dictionary = load_dictionary(extra_paths=args.extra_dict)

    rows = []
    with open(args.manifest, encoding="utf-8") as f:
        docs = [json.loads(line) for line in f if line.strip()]
    for doc in docs:
        txt = Path(doc["txt_path"]).read_text(encoding="utf-8", errors="ignore")
        if args.normalize_linebreaks:
            txt = normalize_linebreaks(txt)
        q = text_quality(txt, dictionary)
        layout = layout_quality(Path(doc["xml_path"]))
        passed, reason = decide(q, layout, args)
        rows.append((doc, passed, reason, q, layout))

    out_jsonl = Path(args.out_jsonl)
    out_csv = Path(args.out_csv)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_jsonl.open("w", encoding="utf-8") as f:
        for doc, passed, reason, q, layout in rows:
            if passed:
                enriched = {**doc, "quality": {**q, **layout, "reason": reason}}
                f.write(json.dumps(enriched, ensure_ascii=False) + "\n")

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["identifier", "passed", "reason", "oov", "singleton_oov",
                         "sym_per_kchar", "upper_frac", "n_lower",
                         "multi_col_pages", "layout_pages", "narrow_line_frac"])
        for doc, passed, reason, q, layout in rows:
            writer.writerow([doc["identifier"], passed, reason, q["oov"], q["singleton_oov"],
                             q["sym_per_kchar"], q["upper_frac"], q["n_lower"],
                             layout["multi_col_pages"], layout["layout_pages"],
                             layout["narrow_line_frac"]])

    n_pass = sum(1 for _, passed, *_ in rows if passed)
    by_reason = Counter(reason.split(":")[0] for _, passed, reason, *_ in rows if not passed)
    print(f"scanned={len(rows)} passed={n_pass} excluded={len(rows) - n_pass} (oov_max={args.oov_max})")
    for reason, count in by_reason.most_common():
        print(f"  excluded[{reason}]={count}")
    print(f"jsonl={out_jsonl}")
    print(f"csv={out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
