#!/usr/bin/env python3
"""(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


COVER_END_RE = re.compile(r"THIS\s+PAGE\s+IS\s+UNCLASSIFIED", re.I)
COVER_SEARCH_CHARS = 1200


LINE_SCRUB_RES = [
    re.compile(r"DISTRIBUTED\s+BY", re.I),                                    # (2)
    re.compile(r"National\s+Tech\w*\s+Information\s+Service", re.I),
    re.compile(r"U\.?\s*S\.?\s+DEPARTMENT\s+OF\s+COMMERCE", re.I),            # (2)
    re.compile(r"Port\s+Royal\s+Road", re.I),
    re.compile(r"Springfield[,\s]+Va", re.I),
    re.compile(r"approved\s+for\s+public\s+release", re.I),                   # (3)
    re.compile(r"distribution\s+(?:is\s+)?unlimited", re.I),                  # (3)



    re.compile(r"Accession\s+For", re.I),
    re.compile(r"\bNTIS\b", re.I),
    re.compile(r"public\s+release\s+and\s+sale", re.I),
    re.compile(r"approved\s+for\s+public", re.I),
    re.compile(r"reviewed\s+and\s+is\s+appro\w*", re.I),
    re.compile(r"Distribution\s+limited\s+to", re.I),
    re.compile(r"^\s*UNCLASSIFIED\s*$", re.I),
    re.compile(r"security\s+class\w*", re.I),
    re.compile(r"^\s*NATIONAL\s+TECHNICAL\s*$", re.I),
    re.compile(r"^\s*INFORMATION\s+SERVICE\s*$", re.I),
    re.compile(r"restrictive\s+markings", re.I),
    re.compile(r"report\s+documentation\s+pag", re.I),
    re.compile(r"SAME\s+AS\s+RPT", re.I),

]




INLINE_SCRUB_RES = [
    re.compile(r"\bunclassif\w*(?:\s+[l1i]ed\b)?[/\w]*", re.I),
]


AD_HEAD_RE = re.compile(r"^\s*A[DP][\s\-?]*\d[\d\s]{4,12}$|^\s*\d{6,7}\s*$", re.I)
AD_HEAD_WINDOW = 5


def scrub_document(text: str) -> tuple[str, list[str]]:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    removed: list[str] = []


    head = text[:COVER_SEARCH_CHARS]
    match = COVER_END_RE.search(head)
    if match:
        cut = text.find("\n", match.end())
        cut = cut + 1 if cut >= 0 else match.end()
        removed.append(f"[cover_block:{cut}chars] " + " / ".join(
            line.strip() for line in text[:cut].splitlines() if line.strip())[:300])
        text = text[cut:]

    lines = text.splitlines()
    kept: list[str] = []
    nonblank_seen = 0
    for line in lines:
        stripped = line.strip()
        if stripped:
            nonblank_seen += 1
        if stripped and nonblank_seen <= AD_HEAD_WINDOW and AD_HEAD_RE.match(stripped):
            removed.append(f"[ad_head] {stripped}")
            continue
        if stripped and any(p.search(stripped) for p in LINE_SCRUB_RES):
            removed.append(f"[line] {stripped}")
            continue
        for pattern in INLINE_SCRUB_RES:
            if pattern.search(line):
                removed.append(f"[inline] {line.strip()}")
                line = pattern.sub("", line)
        kept.append(line)
    return "\n".join(kept), removed


def main() -> None:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, action="append", required=True,
                        help="born-unclassified x00c7 id x00c8(json). x0049 x004a x004b.")
    parser.add_argument("--raw-root", type=Path, action="append", required=True,
                        help="raw_ocr x0047. --candidates x00c9 x00ca x00cb.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit-csv", type=Path, default=None,
                        help="x00cc x00cd x00bf x009e CSV (x0015: out x00ce *_audit.csv)")
    args = parser.parse_args()
    if len(args.candidates) != len(args.raw_root):
        raise SystemExit("--candidates --raw-root x00cf x00d0 x00d1.")

    audit_path = args.audit_csv or args.out.with_name(args.out.stem + "_audit.csv")
    n = 0
    total_removed = 0
    import csv
    with args.out.open("w", encoding="utf-8") as out, \
         audit_path.open("w", encoding="utf-8", newline="") as audit_file:
        audit = csv.writer(audit_file)
        audit.writerow(["identifier", "removed_line"])
        for cand_path, root in zip(args.candidates, args.raw_root):
            for ident in json.loads(cand_path.read_text()):
                txt_path = next((root / ident).glob("*_djvu.txt"), None)
                if txt_path is None:
                    print(f"skip (no txt): {ident}")
                    continue
                raw = txt_path.read_text(errors="replace")
                scrubbed, removed = scrub_document(raw)
                for item in removed:
                    audit.writerow([ident, item])
                total_removed += len(removed)
                record = {
                    "identifier": ident,
                    "label": "UNCLASSIFIED",
                    "status": "u_track_dtic",
                    "source": "dticarchive",
                    "source_label_reason": "born_unclassified(public_release, stamp_audit_clean)",
                    "removed_line_count": len(removed),
                    "removed_lines": removed,
                    "marker_visible_text": raw,
                    "marker_masked_text": scrubbed,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                n += 1
    print(f"records={n}, removed_lines_total={total_removed}")
    print(f"wrote {args.out}")
    print(f"audit {audit_path}")


if __name__ == "__main__":
    main()
