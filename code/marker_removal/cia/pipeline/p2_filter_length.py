#!/usr/bin/env python3
"""(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def main() -> None:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    eligible = []
    for txt_path in sorted(args.root.glob("*/*_djvu.txt")):
        text = txt_path.read_text(encoding="utf-8", errors="replace")
        identifier = txt_path.parent.name
        char_count = len(text)
        word_count = len(re.findall(r"\S+", text))
        is_eligible = args.min_chars <= char_count <= args.max_chars
        row = {
            "identifier": identifier,
            "txt_path": str(txt_path),
            "xml_path": str(find_sibling(txt_path, "_djvu.xml")),
            "hocr_path": str(find_sibling(txt_path, "_hocr.html")),
            "char_count": char_count,
            "word_count": word_count,
            "eligible": str(is_eligible).lower(),
        }
        rows.append(row)
        if is_eligible:
            eligible.append(row)

    with args.csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "identifier",
                "txt_path",
                "xml_path",
                "hocr_path",
                "char_count",
                "word_count",
                "eligible",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with args.out.open("w", encoding="utf-8") as file:
        for row in eligible:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Scanned documents: {len(rows)}")
    print(f"Eligible documents: {len(eligible)}")
    print(f"Wrote eligible manifest: {args.out}")
    print(f"Wrote length stats: {args.csv}")


def parse_args() -> argparse.Namespace:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    parser = argparse.ArgumentParser(
        description="Filter downloaded IA CIA OCR texts by character length."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--min-chars", type=int, default=2000)
    parser.add_argument("--max-chars", type=int, default=10000)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    return parser.parse_args()


def find_sibling(txt_path: Path, suffix: str) -> Path:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    identifier = txt_path.parent.name
    candidates = sorted(txt_path.parent.glob(f"*{suffix}"))
    if candidates:
        return candidates[0]
    return txt_path.parent / f"{identifier}{suffix}"


if __name__ == "__main__":
    main()
