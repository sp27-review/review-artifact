#!/usr/bin/env python3
"""(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""

from __future__ import annotations

import argparse
import csv
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from corelib.textmatch import edit_distance_one_or_less

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # main2_IA_CIA/ (config.py)
from config import LABELS



LABEL_LEVELS = {
    "TOP SECRET": 4,
    "SECRET": 3,
    "CONFIDENTIAL": 2,
    "UNCLASSIFIED": 1,
}

RELEASE_RE = re.compile(
    r"\b(APPROVED\s+FOR\s+RELEASE|DECLASSIFIED|SANITIZED\s+COPY|SANITIZED|CIA[-\s]?RDP|"
    r"RELEASED\s+BY|RELEASE\s+\d{4})\b",
    re.I,
)
CONTROL_RE = re.compile(
    r"\b(NOFORN|ORCON|CONTROL|U\.?S\.?\s+OFFICIALS\s+ONLY|OFFICIALS\s+ONLY|"
    r"LIMITED\s+OFFICIAL\s+USE|EYES\s+ONLY|DISSEM|REL\s+TO)\b",
    re.I,
)
UNCLASS_EXCLUDE_RE = re.compile(r"\b(PROHIBITED|UNLESS|WHEN|IF|MARKED|DOWNGRADE|DECLASSIFY)\b", re.I)


def parse_coords(value: str) -> tuple[int, int, int, int]:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    parts = [int(part) for part in re.split(r"[,\s]+", value.strip()) if part]
    xs = [parts[0], parts[2]]
    ys = [parts[1], parts[3]]
    return min(xs), min(ys), max(xs), max(ys)


def clean_token(value: str) -> str:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    return re.sub(r"[^A-Z0-9]", "", value.upper())




TOP_CONFUSABLES = (
    set("TFILJY17"),
    set("O0QCGDU"),
    set("PFRBND"),
)
TOP_EXCLUDE = {"TO", "TOO", "FOR", "NOT", "NON", "THE", "SO", "GO", "UP"}
SECRET_EXCLUDE = {"SECRETS"}


def looks_like_top(token: str) -> bool:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    if token in TOP_EXCLUDE:
        return False
    if edit_distance_one_or_less(token, "TOP"):
        return True
    if len(token) != 3:
        return False
    return all(ch in allowed for ch, allowed in zip(token, TOP_CONFUSABLES))


def normalize_label(text: str) -> str:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    raw_tokens = re.sub(r"[-_/]+", " ", text).split()
    pairs = []
    for raw in raw_tokens:
        clean = clean_token(raw)
        if not clean:
            continue
        first_alpha = next((ch for ch in raw if ch.isalpha()), "")
        pairs.append((clean, first_alpha.isupper()))

    for idx, (token, is_cap) in enumerate(pairs):
        if not is_cap or token in SECRET_EXCLUDE:
            continue
        if token == "SECRET" or edit_distance_one_or_less(token, "SECRET"):
            if idx > 0:
                prev_token, prev_cap = pairs[idx - 1]
                if prev_cap and looks_like_top(prev_token):
                    return "TOP SECRET"
            return "SECRET"
    for token, is_cap in pairs:
        if is_cap and (token == "CONFIDENTIAL" or edit_distance_one_or_less(token, "CONFIDENTIAL")):
            return "CONFIDENTIAL"
    for token, is_cap in pairs:
        if is_cap and (token == "UNCLASSIFIED" or edit_distance_one_or_less(token, "UNCLASSIFIED")):
            return "UNCLASSIFIED"
    return ""


def iter_lines(xml_path: Path):
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    root = ET.parse(xml_path).getroot()
    for page_index, obj in enumerate(root.iter("OBJECT")):
        width = int(obj.attrib.get("width", "0") or 0)
        height = int(obj.attrib.get("height", "0") or 0)
        for line_index, line in enumerate(obj.iter("LINE")):
            words = []
            for word in line.iter("WORD"):
                text = "".join(word.itertext()).strip()
                coords = word.attrib.get("coords")
                if not text or not coords:
                    continue
                x1, y1, x2, y2 = parse_coords(coords)
                conf = word.attrib.get("x-confidence")
                try:
                    confidence = float(conf) if conf is not None else None
                except ValueError:
                    confidence = None
                words.append((text, x1, y1, x2, y2, confidence))
            if not words:
                continue

            text = " ".join(word[0] for word in words)
            x1 = min(word[1] for word in words)
            y1 = min(word[2] for word in words)
            x2 = max(word[3] for word in words)
            y2 = max(word[4] for word in words)
            confidences = [word[5] for word in words if word[5] is not None]
            avg_confidence = sum(confidences) / len(confidences) if confidences else None
            y_mid_ratio = ((y1 + y2) / 2 / height) if height else None
            yield {
                "page": page_index,
                "line": line_index,
                "page_width": width,
                "page_height": height,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "y_mid_ratio": y_mid_ratio,
                "avg_confidence": avg_confidence,
                "text": text,
                "word_count": len(words),
            }


def region(y_mid_ratio: float | None, top_ratio: float, bottom_ratio: float) -> str:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    if y_mid_ratio is None:
        return "unknown"
    if y_mid_ratio <= top_ratio:
        return "top"
    if y_mid_ratio >= bottom_ratio:
        return "bottom"
    return "body"


def is_clean_marking_line(line: dict, label: str) -> bool:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    text = line["text"]
    if RELEASE_RE.search(text):
        return False
    if label == "UNCLASSIFIED" and UNCLASS_EXCLUDE_RE.search(text):
        return False

    if CONTROL_RE.search(text):
        return True

    words = [clean_token(token) for token in text.split()]
    words = [word for word in words if word]
    if label == "TOP SECRET":
        return line["word_count"] <= 8
    if label in {"SECRET", "CONFIDENTIAL"}:
        return line["word_count"] <= 6
    if label == "UNCLASSIFIED":
        return line["word_count"] <= 4
    return False


def allowed_regions_for(label: str, relax: bool) -> set[str]:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    if relax and label == "CONFIDENTIAL":
        return {"top", "bottom", "body"}
    return {"top", "bottom"}


def decide_label(candidates: list[dict], relax: bool = False) -> tuple[str, str, list[dict]]:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    accepted_by_label: dict[str, list[dict]] = {label: [] for label in LABEL_LEVELS}
    for candidate in candidates:
        label = candidate["label"]
        if not label:
            continue
        if candidate["region"] not in allowed_regions_for(label, relax):
            continue
        if is_clean_marking_line(candidate, label):
            accepted_by_label[label].append(candidate)

    reasons = []
    for label, lines in accepted_by_label.items():
        if not lines:
            continue
        pages = {line["page"] for line in lines}
        regions = {line["region"] for line in lines}
        reasons.append(f"{label}:lines={len(lines)},pages={len(pages)},regions={'+'.join(sorted(regions))}")

    final_candidates = []
    for label, lines in accepted_by_label.items():
        if not lines:
            continue
        pages = {line["page"] for line in lines}
        regions = {line["region"] for line in lines}
        short_or_control = any(line["word_count"] <= 3 or CONTROL_RE.search(line["text"]) for line in lines)


        if label == "TOP SECRET":

            keep = len(lines) >= 1 if relax else (len(lines) >= 1 and short_or_control)
        elif label == "CONFIDENTIAL":

            if relax:
                only_body = regions == {"body"}
                keep = len(lines) >= 2 if only_body else len(lines) >= 1
            else:
                keep = len(lines) >= 1 and short_or_control
        elif label == "SECRET":

            keep = len(lines) >= 2 or len(pages) >= 2 or regions == {"top", "bottom"} or short_or_control
        elif label == "UNCLASSIFIED":
            keep = len(lines) >= 2 or regions == {"top", "bottom"}
        else:
            keep = False

        if keep:
            final_candidates.append(label)

    if not final_candidates:
        return "", "no_high_confidence_marking" + (f"; {'; '.join(reasons)}" if reasons else ""), []

    unique = sorted(set(final_candidates), key=lambda label: LABEL_LEVELS[label], reverse=True)
    chosen = unique[0]
    evidence = accepted_by_label[chosen]

    if len(unique) > 1:
        chosen_lines = len(accepted_by_label[chosen])
        if chosen_lines < 2 and chosen not in {"TOP SECRET", "CONFIDENTIAL"}:
            return "", f"ambiguous_official_markings:{','.join(unique)}; {'; '.join(reasons)}", []
        return chosen, f"chose_highest_with_conflict:{','.join(unique)}; {'; '.join(reasons)}", evidence

    return chosen, f"accepted:{'; '.join(reasons)}", evidence


def read_manifest(path: Path) -> list[dict]:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--top-ratio", type=float, default=LABELS.top_ratio)
    parser.add_argument("--bottom-ratio", type=float, default=LABELS.bottom_ratio)
    parser.add_argument(
        "--relax",
        action="store_true",
        help="x00b4 x00b5 x00b6 x00b7 x00b8 (CONFIDENTIAL x00b9, TOP SECRET keep x00b6, SECRET x00ba x00b0)",
    )
    args = parser.parse_args()

    rows = []
    skipped_no_xml = 0
    for doc in read_manifest(Path(args.manifest)):
        xml_path = Path(doc["xml_path"])
        if not xml_path.exists():


            skipped_no_xml += 1
            rows.append({
                "identifier": doc["identifier"],
                "base": doc.get("base", Path(doc["txt_path"]).stem.replace("_djvu", "")),
                "chars": doc.get("chars", doc.get("char_count", "")),
                "label": "",
                "decision": "excluded",
                "reason": "missing_xml_artifact",
                "evidence": [],
                "txt_path": doc["txt_path"],
                "xml_path": doc["xml_path"],
                "hocr_path": doc["hocr_path"],
            })
            continue
        candidates = []
        for line in iter_lines(xml_path):
            label = normalize_label(line["text"])
            line_region = region(line["y_mid_ratio"], args.top_ratio, args.bottom_ratio)
            if not label and not RELEASE_RE.search(line["text"]):
                continue
            candidates.append({
                "page": line["page"],
                "line": line["line"],
                "region": line_region,
                "label": label,
                "text": line["text"],
                "word_count": line["word_count"],
                "avg_confidence": line["avg_confidence"],
                "y_mid_ratio": line["y_mid_ratio"],
                "bbox": f'{line["x1"]},{line["y1"]},{line["x2"]},{line["y2"]}',
            })

        label, reason, evidence = decide_label(candidates, relax=args.relax)
        rows.append({
            "identifier": doc["identifier"],
            "base": doc.get("base", Path(doc["txt_path"]).stem.replace("_djvu", "")),
            "chars": doc.get("chars", doc.get("char_count", "")),
            "label": label,
            "decision": "accepted" if label else "excluded",
            "reason": reason,
            "evidence": evidence,
            "txt_path": doc["txt_path"],
            "xml_path": doc["xml_path"],
            "hocr_path": doc["hocr_path"],
        })

    out_jsonl = Path(args.out_jsonl)
    out_csv = Path(args.out_csv)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            if row["label"]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_fields = ["identifier", "base", "chars", "decision", "label", "reason", "txt_path", "xml_path", "hocr_path"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in csv_fields})

    accepted = sum(1 for row in rows if row["label"])
    print(f"mode={'relax' if args.relax else 'strict'} scanned={len(rows)} accepted={accepted} excluded={len(rows) - accepted} skipped_no_xml={skipped_no_xml}")
    print(f"jsonl={out_jsonl}")
    print(f"csv={out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
