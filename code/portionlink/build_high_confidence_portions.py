#!/usr/bin/env python3
"""Build a layout-aware, auditable portion-mark sidecar for the span analysis.

The frozen benchmark is read-only.  Portion marks are located in the original
DjVu XML, associated with the XML PARAGRAPH that contains them, and projected
onto the existing r4 ``marker_masked_text`` through monotonic equal-character
alignment.  No text is reconstructed by fuzzy search.

Outputs under ``--output-dir``:
  portion_manifest.jsonl          every detected candidate and its disposition
  high_confidence_portions.jsonl  analysis-ready operational portion labels
  review_sample.csv               deterministic stratified validation sample
  summary.json                    counts, checksums, and extraction parameters
"""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import random
import re
import statistics
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))
from corelib.paths import CIA_BENCHMARK_R4_DIR, CIA_RAW_OCR_DIR, PORTION_DERIVED_DIR, ROOT


SCHEMA_VERSION = "portion-v2-layout-1"
RANDOM_SEED = 7701
MIN_RECOVERABLE_CHARS = 40
MIN_RETAINED_RATIO = 0.50
MIN_MARK_CONFIDENCE = 40.0

# Exact operational forms are primary.  The suffix grammar covers dissemination
# and compartment strings while refusing ordinary parentheticals.
PORTION_RE = re.compile(
    r"\(\s*(?P<label>TS|S|C|U)"
    r"(?P<suffix>(?:(?:\s*//?\s*|\s*-\s*)[A-Z][A-Z0-9-]{0,11})*)\s*\)"
)
FULL_PORTION_RE = re.compile(
    r"\(\s*(?P<label>TOP\s+SECRET|SECRET|CONFIDENTIAL|UNCLASSIFIED)"
    r"(?P<suffix>(?:(?:\s*//?\s*|\s*-\s*)[A-Z][A-Z0-9-]{0,11})*)\s*\)",
    re.I,
)
ENUM_RE = re.compile(r"\(([A-D])\)", re.I)
LEADING_PREFIX_RE = re.compile(r"^\s*(?:(?:\d{1,3}|[A-Za-z])[.)]\s*)?$")
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
ADMIN_STAMP_RE = re.compile(
    r"\b(?:NO\s+CHANGE\s+IN\s+CLASS|CLASS\w*\.?\s+CHANGED|DECLASSIFIED|"
    r"NEXT\s+REVIEW\s+DATE|REVIEWER|AUTH(?:ORITY)?\s*:|DOCUMENT\s+NO\.?|"
    r"APPROVED\s+FOR\s+RELEASE)\b",
    re.I,
)

LABEL_NORMALIZE = {
    "TS": "TS", "TOP SECRET": "TS", "S": "S", "SECRET": "S",
    "C": "C", "CONFIDENTIAL": "C", "U": "U", "UNCLASSIFIED": "U",
}


@dataclass
class Word:
    text: str
    bbox: tuple[int, int, int, int]
    confidence: float
    line_start: int
    line_end: int


@dataclass
class Line:
    page: int
    paragraph: int
    region: int
    line_in_paragraph: int
    text: str
    bbox: tuple[int, int, int, int]
    confidence: float
    page_height: int
    words: list[Word] = field(default_factory=list)
    visible_start: int = -1
    visible_end: int = -1
    render_index: int = -1

    @property
    def zone(self) -> str:
        mid = ((self.bbox[1] + self.bbox[3]) / 2) / max(self.page_height, 1)
        if mid <= 0.15:
            return "top"
        if mid >= 0.85:
            return "bottom"
        return "body"


@dataclass
class Mark:
    match_start: int
    match_end: int
    original: str
    label: str
    form: str
    line: Line
    bbox: tuple[int, int, int, int]
    confidence: float
    placement: str
    direction: str
    exclusion: str = ""

    @property
    def visible_start(self) -> int:
        return self.line.visible_start + self.match_start

    @property
    def visible_end(self) -> int:
        return self.line.visible_start + self.match_end


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_int(value: Any) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return 0


def parse_bbox(raw: str) -> tuple[int, int, int, int]:
    values = [parse_int(value) for value in raw.split(",")]
    if len(values) != 4:
        return 0, 0, 0, 0
    return min(values[0], values[2]), min(values[1], values[3]), max(values[0], values[2]), max(values[1], values[3])


def union_bbox(boxes: Iterable[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    boxes = list(boxes)
    if not boxes:
        return 0, 0, 0, 0
    return min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)


def parse_xml(path: Path) -> list[Line]:
    root = ET.parse(path).getroot()
    lines: list[Line] = []
    for page, obj in enumerate(root.findall(".//OBJECT")):
        page_height = parse_int(obj.attrib.get("height"))
        paragraphs = obj.findall(".//PARAGRAPH")
        for paragraph, para in enumerate(paragraphs):
            parent_region = next(
                (index for index, region in enumerate(obj.findall(".//REGION")) if para in list(region)),
                -1,
            )
            for line_in_paragraph, elem in enumerate(para.findall(".//LINE")):
                words: list[Word] = []
                pieces: list[str] = []
                cursor = 0
                for word_elem in elem.findall(".//WORD"):
                    text = (word_elem.text or "").strip()
                    if not text:
                        continue
                    if pieces:
                        cursor += 1
                    start = cursor
                    cursor += len(text)
                    pieces.append(text)
                    words.append(Word(
                        text=text,
                        bbox=parse_bbox(word_elem.attrib.get("coords", "")),
                        confidence=float(parse_int(word_elem.attrib.get("x-confidence")))
                        if word_elem.attrib.get("x-confidence") is not None else -1.0,
                        line_start=start,
                        line_end=cursor,
                    ))
                if not pieces:
                    continue
                confidences = [word.confidence for word in words if word.confidence >= 0]
                lines.append(Line(
                    page=page,
                    paragraph=paragraph,
                    region=parent_region,
                    line_in_paragraph=line_in_paragraph,
                    text=" ".join(pieces),
                    bbox=union_bbox(word.bbox for word in words),
                    confidence=statistics.fmean(confidences) if confidences else -1.0,
                    page_height=page_height,
                    words=words,
                ))
    return lines


def render_with_offsets(lines: list[Line]) -> str:
    pages: dict[int, list[Line]] = defaultdict(list)
    for line in lines:
        pages[line.page].append(line)
    chunks: list[str] = []
    cursor = 0
    render_index = 0
    for page_number, page in enumerate(sorted(pages)):
        if page_number:
            chunks.append("\n\n")
            cursor += 2
        ordered = sorted(pages[page], key=lambda line: (line.bbox[1], line.bbox[0]))
        for line_number, line in enumerate(ordered):
            if line_number:
                chunks.append("\n")
                cursor += 1
            line.visible_start = cursor
            line.visible_end = cursor + len(line.text)
            line.render_index = render_index
            render_index += 1
            chunks.append(line.text)
            cursor = line.visible_end
    return "".join(chunks)


def mark_geometry(line: Line, start: int, end: int) -> tuple[tuple[int, int, int, int], float]:
    selected = [word for word in line.words if word.line_start < end and start < word.line_end]
    confidences = [word.confidence for word in selected if word.confidence >= 0]
    return union_bbox(word.bbox for word in selected), statistics.fmean(confidences) if confidences else -1.0


def placement_for(line: Line, start: int, end: int) -> tuple[str, str]:
    prefix, suffix = line.text[:start], line.text[end:]
    prefix_empty = not prefix.strip()
    suffix_empty = not suffix.strip()
    if prefix_empty and suffix_empty:
        return "solo_line", "ambiguous"
    if (prefix_empty or LEADING_PREFIX_RE.fullmatch(prefix)) and not suffix_empty:
        return "leading_line", "leading"
    if suffix_empty and TOKEN_RE.search(prefix):
        return "trailing_line", "trailing"
    return "inline", "ambiguous"


def detect_marks(lines: list[Line]) -> list[Mark]:
    marks: list[Mark] = []
    for line in lines:
        found: list[tuple[re.Match[str], str]] = []
        found.extend((match, "abbr") for match in PORTION_RE.finditer(line.text))
        found.extend((match, "full") for match in FULL_PORTION_RE.finditer(line.text))
        found.sort(key=lambda item: (item[0].start(), item[0].end()))
        occupied: list[tuple[int, int]] = []
        for match, form in found:
            if any(match.start() < end and start < match.end() for start, end in occupied):
                continue
            occupied.append((match.start(), match.end()))
            raw_label = re.sub(r"\s+", " ", match.group("label").upper())
            label = LABEL_NORMALIZE[raw_label]
            placement, direction = placement_for(line, match.start(), match.end())
            bbox, confidence = mark_geometry(line, match.start(), match.end())
            mark = Mark(
                match_start=match.start(), match_end=match.end(), original=match.group(0),
                label=label, form=form, line=line, bbox=bbox, confidence=confidence,
                placement=placement, direction=direction,
            )
            post = line.text[match.end():match.end() + 2]
            pre = line.text[max(0, match.start() - 5):match.start()]
            if post.startswith(";"):
                mark.exclusion = "citation_semicolon"
            elif re.search(r"\d\s*\.?\s*$", pre) and placement == "inline":
                mark.exclusion = "numeric_section_reference"
            marks.append(mark)
    mark_enumerator_collisions(lines, marks)
    return marks


def mark_enumerator_collisions(lines: list[Line], marks: list[Mark]) -> None:
    by_page: dict[int, list[Line]] = defaultdict(list)
    for line in lines:
        by_page[line.page].append(line)
    for page_lines in by_page.values():
        page_lines.sort(key=lambda line: line.render_index)
    for mark in marks:
        if mark.label != "C" or mark.exclusion:
            continue
        nearby = [
            line for line in by_page[mark.line.page]
            if abs(line.render_index - mark.line.render_index) <= 3
        ]
        letters: set[str] = set()
        aligned_letters: set[str] = set()
        mark_x = mark.bbox[0]
        for line in nearby:
            for match in ENUM_RE.finditer(line.text):
                letter = match.group(1).upper()
                letters.add(letter)
                bbox, _ = mark_geometry(line, match.start(), match.end())
                if abs(bbox[0] - mark_x) <= 100:
                    aligned_letters.add(letter)
        # Requiring two companion letters and spatial alignment avoids the old
        # document-global rule that removed every C whenever any A/B existed.
        if len(aligned_letters & {"A", "B", "D"}) >= 2 or {"A", "B"} <= aligned_letters:
            mark.exclusion = "local_enumerator_sequence"


def paragraph_groups(lines: list[Line]) -> dict[tuple[int, int], list[Line]]:
    groups: dict[tuple[int, int], list[Line]] = defaultdict(list)
    for line in lines:
        groups[(line.page, line.paragraph)].append(line)
    for group in groups.values():
        group.sort(key=lambda line: line.render_index)
    return groups


def paragraph_bounds(group: list[Line]) -> tuple[int, int, bool]:
    indices = [line.render_index for line in group]
    contiguous = indices == list(range(min(indices), max(indices) + 1))
    return min(line.visible_start for line in group), max(line.visible_end for line in group), contiguous


def portion_bounds(mark: Mark, peer_marks: list[Mark], para_start: int, para_end: int) -> tuple[int, int]:
    peers = sorted(peer_marks, key=lambda item: item.visible_start)
    index = peers.index(mark)
    if mark.direction == "leading":
        end = peers[index + 1].visible_start if index + 1 < len(peers) else para_end
        return mark.visible_end, end
    if mark.direction == "trailing":
        start = peers[index - 1].visible_end if index else para_start
        return start, mark.visible_start
    return mark.visible_end, mark.visible_end


def equal_projection(visible: str, masked: str) -> list[tuple[int, int, int, int]]:
    # ``autojunk=False`` becomes quadratic on a few very long OCR documents with
    # repeated headers.  The default popularity filter removes those repetitive
    # anchors and preserves the long content blocks needed for exact projection.
    matcher = difflib.SequenceMatcher(None, visible, masked, autojunk=True)
    return [
        (i1, i2, j1, j2)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag == "equal" and i2 - i1 >= 4
    ]


def project_interval(
    start: int, end: int, projection: list[tuple[int, int, int, int]], masked: str
) -> tuple[list[list[int]], str, int]:
    spans: list[list[int]] = []
    pieces: list[str] = []
    retained = 0
    for visible_start, visible_end, masked_start, _ in projection:
        overlap_start = max(start, visible_start)
        overlap_end = min(end, visible_end)
        if overlap_end <= overlap_start:
            continue
        out_start = masked_start + (overlap_start - visible_start)
        out_end = out_start + (overlap_end - overlap_start)
        if spans and spans[-1][1] == out_start:
            spans[-1][1] = out_end
        else:
            spans.append([out_start, out_end])
        pieces.append(masked[out_start:out_end])
        retained += out_end - out_start
    return spans, "".join(pieces), retained


def confidence_and_flags(
    mark: Mark,
    contiguous: bool,
    visible_text: str,
    recoverable_text: str,
    retained_ratio: float,
    unique_projection_anchor: bool,
    mixed_directions: bool,
) -> tuple[str, bool, list[str]]:
    flags: list[str] = []
    if mark.exclusion:
        flags.append(mark.exclusion)
    if mark.direction == "ambiguous":
        flags.append("ambiguous_direction")
    if not contiguous:
        flags.append("noncontiguous_xml_paragraph")
    if mark.confidence >= 0 and mark.confidence < MIN_MARK_CONFIDENCE:
        flags.append("low_mark_ocr_confidence")
    if len(visible_text.strip()) < MIN_RECOVERABLE_CHARS:
        flags.append("short_visible_portion")
    if len(recoverable_text.strip()) < MIN_RECOVERABLE_CHARS:
        flags.append("short_recoverable_portion")
    if retained_ratio < MIN_RETAINED_RATIO:
        flags.append("low_retained_ratio")
    if not unique_projection_anchor:
        flags.append("no_unique_projection_anchor")
    if mixed_directions:
        flags.append("mixed_paragraph_directions")
    if mark.line.zone != "body":
        flags.append(f"mark_in_{mark.line.zone}_zone")
    if mark.form == "full":
        inner = mark.original[1:-1].strip()
        if not re.fullmatch(r"(?:TOP\s+SECRET|SECRET|CONFIDENTIAL|UNCLASSIFIED)", inner):
            flags.append("noncanonical_full_form")
    if ADMIN_STAMP_RE.search(visible_text):
        flags.append("administrative_stamp_content")

    blocking = {
        "citation_semicolon", "numeric_section_reference", "local_enumerator_sequence",
        "ambiguous_direction", "noncontiguous_xml_paragraph", "low_mark_ocr_confidence",
        "short_visible_portion", "short_recoverable_portion", "low_retained_ratio",
        "no_unique_projection_anchor", "mixed_paragraph_directions",
        "mark_in_top_zone", "mark_in_bottom_zone", "noncanonical_full_form",
        "administrative_stamp_content",
    }
    analysis_eligible = not any(flag in blocking for flag in flags)
    if analysis_eligible:
        confidence = "high"
    elif not mark.exclusion and mark.direction != "ambiguous":
        confidence = "medium"
    else:
        confidence = "excluded"
    return confidence, analysis_eligible, flags


def resolve_xml(record: dict[str, Any]) -> Path:
    source = record.get("source") or {}
    base = str(source.get("base", ""))
    candidate = CIA_RAW_OCR_DIR / str(record["identifier"]) / f"{base}_djvu.xml"
    if candidate.exists():
        return candidate
    source_path = Path(str(source.get("xml_path", "")))
    if source_path.is_absolute() and source_path.exists():
        return source_path
    raise FileNotFoundError(f"XML not found for {record['identifier']}: {candidate}")


def jsonl_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def build_rows(record: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    xml_path = resolve_xml(record)
    lines = parse_xml(xml_path)
    rendered = render_with_offsets(lines)
    visible = str(record.get("marker_visible_text", "") or "")
    masked = str(record.get("marker_masked_text", "") or "")
    if rendered != visible:
        raise ValueError(f"XML render mismatch for {record['identifier']}")

    marks = detect_marks(lines)
    if not marks:
        return [], {"xml_render_match": 1}
    projection = equal_projection(visible, masked)
    groups = paragraph_groups(lines)
    marks_by_paragraph: dict[tuple[int, int], list[Mark]] = defaultdict(list)
    for mark in marks:
        marks_by_paragraph[(mark.line.page, mark.line.paragraph)].append(mark)

    rows: list[dict[str, Any]] = []
    for index, mark in enumerate(sorted(marks, key=lambda item: item.visible_start)):
        key = (mark.line.page, mark.line.paragraph)
        para_start, para_end, contiguous = paragraph_bounds(groups[key])
        peers = marks_by_paragraph[key]
        portion_start, portion_end = portion_bounds(mark, peers, para_start, para_end)
        visible_portion = visible[portion_start:portion_end].strip()
        masked_spans, recoverable, retained = project_interval(
            portion_start, portion_end, projection, masked
        )
        recoverable = recoverable.strip()
        projected_pieces = [masked[start:end].strip() for start, end in masked_spans]
        longest_anchor = max(projected_pieces, key=len, default="")
        unique_projection_anchor = (
            len(longest_anchor) >= 20 and masked.count(longest_anchor) == 1
        )
        denominator = max(1, portion_end - portion_start)
        retained_ratio = retained / denominator
        peer_directions = {peer.direction for peer in peers if peer.direction != "ambiguous"}
        mixed_directions = len(peer_directions) > 1
        confidence, eligible, flags = confidence_and_flags(
            mark, contiguous, visible_portion, recoverable, retained_ratio,
            unique_projection_anchor, mixed_directions,
        )
        portion_id = (
            f"{record['identifier']}#p{mark.line.page:04d}"
            f"-para{mark.line.paragraph:04d}-m{index:04d}"
        )
        rows.append({
            "schema_version": SCHEMA_VERSION,
            "portion_id": portion_id,
            "doc_id": str(record["identifier"]),
            "document_label": str(record.get("label", "")),
            "source_xml": str(xml_path.relative_to(ROOT)),
            "page_index": mark.line.page,
            "paragraph_index": mark.line.paragraph,
            "line_index": mark.line.render_index,
            "line_in_paragraph": mark.line.line_in_paragraph,
            "marking_original": mark.original,
            "portion_label": mark.label,
            "mark_form": mark.form,
            "mark_bbox": list(mark.bbox),
            "line_bbox": list(mark.line.bbox),
            "paragraph_bbox": list(union_bbox(line.bbox for line in groups[key])),
            "paragraph_line_count": len(groups[key]),
            "xml_paragraph_contiguous": contiguous,
            "mark_ocr_confidence": round(mark.confidence, 2),
            "mark_zone": mark.line.zone,
            "placement": mark.placement,
            "direction": mark.direction,
            "visible_mark_span": [mark.visible_start, mark.visible_end],
            "visible_portion_span": [portion_start, portion_end],
            "masked_portion_spans": masked_spans,
            "visible_portion_chars": len(visible_portion),
            "recoverable_chars": len(recoverable),
            "retained_ratio": round(retained_ratio, 6),
            "redacted_or_masked_ratio": round(1.0 - retained_ratio, 6),
            "unique_projection_anchor": unique_projection_anchor,
            "visible_document_chars": len(visible),
            "masked_document_chars": len(masked),
            "visible_relative_start": round(portion_start / max(len(visible), 1), 6),
            "visible_relative_end": round(portion_end / max(len(visible), 1), 6),
            "visible_portion_sha256": sha256_text(visible_portion),
            "recoverable_text_sha256": sha256_text(recoverable),
            "visible_portion_text": visible_portion,
            "recoverable_text": recoverable,
            "confidence": confidence,
            "analysis_eligible": eligible,
            "quality_flags": flags,
        })
    return rows, {"xml_render_match": 1}


def write_review_sample(rows: list[dict[str, Any]], path: Path, per_stratum: int) -> int:
    rng = random.Random(RANDOM_SEED)
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[(row["portion_label"], row["confidence"])].append(row)
    selected: list[dict[str, Any]] = []
    for key in sorted(strata):
        candidates = sorted(strata[key], key=lambda row: row["portion_id"])
        rng.shuffle(candidates)
        selected.extend(candidates[:per_stratum])
    selected.sort(key=lambda row: (row["portion_label"], row["confidence"], row["portion_id"]))
    judgment_fields = [
        "is_true_portion_mark", "boundary_correct", "label_correct",
        "visible_sensitive_evidence", "reviewer", "review_notes",
    ]
    existing: dict[str, dict[str, str]] = {}
    if path.exists():
        with path.open(encoding="utf-8", newline="") as handle:
            existing = {str(row["portion_id"]): row for row in csv.DictReader(handle)}
        selected_ids = {str(row["portion_id"]) for row in selected}
        reviewed_dropped = [
            portion_id for portion_id, row in existing.items()
            if portion_id not in selected_ids and any(str(row.get(field, "")).strip() for field in judgment_fields)
        ]
        if reviewed_dropped:
            raise ValueError(
                f"refusing to discard {len(reviewed_dropped)} reviewed sample rows; "
                "use a new --output-dir or preserve the existing review file"
            )
    fields = [
        "portion_id", "doc_id", "page_index", "paragraph_index", "portion_label",
        "marking_original", "placement", "direction", "mark_ocr_confidence",
        "confidence", "analysis_eligible", "quality_flags", "visible_excerpt",
        "is_true_portion_mark", "boundary_correct", "label_correct",
        "visible_sensitive_evidence", "reviewer", "review_notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in selected:
            writer.writerow({
                "portion_id": row["portion_id"], "doc_id": row["doc_id"],
                "page_index": row["page_index"], "paragraph_index": row["paragraph_index"],
                "portion_label": row["portion_label"], "marking_original": row["marking_original"],
                "placement": row["placement"], "direction": row["direction"],
                "mark_ocr_confidence": row["mark_ocr_confidence"],
                "confidence": row["confidence"], "analysis_eligible": row["analysis_eligible"],
                "quality_flags": "|".join(row["quality_flags"]),
                "visible_excerpt": row["visible_portion_text"][:500].replace("\n", " "),
                **{
                    field: str(existing.get(str(row["portion_id"]), {}).get(field, ""))
                    for field in judgment_fields
                },
            })
    return len(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=CIA_BENCHMARK_R4_DIR / "benchmark.jsonl")
    parser.add_argument("--output-dir", type=Path, default=PORTION_DERIVED_DIR / "portion_v2_layout")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--review-per-stratum", type=int, default=50)
    args = parser.parse_args()

    dataset = args.dataset.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    document_ids: set[str] = set()

    for record_index, record in enumerate(jsonl_records(dataset)):
        if args.max_docs is not None and record_index >= args.max_docs:
            break
        stats["documents_scanned"] += 1
        visible = str(record.get("marker_visible_text", "") or "")
        if not PORTION_RE.search(visible) and not FULL_PORTION_RE.search(visible):
            continue
        stats["candidate_documents"] += 1
        try:
            rows, local = build_rows(record)
        except (ET.ParseError, FileNotFoundError, ValueError) as error:
            stats[f"error:{type(error).__name__}"] += 1
            continue
        stats.update(local)
        if rows:
            document_ids.add(str(record["identifier"]))
            all_rows.extend(rows)

    all_rows.sort(key=lambda row: row["portion_id"])
    high_rows = [row for row in all_rows if row["analysis_eligible"]]
    for row in all_rows:
        stats["marks_total"] += 1
        stats[f"label:{row['portion_label']}"] += 1
        stats[f"confidence:{row['confidence']}"] += 1
        stats[f"placement:{row['placement']}"] += 1
        for flag in row["quality_flags"]:
            stats[f"flag:{flag}"] += 1
    stats["documents_with_marks"] = len(document_ids)
    stats["high_confidence_portions"] = len(high_rows)
    stats["high_confidence_documents"] = len({row["doc_id"] for row in high_rows})

    manifest = output_dir / "portion_manifest.jsonl"
    high_manifest = output_dir / "high_confidence_portions.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with high_manifest.open("w", encoding="utf-8") as handle:
        for row in high_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    review_count = write_review_sample(
        all_rows, output_dir / "review_sample.csv", args.review_per_stratum
    )

    outputs = {}
    for path in (manifest, high_manifest, output_dir / "review_sample.csv"):
        outputs[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "dataset": str(dataset.relative_to(ROOT)),
        "dataset_sha256": sha256_file(dataset),
        "parameters": {
            "min_recoverable_chars": MIN_RECOVERABLE_CHARS,
            "min_retained_ratio": MIN_RETAINED_RATIO,
            "min_mark_confidence": MIN_MARK_CONFIDENCE,
            "random_seed": RANDOM_SEED,
            "review_per_stratum": args.review_per_stratum,
        },
        "counts": dict(sorted(stats.items())),
        "review_sample_rows": review_count,
        "outputs": outputs,
        "interpretation": (
            "High-confidence rows are validated-by-rule operational portion-label candidates, "
            "not original-classifier rationales. Human validation fields remain blank."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
