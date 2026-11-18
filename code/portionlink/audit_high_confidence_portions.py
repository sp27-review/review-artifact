#!/usr/bin/env python3
"""Verify integrity and characterize the layout-aware portion sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))
from corelib.paths import CIA_BENCHMARK_R4_DIR, PORTION_DERIVED_DIR, ROOT


LABEL_ORDER = {"U": 0, "C": 1, "S": 2, "TS": 3}
DOC_LABEL = {"UNCLASSIFIED": "U", "CONFIDENTIAL": "C", "SECRET": "S", "TOP SECRET": "TS"}


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return round(ordered[index], 6)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=PORTION_DERIVED_DIR / "portion_v2_layout")
    parser.add_argument("--dataset", type=Path, default=CIA_BENCHMARK_R4_DIR / "benchmark.jsonl")
    args = parser.parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    dataset_path = args.dataset.expanduser().resolve()

    all_rows = list(read_jsonl(input_dir / "portion_manifest.jsonl"))
    high_rows = list(read_jsonl(input_dir / "high_confidence_portions.jsonl"))
    high_ids = {row["portion_id"] for row in high_rows}
    all_by_id = {row["portion_id"]: row for row in all_rows}
    if len(all_by_id) != len(all_rows) or len(high_ids) != len(high_rows):
        raise ValueError("duplicate portion_id")
    if not high_ids <= set(all_by_id):
        raise ValueError("high-confidence manifest is not a subset of the full manifest")
    if high_ids != {row["portion_id"] for row in all_rows if row["analysis_eligible"]}:
        raise ValueError("high-confidence manifest does not equal analysis_eligible rows")

    doc_ids = {row["doc_id"] for row in high_rows}
    documents = {
        str(record["identifier"]): record
        for record in read_jsonl(dataset_path)
        if str(record["identifier"]) in doc_ids
    }
    if set(documents) != doc_ids:
        raise ValueError("dataset is missing high-confidence documents")

    errors: Counter[str] = Counter()
    features: Counter[str] = Counter()
    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    coverage: list[float] = []
    for row in high_rows:
        doc = documents[row["doc_id"]]
        masked = str(doc.get("marker_masked_text", "") or "")
        spans = row["masked_portion_spans"]
        reconstructed = "".join(masked[int(start):int(end)] for start, end in spans).strip()
        if reconstructed != row["recoverable_text"]:
            errors["recoverable_text_mismatch"] += 1
        if hashlib.sha256(reconstructed.encode()).hexdigest() != row["recoverable_text_sha256"]:
            errors["recoverable_hash_mismatch"] += 1
        if any(start < 0 or end < start or end > len(masked) for start, end in spans):
            errors["invalid_masked_span"] += 1
        if row["confidence"] != "high" or not row["analysis_eligible"]:
            errors["non_high_row"] += 1
        features[f"label:{row['portion_label']}"] += 1
        features[f"direction:{row['direction']}"] += 1
        features[f"form:{row['mark_form']}"] += 1
        features[f"zone:{row['mark_zone']}"] += 1
        by_doc[row["doc_id"]].append(row)

    overlap_pairs = 0
    consistency = Counter()
    for doc_id, rows in by_doc.items():
        doc = documents[doc_id]
        masked_len = len(str(doc.get("marker_masked_text", "") or ""))
        intervals = sorted(
            (int(start), int(end), row["portion_id"])
            for row in rows for start, end in row["masked_portion_spans"] if end > start
        )
        for left, right in zip(intervals, intervals[1:]):
            if right[0] < left[1] and right[2] != left[2]:
                overlap_pairs += 1
        merged: list[list[int]] = []
        for start, end, _ in intervals:
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        coverage.append(sum(end - start for start, end in merged) / max(masked_len, 1))
        max_portion = max(rows, key=lambda row: LABEL_ORDER[row["portion_label"]])["portion_label"]
        document_label = DOC_LABEL.get(str(doc.get("label", "")).upper(), "")
        consistency["comparable_documents"] += bool(document_label)
        consistency["highest_portion_equals_document"] += max_portion == document_label

    audit = {
        "schema_version": "portion-v2-layout-audit-1",
        "inputs": {
            "manifest": str((input_dir / "portion_manifest.jsonl").relative_to(ROOT)),
            "high_confidence": str((input_dir / "high_confidence_portions.jsonl").relative_to(ROOT)),
            "dataset": str(dataset_path.relative_to(ROOT)),
        },
        "counts": {
            "all_candidates": len(all_rows),
            "high_confidence_portions": len(high_rows),
            "high_confidence_documents": len(by_doc),
            "cross_portion_overlap_pairs": overlap_pairs,
            **dict(sorted(features.items())),
        },
        "integrity_errors": dict(errors),
        "document_consistency": {
            **dict(consistency),
            "rate": round(
                consistency["highest_portion_equals_document"]
                / max(consistency["comparable_documents"], 1), 6
            ),
        },
        "masked_text_coverage_per_document": {
            "mean": round(statistics.fmean(coverage), 6) if coverage else None,
            "p10": percentile(coverage, 0.10),
            "median": percentile(coverage, 0.50),
            "p90": percentile(coverage, 0.90),
        },
        "gate_passed": not errors and overlap_pairs == 0,
        "interpretation": (
            "A passed gate establishes internal extraction/projection integrity only; "
            "manual precision/recall validation is still required."
        ),
    }
    output = input_dir / "audit_summary.json"
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if not audit["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
