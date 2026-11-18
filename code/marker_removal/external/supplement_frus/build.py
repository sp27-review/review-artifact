#!/usr/bin/env python3
"""CPU-only supplemental masking. Never overwrites source, old rules or existing outputs.

Example: python scripts/external_marking_supplement/build.py --source .../evaluation.jsonl
  --out .../supplemental_mask_20261103_v1 --g3-input .../inputs/frus_masked.jsonl
Review decisions, if supplied, use doc_id, input_sha256, start, end, text, action
(remove/retain), and reason. Coordinates refer to this stage's input, not raw CIA/FRUS.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import uuid

from remover import apply, detect
from residual_scanner import scan


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1<<20), b""):
            h.update(b)
    return h.hexdigest()


def rows(path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def emit(f, row):
    f.write(json.dumps(row, ensure_ascii=False)+"\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--g3-input", type=Path)
    p.add_argument("--review-decisions", type=Path)
    args = p.parse_args()
    args.source, args.out = args.source.resolve(), args.out.resolve()
    if args.out.exists() or args.source.is_relative_to(args.out):
        raise SystemExit("Output must be a new directory separate from source")
    started = time.monotonic()
    hashes = {str(args.source): file_sha(args.source)}
    if args.g3_input:
        args.g3_input = args.g3_input.resolve()
        hashes[str(args.g3_input)] = file_sha(args.g3_input)
    decisions, seen_decisions = {}, set()
    if args.review_decisions:
        args.review_decisions = args.review_decisions.resolve()
        hashes[str(args.review_decisions)] = file_sha(args.review_decisions)
        for d in rows(args.review_decisions):
            if d["action"] not in {"remove", "retain"} or not d.get("reason"):
                raise ValueError("Review requires remove/retain action and a reason")
            decisions.setdefault(d["doc_id"], []).append(d)
    stage = args.out.with_name(args.out.name+".partial-"+uuid.uuid4().hex[:8])
    stage.mkdir(parents=True)
    (stage/"code").mkdir()
    for path in Path(__file__).parent.iterdir():
        if path.suffix in {".py", ".json"}:
            shutil.copyfile(path, stage/"code"/path.name)
    if args.review_decisions:
        shutil.copyfile(args.review_decisions, stage/"review_decisions.jsonl")
    names = ["masked.jsonl", "deletion_audit.jsonl", "residual_audit.jsonl", "review_queue.jsonl", "deletion_review.jsonl"]
    files = {n: (stage/n).open("w", encoding="utf-8") for n in names}
    stats, rule_counts, candidate_counts = Counter(), Counter(), Counter()
    seen, projected, negatives = set(), {}, []
    g3 = list(rows(args.g3_input)) if args.g3_input else []
    g3_map = {x["identifier"]: x for x in g3}
    if len(g3_map) != len(g3):
        raise ValueError("Duplicate G3 IDs")
    try:
        for row in rows(args.source):
            did, text = row["doc_id"], row["text"]
            if did in seen or sha(text) != row["text_sha256"]:
                raise ValueError(("Duplicate ID or input hash mismatch", did))
            seen.add(did)
            edits = detect(text)
            retained = []
            for d in decisions.get(did, []):
                seen_decisions.add(did)
                if d["input_sha256"] != sha(text) or not 0 <= d["start"] < d["end"] <= len(text) or text[d["start"]:d["end"]] != d["text"]:
                    raise ValueError(("Stale review coordinates", did))
                if d["action"] == "remove":
                    edits.append({k:d[k] for k in ("start", "end", "text")} | dict(rules=["reviewed_marking"], reason=d["reason"]))
                else:
                    retained.append(d)
            edits.sort(key=lambda x: (x["start"], x["end"]))
            for d in retained:
                if any(e["start"] < d["end"] and d["start"] < e["end"] for e in edits):
                    raise ValueError(("Retain decision conflicts with removal", did))
            masked = apply(text, edits)
            if not masked.strip():
                raise ValueError(("Empty result; inspect source", did))
            if detect(masked):
                raise ValueError(("Not idempotent; inspect rules", did, detect(masked)))
            out = dict(row, text=masked, text_sha256=sha(masked), supplement_source_text_sha256=sha(text))
            emit(files["masked.jsonl"], out)
            emit(files["deletion_audit.jsonl"], dict(doc_id=did, input_sha256=sha(text), output_sha256=sha(masked), deletions=edits))
            candidates = scan(masked)
            for c in candidates:
                for d in retained:
                    offset = sum(e["end"]-e["start"] for e in edits if e["end"] <= d["start"])
                    if (c["start"], c["end"]) == (d["start"]-offset, d["end"]-offset):
                        c.update(status="reviewed_retained", reason=d["reason"])
            pending = [c for c in candidates if c["status"] == "review"]
            if candidates:
                emit(files["residual_audit.jsonl"], dict(doc_id=did, text_sha256=sha(masked), candidates=candidates))
            if pending:
                # Review queue includes both coordinate spaces; decisions use input spans.
                def original_position(pos, right_boundary):
                    delta = 0
                    for e in edits:
                        anchor = e["start"]-delta
                        if anchor < pos or (anchor == pos and right_boundary):
                            delta += e["end"]-e["start"]
                        else:
                            break
                    return pos+delta
                review_candidates = []
                for c in pending:
                    a = original_position(c["start"], True)
                    b = original_position(c["end"], False)
                    review_candidates.append(c | dict(input_span=dict(start=a, end=b, text=text[a:b])))
                emit(files["review_queue.jsonl"], dict(doc_id=did, input_sha256=sha(text), output_sha256=sha(masked), candidates=review_candidates))
            if edits:
                emit(files["deletion_review.jsonl"], dict(doc_id=did, input_sha256=sha(text), deletions=[e | dict(context=text[max(0,e["start"]-120):e["end"]+160]) for e in edits]))
            if not candidates:
                negatives.append((sha(did+":supplement-v1-negative-sample"), dict(doc_id=did, text_sha256=sha(masked), text=masked)))
                if len(negatives) > 60:
                    negatives.sort(key=lambda x:x[0]); negatives = negatives[:50]
            stats.update(documents=1, changed_documents=text!=masked, removed_spans=len(edits),
                         removed_characters=len(text)-len(masked), review_documents=bool(pending),
                         review_candidates=len(pending), scanner_negative_documents=not candidates)
            for edit in edits:
                rule_counts.update(edit["rules"])
            candidate_counts.update(c["kind"] for c in pending)
            if did in g3_map:
                if text != g3_map[did]["marker_masked_text"]:
                    raise ValueError(("G3 baseline differs from supplemental input", did))
                projected[did] = dict(g3_map[did], marker_masked_text=masked, text_source=str(args.out/"masked.jsonl"))
                stats.update(g3_documents=1, g3_changed_documents=text!=masked, g3_review_documents=bool(pending))
        if seen_decisions != decisions.keys():
            raise ValueError("Review decisions refer to missing documents")
        if projected.keys() != g3_map.keys():
            raise ValueError("Some fixed G3 IDs missing from source")
    finally:
        for f in files.values():
            f.close()
    with (stage/"negative_review_sample.jsonl").open("w") as f:
        for _, row in sorted(negatives, key=lambda x:x[0])[:50]:
            emit(f, row)
    if args.g3_input:
        with (stage/"g3_5000_masked.jsonl").open("w") as f:
            for old in g3:
                emit(f, projected[old["identifier"]])
    for path, before in hashes.items():
        if file_sha(Path(path)) != before:
            raise ValueError(("Source changed during build", path))
    outputs = {str(p.relative_to(stage)):dict(sha256=file_sha(p), bytes=p.stat().st_size) for p in stage.rglob("*") if p.is_file()}
    manifest = dict(version="supplement-v1", created_utc=datetime.now(timezone.utc).isoformat(),
                    status="built_with_review_queue", source=str(args.source), source_files=hashes,
                    g3_input=str(args.g3_input) if args.g3_input else None, stats=dict(stats),
                    rules=dict(rule_counts), review_kinds=dict(candidate_counts), outputs=outputs,
                    elapsed_seconds=time.monotonic()-started, dataset_membership="unchanged; no exclusions or resampling",
                    inference_ready=False, review_complete=False, idempotence_verified=True,
                    constraints="CPU only; no model imports or inference; label-blind detection; no normalization of actual input; only logged spans removed; old pipeline not imported or modified",
                    limitations="Broad scanner candidates are not proven leaks. Manual review and new backend length validation required before a new inference run. Existing G3/H results are unchanged.")
    (stage/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    from verify import verify
    verification = verify(stage)
    (stage/"verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2)+"\n")
    stage.rename(args.out)
    print(json.dumps(dict(out=str(args.out), stats=dict(stats), verification=verification["status"], seconds=manifest["elapsed_seconds"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
