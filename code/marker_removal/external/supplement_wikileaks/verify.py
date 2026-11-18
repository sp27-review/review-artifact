#!/usr/bin/env python3
"""Independent streaming verifier; does not import the builder or span grammar."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from itertools import zip_longest
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1<<20), b""):
            h.update(block)
    return h.hexdigest()


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def rows(path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(out):
    out = Path(out)
    manifest = json.loads((out/"manifest.json").read_text())
    for name, expected in manifest["source_files"].items():
        require(digest(Path(name)) == expected, ("source modified", name))
    for name, info in manifest["outputs"].items():
        require((out/name).stat().st_size == info["bytes"] and digest(out/name) == info["sha256"], ("output hash", name))
    seen, stats = {}, Counter()
    for raw, new, audit in zip_longest(rows(Path(manifest["source"])), rows(out/"masked.jsonl"), rows(out/"deletion_audit.jsonl")):
        require(all(x is not None for x in (raw,new,audit)), "row count mismatch")
        did = raw["doc_id"]
        require(did == new["doc_id"] == audit["doc_id"] and did not in seen, ("membership",did))
        require(set(new) == set(raw) | {"supplement_source_text_sha256"}, ("new metadata",did))
        for key in raw.keys()-{"text","text_sha256","supplement_source_text_sha256"}:
            require(raw[key] == new[key], ("changed metadata",did,key))
        text, masked = raw["text"], new["text"]
        require(sha(text) == raw["text_sha256"] == audit["input_sha256"] == new["supplement_source_text_sha256"], ("input hash",did))
        require(sha(masked) == new["text_sha256"] == audit["output_sha256"], ("output text hash",did))
        pieces, previous = [], 0
        for e in audit["deletions"]:
            a,b = e["start"],e["end"]
            require(previous <= a < b <= len(text) and text[a:b] == e["text"], ("span",did,e))
            pieces.append(text[previous:a]); previous = b
        pieces.append(text[previous:])
        require("".join(pieces) == masked and bool(masked.strip()), ("reconstruction",did))
        seen[did] = new["text_sha256"]
        stats.update(documents=1, changed_documents=text!=masked, removed_spans=len(audit["deletions"]), removed_characters=len(text)-len(masked))
    for k,v in stats.items():
        require(manifest["stats"][k] == v, ("stats",k))
    pending_ids, pending_candidates, residual = set(), 0, {}
    for row in rows(out/"residual_audit.jsonl"):
        did = row["doc_id"]
        require(did in seen and did not in residual and row["text_sha256"] == seen[did], ("residual doc",did))
        residual[did] = row["candidates"]
    # Re-read outputs to independently check every candidate's character coordinates.
    for row in rows(out/"masked.jsonl"):
        for c in residual.get(row["doc_id"], []):
            require(0 <= c["start"] < c["end"] <= len(row["text"]) and row["text"][c["start"]:c["end"]] == c["text"], ("residual offset",row["doc_id"]))
    queue = {}
    for row in rows(out/"review_queue.jsonl"):
        did = row["doc_id"]
        require(did not in pending_ids and did in seen and row["output_sha256"] == seen[did], ("queue ID",did))
        require([{k:v for k,v in c.items() if k!="input_span"} for c in row["candidates"]] == [c for c in residual[did] if c["status"] == "review"], ("queue contents",did))
        pending_ids.add(did); pending_candidates += len(row["candidates"])
        queue[did] = row
    for original, audit in zip(rows(Path(manifest["source"])), rows(out/"deletion_audit.jsonl")):
        did = original["doc_id"]
        if did not in queue:
            continue
        require(queue[did]["input_sha256"] == sha(original["text"]), ("queue input hash",did))
        def projected(pos):
            return pos-sum(max(0,min(pos,e["end"])-e["start"]) for e in audit["deletions"])
        for c in queue[did]["candidates"]:
            span=c["input_span"]
            require(original["text"][span["start"]:span["end"]] == span["text"], ("review source span",did))
            require((projected(span["start"]),projected(span["end"])) == (c["start"],c["end"]), ("review coordinate mapping",did))
    require(pending_ids == {i for i,cs in residual.items() if any(c["status"]=="review" for c in cs)}, "missing queue rows")
    require(len(pending_ids) == manifest["stats"]["review_documents"] and pending_candidates == manifest["stats"]["review_candidates"], "review counts")
    sample_ids = set()
    for row in rows(out/"negative_review_sample.jsonl"):
        did = row["doc_id"]
        require(did not in sample_ids and did in seen and did not in residual and sha(row["text"]) == row["text_sha256"] == seen[did], ("negative sample",did))
        sample_ids.add(did)
    g3_count = 0
    if manifest["g3_input"]:
        for old, new in zip_longest(rows(Path(manifest["g3_input"])), rows(out/"g3_5000_masked.jsonl")):
            require(old is not None and new is not None and old.keys()==new.keys(), "G3 shape")
            for key in old.keys()-{"marker_masked_text","text_source"}:
                require(old[key] == new[key], ("G3 metadata",key))
            require(sha(new["marker_masked_text"]) == seen[new["identifier"]], ("G3 text",new["identifier"]))
            g3_count += 1
        require(g3_count == manifest["stats"]["g3_documents"], "G3 count")
    return dict(status="passed", verified_utc=datetime.now(timezone.utc).isoformat(),
                stats=dict(stats), g3_ids_order_labels_preserved=g3_count,
                input_files_unchanged=True, nondeleted_characters_exactly_preserved=True,
                metadata_and_membership_unchanged=True, all_candidate_offsets_verified=True,
                review_documents=len(pending_ids), review_candidates=pending_candidates,
                negative_sample_documents=len(sample_ids),
                limitation="Mechanical integrity passed; this is not a claim that every marking is removed or every candidate is a marking.")


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("out",type=Path); args=p.parse_args()
    print(json.dumps(verify(args.out),ensure_ascii=False,indent=2))
