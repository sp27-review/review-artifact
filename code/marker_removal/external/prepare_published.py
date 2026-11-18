"""Prepare downloaded published corpora; stdlib only, no model execution.

Run from any directory. Source archives and all text are preserved. Outputs are
versioned and write-once; incompatible reruns fail instead of replacing data.
"""
from pathlib import Path
import collections
import csv
import hashlib
import io
import json
import re
import zipfile

DATA = Path(__file__).resolve().parents[2]
DATE = "2026-10-30"
US_LABELS = {"U": 0, "C": 1, "S": 2, "TS": 3}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def put(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"Refusing to replace different existing file: {path}")
    else:
        path.write_bytes(payload)


def dump(path, obj):
    put(path, (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode())


def row(name, sid, text, label, source_path, **meta):
    assert text.strip(), (name, sid)
    return dict(doc_id=f"{name}:{sid}", dataset=name, source_id=sid,
                text=text, label=label, language="en", source_path=source_path,
                source_split="unspecified", text_sha256=sha(text.encode()),
                duplicate_group=sha(text.strip().encode()), **meta)


def relabel(rows, fn):
    out = []
    for r in rows:
        label = fn(r)
        if label is not None:
            out.append({**r, "label": label})
    return out


def export(name, views, notes):
    out = DATA / "datasets" / name / "v1"
    primary, primary_labels = views["all.jsonl"]
    groups = collections.defaultdict(list)
    for r in primary:
        groups[r["duplicate_group"]].append(r)
    duplicates, unique = [], []
    for key, rr in groups.items():
        conflict = len({r["label"] for r in rr}) > 1
        if not conflict:
            unique.append(rr[0])
        if len(rr) > 1:
            duplicates.append(dict(duplicate_group=key, doc_ids=[r["doc_id"] for r in rr],
                                   labels=[r["label"] for r in rr], conflict=conflict,
                                   kept_doc_id=None if conflict else rr[0]["doc_id"]))
    views["unique.jsonl"] = (unique, primary_labels)
    stats = {}
    for filename, (rows, labels) in views.items():
        assert len(rows) == len({r["doc_id"] for r in rows}), (name, filename)
        records = [{**r, "label_id": labels[r["label"]]} for r in rows]
        for r in records:
            assert r["text"].strip() and sha(r["text"].encode()) == r["text_sha256"]
        payload = b"".join((json.dumps(r, ensure_ascii=False) + "\n").encode() for r in records)
        put(out / filename, payload)
        # Verify the saved JSONL, counts and label encoding independently of serialization.
        with (out / filename).open() as f:
            decoded = [json.loads(line) for line in f]
        assert decoded == records
        stats[filename] = dict(rows=len(records), bytes=len(payload), sha256=sha(payload),
                               labels=dict(collections.Counter(r["label"] for r in records)),
                               label_map=labels)
    raw = DATA / "raw" / name
    archives = []
    for archive in sorted(raw.glob("*.zip")):
        archives.append(dict(path=str(archive.relative_to(DATA)), bytes=archive.stat().st_size,
                             sha256=sha(archive.read_bytes())))
    manifest = dict(dataset=name, prepared_on_client_date=DATE, input_field="text",
                    gold_field="label_id", label_map=primary_labels, outputs=stats,
                    raw_archives=archives, provenance=json.loads((raw / "download_manifest.json").read_text()),
                    text_processing="Decode UTF-8 BOM if present; parse source fields. No masking, rewriting, trimming or truncation of text.",
                    deduplication="Exact text.strip() groups; unique.jsonl keeps first if labels agree, excludes entire conflicting groups.",
                    duplicate_groups=len(duplicates), conflicting_label_groups=sum(d["conflict"] for d in duplicates),
                    split_policy="No train/dev/test split invented; source batch/administration metadata retained.",
                    notes=notes, builder=str(Path(__file__).relative_to(DATA)),
                    builder_sha256=sha(Path(__file__).read_bytes()))
    dump(out / "labels.json", primary_labels)
    dump(out / "duplicates.json", duplicates)
    dump(out / "manifest.json", manifest)
    print(json.dumps(dict(dataset=name, outputs=stats, duplicate_groups=len(duplicates),
                          conflicting_label_groups=manifest["conflicting_label_groups"])), flush=True)


def frus():
    name = "frus_sachn"
    rows = []
    excluded = []
    with zipfile.ZipFile(DATA / "raw" / name / f"{name}.zip") as z:
        names = set(z.namelist())
        for path in sorted(n for n in names if n.endswith(".csv")):
            for r in csv.DictReader(io.StringIO(z.read(path).decode("utf-8-sig"))):
                text_path = "FRUS/" + r["content"].replace("\\", "/")
                assert text_path in names
                text = z.read(text_path).decode("utf-8-sig")
                if not text.strip():
                    excluded.append(dict(reason="empty_text", source_path=text_path, source_metadata=r))
                    continue
                rows.append(row(name, r["id"], text,
                                r["document_confidentiality"], text_path,
                                source_label=r["document_confidentiality"],
                                source_metadata=r, source_csv=path))
    assert len(rows) == 24676 and len(excluded) == 30
    dump(DATA / "datasets" / name / "v1" / "excluded.json", excluded)
    raw_labels = {s: i for i, s in enumerate(sorted({r["source_label"] for r in rows}))}
    explicit = {"Unclassified": "U", "Confidential": "C", "Secret": "S", "Top Secret": "TS"}
    primary = {**explicit, "No classification marking": "U"}
    card = {**primary, "Limited Official Use": "U"}
    export(name, {
        "all.jsonl": (relabel(rows, lambda r: primary.get(r["source_label"])), US_LABELS),
        "explicit_labels.jsonl": (relabel(rows, lambda r: explicit.get(r["source_label"])), US_LABELS),
        "card_protocol.jsonl": (relabel(rows, lambda r: card[r["source_label"]]), US_LABELS),
        "source_all.jsonl": (rows, raw_labels),
    }, [
        "Official author release for SACHN-DeBERTa, doi:10.3390/app16115661; source CSV + TXT count 24,706.",
        "30 empty TXT files are excluded from every JSONL view and recorded in excluded.json; usable source_all has 24,676 rows.",
        "all.jsonl follows paper Section 3.2.2 / card example: no marking -> U; exclude 609 LOU. After empty-text filtering 24,067 rows, not headline 24,706.",
        "explicit_labels.jsonl contains only explicit U/C/S/TS (22,136 nonempty documents); recommended for unambiguous external grade evaluation.",
        "card_protocol.jsonl additionally maps LOU -> U (24,676 nonempty), matching the card's policy table but contradicting its example and paper's stated exclusion.",
        "Mapping absent markings to U is a benchmark policy, not an independently verified classification judgment.",
        "Kaggle licenseName is CC BY-NC-SA 4.0, while description says CC BY 4.0; conflict retained, no resolution asserted.",
        "Text retains original whitespace, archival notes and any remaining classification marks; masking and CIA cross-corpus overlap audit still required for fair content-only evaluation.",
    ])


def wikileaks():
    name = "wikileaks_cable_classifier"
    rows = []
    with zipfile.ZipFile(DATA / "raw" / name / f"{name}.zip") as z:
        for line in z.read("wikileaks_docs.jl").splitlines():
            r = json.loads(line)
            raw = r["Original Classification:"].strip()
            base = re.sub(r"\s*\(\d+ documents\)\s*$", "", raw).split(",")[0]
            label = {"UNCLASSIFIED": "U", "CONFIDENTIAL": "C", "SECRET": "S"}[base]
            rows.append(row(name, r["Canonical ID:"].strip(), r["Content"], label,
                            "wikileaks_docs.jl", source_label=raw,
                            source_metadata={k: v for k, v in r.items() if k != "Content"}))
    assert len(rows) == 9005
    export(name, {"all.jsonl": (rows, US_LABELS)}, [
        "Kaggle yilouislu/wikileaks, input of the Wikileaks Cable Classifier notebook cited by SACHN-DeBERTa; 9,005 records.",
        "Use Original Classification metadata, not Current Classification. FOUO/NOFORN retained as metadata and removed only from target base grade.",
        "No TS gold samples; the US four-class output vocabulary is retained. Do not suppress TS predictions when scoring the existing head.",
        "Historical WikiLeaks cable collection, not a declassified archive equivalent to FRUS.",
        "Raw Content is preserved; inline marks remain. Source metadata must never be included in model input.",
        "Kaggle dataset licenseName is Unknown; notebook Apache-2.0 license is not asserted to license the dataset.",
    ])


def foia_sentence():
    name = "foia_sentence"
    rows = []
    with zipfile.ZipFile(next((DATA / "raw" / name).glob("*.zip"))) as z:
        for path in sorted(n for n in z.namelist() if n.endswith(".txt")):
            batch = re.search(r"foia_([KR]\d)", path).group(1)
            block = 0
            reader = csv.reader(io.StringIO(z.read(path).decode("utf-8-sig")), delimiter="|")
            next(reader)
            for lineno, r in enumerate(reader, 2):
                assert len(r) == 6
                if not any(r):
                    block += 1
                    continue
                assert r[0] in ("D0", "D1") and r[2] in ("", "1") and r[3] in ("", "1")
                rows.append(row(name, f"{batch}:{lineno}", r[1], "AD" if r[2] == "1" else "non_AD", path,
                                source_line=lineno, source_batch=batch, source_block=f"{batch}:{block}",
                                original_paragraph_label=r[0], always_deliberative=r[2] == "1",
                                non_iia=r[3] == "1", source_sentiment=r[4], source_subjectivity=r[5]))
    labels = {"non_AD": 0, "AD": 1}
    iia = [r for r in rows if not r["non_iia"]]
    primary = [r for r in iia if r["source_batch"] != "K5"]
    later = [r for r in rows if r["source_batch"] != "K5"]
    assert (len(rows), len(iia), len(primary), len(later)) == (5811, 5010, 3962, 4504)
    assert sum(r["label"] == "AD" for r in primary) == 1291
    export(name, {"all.jsonl": (primary, labels), "including_k5.jsonl": (iia, labels),
                  "unfiltered.jsonl": (rows, labels), "later_four_batches.jsonl": (later, labels)}, [
        "Branting et al., Decision support for detecting sensitive text in government records, doi:10.1007/s10506-023-09383-6 (online 2023; journal 2025).",
        "Target is Always Deliberative column, NOT D0/D1 inherited paragraph label. This is deliberative-language classification, not U/C/S/TS.",
        "all.jsonl excludes non-IIA sentences and K5, matching original paper Section 4.1 and Table 1; 3,962 sentences, AD 1,291.",
        "later_four_batches includes non-IIA and excludes K5: actual sum 4,504. To Redact (arxiv:2605.10211) Table 1 lists the same batch counts but total 4,502 (arithmetic mismatch); not silently reproduced.",
        "FOIA paragraph and sentence sets share Clinton White House source documents; not two independent corpora and must not be mixed across train/test.",
        "Source approved for public release, distribution unlimited, MITRE Case 23-1958; README retains copyright, no standard open license asserted.",
    ])


def foia_paragraph():
    name = "foia_paragraph"
    rows = []
    with zipfile.ZipFile(next((DATA / "raw" / name).glob("*.zip"))) as z:
        for path in sorted(n for n in z.namelist() if n.endswith(".tsv")):
            batch = Path(path).parent.name
            reader = csv.DictReader(io.StringIO(z.read(path).decode("utf-8-sig")), delimiter="\t")
            for lineno, r in enumerate(reader, 2):
                assert set(r) == {"Annotator1", "Annotator2", "Resolve1&2", "Paragraph"}
                selected = r["Resolve1&2"].strip() or r["Annotator1"].strip()
                label = selected.removesuffix("//")
                assert label in ("D0", "D1", "T0", "E0")
                rows.append(row(name, f"{batch}:{lineno}", r["Paragraph"], label, path,
                                source_batch=batch, source_line=lineno, source_label=selected,
                                annotations={k: r[k] for k in ("Annotator1", "Annotator2", "Resolve1&2")}))
    core = [r for r in rows if r["label"] in ("D0", "D1")]
    assert len(rows) == 3019 and len(core) == 2237
    export(name, {"all.jsonl": (core, {"D0": 0, "D1": 1}),
                  "source_all.jsonl": (rows, {"D0": 0, "D1": 1, "T0": 2, "E0": 3})}, [
        "Baron, Sayed & Oard, Providing More Efficient Access To Government Records, arxiv:2011.07203; ACM JOCCH 15(1), 2022.",
        "all.jsonl includes D0/D1 only; excludes T0/E0 without inventing a binary collapse. source_all retains all four original categories.",
        "Use Resolve1&2 if provided, otherwise Annotator1; normalize terminal // in K1 labels. Original annotations retained.",
        "Source PDFs and XLSX are preserved inside the raw ZIP; JSONL uses the author's TSV text, no fresh OCR.",
        "Shares source corpus with foia_sentence. Batches are not original document identifiers; no independent random sentence/document split asserted.",
        "Public repository has no LICENSE file in downloaded tree; no standard redistribution license inferred.",
    ])


if __name__ == "__main__":
    frus()
    wikileaks()
    foia_sentence()
    foia_paragraph()
