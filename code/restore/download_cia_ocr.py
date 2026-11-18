#!/usr/bin/env python3
"""Plan or perform re-acquisition of the CIA Reading Room OCR text files.

Documents are referenced by their Internet Archive item identifiers
(``cia-readingroom-document-<id>``).  The default is a read-only plan.
Downloading requires both --download and --accept-source-terms; users remain
responsible for the source sites' terms and for rights in the documents.

Each downloaded ``*_djvu.txt`` is verified against the SHA-256 recorded when
the corpus was built (``source_ocr_sha256`` in data/documents/cia_documents.csv.gz).
"""
import argparse
import csv
import gzip
import hashlib
import urllib.request
from pathlib import Path
from urllib.parse import quote

DEFAULT = Path(__file__).resolve().parents[2] / 'data/documents/cia_documents.csv.gz'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, default=DEFAULT)
    ap.add_argument('--split', action='append', help='restrict to a split (train/dev/test/portion_eval); repeatable')
    ap.add_argument('--output-dir', type=Path)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--download', action='store_true')
    ap.add_argument('--accept-source-terms', action='store_true')
    a = ap.parse_args()
    if a.download and (not a.accept_source_terms or a.output_dir is None):
        raise SystemExit('--download requires --accept-source-terms and --output-dir')
    with gzip.open(a.manifest, 'rt', encoding='utf-8', newline='') as h:
        rows = [r for r in csv.DictReader(h) if not a.split or r['split'] in a.split]
    if a.limit is not None:
        rows = rows[:a.limit]
    print(f'documents selected: {len(rows)}')
    url = lambda r: f"https://archive.org/download/{quote(r['document_id'])}/{quote(r['source_ocr_filename'])}"
    if not a.download:
        for r in rows[:10]:
            print(f"{r['document_id']}\t{url(r)}\tsha256={r['source_ocr_sha256']}")
        if len(rows) > 10:
            print(f'... {len(rows) - 10} more; pass --download only after reviewing the source terms')
        return
    failures = []
    for k, r in enumerate(rows, 1):
        target = a.output_dir / r['document_id'] / r['source_ocr_filename']
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(url(r), target)
            if hashlib.sha256(target.read_bytes()).hexdigest() != r['source_ocr_sha256']:
                failures.append((r['document_id'], 'sha256 mismatch'))
            else:
                print(f'[{k}/{len(rows)}] verified {r["document_id"]}')
        except Exception as exc:  # network and source availability are external state
            failures.append((r['document_id'], str(exc)))
    for doc, reason in failures:
        print(f'FAILED {doc}: {reason}')
    raise SystemExit(1 if failures else 0)


if __name__ == '__main__':
    main()
