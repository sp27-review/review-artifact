#!/usr/bin/env python3
"""Check locally restored texts against the released SHA-256 manifests.

Input: a JSON-lines file with one object per document containing
``document_id`` and ``text`` (the restored text).  The script compares
sha256(text) with a hash column of a document manifest.

Examples
  verify_text_hashes.py --manifest data/documents/cia_documents.csv.gz \
      --column masked_text_sha256 restored_cia_masked.jsonl
  verify_text_hashes.py --manifest data/documents/wikileaks_documents.csv.gz \
      --column masked_text_sha256 restored_wikileaks.jsonl
"""
import argparse
import csv
import gzip
import hashlib
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--column', required=True)
    ap.add_argument('--id-field', default='document_id')
    ap.add_argument('--text-field', default='text')
    ap.add_argument('restored')
    a = ap.parse_args()
    with gzip.open(a.manifest, 'rt', encoding='utf-8', newline='') as h:
        want = {r['document_id']: r[a.column] for r in csv.DictReader(h)}
    seen = ok = bad = unknown = 0
    with open(a.restored, encoding='utf-8') as h:
        for line in h:
            if not line.strip():
                continue
            row = json.loads(line)
            seen += 1
            doc = row[a.id_field]
            if doc not in want:
                unknown += 1
            elif hashlib.sha256(row[a.text_field].encode('utf-8')).hexdigest() == want[doc]:
                ok += 1
            else:
                bad += 1
    print(f'restored rows {seen}: match {ok}, mismatch {bad}, not in manifest {unknown}; '
          f'manifest documents {len(want)}, covered {ok}/{len(want)}')
    raise SystemExit(0 if bad == 0 and seen else 1)


if __name__ == '__main__':
    main()
