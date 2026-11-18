#!/usr/bin/env python3
"""Rebuild edited and control inputs from restored baseline texts.

The release contains no document text.  ``data/edits/edit_operations.jsonl.gz``
stores every valid edit variant as text-free operations over the baseline
(marker-removed) text of a document.  Given restored baseline texts, this
script reconstructs each variant and verifies its SHA-256.

Input: JSON lines with ``document_id`` and ``text`` (baseline text whose
SHA-256 equals ``masked_text_sha256`` in the document manifest).
Output (optional): JSON lines with ``document_id``, ``variant``, ``text``.

Example
  apply_edits.py --variant 'selected_combo:blank_collapse_native+capital_title_case' \
      --out edited.jsonl restored_wikileaks.jsonl
"""
import argparse
import gzip
import json
from pathlib import Path

import edit_ops

HERE = Path(__file__).resolve().parent
DEFAULT_OPS = HERE.parents[1] / 'data/edits/edit_operations.jsonl.gz'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ops', default=str(DEFAULT_OPS))
    ap.add_argument('--variant', action='append', help='variant name (repeatable); default: all')
    ap.add_argument('--out')
    ap.add_argument('restored')
    a = ap.parse_args()
    texts = {}
    with open(a.restored, encoding='utf-8') as h:
        for line in h:
            if line.strip():
                row = json.loads(line)
                texts[row['document_id']] = row['text']
    out = open(a.out, 'w', encoding='utf-8') if a.out else None
    seen = ok = bad = opaque = missing = 0
    with gzip.open(a.ops, 'rt', encoding='utf-8') as h:
        for line in h:
            rec = json.loads(line)
            if a.variant and rec['variant'] not in a.variant:
                continue
            if rec['document_id'] not in texts:
                missing += 1
                continue
            seen += 1
            edited = edit_ops.apply_ops(texts[rec['document_id']], rec['ops'])
            if edited is None:
                opaque += 1
            elif edit_ops.sha256(edited) == rec['text_sha256']:
                ok += 1
                if out:
                    out.write(json.dumps(dict(document_id=rec['document_id'], variant=rec['variant'], text=edited)) + '\n')
            else:
                bad += 1
    if out:
        out.close()
    print(f'variants with restored baseline {seen}: verified {ok}, hash mismatch {bad}, '
          f'not reconstructible from operations {opaque}; baseline not provided {missing}')
    raise SystemExit(0 if bad == 0 else 1)


if __name__ == '__main__':
    main()
