#!/usr/bin/env python3
"""Verify every file listed in checksums.sha256 (run from the artifact root)."""
import hashlib
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
bad = n = 0
for line in (root / 'checksums.sha256').read_text().splitlines():
    digest, rel = line.split('  ', 1)
    n += 1
    if hashlib.sha256((root / rel).read_bytes()).hexdigest() != digest:
        bad += 1
        print('MISMATCH', rel)
print(f'checksums: {n - bad}/{n} files match')
sys.exit(1 if bad else 0)
