#!/usr/bin/env python3
"""(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from p1_download import (
    DEFAULT_FORMAT_SUFFIXES,
    IA_DOWNLOAD_URL,
    IA_METADATA_URL,
    derive_base_name,
    download_file,
    fetch_json,
    select_file,
)


def find_incomplete(root: Path, suffixes: tuple[str, ...]) -> list[tuple[Path, list[str]]]:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    out: list[tuple[Path, list[str]]] = []
    for entry in os.scandir(root):
        if not entry.is_dir():
            continue
        names = {f.name: f.stat().st_size for f in os.scandir(entry.path)}
        missing = [s for s in suffixes
                   if not any(n.endswith(s) and size > 0 for n, size in names.items())]
        if missing:
            out.append((Path(entry.path), missing))
    return out


def cleanup_tmp(root: Path) -> int:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    removed = 0
    for entry in os.scandir(root):
        if not entry.is_dir():
            continue
        for name in os.listdir(entry.path):
            if name.endswith(".tmp"):
                (Path(entry.path) / name).unlink(missing_ok=True)
                removed += 1
    return removed


def repair_one(item_dir: Path, missing: list[str], strip_prefix: str) -> dict:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    identifier = item_dir.name
    result = {"identifier": identifier, "missing": missing,
              "repaired": [], "unavailable": [], "errors": []}
    base = derive_base_name(identifier, strip_prefix)

    def url_for(name: str) -> str:
        return IA_DOWNLOAD_URL.format(identifier=urllib.parse.quote(identifier),
                                      filename=urllib.parse.quote(name))

    metadata_names: list[str] | None = None
    for suffix in missing:
        name = f"{base}{suffix}"
        target = item_dir / name
        try:
            download_file(url_for(name), target)
            result["repaired"].append(suffix)
            continue
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if "HTTP 404" not in str(exc):
                result["errors"].append({"suffix": suffix, "error": str(exc)})
                continue


        if metadata_names is None:
            try:
                meta = fetch_json(IA_METADATA_URL.format(
                    identifier=urllib.parse.quote(identifier)))
                metadata_names = [str(f.get("name", "")) for f in meta.get("files", [])]
            except Exception as exc:  # pylint: disable=broad-exception-caught
                result["errors"].append({"suffix": suffix, "error": f"metadata: {exc}"})
                metadata_names = []
        real = select_file(metadata_names, suffix) if metadata_names else ""
        if not real:
            result["unavailable"].append(suffix)
            continue
        try:
            download_file(url_for(real), item_dir / real)
            result["repaired"].append(suffix)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            result["errors"].append({"suffix": suffix, "error": str(exc)})
    return result


def main() -> int:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--name-strip-prefix", default="")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="x008b x008c x008d x008e x008f x0090.")
    args = parser.parse_args()

    tmp_removed = 0 if args.dry_run else cleanup_tmp(args.root)
    incomplete = find_incomplete(args.root, DEFAULT_FORMAT_SUFFIXES)
    print(f"x0091 x0044 {len(incomplete)} (.tmp x0092 {tmp_removed})", flush=True)
    if args.dry_run or not incomplete:
        for path, missing in incomplete[:20]:
            print(f"  {path.name}: x008b {missing}")
        return 0

    from concurrent.futures import ThreadPoolExecutor

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(repair_one, p, m, args.name_strip_prefix)
                   for p, m in incomplete]
        for i, future in enumerate(futures, 1):
            try:
                results.append(future.result())
            except Exception as exc:  # pylint: disable=broad-exception-caught
                results.append({"identifier": "?", "errors": [{"error": str(exc)}]})
            if i % 50 == 0:
                print(f"  x0093 {i}/{len(futures)}", flush=True)

    repaired = sum(len(r["repaired"]) for r in results)
    unavailable = sum(len(r.get("unavailable", [])) for r in results)
    failed = sum(len(r.get("errors", [])) for r in results)
    still = find_incomplete(args.root, DEFAULT_FORMAT_SUFFIXES)
    print(f"x0094 x005d {repaired} | x0095 x0096 {unavailable} | x0004 {failed}")
    print(f"x0094 x0097 x0091 x0044: {len(still)} (x0094 x0098 {len(incomplete)})")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w", encoding="utf-8") as handle:
            for record in results:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"report={args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
