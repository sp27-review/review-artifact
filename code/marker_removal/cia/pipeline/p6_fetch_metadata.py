#!/usr/bin/env python3
"""(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ADVANCED_SEARCH = "https://archive.org/advancedsearch.php"


def load_done(out_path: Path) -> set[str]:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    done: set[str] = set()
    if out_path.exists():
        for line in out_path.open(encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["identifier"])
                except Exception:
                    pass
    return done


def fetch_batch(ids: list[str], timeout: int) -> list[dict]:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    query = "identifier:(" + " OR ".join(f'"{i}"' for i in ids) + ")"
    qs = (
        "q=" + urllib.parse.quote(query)
        + f"&rows={len(ids)}&output=json"
        + "&fl[]=identifier&fl[]=title&fl[]=subject&fl[]=description"
    )
    url = ADVANCED_SEARCH + "?" + qs
    raw = urllib.request.urlopen(url, timeout=timeout).read().decode()
    docs = json.loads(raw).get("response", {}).get("docs", [])
    records = []
    for doc in docs:
        subject = doc.get("subject", "")
        if isinstance(subject, list):
            subject = "; ".join(subject)
        description = doc.get("description", "")
        if isinstance(description, list):
            description = " ".join(description)
        records.append({
            "identifier": doc["identifier"],
            "title": doc.get("title", ""),
            "subject": subject,
            "description": (description or "")[:400],
        })
    return records


def main() -> None:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", type=Path, help="x00d2 identifier x00c8")
    parser.add_argument("--manifest", type=Path, help="track_manifest.jsonl (doc_id x00d3)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--sleep", type=float, default=0.4,
                        help="x00d4 x0024 x0015 x00d5(x0080). x00d6 x00d7.")
    args = parser.parse_args()

    ids: list[str] = []
    if args.manifest:
        ids += [json.loads(l)["doc_id"] for l in args.manifest.open() if l.strip()]
    if args.ids:
        ids += [l.strip() for l in args.ids.open() if l.strip()]

    ids = list(dict.fromkeys(ids))

    done = load_done(args.out)
    todo = [i for i in ids if i not in done]
    print(f"x009d {len(ids)}, x0042 {len(done)}, x00d8 x00d9 {len(todo)}")

    got = 0
    with args.out.open("a", encoding="utf-8") as out:
        for start in range(0, len(todo), args.batch):
            batch = todo[start:start + args.batch]

            for attempt in range(args.retries):
                try:
                    records = fetch_batch(batch, args.timeout)
                except Exception as exc:
                    records = None
                    reason = str(exc)
                if records:
                    for rec in records:
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        got += 1
                    out.flush()
                    break
                if attempt == args.retries - 1:
                    print(f"  batch {start//args.batch} x00da/x0004 (id {len(batch)} x00db)")
                else:
                    time.sleep(2 ** attempt)
            time.sleep(args.sleep)
            if (start // args.batch) % 50 == 0:
                print(f"...batch {start//args.batch}, got {got}", flush=True)
    print(f"x00dc: +{got} -> {args.out}")


if __name__ == "__main__":
    main()
