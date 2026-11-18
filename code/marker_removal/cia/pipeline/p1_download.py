#!/usr/bin/env python3
"""(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_QUERY = "identifier:cia-readingroom-document-* AND item_size:[2000000 TO 50000000]"
DEFAULT_FORMAT_SUFFIXES = ("_djvu.txt", "_djvu.xml", "_hocr.html")
IA_ADVANCED_SEARCH_URL = "https://archive.org/advancedsearch.php"
IA_SCRAPE_URL = "https://archive.org/services/search/v1/scrape"
IA_METADATA_URL = "https://archive.org/metadata/{identifier}"
IA_DOWNLOAD_URL = "https://archive.org/download/{identifier}/{filename}"


def main() -> None:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / args.manifest_name
    if manifest_path.exists():
        raise SystemExit(f"manifest already exists (x0001 x0002): {manifest_path}")

    skip = build_skip_set(args)

    if args.workers > 1:
        stats = run_parallel(args, manifest_path, skip)
    else:
        stats = run_sequential(args, manifest_path, skip)

    print(f"Wrote manifest: {manifest_path}")
    print(json.dumps(stats, ensure_ascii=False))


def build_skip_set(args: argparse.Namespace) -> set[str]:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    skip: set[str] = set()
    for skip_dir in args.skip_existing_dir:
        if skip_dir.is_dir():
            before = len(skip)
            skip.update(child.name for child in os.scandir(skip_dir) if child.is_dir())
            print(f"skip set += {len(skip) - before} from {skip_dir}", flush=True)

    for pattern in args.seen_manifest:
        for path in sorted(glob.glob(str(pattern))):
            before = len(skip)
            bad = retried = 0
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        bad += 1
                        continue
                    identifier = record.get("identifier", "")
                    if not identifier or identifier == "?":
                        continue




                    if record.get("status") == "error" or record.get("errors"):
                        retried += 1
                        continue
                    skip.add(identifier)
            note = f" (x0003 x0004 {bad} x0005)" if bad else ""
            if retried:
                note += f" (x0004 {retried} x0006 x0007 x0008)"
            print(f"skip set += {len(skip) - before} from {path}{note}", flush=True)

    print(f"skip set total = {len(skip)}", flush=True)
    return skip


def make_id_iterator(args: argparse.Namespace):
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    if args.search_api == "scrape":
        return iter_scrape_identifiers(query=args.query, rows=args.rows, delay=args.delay,
                                       cursor_file=args.cursor_file)
    return iter_search_identifiers(
        query=args.query, rows=args.rows, sort=args.sort, delay=args.delay
    )


def new_stats() -> dict[str, int]:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    return {"accepted": 0, "skipped_known": 0, "skipped_size": 0, "skipped_no_txt": 0,
            "errors": 0, "scanned": 0, "search_failed": 0}


def tally(stats: dict[str, int], status: str) -> None:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    key = ("accepted" if status == "accepted" else
           "skipped_size" if status == "skipped_size" else
           "skipped_no_txt" if status == "skipped_no_txt" else "errors")
    stats[key] += 1


def fetch_one(args: argparse.Namespace, identifier: str) -> dict[str, Any]:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    return download_item(
        identifier=identifier,
        item_dir=args.out_dir / identifier,
        include_pdf=args.include_pdf,
        delay=args.delay,
        txt_bytes_min=args.txt_bytes_min,
        txt_bytes_max=args.txt_bytes_max,
        gate_via=args.gate_via,
        strip_prefix=args.name_strip_prefix,
    )


def run_sequential(args: argparse.Namespace, manifest_path: Path, skip: set[str]) -> dict[str, int]:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    stats = new_stats()
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for identifier in make_id_iterator(args):
            if stats["accepted"] >= args.max_items:
                break
            stats["scanned"] += 1
            if identifier in skip:
                stats["skipped_known"] += 1
                continue

            record = fetch_one(args, identifier)
            tally(stats, record["status"])
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            manifest.flush()
            if record["status"] == "accepted" or stats["scanned"] % 50 == 0:
                print(
                    f"[accepted {stats['accepted']}/{args.max_items} | scanned {stats['scanned']}] "
                    f"{identifier}: {record['status']} txt_bytes={record.get('txt_bytes')}",
                    flush=True,
                )
    return stats


def run_parallel(args: argparse.Namespace, manifest_path: Path, skip: set[str]) -> dict[str, int]:
    """Thread-pool downloader. archive.org is latency-bound, so N concurrent
    metadata+file requests give a near-linear speedup up to the server's rate limit.
    A single lock guards the shared search generator, stats, and manifest writes;
    the network work (download_item) runs outside the lock."""
    import threading
    from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

    stats = new_stats()
    lock = threading.Lock()
    id_gen = make_id_iterator(args)

    exhausted = False
    started = time.time()
    last_beat = [started]

    def next_identifier() -> str | None:
        # Serialize consumption of the search generator (not thread-safe) and
        # skip/scanned bookkeeping. Page fetches happen here but are infrequent.
        nonlocal exhausted
        with lock:
            if exhausted or stats["accepted"] >= args.max_items:
                return None
            try:
                for identifier in id_gen:
                    stats["scanned"] += 1
                    if identifier in skip:
                        stats["skipped_known"] += 1
                        continue
                    return identifier
            except Exception as exc:  # pylint: disable=broad-exception-caught


                print(f"[search-fatal] identifier x0009 x000a: {exc}", file=sys.stderr, flush=True)
                stats["search_failed"] = 1
            exhausted = True
            return None

    def heartbeat(force: bool = False) -> None:

        now = time.time()
        if not force and now - last_beat[0] < args.heartbeat_seconds:
            return
        last_beat[0] = now
        elapsed = now - started
        rate = stats["accepted"] / elapsed if elapsed > 0 else 0.0
        print(
            f"[heartbeat +{elapsed / 3600:.2f}h] accepted={stats['accepted']} "
            f"scanned={stats['scanned']} skipped_known={stats['skipped_known']} "
            f"skipped_size={stats['skipped_size']} skipped_no_txt={stats['skipped_no_txt']} "
            f"errors={stats['errors']} | {rate * 3600:.0f} accepted/h",
            flush=True,
        )

    manifest = manifest_path.open("w", encoding="utf-8")
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            inflight = {
                executor.submit(fetch_one, args, idf)
                for idf in filter(None, (next_identifier() for _ in range(args.workers * 2)))
            }
            while inflight:
                done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                for future in done:
                    try:
                        record = future.result()
                    except Exception as exc:  # pylint: disable=broad-exception-caught


                        record = {"identifier": "?", "status": "error",
                                  "errors": [{"stage": "worker", "error": str(exc)}]}
                    with lock:
                        tally(stats, record["status"])
                        manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                        manifest.flush()
                        accepted, scanned = stats["accepted"], stats["scanned"]
                        if record["status"] == "accepted" or scanned % 50 == 0:
                            print(
                                f"[accepted {accepted}/{args.max_items} | scanned {scanned}] "
                                f"{record['identifier']}: {record['status']} "
                                f"txt_bytes={record.get('txt_bytes')}",
                                flush=True,
                            )
                        heartbeat()
                    identifier = next_identifier()
                    if identifier is not None:
                        inflight.add(executor.submit(fetch_one, args, identifier))
    finally:
        manifest.close()
        heartbeat(force=True)
    return stats


def parse_args() -> argparse.Namespace:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    parser = argparse.ArgumentParser(
        description="Download IA CIA Reading Room OCR artifacts."
    )
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--max-items", type=int, default=1000,
                        help="x000b x000c 'x000d x000e x000f + x0010 x0011' x0012 x0013 x000a")
    parser.add_argument("--rows", type=int, default=100)
    parser.add_argument("--search-api", choices=("scrape", "advanced"), default="scrape",
                        help="identifier x0014 API. x0015 scrape cursor x0016 "
                             "advancedsearch deep-pagination 10,000 x0017 x0018. "
                             "advanced x0019 x001a(x001b/x001c).")
    parser.add_argument("--sort", default="identifier asc",
                        help="advancedsearch x001d(--search-api advanced x001e x001f). "
                             "x0015 'identifier asc'x0020 x0021 x0022 x0023 x0024 "
                             "x0025 identifier x0026. x0027 x0028 x0029 x002a('') x002b.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1,
                        help="x002c x0010 x002d x002e. 1 x002f(x0030). archive.org "
                             "x0031 x0032 16~32 x0033 x0034 x0035. x0036 x0037 rate limit(429).")
    parser.add_argument("--delay", type=float, default=0.1)
    parser.add_argument("--include-pdf", action="store_true")
    parser.add_argument("--txt-bytes-min", type=int, default=1900,
                        help="metadata _djvu.txt x0038(x0039) x003a. txt x0039≈x003b "
                             "2000~10000 x003c x003d(1900~12000) x003e x003f x0040 x0041.")
    parser.add_argument("--txt-bytes-max", type=int, default=12000)
    parser.add_argument("--skip-existing-dir", type=Path, action="append", default=[],
                        help="x0042 x0043 x0044 x0045 x0046 x0047(x0048: raw_ocr). x0049 x004a x004b.")
    parser.add_argument("--gate-via", choices=("metadata", "head"), default="metadata",
                        help="x000d x004c x000e x004d x001a. metadata x0030 x004e(x004f "
                             "/metadata/ 1). head `_djvu.txt`x0050 HEAD Content-Length "
                             "x0051 /metadata/ x0052(~1.1 req/s, x0053 x0054) x0055. "
                             "x0056 x0057 x0058 x0059 metadata x005a x005b x005c.")
    parser.add_argument("--name-strip-prefix", default="",
                        help="--gate-via head x005d x005e x005f x0060 identifier "
                             "x0061 x0062. CIA Reading Room "
                             "'cia-readingroom-document-', DTIC x0029 x002a.")
    parser.add_argument("--seen-manifest", action="append", default=[],
                        help="x0019 x0063 manifest jsonl x0064(glob x004b). x0049 x004a x004b. "
                             "x000d x000e x0065(skipped_size x0066) x0067 x0068 "
                             "x0069 x006a metadata x006b x006c.")
    parser.add_argument("--cursor-file", type=Path, default=None,
                        help="scrape cursor x006d x0064. x006e x006f x0070 "
                             "x0071 x0072 x0073 x0074 x0075 x0076(x0077 x0078 x0079).")
    parser.add_argument("--heartbeat-seconds", type=float, default=300.0,
                        help="x007a x007b x006a x007c x007d x007e x007f(x0080).")
    parser.add_argument("--manifest-name", default=f"download_manifest_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    return parser.parse_args()


def iter_scrape_identifiers(query: str, rows: int, delay: float,
                            cursor_file: Path | None = None):
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    cursor = ""
    if cursor_file is not None and cursor_file.exists():
        cursor = cursor_file.read_text(encoding="utf-8").strip()
        if cursor:
            print(f"[resume] x0081 cursor x0014 x0069: {cursor_file}", flush=True)




    count = max(500, min(rows * 100, 5000))
    page = 0




    seen_ids: set[str] = set()
    stale_pages = 0
    while True:
        params: list[tuple[str, str]] = [
            ("q", query),
            ("fields", "identifier"),
            ("count", str(count)),
        ]
        if cursor:
            params.append(("cursor", cursor))
        url = IA_SCRAPE_URL + "?" + urllib.parse.urlencode(params)
        payload = fetch_json_persistent(url, f"scrape page={page}")
        items = payload.get("items", [])
        page += 1
        if not items:
            return

        fresh = 0
        for item in items:
            identifier = str(item.get("identifier", "")).strip()
            if not identifier or identifier in seen_ids:
                continue
            seen_ids.add(identifier)
            fresh += 1
            yield identifier
        print(f"[search] page={page} items={len(items)} x0082={fresh} "
              f"x0083={len(seen_ids)} total={payload.get('total')}", flush=True)


        if fresh == 0:
            stale_pages += 1
            if stale_pages >= 3:
                print(f"[search] cursor {stale_pages}x0023 x0084 x0085 x0086 x0075 x0087 "
                      f"(x0083 {len(seen_ids)}). count={count}", file=sys.stderr, flush=True)
                return
        else:
            stale_pages = 0

        cursor = str(payload.get("cursor", "") or "")
        if cursor_file is not None:

            tmp = cursor_file.with_suffix(".tmp")
            tmp.write_text(cursor, encoding="utf-8")
            tmp.replace(cursor_file)
        if not cursor:
            return
        sleep(delay)


def iter_search_identifiers(query: str, rows: int, sort: str, delay: float):
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    page = 1
    rows = max(1, min(rows, 1000))
    while True:
        params: list[tuple[str, str]] = [
            ("q", query),
            ("fl[]", "identifier"),
            ("rows", str(rows)),
            ("page", str(page)),
            ("output", "json"),
        ]
        if sort:
            params.append(("sort[]", sort))
        url = IA_ADVANCED_SEARCH_URL + "?" + urllib.parse.urlencode(params)
        payload = fetch_json(url)
        docs = payload.get("response", {}).get("docs", [])
        if not docs:
            return
        for doc in docs:
            identifier = str(doc.get("identifier", "")).strip()
            if identifier:
                yield identifier
        page += 1
        sleep(delay)


def head_content_length(url: str, retries: int = 3) -> int | None:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={
                "User-Agent": "ia-cia-downloader"}, method="HEAD")
            with urllib.request.urlopen(request, timeout=60) as response:
                return int(response.headers.get("Content-Length") or 0)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            if exc.code not in RETRYABLE_HTTP or attempt == retries - 1:
                raise RuntimeError(f"HEAD x0004 {url}: HTTP {exc.code}") from exc
            sleep(backoff_seconds(attempt, exc.code in RATE_LIMIT_HTTP))
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries - 1:
                raise RuntimeError(f"HEAD x0004 {url}: {exc}") from exc
            sleep(backoff_seconds(attempt, False))
    return None


def derive_base_name(identifier: str, strip_prefix: str) -> str:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    if strip_prefix and identifier.startswith(strip_prefix):
        return identifier[len(strip_prefix):]
    return identifier


def head_gate_item(
    identifier: str,
    item_dir: Path,
    include_pdf: bool,
    delay: float,
    txt_bytes_min: int,
    txt_bytes_max: int,
    strip_prefix: str,
) -> dict[str, Any] | None:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    base = derive_base_name(identifier, strip_prefix)
    record: dict[str, Any] = {
        "identifier": identifier,
        "metadata_url": None,
        "item_dir": str(item_dir),
        "downloaded_files": [],
        "missing_suffixes": [],
        "errors": [],
        "status": "accepted",
        "txt_bytes": None,
        "gate": "head",
    }

    def file_url(name: str) -> str:
        return IA_DOWNLOAD_URL.format(identifier=urllib.parse.quote(identifier),
                                      filename=urllib.parse.quote(name))

    try:
        txt_size = head_content_length(file_url(f"{base}_djvu.txt"))
    except Exception as exc:  # pylint: disable=broad-exception-caught
        record["errors"].append({"stage": "head", "error": str(exc)})
        record["status"] = "error"
        return record

    if txt_size is None:
        return None
    record["txt_bytes"] = txt_size
    if txt_bytes_min and txt_size < txt_bytes_min or txt_bytes_max and txt_size > txt_bytes_max:
        record["status"] = "skipped_size"
        return record

    item_dir.mkdir(parents=True, exist_ok=True)
    suffixes = list(DEFAULT_FORMAT_SUFFIXES)
    if include_pdf:
        suffixes.append(".pdf")

    for suffix in suffixes:
        name = f"{base}{suffix}"
        target = item_dir / name
        if target.exists() and target.stat().st_size > 0:
            record["downloaded_files"].append(str(target))
            continue
        try:
            download_file(file_url(name), target)
            record["downloaded_files"].append(str(target))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if "HTTP 404" in str(exc):


                return None
            record["errors"].append(
                {"stage": "download", "file": name, "error": str(exc)})
        sleep(delay)

    return record


def download_item(
    identifier: str,
    item_dir: Path,
    include_pdf: bool,
    delay: float,
    txt_bytes_min: int = 0,
    txt_bytes_max: int = 0,
    gate_via: str = "metadata",
    strip_prefix: str = "",
) -> dict[str, Any]:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    if gate_via == "head":
        fast = head_gate_item(identifier, item_dir, include_pdf, delay,
                              txt_bytes_min, txt_bytes_max, strip_prefix)
        if fast is not None:
            return fast

    metadata_url = IA_METADATA_URL.format(identifier=urllib.parse.quote(identifier))
    record: dict[str, Any] = {
        "identifier": identifier,
        "metadata_url": metadata_url,
        "item_dir": str(item_dir),
        "downloaded_files": [],
        "missing_suffixes": [],
        "errors": [],
        "status": "accepted",
        "txt_bytes": None,
    }

    try:
        metadata = fetch_json(metadata_url)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        record["errors"].append({"stage": "metadata", "error": str(exc)})
        record["status"] = "error"
        return record

    files = metadata.get("files", [])
    file_names = [str(file_info.get("name", "")) for file_info in files]




    txt_name = select_file(file_names, "_djvu.txt")
    if not txt_name:
        record["status"] = "skipped_no_txt"
        return record
    txt_size = next(
        (int(f.get("size", 0) or 0) for f in files if str(f.get("name")) == txt_name), 0
    )
    record["txt_bytes"] = txt_size
    if txt_bytes_min and txt_size < txt_bytes_min or txt_bytes_max and txt_size > txt_bytes_max:
        record["status"] = "skipped_size"
        return record

    item_dir.mkdir(parents=True, exist_ok=True)
    suffixes = list(DEFAULT_FORMAT_SUFFIXES)
    if include_pdf:
        suffixes.append(".pdf")

    for suffix in suffixes:
        selected = select_file(file_names, suffix)
        if not selected:
            record["missing_suffixes"].append(suffix)
            continue

        target = item_dir / selected
        if target.exists() and target.stat().st_size > 0:
            record["downloaded_files"].append(str(target))
            continue

        url = IA_DOWNLOAD_URL.format(
            identifier=urllib.parse.quote(identifier),
            filename=urllib.parse.quote(selected),
        )
        try:
            download_file(url, target)
            record["downloaded_files"].append(str(target))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            record["errors"].append(
                {"stage": "download", "file": selected, "url": url, "error": str(exc)}
            )
        sleep(delay)

    return record


def select_file(file_names: list[str], suffix: str) -> str:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    candidates = [name for name in file_names if name.endswith(suffix)]
    if not candidates:
        return ""
    candidates.sort(key=lambda name: (len(name), name))
    return candidates[0]




RETRYABLE_HTTP = frozenset({408, 425, 429, 500, 502, 503, 504})
RATE_LIMIT_HTTP = frozenset({429, 503})


def backoff_seconds(attempt: int, rate_limited: bool) -> float:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    if rate_limited:
        return min(300.0, 5.0 * (2.0 ** attempt))
    return min(60.0, 1.5 * (attempt + 1))


def fetch_json(url: str, retries: int = 5) -> dict[str, Any]:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "ia-cia-downloader"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as exc:

            if exc.code not in RETRYABLE_HTTP or attempt == retries - 1:
                raise RuntimeError(f"Could not fetch JSON from {url}: HTTP {exc.code}") from exc
            sleep(backoff_seconds(attempt, exc.code in RATE_LIMIT_HTTP))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            if attempt == retries - 1:
                raise RuntimeError(f"Could not fetch JSON from {url}: {exc}") from exc
            sleep(backoff_seconds(attempt, False))
    return {}


def fetch_json_persistent(url: str, label: str, attempts: int = 40) -> dict[str, Any]:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return fetch_json(url, retries=2)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            last = exc
            wait = min(300.0, 10.0 * (1.6 ** min(attempt, 10)))
            print(f"[search-retry {attempt + 1}/{attempts}] {label}: {exc} -> {wait:.0f}s x0088",
                  file=sys.stderr, flush=True)
            sleep(wait)
    raise RuntimeError(f"search API x0089 {attempts} x008a x0004: {label}: {last}")


def download_file(url: str, target: Path, retries: int = 4) -> None:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    tmp = target.with_suffix(f"{target.suffix}.{os.getpid()}.tmp")
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "ia-cia-downloader"},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                with tmp.open("wb") as file:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        file.write(chunk)
            tmp.replace(target)
            return
        except urllib.error.HTTPError as exc:
            tmp.unlink(missing_ok=True)
            if exc.code not in RETRYABLE_HTTP or attempt == retries - 1:
                raise RuntimeError(f"Could not download {url}: HTTP {exc.code}") from exc
            sleep(backoff_seconds(attempt, exc.code in RATE_LIMIT_HTTP))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            tmp.unlink(missing_ok=True)
            if attempt == retries - 1:
                raise RuntimeError(f"Could not download {url}: {exc}") from exc
            sleep(backoff_seconds(attempt, False))


def sleep(seconds: float) -> None:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    if seconds > 0:
        time.sleep(seconds)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise
