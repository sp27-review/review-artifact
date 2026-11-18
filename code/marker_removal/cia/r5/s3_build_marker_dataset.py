#!/usr/bin/env python3
"""(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
from __future__ import annotations
import csv, hashlib, importlib.util, json, os, sys, time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

ROOT = Path("/path/to/workspace")
HERE = Path(__file__).resolve().parent
SRC = ROOT / "data/datasets/cia_benchmark/r4/benchmark.jsonl"
OUT_DIR = ROOT / "data/datasets/cia_benchmark/r5"
OUT = OUT_DIR / "benchmark.jsonl"
WORKERS = int(os.environ.get("S3_WORKERS", "16"))

_spec = importlib.util.spec_from_file_location("s1", HERE / "s1_masking_rules_r5.py")
S1 = importlib.util.module_from_spec(_spec); sys.modules["s1"] = S1; _spec.loader.exec_module(S1)


def process(line: str):
    d = json.loads(line)
    text = str(d.get("marker_masked_text", "") or "")
    if not text:
        return json.dumps(d, ensure_ascii=False), None
    new, st = S1.scrub_r5(text)
    n = st["portion"] + st["banner"]
    if n:
        d["marker_masked_text"] = new
        d["r5_patch"] = dict(portion_removed=st["portion"], banner_removed=st["banner"],
                             chars_removed=st["chars"],
                             r4_len=len(text), r5_len=len(new))
        return json.dumps(d, ensure_ascii=False), dict(
            doc_id=str(d["identifier"]), label=str(d.get("label", "?")),
            portion_removed=st["portion"], banner_removed=st["banner"],
            chars_removed=st["chars"], r4_len=len(text), r5_len=len(new))
    return json.dumps(d, ensure_ascii=False), None


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = SRC.read_text(encoding="utf-8").splitlines()
    print(f"r4 x00f5 {len(lines):,} x00f6 ({time.time()-t0:.0f}s)", flush=True)
    changed, stats = [], Counter()
    h = hashlib.sha256()
    with Pool(WORKERS) as pool, OUT.open("w", encoding="utf-8") as f:
        for out_line, ch in pool.imap(process, lines, chunksize=200):
            f.write(out_line + "\n")
            h.update((out_line + "\n").encode("utf-8"))
            stats["docs"] += 1
            if ch:
                changed.append(ch)
                stats["changed"] += 1
                stats["portion"] += ch["portion_removed"]
                stats["banner"] += ch["banner_removed"]
                stats["chars"] += ch["chars_removed"]
    with (HERE / "s3_changed_docs.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(changed[0])); w.writeheader(); w.writerows(changed)
    by_label = Counter(c["label"] for c in changed)
    report = dict(generated="2026-09-09", source=str(SRC.relative_to(ROOT)), output=str(OUT.relative_to(ROOT)),
                  rules="s1_masking_rules_r5.py (x00f7 12, s2 6 x000e x000f 200/200)",
                  docs=stats["docs"], changed_docs=stats["changed"],
                  changed_share=round(stats["changed"] / stats["docs"], 4),
                  spans_removed=dict(portion=stats["portion"], banner=stats["banner"]),
                  chars_removed=stats["chars"],
                  chars_removed_mean_per_changed_doc=round(stats["chars"] / max(stats["changed"], 1), 1),
                  changed_by_label=dict(by_label),
                  output_sha256=h.hexdigest(), output_bytes=OUT.stat().st_size,
                  workers=WORKERS, elapsed_s=round(time.time() - t0, 1))
    (HERE / "s3_build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"x0099 {stats['docs']:,} | x00f8 {stats['changed']:,} ({stats['changed']/stats['docs']*100:.2f}%)")
    print(f"x00cc x00f9 x00fa {stats['portion']:,} / x00fb {stats['banner']:,} | x00fc {stats['chars']:,}")
    print(f"x00fd x00f8: {dict(by_label)}")
    print(f"x00e9 {OUT.relative_to(ROOT)} ({report['output_bytes']/1048576:.0f}MB)")
    print(f"sha256 {report['output_sha256'][:32]}…  ({report['elapsed_s']}s, {WORKERS})")


if __name__ == "__main__":
    main()
