#!/usr/bin/env python3
"""(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
from __future__ import annotations
import json, re, sys, time
from collections import Counter
from pathlib import Path

ROOT = Path("/path/to/workspace")
HERE = Path(__file__).resolve().parent
GRADE = re.compile(r"(?<![A-Za-z])(TOP\s{0,3}SECRET|SECRET|CONFIDENTIAL|UNCLASSIFIED)(?![A-Za-z])")
PROPER = re.compile(r"SECRET\s+SERVICE", re.I)
WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
PORT = re.compile(r"\(\s*(TS|S|C|U)\s*(?://|/|-|\s)\s*([A-Za-z0-9]{0,12})")
DISSEM = re.compile(r"^(N[FX]|NOFORN|FOUO|FOU0|F0UO|OC|ORCON|REL|RD|TK|SI|HCS|NC|PR|XF|X[0-9]|B[I1]8?O|LEOUO|HOUO|ATO|NFIB|WN|WNINTEL|OADR)", re.I)


def para(t, i):
    ls = t.rfind("\n", 0, i) + 1
    b = t[ls:i].strip()
    if b == "":
        return True
    if re.fullmatch(r"[0-9]{1,3}\s*[.,)]|[a-z]\s*[.,)]|[IVXivx]{1,4}\s*[.,)]", b):
        return True
    return bool(re.search(r"[.!?:;\"”]\s*$", b))


def f1_hits(t):
    n = 0
    for m in GRADE.finditer(t):
        ls = t.rfind("\n", 0, m.start()) + 1
        le = t.find("\n", m.end()); le = len(t) if le < 0 else le
        line = t[ls:le].strip()
        if len(WORD.findall(line)) > 8:
            b, a = line[:m.start() - ls], line[m.end() - ls:]
            if len(WORD.findall(b)) >= 3 and len(WORD.findall(a)) >= 3:
                continue
        if PROPER.search(line):
            continue
        n += 1
    return n


def f2_hits(t):
    n = 0
    for m in PORT.finditer(t):
        tail = m.group(2)
        if (DISSEM.match(tail) if tail else False) or (para(t, m.start()) and len(tail) <= 2):
            n += 1
    return n


def main() -> None:
    t0 = time.time()
    r4 = {}
    with (ROOT / "data/datasets/cia_benchmark/r4/benchmark.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                r4[str(d["identifier"])] = str(d.get("marker_masked_text", "") or "")
    s = Counter(); bad = []
    with (ROOT / "data/datasets/cia_benchmark/r5/benchmark.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line); did = str(d["identifier"])
            new = str(d.get("marker_masked_text", "") or ""); old = r4.get(did, "")
            s["docs"] += 1
            for tag, txt in (("r4", old), ("r5", new)):
                a, b = f1_hits(txt), f2_hits(txt)
                s[f"{tag}_f1"] += a; s[f"{tag}_f2"] += b
                s[f"{tag}_f1_docs"] += a > 0; s[f"{tag}_f2_docs"] += b > 0
            if d.get("r5_patch"):
                s["changed"] += 1

                sq_old = re.sub(r"\s+", "", old); sq_new = re.sub(r"\s+", "", new)
                it = iter(sq_old)
                if not all(c in it for c in sq_new):
                    s["not_subsequence"] += 1
                    if len(bad) < 5:
                        bad.append(did)
    rep = dict(generated="2026-09-09", docs=s["docs"], changed=s["changed"],
               f1_banner_grade=dict(r4_hits=s["r4_f1"], r5_hits=s["r5_f1"],
                                    r4_docs=s["r4_f1_docs"], r5_docs=s["r5_f1_docs"]),
               f2_portion=dict(r4_hits=s["r4_f2"], r5_hits=s["r5_f2"],
                               r4_docs=s["r4_f2_docs"], r5_docs=s["r5_f2_docs"]),
               subsequence_violations=s["not_subsequence"], violation_examples=bad,
               elapsed_s=round(time.time() - t0, 1))
    (HERE / "s3b_verify_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"x0099 {s['docs']:,} (x00f8 {s['changed']:,})\n")
    print(f"{'x00b2':22s} {'r4':>10s} {'r5':>10s} {'x00fe':>10s}")
    for lab, a, b in (("F1 x00fb x00ff x0100", s['r4_f1'], s['r5_f1']), ("F1 x0099", s['r4_f1_docs'], s['r5_f1_docs']),
                      ("F2 x0101 x0100", s['r4_f2'], s['r5_f2']), ("F2 x0099", s['r4_f2_docs'], s['r5_f2_docs'])):
        print(f"{lab:22s} {a:10,} {b:10,} {(a-b)/max(a,1)*100:9.1f}%")
    print(f"\n x0102 x0103(x00ad x0104): {s['not_subsequence']} {bad if bad else ''}")
    print(f"({rep['elapsed_s']}s)")


if __name__ == "__main__":
    main()
