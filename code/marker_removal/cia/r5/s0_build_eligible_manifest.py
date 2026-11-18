#!/usr/bin/env python3
"""(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
from __future__ import annotations
import json, time
from collections import Counter
from pathlib import Path

ROOT = Path("/path/to/workspace")
OUT = Path(__file__).resolve().parent
SRC = ROOT / "data/intermediate/cia_reading_room/manifests/high_conf_labels_v5full_gate08.jsonl"
OCR = ROOT / "data/raw/cia_reading_room/ocr"
OLD_PREFIX = "_experiments/main2_IA_CIA/data/ia_cia_readingroom/raw_ocr"


def remap(path: str, identifier: str, base: str, suffix: str) -> str:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    return str((OCR / identifier / f"{base}{suffix}").relative_to(ROOT))


def main() -> None:
    t0 = time.time()
    rows, stats = [], Counter()
    missing = []
    with SRC.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            stats["input"] += 1
            ident, base = str(d["identifier"]), str(d["base"])
            for field, suffix in (("xml_path", "_djvu.xml"), ("txt_path", "_djvu.txt"), ("hocr_path", "_hocr.html")):
                old = str(d.get(field, ""))
                if old.startswith(OLD_PREFIX):
                    stats[f"remapped_{field}"] += 1
                d[field] = remap(old, ident, base, suffix)
            xml = ROOT / d["xml_path"]
            if xml.exists():
                stats["xml_found"] += 1
            else:
                stats["xml_missing"] += 1
                if len(missing) < 20:
                    missing.append(dict(identifier=ident, expected=d["xml_path"]))
            rows.append(d)
    out_path = OUT / "eligible_v5full_r5.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for d in rows:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    report = dict(generated="2026-09-08", source=str(SRC.relative_to(ROOT)),
                  output=str(out_path.relative_to(ROOT)), ocr_root=str(OCR.relative_to(ROOT)),
                  stats=dict(stats), missing_examples=missing,
                  ready=stats["xml_missing"] == 0, elapsed_s=round(time.time() - t0, 1))
    (OUT / "s0_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"x00af {stats['input']:,}")
    print(f"  x0064 x00f2 xml/txt/hocr = {stats['remapped_xml_path']:,}/{stats['remapped_txt_path']:,}/{stats['remapped_hocr_path']:,}")
    print(f"  XML x00f3 {stats['xml_found']:,}  x008b {stats['xml_missing']:,}")
    for m in missing[:5]:
        print("   x008b:", m["expected"])
    print(f"  → {out_path.name} x00f4, ready={report['ready']} ({report['elapsed_s']}s)")


if __name__ == "__main__":
    main()
