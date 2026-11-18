#!/usr/bin/env python3
"""(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
from __future__ import annotations
import csv, json, sys
from collections import Counter
from pathlib import Path
import numpy as np

csv.field_size_limit(10**9)
LABELS = ["Unclassified", "Confidential", "Secret", "Top Secret"]; ORD = {l: i for i, l in enumerate(LABELS)}; B, SEED, MAX_TOKENS = 10000, 7701, 4096


def load(p: Path):
    with p.open(encoding="utf-8", newline="") as f:
        return {r["doc_id"]: {"gold": r["gold_label"], "pred": (r.get("pred_label") or "").strip(), "tokens": int(r.get("tokens") or 0)} for r in csv.DictReader(f)}


def cond_stats(base, v, ids):
    gold = np.array([base[i]["gold"] for i in ids]); bc = np.array([base[i]["pred"] == base[i]["gold"] for i in ids]); prot = np.isin(gold, LABELS[1:])
    corr = np.array([v[i]["pred"] == v[i]["gold"] for i in ids]); toU = np.array([v[i]["pred"] == "Unclassified" for i in ids])
    up = np.array([v[i]["pred"] in ORD and ORD[v[i]["pred"]] > ORD[v[i]["gold"]] for i in ids]); down = np.array([v[i]["pred"] in ORD and ORD[v[i]["pred"]] < ORD[v[i]["gold"]] for i in ids])
    return bc, prot, corr, bc & ~corr, bc & prot & toU, bc & up, bc & down


def pair(base, end, mid, ids=None):
    ids = sorted(set(base) & set(end) & set(mid)) if ids is None else sorted(ids); n = len(ids)
    bc, prot, ec, ea, ep, eu, ed = cond_stats(base, end, ids); _, _, mc, ma, mp_, mu, md = cond_stats(base, mid, ids)
    rng = np.random.default_rng(SEED); d_asr, d_pu = [], []
    for _ in range(B):
        idx = rng.integers(0, n, n); b = bc[idx].sum(); p = (bc & prot)[idx].sum()
        d_asr.append((ea[idx].sum() - ma[idx].sum()) / max(1, b)); d_pu.append((ep[idx].sum() - mp_[idx].sum()) / max(1, p))
    pct = lambda a, b: round(100 * a / max(1, b), 2)
    return {"n_pairs": n, "baseline_correct": int(bc.sum()), "baseline_protected_correct": int((bc & prot).sum()),
            "accuracy": {"baseline": pct(bc.sum(), n), "move_end": pct(ec.sum(), n), "move_middle": pct(mc.sum(), n)},
            "asr": {"move_end": {"num": int(ea.sum()), "den": int(bc.sum()), "pct": pct(ea.sum(), bc.sum()), "errors_up": int(eu.sum()), "errors_down": int(ed.sum())},
                    "move_middle": {"num": int(ma.sum()), "den": int(bc.sum()), "pct": pct(ma.sum(), bc.sum()), "errors_up": int(mu.sum()), "errors_down": int(md.sum())},
                    "end_minus_middle_pp": pct(ea.sum() - ma.sum(), bc.sum()), "end_minus_middle_ci95_pp": [round(float(100 * x), 2) for x in np.percentile(d_asr, [2.5, 97.5])]},
            "protected_to_U": {"move_end": {"num": int(ep.sum()), "den": int((bc & prot).sum()), "pct": pct(ep.sum(), (bc & prot).sum())},
                               "move_middle": {"num": int(mp_.sum()), "den": int((bc & prot).sum()), "pct": pct(mp_.sum(), (bc & prot).sum())},
                               "end_minus_middle_pp": pct(ep.sum() - mp_.sum(), (bc & prot).sum()), "end_minus_middle_ci95_pp": [round(float(100 * x), 2) for x in np.percentile(d_pu, [2.5, 97.5])]},
            "label_changed_vs_baseline": {"move_end": int(sum(end[i]["pred"] != base[i]["pred"] for i in ids)), "move_middle": int(sum(mid[i]["pred"] != base[i]["pred"] for i in ids))}}


def main():
    run = Path(sys.argv[1]); out = {"run": run.name, "models": {}}
    for sd in sorted(p for p in (run / "results").iterdir() if p.is_dir()):
        m = {"conditions": {}, "pairs": {}}
        for cd in sorted(p for p in sd.iterdir() if p.is_dir()):
            f = cd / "final/baseline_predictions.csv"
            if not f.exists(): continue
            rows = load(f); s = json.loads((cd / "final/merge_summary.json").read_text()) if (cd / "final/merge_summary.json").exists() else {}; t = json.loads((cd / "timing.json").read_text()) if (cd / "timing.json").exists() else {}
            golds = Counter(r["gold"] for r in rows.values()); maj = golds.most_common(1)[0]
            m["conditions"][cd.name] = {"n": len(rows), "accuracy": round(100 * sum(r["pred"] == r["gold"] for r in rows.values()) / len(rows), 2), "majority_label": maj[0], "majority_baseline_pct": round(100 * maj[1] / len(rows), 2),
                                        "recall": {l: round(100 * sum(r["pred"] == l for r in rows.values() if r["gold"] == l) / max(1, golds[l]), 2) if golds[l] else None for l in LABELS},
                                        "empty": sum(not r["pred"] for r in rows.values()), "parse_path": s.get("parse_path"), "prediction_sources": s.get("prediction_sources"), "docs_per_min": t.get("docs_per_min"), "truncation_lines": t.get("ollama_truncation_lines"), "gate": s.get("gate")}
        for src in ("frus", "wikileaks", "cia"):
            b, e, md = (sd / f"pos_{src}_{c}/final/baseline_predictions.csv" for c in ("base", "move_end", "move_middle"))
            if b.exists() and e.exists() and md.exists():
                B_, E_, M_ = load(b), load(e), load(md); m["pairs"][src] = pair(B_, E_, M_)
                keep = [i for i in set(B_) & set(E_) & set(M_) if max(B_[i]["tokens"], E_[i]["tokens"], M_[i]["tokens"]) < MAX_TOKENS]
                m["pairs"][src]["n_truncated_any"] = m["pairs"][src]["n_pairs"] - len(keep)
                if m["pairs"][src]["n_truncated_any"]: m["pairs"][src + "_no_trunc"] = pair(B_, E_, M_, keep)
        out["models"][sd.name] = m
    (run / "analysis").mkdir(exist_ok=True); (run / "analysis/position_move_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = [f"# x018e x018f x017e ({run.name})", "", "| x00ae | x0184 | x0181 | x00b2 x0185 | x018e | x0190 | ASR x018e (n/x012b; x0191/x0192) | ASR x0190 (x0191/x0192) | x018e−x0190 Δ [CI] | x010d→U x018e / x0190 (x012b) | Δ [CI] |", "|---|---|---:|---:|---:|---:|---|---|---|---|---|"]
    for slug, m in out["models"].items():
        for src, v in m["pairs"].items():
            a, p = v["asr"], v["protected_to_U"]
            md.append(f"| {slug} | {src} | {v['n_pairs']} | {v['accuracy']['baseline']} | {v['accuracy']['move_end']} | {v['accuracy']['move_middle']} | {a['move_end']['pct']} ({a['move_end']['num']}/{a['move_end']['den']}; {a['move_end']['errors_up']}/{a['move_end']['errors_down']}) | {a['move_middle']['pct']} ({a['move_middle']['errors_up']}/{a['move_middle']['errors_down']}) | {a['end_minus_middle_pp']} {a['end_minus_middle_ci95_pp']} | {p['move_end']['num']} / {p['move_middle']['num']} ({p['move_end']['den']}) | {p['end_minus_middle_pp']} {p['end_minus_middle_ci95_pp']} |")
    md += ["", "| x00ae | x0187 | n | x0185 | x0188 x018a | U/C/S/TS recall | x0029 x0161 | parse | x018b | x018c/x018d |", "|---|---|---:|---:|---:|---|---:|---|---:|---:|"]
    for slug, m in out["models"].items():
        for c, v in m["conditions"].items():
            md.append(f"| {slug} | {c} | {v['n']} | {v['accuracy']} | {v['majority_label']} {v['majority_baseline_pct']} | {'/'.join(str(v['recall'][l]) for l in LABELS)} | {v['empty']} | {v['parse_path']} | {v['truncation_lines']} | {v['docs_per_min']} |")
    (run / "analysis/position_move_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({s: {k: (v["asr"]["move_end"]["pct"], v["asr"]["move_middle"]["pct"], v["asr"]["end_minus_middle_pp"]) for k, v in m["pairs"].items()} for s, m in out["models"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
