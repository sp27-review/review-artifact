#!/usr/bin/env python3
"""(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
from __future__ import annotations
import csv, json, sys
from collections import Counter
from pathlib import Path
import numpy as np

csv.field_size_limit(10**9)
LABELS = ["Unclassified", "Confidential", "Secret", "Top Secret"]
B = 10000; SEED = 7701; MAX_TOKENS = 4096


def load(p: Path) -> dict[str, dict]:
    with p.open(encoding="utf-8", newline=""):
        pass
    with p.open(encoding="utf-8", newline="") as f:
        return {r["doc_id"]: {"gold": r["gold_label"], "pred": (r.get("pred_label") or "").strip(), "src": r.get("prediction_source", ""), "tokens": int(r.get("tokens") or 0)} for r in csv.DictReader(f)}


def recall(rows):
    out = {}
    for l in LABELS:
        g = [r for r in rows.values() if r["gold"] == l]
        out[l] = {"n": len(g), "recall": round(100 * sum(r["pred"] == l for r in g) / len(g), 2) if g else None}
    return out


def pair_metrics(base: dict, edit: dict, ctrl: dict | None) -> dict:
    ids = sorted(set(base) & set(edit) & (set(ctrl) if ctrl else set(edit)))
    gold = np.array([base[i]["gold"] for i in ids]); bc = np.array([base[i]["pred"] == base[i]["gold"] for i in ids])
    prot = np.isin(gold, ["Confidential", "Secret", "Top Secret"])
    def cond_stats(v):
        corr = np.array([v[i]["pred"] == v[i]["gold"] for i in ids]); toU = np.array([v[i]["pred"] == "Unclassified" for i in ids])
        asr_num = bc & ~corr; pu_num = bc & prot & toU
        return corr, asr_num, pu_num
    ec, ea, ep = cond_stats(edit)
    out = {"n_pairs": len(ids), "baseline_correct": int(bc.sum()), "baseline_protected_correct": int((bc & prot).sum()),
           "accuracy": {"baseline": round(100 * bc.mean(), 2), "edit": round(100 * ec.mean(), 2)},
           "asr": {"edit": {"num": int(ea.sum()), "den": int(bc.sum()), "pct": round(100 * ea.sum() / max(1, bc.sum()), 2)}},
           "protected_to_U": {"edit": {"num": int(ep.sum()), "den": int((bc & prot).sum()), "pct": round(100 * ep.sum() / max(1, (bc & prot).sum()), 2)}},
           "label_changed_vs_baseline": {"edit": int(sum(edit[i]["pred"] != base[i]["pred"] for i in ids))}}
    if ctrl:
        cc, ca, cp = cond_stats(ctrl)
        out["accuracy"]["control"] = round(100 * cc.mean(), 2)
        out["asr"]["control"] = {"num": int(ca.sum()), "den": int(bc.sum()), "pct": round(100 * ca.sum() / max(1, bc.sum()), 2)}
        out["protected_to_U"]["control"] = {"num": int(cp.sum()), "den": int((bc & prot).sum()), "pct": round(100 * cp.sum() / max(1, (bc & prot).sum()), 2)}
        out["label_changed_vs_baseline"]["control"] = int(sum(ctrl[i]["pred"] != base[i]["pred"] for i in ids))
        rng = np.random.default_rng(SEED); n = len(ids); d_asr, d_pu = [], []
        for _ in range(B):
            idx = rng.integers(0, n, n); b = bc[idx]; p = (bc & prot)[idx]
            d_asr.append((ea[idx].sum() - ca[idx].sum()) / max(1, b.sum())); d_pu.append((ep[idx].sum() - cp[idx].sum()) / max(1, p.sum()))
        out["asr"]["edit_minus_control_pp"] = round(100 * (ea.sum() - ca.sum()) / max(1, bc.sum()), 2)
        out["asr"]["edit_minus_control_ci95_pp"] = [round(float(100 * x), 2) for x in np.percentile(d_asr, [2.5, 97.5])]
        out["protected_to_U"]["edit_minus_control_pp"] = round(100 * (ep.sum() - cp.sum()) / max(1, (bc & prot).sum()), 2)
        out["protected_to_U"]["edit_minus_control_ci95_pp"] = [round(float(100 * x), 2) for x in np.percentile(d_pu, [2.5, 97.5])]
    return out


def cond_summary(run: Path, slug: str, cond: str) -> dict | None:
    f = run / "results" / slug / cond / "final/baseline_predictions.csv"
    if not f.exists(): return None
    rows = load(f); ms = run / "results" / slug / cond / "final/merge_summary.json"; tm = run / "results" / slug / cond / "timing.json"
    s = json.loads(ms.read_text()) if ms.exists() else {}; t = json.loads(tm.read_text()) if tm.exists() else {}
    golds = Counter(r["gold"] for r in rows.values()); maj = golds.most_common(1)[0]
    return {"rows": rows, "n": len(rows), "accuracy": round(100 * sum(r["pred"] == r["gold"] for r in rows.values()) / len(rows), 2),
            "majority_label": maj[0], "majority_baseline_pct": round(100 * maj[1] / len(rows), 2), "recall": recall(rows),
            "pred_counts": dict(Counter(r["pred"] or "(empty)" for r in rows.values())), "empty": sum(not r["pred"] for r in rows.values()),
            "parse_path": s.get("parse_path"), "stop_reason": s.get("stop_reason"), "prediction_sources": s.get("prediction_sources"), "mapping_per_row": s.get("mapping_per_row"),
            "docs_per_min": t.get("docs_per_min"), "truncation_lines": t.get("ollama_truncation_lines"), "gate": s.get("gate")}


def main() -> None:
    run = Path(sys.argv[1]); out = {"run": run.name, "models": {}}
    for sd in sorted(p for p in (run / "results").iterdir() if p.is_dir()):
        slug = sd.name; m = {"conditions": {}, "pairs": {}}
        for cd in sorted(p for p in sd.iterdir() if p.is_dir()):
            cs = cond_summary(run, slug, cd.name)
            if cs: m["conditions"][cd.name] = {k: v for k, v in cs.items() if k != "rows"}
        def rows(c):
            cs = cond_summary(run, slug, c); return cs["rows"] if cs else None
        def no_trunc(name, b, e, c):
            ids = set(b) & set(e) & (set(c) if c else set(e)); keep = {i for i in ids if max(b[i]["tokens"], e[i]["tokens"], (c[i]["tokens"] if c else 0)) < MAX_TOKENS}
            m["pairs"][name]["n_truncated_any"] = len(ids) - len(keep)
            if keep != ids: m["pairs"][name + " (no_trunc)"] = pair_metrics({k: v for k, v in b.items() if k in keep}, {k: v for k, v in e.items() if k in keep}, {k: v for k, v in c.items() if k in keep} if c else None)
        for src in ("wikileaks", "frus", "cia"):
            b, e, c = rows(f"{src}_base"), rows(f"{src}_combo"), rows(f"{src}_combo_ctrl")
            if b and e: m["pairs"][f"{src}/combo"] = pair_metrics(b, e, c); no_trunc(f"{src}/combo", b, e, c)
            if src != "cia":
                e2, c2 = rows(f"{src}_single"), rows(f"{src}_single_ctrl")
                if b and e2: b2 = {k: v for k, v in b.items() if k in e2}; m["pairs"][f"{src}/single"] = pair_metrics(b2, e2, c2); no_trunc(f"{src}/single", b2, e2, c2)
        b, e, c = rows("orig_wikileaks_base"), rows("orig_wikileaks_combo"), rows("orig_wikileaks_combo_ctrl")
        if b and e:
            m["pairs"]["wikileaks/combo (original ClassifyIt symbols)"] = pair_metrics(b, e, c)
            pb = rows("wikileaks_base")
            if pb:
                ids = sorted(set(pb) & set(b)); m["orig_vs_patched_baseline_agreement_pct"] = round(100 * sum(pb[i]["pred"] == b[i]["pred"] for i in ids) / len(ids), 2)
        out["models"][slug] = m
    (run / "analysis").mkdir(exist_ok=True)
    (run / "analysis/zs_format_edit_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = [f"# x017d 1 — x017e P0 + FT3 x017f x0180 ({run.name})", "", "## x0181 x0182 (x00b2→x0180 / x00b2→x0183)", "",
          "| x00ae | x0184/x0180 | x0181 | x00b2 x0185 | x0180 | x0183 | ASR x0180 (n/x012b) | ASR x0183 | Δ(pp) [CI] | x010d→U x0180 (n/x012b) | x010d→U x0183 | Δ(pp) [CI] |", "|---|---|---:|---:|---:|---:|---|---|---|---|---|---|"]
    for slug, m in out["models"].items():
        for k, v in m["pairs"].items():
            a, p = v["asr"], v["protected_to_U"]; ce = v["accuracy"].get("control", "—")
            md.append(f"| {slug} | {k} | {v['n_pairs']} | {v['accuracy']['baseline']} | {v['accuracy']['edit']} | {ce} | {a['edit']['pct']} ({a['edit']['num']}/{a['edit']['den']}) | {a.get('control',{}).get('pct','—')} | {a.get('edit_minus_control_pp','—')} {a.get('edit_minus_control_ci95_pp','')} | {p['edit']['pct']} ({p['edit']['num']}/{p['edit']['den']}) | {p.get('control',{}).get('pct','—')} | {p.get('edit_minus_control_pp','—')} {p.get('edit_minus_control_ci95_pp','')} |")
    md += ["", "## x0186", "", "| x00ae | x0187 | n | x0185 | x0188 x0189 x018a | U/C/S/TS recall | x0029 x0161 | parse | x018b | x018c/x018d |", "|---|---|---:|---:|---:|---|---:|---|---:|---:|"]
    for slug, m in out["models"].items():
        for c, v in m["conditions"].items():
            r = v["recall"]; rec = "/".join(str(r[l]["recall"]) for l in LABELS)
            md.append(f"| {slug} | {c} | {v['n']} | {v['accuracy']} | {v['majority_label']} {v['majority_baseline_pct']} | {rec} | {v['empty']} | {v['parse_path']} | {v['truncation_lines']} | {v['docs_per_min']} |")
    (run / "analysis/zs_format_edit_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({s: {k: (v["asr"]["edit"]["pct"], v["asr"].get("control", {}).get("pct"), v["protected_to_U"]["edit"]["pct"], v["protected_to_U"].get("control", {}).get("pct")) for k, v in m["pairs"].items()} for s, m in out["models"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
