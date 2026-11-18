#!/usr/bin/env python3
"""(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
ROOT = Path('/path/to/workspace'); RUN = Path(__file__).resolve().parents[1]
YR = ROOT / 'experiments/Y_small_models_robustness/runs/run_20261112_r5ext_cia_external_5small_seed7701_training'
REAGG = ROOT / 'experiments/_reaggregate_strict_combo_20261115/per_document.csv'
SMALL = [('Llama-3.1-8B', 'llama8b'), ('Qwen2.5-7B', 'qwen7b'), ('Mistral-7B', 'mistral7b'), ('Gemma-2-9B', 'gemma9b'), ('Phi-3.5-mini', 'phi35')]
CONDS = ['x0193', 'N1 x0170', 'N2 x0170', 'N2 x0194+x0112']; SRCS = ['wikileaks', 'frus', 'cia']; DISP = {'Unclassified': 0, 'Confidential': 1, 'Secret': 2, 'Top Secret': 3}
B, SEED = 10000, 7701; csv.field_size_limit(10**9)


def load(p):
    with open(p, encoding='utf-8', newline='') as h: return {r['doc_id']: (DISP.get((r.get('pred_label') or '').strip()), DISP[r['gold_label']]) for r in csv.DictReader(h)}


def cdir(key, cond):
    return {'x0193': YR / 'results' / f'{key}_cia_ext_seed7701', 'N1 x0170': RUN / 'results' / f'{key}_cia_ext_N1_inferonly_seed7701', 'N2 x0170': RUN / 'results' / f'{key}_cia_ext_N2_inferonly_seed7701', 'N2 x0194+x0112': RUN / 'results' / f'{key}_cia_ext_N2_seed7701'}[cond]


def boot(num_a, num_c, den):
    n = len(den); rng = np.random.default_rng(SEED); out = []
    for _ in range(10):
        idx = rng.integers(0, n, (B // 10, n)); out.append(100 * (num_a[idx].sum(1) - num_c[idx].sum(1)) / np.maximum(den[idx].sum(1), 1))
    d = np.concatenate(out); return dict(point=round(100 * (num_a.sum() - num_c.sum()) / max(1, den.sum()), 2), ci95=[round(float(x), 2) for x in np.percentile(d, [2.5, 97.5])], den=int(den.sum()))


def triple(base, att, ctl):
    ids = sorted(set(base) & set(att) & set(ctl)); assert all(base[i][0] is not None and att[i][0] is not None and ctl[i][0] is not None for i in ids)
    g = np.array([base[i][1] for i in ids]); P = dict(base=np.array([base[i][0] for i in ids]), attack=np.array([att[i][0] for i in ids]), control=np.array([ctl[i][0] for i in ids]))
    prot = g > 0; isU = g == 0; corr = {k: P[k] == g for k in P}; den = prot & corr['base']
    miss = {k: prot & (P[k] == 0) for k in P}; over = {k: isU & (P[k] > 0) for k in P}
    return dict(n=len(ids), n_protected=int(prot.sum()), n_U=int(isU.sum()), correct={k: int(corr[k].sum()) for k in P}, accuracy={k: round(100 * corr[k].mean(), 2) for k in P},
                protected_missed={k: int(miss[k].sum()) for k in P}, overprotect={k: int(over[k].sum()) for k in P},
                protected_missed_diff=boot(miss['attack'], miss['control'], prot), overprotect_diff=boot(over['attack'], over['control'], isU) if isU.sum() else None,
                P=int(den.sum()), protected_to_U=dict(attack=int((den & (P['attack'] == 0)).sum()), control=int((den & (P['control'] == 0)).sum())), protected_to_U_diff=boot(den & (P['attack'] == 0), den & (P['control'] == 0), den),
                label_changed=dict(attack=int((P['attack'] != P['base']).sum()), control=int((P['control'] != P['base']).sum())),
                new_errors=dict(attack=int((corr['base'] & ~corr['attack']).sum()), control=int((corr['base'] & ~corr['control']).sum())), new_errors_diff=boot(corr['base'] & ~corr['attack'], corr['base'] & ~corr['control'], corr['base']))


def position(d, src):
    ps = [d / f'pos_{src}_{c}/final/baseline_predictions.csv' for c in ('base', 'move_end', 'move_middle')]
    if not all(p.exists() for p in ps): return None
    base, end, mid = map(load, ps); ids = sorted(set(base) & set(end) & set(mid)); g = np.array([base[i][1] for i in ids]); pb, pe, pm = (np.array([x[i][0] for i in ids]) for x in (base, end, mid))
    bc = pb == g; e = bc & (pe != g); m = bc & (pm != g); prot = bc & (g > 0)
    return dict(n=len(ids), baseline_correct=int(bc.sum()), accuracy=dict(base=round(100 * bc.mean(), 2), end=round(100 * (pe == g).mean(), 2), middle=round(100 * (pm == g).mean(), 2)), end_new_errors=int(e.sum()), middle_new_errors=int(m.sum()), diff=boot(e, m, bc),
                end_up=int((e & (pe > g)).sum()), end_down=int((e & (pe < g)).sum()), protected_to_U=dict(end=int((prot & (pe == 0)).sum()), middle=int((prot & (pm == 0)).sum()), den=int(prot.sum())))


def main():
    res = dict(bootstrap=dict(B=B, seed=SEED, unit='document pairs (paired resampling)'), models={}, position={})
    for label, key in SMALL:
        res['models'][label] = {c: {s: triple(*[load(cdir(key, c) / f'{s}_{x}/final/baseline_predictions.csv') for x in ('base', 'combo', 'combo_ctrl')]) for s in SRCS if (cdir(key, c) / f'{s}_combo_ctrl/final/baseline_predictions.csv').exists()} for c in CONDS}
        res['position'][label] = {c: {s: position(cdir(key, c), s) for s in SRCS} for c in ('x0193', 'N2 x0194+x0112')}
    L = {'U': 0, 'C': 1, 'S': 2, 'TS': 3}; base, att, ctl = {}, {}, {}
    with open(REAGG, encoding='utf-8', newline='') as h:
        for r in csv.DictReader(h):
            if r['model'] == 'Llama 3.3 70B' and r['source'] == 'wikileaks' and r['in_E'] == '1': g = L[r['true_label']]; base[r['doc_id']] = (L[r['baseline_pred']], g); att[r['doc_id']] = (L[r['attack_pred']], g); ctl[r['doc_id']] = (L[r['control_pred']], g)
    m70 = {'x0193': {'wikileaks': triple(base, att, ctl)}}
    for c, N in (('N1 x0170', 'N1'), ('N2 x0170', 'N2')):
        d = RUN / 'results' / f'llama70b_cia_only_{N}_inferonly_seed7701'; m70[c] = {'wikileaks': triple(*[load(d / f'wikileaks_{x}/final/baseline_predictions.csv') for x in ('base', 'combo', 'combo_ctrl')])}
    res['models']['Llama 3.3 70B (CIA x0194)'] = m70
    (RUN / 'analysis/norm_same_docs.json').write_text(json.dumps(res, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    f = lambda d: f"{d['point']:+.2f} [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]" if d else ''
    md = ['# x00a9 x017d — x00c9 x0099 x00b2 x0195 (2026-11-17)', '', 'x015d x0196 x0180 x0181 x0197 E x0198 x017f x0199 x019a(WikiLeaks 2,528 = x010d 1,136 + U 1,392; FRUS 431; CIA 597). B = x00b2(x019b), A = x0180(x0029 x019c x019d+x019e x019f), C = x0183. x011a x01a0 x015a x01a1 x01a2 x01a3 x0161. x01a4 = A − C, 95% CI x0099 x0181 bootstrap 10,000·seed 7701. x01a5 x01a6 x01a7.', '']
    for s, title in (('wikileaks', 'WikiLeaks E=2,528 (x010d 1,136 · U 1,392)'), ('frus', 'FRUS E=431'), ('cia', 'CIA E=597')):
        md += [f'## {title}', '', '| x00ae | x0187 | x0160 x002e B / A / C (x0185 %) | x010d x008b B / A / C | x008b A−C pp [CI] | x01a8 B / A / C | x01a9 A−C pp [CI] | x010d→U A / C (P) | x010d→U A−C pp [CI] | x004d x00f8 A / C | x01aa x01ab A / C | x01aa x01ab A−C pp [CI] |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
        for label, m in res['models'].items():
            for c in CONDS:
                v = m.get(c, {}).get(s)
                if not v: continue
                md.append(f"| {label} | {c} | {v['correct']['base']} / {v['correct']['attack']} / {v['correct']['control']} ({v['accuracy']['base']} / {v['accuracy']['attack']} / {v['accuracy']['control']}) | {v['protected_missed']['base']} / {v['protected_missed']['attack']} / {v['protected_missed']['control']} (/{v['n_protected']}) | {f(v['protected_missed_diff'])} | {v['overprotect']['base']} / {v['overprotect']['attack']} / {v['overprotect']['control']} (/{v['n_U']}) | {f(v['overprotect_diff'])} | {v['protected_to_U']['attack']} / {v['protected_to_U']['control']} ({v['P']}) | {f(v['protected_to_U_diff'])} | {v['label_changed']['attack']} / {v['label_changed']['control']} | {v['new_errors']['attack']} / {v['new_errors']['control']} (/{v['correct']['base']}) | {f(v['new_errors_diff'])} |")
        md.append('')
    md += ['## x018e x018f — x018e vs x0190 (x00c9 x0141 x018f; x01aa x01ab = x00b2 x0160 → x01ac)', '', '| x00ae | x0187 | x0106 | x0181 | x00b2 x0160(x012b) | x018e x01aa x01ab | x0190 x01aa x01ab | x018e−x0190 pp [CI] | x018e x01ab x0191 / x0192 | x010d→U x018e / x0190 (x012b) | x0185 x00b2 / x018e / x0190 |', '|---|---|---|---:|---:|---:|---:|---|---|---|---|']
    for label, m in res['position'].items():
        for c, bys in m.items():
            for s in ('frus', 'wikileaks', 'cia'):
                v = bys.get(s)
                if v: md.append(f"| {label} | {c} | {s} | {v['n']} | {v['baseline_correct']} | {v['end_new_errors']} | {v['middle_new_errors']} | {f(v['diff'])} | {v['end_up']} / {v['end_down']} | {v['protected_to_U']['end']} / {v['protected_to_U']['middle']} ({v['protected_to_U']['den']}) | {v['accuracy']['base']} / {v['accuracy']['end']} / {v['accuracy']['middle']} |")
    (RUN / 'analysis/norm_same_docs.md').write_text('\n'.join(md) + '\n', encoding='utf-8'); print('written')


if __name__ == '__main__':
    main()
