#!/usr/bin/env python3
"""(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
from __future__ import annotations
import csv, glob, json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path('/path/to/workspace'); OUT = Path(__file__).resolve().parent
H = ROOT / 'experiments/H_format_edit_screen/runs'
EXT70 = H / 'run_20261102_r5_fullpool_firstseed/results/{m}/external/documents'
CIA70 = H / 'run_20261104_r5_fullpool_seed_repeats/seed_runs/7701/results/{m}/cia/documents'
PILOT = H / 'run_20261031_r5_external_nocopy_diagnostics/external_manifest.json'
E1 = H / 'run_20261110_ft3pairs3556_3model_P0_zeroshot_format_edit/results'
Y = ROOT / 'experiments/Y_small_models_robustness/runs/run_20261112_r5ext_cia_external_5small_seed7701_training/results'
SC = 'selected_combo:blank_collapse_native+capital_title_case'; RC = 'random:' + SC
LAB = ['U', 'C', 'S', 'TS']; DISP = {'Unclassified': 'U', 'Confidential': 'C', 'Secret': 'S', 'Top Secret': 'TS', '': ''}
SOURCES = ['wikileaks', 'frus', 'cia']; B, SEED = 10000, 7701
MODELS = [  # (model_label, train_setting, kind, key)
    ('Llama 3.3 70B', 'cia_only_70b', '70b', 'llama'), ('Qwen 2.5 72B', 'cia_only_70b', '70b', 'qwen'), ('Nemotron 70B', 'cia_only_70b', '70b', 'nemotron'),
    ('Llama-3.1-8B', 'cia_ext_small', 'y', 'llama8b_cia_ext_seed7701'), ('Qwen2.5-7B', 'cia_ext_small', 'y', 'qwen7b_cia_ext_seed7701'), ('Mistral-7B', 'cia_ext_small', 'y', 'mistral7b_cia_ext_seed7701'),
    ('Gemma-2-9B', 'cia_ext_small', 'y', 'gemma9b_cia_ext_seed7701'), ('Phi-3.5-mini', 'cia_ext_small', 'y', 'phi35_cia_ext_seed7701'),
    ('Llama 3.3 70B (zero-shot)', 'zero_shot_70b', 'zs', 'llama33_70b'), ('Qwen 2.5 72B (zero-shot)', 'zero_shot_70b', 'zs', 'qwen25_72b'), ('Nemotron 70B (zero-shot)', 'zero_shot_70b', 'zs', 'nemotron_70b'),
    ('Llama-3.1-8B (CIA only)', 'cia_only_small', 'y', 'llama8b_cia_only_seed7701'), ('Qwen2.5-7B (CIA only)', 'cia_only_small', 'y', 'qwen7b_cia_only_seed7701')]
MAIN = [m for m in MODELS if m[1] != 'cia_only_small']


def load70(model):
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    rows = {}; na = Counter()
    for pattern in (EXT70, CIA70):
        for f in glob.glob(str(pattern).format(m=model) + '/*.json'):
            d = json.load(open(f)); c = {x['name']: x for x in d['conditions']}; a = c.get(SC); r = c.get(RC)
            if not a or a.get('status') != 'valid': na[d['dataset']] += 1; continue
            reason = []
            if d['clean'].get('status') != 'valid': reason.append('clean_invalid')
            if not r or r.get('status') != 'valid': reason.append('control_unavailable')
            if model in (a.get('length_excluded_models') or []) or (r and model in (r.get('length_excluded_models') or [])): reason.append('length_excluded')
            rows[d['identifier']] = dict(source=d['dataset'], gold=LAB[d['gold']], base=LAB[d['clean']['prediction']] if d['clean'].get('prediction') is not None else '',
                                         attack=LAB[a['prediction']] if a.get('prediction') is not None else '', control=LAB[r['prediction']] if r and r.get('prediction') is not None else '',
                                         in_E=not reason, reason=';'.join(reason))
    return rows, na


def load_csv(p):
    csv.field_size_limit(10**9)
    with open(p, encoding='utf-8', newline='') as h: return {r['doc_id']: DISP.get((r.get('pred_label') or '').strip(), (r.get('pred_label') or '').strip()) for r in csv.DictReader(h)}


def load_cond(kind, key, universe):
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    base = Y if kind == 'y' else E1; rows = {}
    for src in SOURCES:
        pb, pa, pc = (base / key / f'{src}_{c}/final/baseline_predictions.csv' for c in ('base', 'combo', 'combo_ctrl'))
        if not (pb.exists() and pa.exists() and pc.exists()): continue
        b, a, c = load_csv(pb), load_csv(pa), load_csv(pc)
        for i, u in universe.items():
            if u['source'] != src: continue
            if i in b and i in a and i in c:
                rows[i] = dict(base=b[i], attack=a[i], control=c[i], in_E=bool(b[i] and a[i] and c[i]), reason='' if (b[i] and a[i] and c[i]) else 'empty_prediction')
            else:
                rows[i] = dict(base=b.get(i, ''), attack=a.get(i, ''), control=c.get(i, ''), in_E=False, reason='not_in_input' + (';control_unavailable' if 'control_unavailable' in u['reason'] else ''))
    return rows


def prot(l): return l in ('C', 'S', 'TS')


def agg(rows):
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    n = len(rows); out = {'n_E': n}
    if n == 0: return out
    g = np.array([r['gold'] for r in rows]); P = {k: np.array([r[k] for r in rows]) for k in ('base', 'attack', 'control')}
    out['confusion'] = {k: {gl: {pl: int(((g == gl) & (P[k] == pl)).sum()) for pl in LAB} for gl in LAB} for k in P}
    isprot = np.isin(g, ['C', 'S', 'TS']); isU = g == 'U'
    out['n_protected'] = int(isprot.sum()); out['n_gold_U'] = int(isU.sum())
    corr = {k: P[k] == g for k in P}
    out['accuracy'] = {k: round(100 * corr[k].mean(), 2) for k in P}

    out['protected_missed'] = {k: {'n': int((isprot & (P[k] == 'U')).sum()), 'den': int(isprot.sum()), 'pct': round(100 * (isprot & (P[k] == 'U')).sum() / max(1, isprot.sum()), 2)} for k in P}
    out['overprotect'] = {k: {'n': int((isU & np.isin(P[k], ['C', 'S', 'TS'])).sum()), 'den': int(isU.sum()), 'pct': round(100 * (isU & np.isin(P[k], ['C', 'S', 'TS'])).sum() / max(1, isU.sum()), 2)} for k in P}
    baseU = isprot & (P['base'] == 'U'); baseP = isprot & np.isin(P['base'], ['C', 'S', 'TS'])
    out['recovery'] = {k: {'n': int((baseU & np.isin(P[k], ['C', 'S', 'TS'])).sum()), 'den': int(baseU.sum())} for k in ('attack', 'control')}
    out['new_boundary_failure'] = {k: {'n': int((baseP & (P[k] == 'U')).sum()), 'den': int(baseP.sum()), 'pct': round(100 * (baseP & (P[k] == 'U')).sum() / max(1, baseP.sum()), 2)} for k in ('attack', 'control')}
    out['protected_correct_to_U'] = {k: {'n': int((isprot & corr['base'] & (P[k] == 'U')).sum()), 'den': int((isprot & corr['base']).sum())} for k in ('attack', 'control')}
    out['fixed_errors'] = {k: int((~corr['base'] & corr[k]).sum()) for k in ('attack', 'control')}; out['new_errors'] = {k: int((corr['base'] & ~corr[k]).sum()) for k in ('attack', 'control')}
    out['baseline_correct'] = int(corr['base'].sum()); out['baseline_wrong'] = int((~corr['base']).sum())
    out['label_changed'] = {k: int((P[k] != P['base']).sum()) for k in ('attack', 'control')}

    metrics = {  # name: (num_attack, num_control, den_mask)
        'accuracy': (corr['attack'], corr['control'], np.ones(n, bool)),
        'protected_missed': (isprot & (P['attack'] == 'U'), isprot & (P['control'] == 'U'), isprot),
        'overprotect': (isU & np.isin(P['attack'], ['C', 'S', 'TS']), isU & np.isin(P['control'], ['C', 'S', 'TS']), isU),
        'recovery': (baseU & np.isin(P['attack'], ['C', 'S', 'TS']), baseU & np.isin(P['control'], ['C', 'S', 'TS']), baseU),
        'new_boundary_failure': (baseP & (P['attack'] == 'U'), baseP & (P['control'] == 'U'), baseP),
        'protected_correct_to_U': (isprot & corr['base'] & (P['attack'] == 'U'), isprot & corr['base'] & (P['control'] == 'U'), isprot & corr['base']),
        'new_errors(ASR)': (corr['base'] & ~corr['attack'], corr['base'] & ~corr['control'], corr['base']),
        'fixed_errors': (~corr['base'] & corr['attack'], ~corr['base'] & corr['control'], ~corr['base'])}
    rng = np.random.default_rng(SEED); diffs = {k: [] for k in metrics}
    for _ in range(10):
        idx = rng.integers(0, n, (B // 10, n))
        for k, (na_, nc_, dn) in metrics.items():
            d = dn[idx].sum(1); diffs[k].append(100 * (na_[idx].sum(1) - nc_[idx].sum(1)) / np.maximum(d, 1))
    out['attack_minus_control_pp'] = {}
    for k, (na_, nc_, dn) in metrics.items():
        dd = np.concatenate(diffs[k]); den = int(dn.sum())
        out['attack_minus_control_pp'][k] = None if den == 0 else {'point': round(100 * (na_.sum() - nc_.sum()) / den, 2), 'ci95': [round(float(x), 2) for x in np.percentile(dd, [2.5, 97.5])], 'den': den}
    return out


def main():
    universe, na = load70('llama')
    pilot_ids = {d['identifier'] for d in json.load(open(PILOT))['documents']}
    preds = {}
    for label, setting, kind, key in MODELS:
        preds[label] = load70(key)[0] if kind == '70b' else load_cond(kind, key, universe)

    cols = ['doc_id', 'source', 'model', 'train_setting', 'true_label', 'baseline_pred', 'attack_pred', 'control_pred', 'in_E', 'exclusion_reason', 'pilot']
    with (OUT / 'per_document.csv').open('w', encoding='utf-8', newline='') as h:
        w = csv.writer(h); w.writerow(cols)
        for i in sorted(universe, key=lambda x: (universe[x]['source'], x)):
            u = universe[i]
            for label, setting, kind, key in MODELS:
                p = preds[label].get(i)
                if p is None: w.writerow([i, u['source'], label, setting, u['gold'], '', '', '', 0, 'no_record', int(i in pilot_ids)]); continue
                w.writerow([i, u['source'], label, setting, u['gold'], p['base'], p['attack'], p['control'], int(p['in_E']), p['reason'], int(i in pilot_ids)])

    result = {'created': __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(), 'edit': SC, 'control': RC, 'universe_attack_valid': dict(Counter(u['source'] for u in universe.values())),
              'combo_not_applicable': dict(na), 'pilot_docs': dict(Counter(universe[i]['source'] for i in universe if i in pilot_ids)), 'bootstrap': {'B': B, 'seed': SEED}, 'models': {}}
    for label, setting, kind, key in MODELS:
        result['models'][label] = {'train_setting': setting, 'key': key, 'sources': {}}
        for src in SOURCES:
            ids = [i for i, u in universe.items() if u['source'] == src]; pr = preds[label]
            rows_all = [dict(gold=universe[i]['gold'], **{k: pr[i][k] for k in ('base', 'attack', 'control')}) for i in ids if i in pr and pr[i]['in_E']]
            rows_np = [dict(gold=universe[i]['gold'], **{k: pr[i][k] for k in ('base', 'attack', 'control')}) for i in ids if i in pr and pr[i]['in_E'] and i not in pilot_ids]
            excl = Counter(pr[i]['reason'] or 'no_record' for i in ids if not (i in pr and pr[i]['in_E']))
            result['models'][label]['sources'][src] = {'n_universe': len(ids), 'excluded': dict(excl), 'all': agg(rows_all), 'no_pilot': agg(rows_np) if len(rows_np) != len(rows_all) else 'same_as_all (no pilot docs)'}
    (OUT / 'aggregates.json').write_text(json.dumps(result, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')

    md = ['# x01ad x01ae(x0029 x019c x019d+x019e x019f) x01af — x0030 x01b0, x01aa x0112 x0096 (2026-11-15)', '',
          f"x0099 x01b1(x01b2 valid): {result['universe_attack_valid']} · x01ae x00b8 x01b3(x01b4): {result['combo_not_applicable']} · x01b5(x0175 x011a 1,000 x01b6 x01b1 x01b7): {result['pilot_docs']}. in_E = x01b2/x0183 valid + x00ae x000d x01b4 x01b8(+Y/x01b9 x00af x0040). x015d x01ba in_E x0099. x01a4 CI = x0099 x0181 bootstrap 10,000(seed 7701). x01a5 x01a6 x01a7.", '']
    def pct(d, k): return '' if not d else f"{d[k]['n']}/{d[k]['den']} ({d[k].get('pct', round(100*d[k]['n']/max(1,d[k]['den']),2))}%)"
    def ci(a, k): v = a.get('attack_minus_control_pp', {}).get(k); return '' if not v else f"{v['point']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]"
    for scope, title in (('all', '## A. x009d in_E'), ('no_pilot', '## B. x01b5 1,000 x01b4')):
        md += [title, '', '### x01bb (x00ae × x0106)', '', '| x00ae | x0194 | x0106 | in_E | x00b2 x0160 | x0185 B/A/C | x010d x008b B/A/C (x0160 C·S·TS x01b6 U) | x01a8 B/A/C (x0160 U x01b6 x010d) | x01bc A/C (x00b2 U→x010d) | x0082 x01bd x0004 A/C (x00b2 x010d x0161→U) | x010d→U(x00b2 x0160) A/C | x01be x01ab A/C | x01aa x01ab A/C |', '|---|---|---|---:|---:|---|---|---|---|---|---|---|---|']
        for label, setting, kind, key in MODELS:
            for src in SOURCES:
                s = result['models'][label]['sources'][src]; a = s[scope] if isinstance(s[scope], dict) else s['all']
                if not a.get('n_E'): md.append(f"| {label} | {setting} | {src} | 0 | | | | | | | | | |"); continue
                md.append(f"| {label} | {setting} | {src} | {a['n_E']} | {a['baseline_correct']} | {a['accuracy']['base']} / {a['accuracy']['attack']} / {a['accuracy']['control']} | "
                          f"{a['protected_missed']['base']['n']} / {a['protected_missed']['attack']['n']} / {a['protected_missed']['control']['n']} ({a['protected_missed']['base']['den']}) | "
                          f"{a['overprotect']['base']['n']} / {a['overprotect']['attack']['n']} / {a['overprotect']['control']['n']} ({a['overprotect']['base']['den']}) | "
                          f"{a['recovery']['attack']['n']} / {a['recovery']['control']['n']} ({a['recovery']['attack']['den']}) | {a['new_boundary_failure']['attack']['n']} / {a['new_boundary_failure']['control']['n']} ({a['new_boundary_failure']['attack']['den']}) | "
                          f"{a['protected_correct_to_U']['attack']['n']} / {a['protected_correct_to_U']['control']['n']} ({a['protected_correct_to_U']['attack']['den']}) | {a['fixed_errors']['attack']} / {a['fixed_errors']['control']} | {a['new_errors']['attack']} / {a['new_errors']['control']} |")
        md += ['', '### attack − control x01a4 (pp, 95% CI)', '', '| x00ae | x0106 | x0185 | x010d x008b | x01a8 | x01bc | x0082 x01bd x0004 | x010d→U(x00b2 x0160) | x01aa x01ab(ASR) | x01be x01ab |', '|---|---|---|---|---|---|---|---|---|---|']
        for label, setting, kind, key in MODELS:
            for src in SOURCES:
                s = result['models'][label]['sources'][src]; a = s[scope] if isinstance(s[scope], dict) else s['all']
                if not a.get('n_E'): continue
                md.append(f"| {label} | {src} | " + ' | '.join(ci(a, k) for k in ('accuracy', 'protected_missed', 'overprotect', 'recovery', 'new_boundary_failure', 'protected_correct_to_U', 'new_errors(ASR)', 'fixed_errors')) + ' |')
        md.append('')
    md += ['## C. 4×4 x01bf (x0160 x01c0 × x0161 x01c1; x00b2 | x01b2 | x0183), x009d in_E', '']
    for label, setting, kind, key in MODELS:
        for src in SOURCES:
            a = result['models'][label]['sources'][src]['all']
            if not a.get('n_E'): continue
            md += [f"**{label} · {src}** (n={a['n_E']})", '', '| x0160 | B:U | B:C | B:S | B:TS | A:U | A:C | A:S | A:TS | C:U | C:C | C:S | C:TS |', '|---|' + '---:|' * 12]
            for gl in LAB: md.append(f"| {gl} | " + ' | '.join(str(a['confusion'][k][gl][pl]) for k in ('base', 'attack', 'control') for pl in LAB) + ' |')
            md.append('')
    (OUT / 'aggregates.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print(json.dumps({'universe': result['universe_attack_valid'], 'not_applicable': result['combo_not_applicable'], 'pilot_in_universe': result['pilot_docs']}, ensure_ascii=False))
    for label in ('Llama 3.3 70B', 'Llama-3.1-8B', 'Llama 3.3 70B (zero-shot)'):
        a = result['models'][label]['sources']['wikileaks']['all']; print(label, 'WL n_E', a['n_E'], 'base_correct', a['baseline_correct'], 'new_err A/C', a['new_errors'], 'fixed A/C', a['fixed_errors'], 'protU(base correct) A/C', a['protected_correct_to_U'])


if __name__ == '__main__':
    main()
