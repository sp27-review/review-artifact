#!/usr/bin/env python3
"""(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
from __future__ import annotations
import glob, json, re, statistics, sys, datetime
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path('/path/to/workspace'); OUT = Path(__file__).resolve().parent
Y = ROOT / 'experiments/Y_small_models_robustness/runs/run_20261112_r5ext_cia_external_5small_seed7701_training'
BENCH = ROOT / 'data/datasets/cia_benchmark/r5/benchmark.jsonl'
CIA_SPLIT = ROOT / 'experiments/X_nonllm_classifiers/runs/run_20261021_r5_noprompt_llama33_70b_training_smoke/manifests/split.jsonl'
HEXT = ROOT / 'experiments/H_format_edit_screen/runs/run_20261102_r5_fullpool_firstseed/inputs/external'
E1 = ROOT / 'experiments/H_format_edit_screen/runs/run_20261110_ft3pairs3556_3model_P0_zeroshot_format_edit/inputs'
LAB = ['UNCLASSIFIED', 'CONFIDENTIAL', 'SECRET', 'TOP SECRET']; L = {l: i for i, l in enumerate(LAB)}
MONTH_RX = r'(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?'
DATE_RX = re.compile(r'(?<!\w)(\d{1,2})\s+' + MONTH_RX + r',?\s+(\d{4})(?!\d)|(?<!\w)' + MONTH_RX + r'\s+(\d{1,2}),?\s+(\d{4})(?!\d)', re.I)
ONE_WORD_TITLES = set('SUMMARY BACKGROUND INTRODUCTION CONCLUSION CONCLUSIONS COMMENT COMMENTS DISCUSSION ANALYSIS ASSESSMENT OVERVIEW RECOMMENDATIONS CONTENTS ANNEX APPENDIX'.split())
SECTION_RX = re.compile(r'^\s*(\d{1,2}[.)]|[IVXLC]{1,5}[.)]|[A-Za-z][.)])\s+\S')
FEATS = ['n_blank', 'blank_ratio', 'blank_runs', 'max_blank_run', 'caps_lines', 'caps_ratio', 'capital_title_targets', 'section_header_lines', 'standalone_date_lines', 'line_len_mean', 'line_len_var', 'n_chars', 'n_lines']
LOG1P = {'n_blank', 'blank_runs', 'max_blank_run', 'caps_lines', 'capital_title_targets', 'section_header_lines', 'standalone_date_lines', 'line_len_mean', 'line_len_var', 'n_chars', 'n_lines'}
BUNDLES = {'length': ['n_chars', 'n_lines'], 'edit': FEATS[:7], 'format_all': FEATS[:11], 'format_all+length': FEATS}
B, SEED = 10000, 7701


def boundary(s): return (not s) or bool(re.fullmatch(r'[-_=*~. ]+', s))


def features(text):
    lines = text.replace('\r\n', '\n').replace('\r', '\n').split('\n'); n = len(lines)
    blank = [l.strip() == '' for l in lines]; nb = sum(blank)
    runs, mx, cur = 0, 0, 0
    for b in blank:
        if b: cur += 1; mx = max(mx, cur)
        else:
            if cur: runs += 1
            cur = 0
    if cur: runs += 1
    nonblank = [l for l, b in zip(lines, blank) if not b]; nnb = max(1, len(nonblank))
    caps = 0; targets = 0; sections = 0; dates = 0
    for i, l in enumerate(lines):
        if blank[i]: continue
        letters = [ch for ch in l if ch.isalpha()]
        is_caps = len(letters) >= 3 and all(ch.isupper() for ch in letters)
        if is_caps:
            caps += 1
            words = re.findall('[A-Z]+', l); s = l.strip()
            ok_words = (2 <= len(words) <= 14) or (len(words) == 1 and words[0] in ONE_WORD_TITLES)
            prev_b = i == 0 or boundary(lines[i - 1].strip()); next_b = i == n - 1 or boundary(lines[i + 1].strip())
            isolated = prev_b and next_b
            if len(s) <= 120 and not re.search(r'[0-9/]', s) and ok_words and (isolated or (len(words) == 1 and words[0] in ONE_WORD_TITLES)): targets += 1
        if len(l) <= 80 and (SECTION_RX.match(l) or (l.rstrip().endswith(':') and len(l.split()) <= 8)): sections += 1
        core = l.strip().strip('()[].,;: ')
        if core and DATE_RX.fullmatch(core): dates += 1
    lens = [len(l) for l in nonblank] or [0]
    return dict(n_blank=nb, blank_ratio=nb / max(1, n), blank_runs=runs, max_blank_run=mx, caps_lines=caps, caps_ratio=caps / nnb, capital_title_targets=targets,
                section_header_lines=sections, standalone_date_lines=dates, line_len_mean=statistics.fmean(lens), line_len_var=statistics.pvariance(lens) if len(lens) > 1 else 0.0, n_chars=len(text), n_lines=n)


def X(rows, names): return np.array([[np.log1p(r['f'][k]) if k in LOG1P else r['f'][k] for k in names] for r in rows], dtype=float)


def records(p):
    with open(p, encoding='utf-8') as h:
        for line in h:
            if line.strip(): yield json.loads(line)


def load():
    data = {s: {'train': [], 'dev': [], 'test': [], 'heldout': []} for s in ('cia', 'wikileaks', 'frus')}
    split = {r['identifier']: r['split'] for r in records(CIA_SPLIT)}
    for r in records(BENCH):
        sp = split.get(r['identifier'])
        if sp in ('train', 'dev', 'test'): data['cia'][sp].append(dict(id=r['identifier'], y=L[r['label']], f=features(r['marker_masked_text'])))
    for r in records(Y / 'inputs/external_texts.jsonl'):
        data[r['source']][r['split']].append(dict(id=r['identifier'], y=L[r['label']], f=features(r['marker_masked_text'])))
    held = set(json.load(open(Y / 'manifests/held_out_eval_ids.json'))['ids'])
    for f in glob.glob(str(HEXT / '*.json')):
        d = json.load(open(f, encoding='utf-8'))
        if d['identifier'] in held: data[d['dataset']]['heldout'].append(dict(id=d['identifier'], y=int(d['gold']), f=features(d['text'])))
    sim = {}
    for s in ('wikileaks', 'frus', 'cia'):
        sim[s] = {c: {r['identifier']: dict(y=L[r['label'].upper()], f=features(r['marker_masked_text'])) for r in records(E1 / f'{s}_{c}.jsonl')} for c in ('base', 'combo', 'combo_ctrl')}
    return data, sim


def evaluate(clf, sc, rows, names, maj):
    if not rows: return None
    y = np.array([r['y'] for r in rows]); P = clf.predict_proba(sc.transform(X(rows, names))); pred = clf.classes_[P.argmax(1)]
    pU = P[:, list(clf.classes_).index(0)] if 0 in clf.classes_ else np.zeros(len(y))
    out = dict(n=len(y), accuracy=round(100 * (pred == y).mean(), 2), macro_f1=round(100 * f1_score(y, pred, average='macro', labels=sorted(set(y))), 2),
               majority_accuracy=round(100 * (y == maj).mean(), 2), majority_macro_f1=round(100 * f1_score(y, np.full_like(y, maj), average='macro', labels=sorted(set(y))), 2),
               auc_U_vs_protected=None if len(set(y == 0)) < 2 else round(roc_auc_score((y == 0).astype(int), pU), 4), pred_dist={LAB[k]: int(v) for k, v in Counter(pred.tolist()).items()}, U_ratio=round(100 * (y == 0).mean(), 2))
    return out


def simulate(clf, sc, names, simsrc):
    ids = sorted(set(simsrc['base']) & set(simsrc['combo']) & set(simsrc['combo_ctrl'])); n = len(ids)
    g = np.array([simsrc['base'][i]['y'] for i in ids]); pred = {}
    for c in ('base', 'combo', 'combo_ctrl'):
        rows = [simsrc[c][i] for i in ids]; P = clf.predict_proba(sc.transform(X(rows, names))); pred[c] = clf.classes_[P.argmax(1)]
    bc = pred['base'] == g; prot = g > 0; den = bc & prot
    aU = den & (pred['combo'] == 0); cU = den & (pred['combo_ctrl'] == 0); aw = bc & (pred['combo'] != g); cw = bc & (pred['combo_ctrl'] != g)
    rng = np.random.default_rng(SEED); dU, dA = [], []
    for _ in range(10):
        idx = rng.integers(0, n, (B // 10, n)); d1 = np.maximum(den[idx].sum(1), 1); d2 = np.maximum(bc[idx].sum(1), 1)
        dU.append(100 * (aU[idx].sum(1) - cU[idx].sum(1)) / d1); dA.append(100 * (aw[idx].sum(1) - cw[idx].sum(1)) / d2)
    dU, dA = np.concatenate(dU), np.concatenate(dA)
    pct = lambda a, b: round(100 * a / max(1, b), 2)
    return dict(n_pairs=n, baseline_accuracy=pct(bc.sum(), n), attack_accuracy=pct((pred['combo'] == g).sum(), n), control_accuracy=pct((pred['combo_ctrl'] == g).sum(), n),
                label_changed=dict(attack=int((pred['combo'] != pred['base']).sum()), control=int((pred['combo_ctrl'] != pred['base']).sum())),
                protected_to_U=dict(attack=dict(n=int(aU.sum()), den=int(den.sum()), pct=pct(aU.sum(), den.sum())), control=dict(n=int(cU.sum()), den=int(den.sum()), pct=pct(cU.sum(), den.sum())),
                                    diff_pp=pct(aU.sum() - cU.sum(), den.sum()), ci95=[round(float(x), 2) for x in np.percentile(dU, [2.5, 97.5])]),
                asr=dict(attack=dict(n=int(aw.sum()), den=int(bc.sum()), pct=pct(aw.sum(), bc.sum())), control=dict(n=int(cw.sum()), den=int(bc.sum()), pct=pct(cw.sum(), bc.sum())),
                         diff_pp=pct(aw.sum() - cw.sum(), bc.sum()), ci95=[round(float(x), 2) for x in np.percentile(dA, [2.5, 97.5])]))


def main():
    data, sim = load(); res = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), features=FEATS, log1p=sorted(LOG1P), bundles=BUNDLES, model='StandardScaler + LogisticRegression(C=1.0, lbfgs, max_iter=5000)', bootstrap=dict(B=B, seed=SEED), sources={})
    for s in ('wikileaks', 'frus', 'cia'):
        d = data[s]; tr = d['train']; ytr = np.array([r['y'] for r in tr]); maj = int(Counter(ytr.tolist()).most_common(1)[0][0])
        src = dict(n=dict(train=len(tr), dev=len(d['dev']), test=len(d['test']), heldout=len(d['heldout'])), U_ratio_pct={k: round(100 * np.mean([r['y'] == 0 for r in v]), 2) for k, v in d.items() if v},
                   label_dist={k: {LAB[c]: n for c, n in sorted(Counter(r['y'] for r in v).items())} for k, v in d.items() if v}, majority_train=LAB[maj], bundles={},
                   feature_by_grade_train={LAB[c]: {k: dict(median=round(float(np.median(vals)), 4), q1=round(float(np.percentile(vals, 25)), 4), q3=round(float(np.percentile(vals, 75)), 4), n=len(vals))
                                                   for k in ('blank_ratio', 'caps_ratio', 'capital_title_targets', 'n_blank', 'n_lines') for vals in [[r['f'][k] for r in tr if r['y'] == c]] if vals} for c in range(4) if any(r['y'] == c for r in tr)})
        for bname, names in BUNDLES.items():
            sc = StandardScaler().fit(X(tr, names)); clf = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(X(tr, names)), ytr)
            src['bundles'][bname] = dict(train=evaluate(clf, sc, tr, names, maj), dev=evaluate(clf, sc, d['dev'], names, maj), test=evaluate(clf, sc, d['test'], names, maj), heldout=evaluate(clf, sc, d['heldout'], names, maj),
                                         coef={LAB[c]: {k: round(float(w), 3) for k, w in zip(names, clf.coef_[i])} for i, c in enumerate(clf.classes_)} if len(clf.classes_) > 2 else {k: round(float(w), 3) for k, w in zip(names, clf.coef_[0])},
                                         edit_simulation=simulate(clf, sc, names, sim[s]))
        res['sources'][s] = src; print(s, 'done', {b: (v['test'] or v['heldout'] or {}).get('accuracy') for b, v in src['bundles'].items()}, flush=True)
    (OUT / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    md = ['# x01c2 x0189 x01c3 — x01c4 x01c5 (2026-11-15, CPU)', '', f"x012d(x017f): {', '.join(FEATS)} · log1p: x01c6/x000d · x01c7: {', '.join(BUNDLES)} · x00ae {res['model']} · test x01c8 x01c9 x0096.", '']
    for s, src in res['sources'].items():
        ev = 'test' if s == 'cia' else 'heldout'
        md += [f"## {s} — train {src['n']['train']} / dev {src['n']['dev']} / {ev} {src['n'][ev]}; U x01ca(%) {src['U_ratio_pct']}; x0188(train) {src['majority_train']}", '',
               f"### a. x01cb ({ev}; x01cc dev) — x0188 x018a x0185 {src['bundles']['length'][ev]['majority_accuracy']} / macro-F1 {src['bundles']['length'][ev]['majority_macro_f1']}", '',
               '| x01c7 | x0185 | macro-F1 | U x01cd x010d AUC | x0161 x00f1 |', '|---|---:|---:|---:|---|']
        for b, v in src['bundles'].items():
            e, dv = v[ev], v['dev']; md.append(f"| {b} | {e['accuracy']} ({dv['accuracy']}) | {e['macro_f1']} ({dv['macro_f1']}) | {e['auc_U_vs_protected']} ({dv['auc_U_vs_protected']}) | {e['pred_dist']} |")
        md += ['', '### b. x01ce x012d x00f1 (train; x014d [Q1, Q3])', '', '| x0189 | n | blank_ratio | caps_ratio | capital_title_targets | n_blank | n_lines |', '|---|---:|---|---|---|---|---|']
        for g, fs in src['feature_by_grade_train'].items():
            md.append(f"| {g} | {fs['blank_ratio']['n']} | " + ' | '.join(f"{fs[k]['median']} [{fs[k]['q1']}, {fs[k]['q3']}]" for k in ('blank_ratio', 'caps_ratio', 'capital_title_targets', 'n_blank', 'n_lines')) + ' |')
        md += ['', f"### c. x0189 x00f1 — {src['label_dist']}", '', '### d. x0180 x01cf (x01ad x01ae base/attack/control, Y x00c9 x0181 x0197)', '', '| x01c7 | x0181 | x0185 B/A/C | x01d0 x00f8 A/C | x010d→U A / C (x012b) | Δ pp [95% CI] | ASR A / C | Δ [CI] |', '|---|---:|---|---|---|---|---|---|']
        for b, v in src['bundles'].items():
            e = v['edit_simulation']; pu = e['protected_to_U']; a = e['asr']
            md.append(f"| {b} | {e['n_pairs']} | {e['baseline_accuracy']} / {e['attack_accuracy']} / {e['control_accuracy']} | {e['label_changed']['attack']} / {e['label_changed']['control']} | {pu['attack']['n']} / {pu['control']['n']} ({pu['attack']['den']}) = {pu['attack']['pct']} / {pu['control']['pct']}% | {pu['diff_pp']:+} {pu['ci95']} | {a['attack']['pct']} / {a['control']['pct']} | {a['diff_pp']:+} {a['ci95']} |")
        md.append('')
    (OUT / 'results.md').write_text('\n'.join(md) + '\n', encoding='utf-8'); print('written')


if __name__ == '__main__':
    main()
