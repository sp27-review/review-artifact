#!/usr/bin/env python3
"""Recompute the paper's headline counts from the released per-document
predictions and compare them with the frozen summaries.

No model inference and no document text are needed.  Standard library only.

  python code/tools/reaggregate.py            # print tables
  python code/tools/reaggregate.py --check    # also compare with data/summaries/*.json

Definitions (all on a fixed set of documents):
  baseline/attack/control  prediction on the original text, the strict
                           blank-line + capital-heading edit, and its matched
                           random control
  protected-to-U           among protected (C/S/TS) documents whose baseline
                           prediction is correct, the number predicted U
  protected missed         among all protected documents, the number predicted U
  over-protection          among U documents, the number predicted C/S/TS
  new/fixed errors         baseline correct -> wrong / baseline wrong -> correct
"""
import argparse
import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRED = ROOT / 'data/predictions'
ATTACK = 'selected_combo:blank_collapse_native+capital_title_case'
CONTROL = 'random:' + ATTACK
SOURCES = ('wikileaks', 'frus', 'cia')
csv.field_size_limit(10 ** 9)


def rows(path):
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as h:
        yield from csv.DictReader(h)


def triple_stats(base, attack, control):
    """base/attack/control: {doc: (prediction, gold)} restricted to the paired set."""
    ids = sorted(set(base) & set(attack) & set(control))
    out = dict(n=len(ids), correct=[0, 0, 0], protected_missed=[0, 0, 0], overprotect=[0, 0, 0],
               n_protected=0, n_U=0, P=0, protected_to_U=[0, 0], new_errors=[0, 0], fixed_errors=[0, 0])
    for i in ids:
        gold = base[i][1]
        preds = (base[i][0], attack[i][0], control[i][0])
        prot = gold != 'U'
        out['n_protected'] += prot
        out['n_U'] += not prot
        for k, p in enumerate(preds):
            out['correct'][k] += p == gold
            out['protected_missed'][k] += prot and p == 'U'
            out['overprotect'][k] += (not prot) and p in ('C', 'S', 'TS')
        base_ok = preds[0] == gold
        if prot and base_ok:
            out['P'] += 1
        for k, p in enumerate(preds[1:]):
            out['protected_to_U'][k] += prot and base_ok and p == 'U'
            out['new_errors'][k] += base_ok and p != gold
            out['fixed_errors'][k] += (not base_ok) and p == gold
    return out


def load_small(tag):
    by = defaultdict(dict)
    path = PRED / 'small_models' / f'{tag}.csv.gz'
    if not path.exists():
        return None
    for r in rows(path):
        by[r['condition']][r['document_id']] = (r['prediction'], r['gold'])
    return by


def small_triples(tag):
    by = load_small(tag)
    if by is None:
        return {}
    return {s: (by[f'{s}_base'], by[f'{s}_combo'], by[f'{s}_combo_ctrl']) for s in SOURCES if f'{s}_combo_ctrl' in by}


def zero_shot_triples(model):
    by = defaultdict(dict)
    for r in rows(PRED / 'zero_shot/zsc_format_edit.csv.gz'):
        if r['model'] == model:
            by[r['condition']][r['document_id']] = (r['prediction'], r['gold'])
    return {s: (by[f'{s}_base'], by[f'{s}_combo'], by[f'{s}_combo_ctrl']) for s in SOURCES if f'{s}_combo_ctrl' in by}


def trained_triples(model):
    out = {}
    for scope in ('external', 'cia'):
        docs = defaultdict(dict)
        for r in rows(PRED / 'trained_classifiers_edits' / f'{scope}_{model}.csv.gz'):
            if r['condition'] in ('clean', ATTACK, CONTROL):
                docs[(r['source'], r['document_id'])][r['condition']] = r
        for (source, doc), c in docs.items():
            if all(k in c and c[k]['status'] == 'valid' and c[k]['prediction'] for k in ('clean', ATTACK, CONTROL)):
                b, a, ct = out.setdefault(source, ({}, {}, {}))
                gold = c['clean']['gold']
                b[doc], a[doc], ct[doc] = (c['clean']['prediction'], gold), (c[ATTACK]['prediction'], gold), (c[CONTROL]['prediction'], gold)
    return out


def position_stats(by, source):
    keys = [f'pos_{source}_{c}' for c in ('base', 'move_end', 'move_middle')]
    if not all(k in by for k in keys):
        return None
    base, end, mid = (by[k] for k in keys)
    ids = sorted(set(base) & set(end) & set(mid))
    den = e = m = 0
    for i in ids:
        gold = base[i][1]
        if base[i][0] == gold:
            den += 1
            e += end[i][0] != gold
            m += mid[i][0] != gold
    return dict(n=len(ids), baseline_correct=den, end_new_errors=e, middle_new_errors=m)


def fmt(s):
    acc = '/'.join(f"{100 * c / s['n']:.2f}" for c in s['correct'])
    return (f"n={s['n']} acc B/A/C {acc} | protected-to-U A/C {s['protected_to_U'][0]}/{s['protected_to_U'][1]} (P={s['P']}) | "
            f"missed B/A/C {'/'.join(map(str, s['protected_missed']))} (/{s['n_protected']}) | "
            f"over-protect B/A/C {'/'.join(map(str, s['overprotect']))} (/{s['n_U']}) | new A/C {s['new_errors'][0]}/{s['new_errors'][1]} | fixed A/C {s['fixed_errors'][0]}/{s['fixed_errors'][1]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    checks = ok = 0
    mismatches = []

    def compare(label, got, want):
        nonlocal checks, ok
        checks += 1
        if got == want:
            ok += 1
        else:
            mismatches.append((label, got, want))

    frozen = json.loads((ROOT / 'data/summaries/strict_combo_aggregates.json').read_text())['models']
    print('== Strict combined edit (blank-line collapse + capital-heading title case): baseline / attack / control ==')
    for label, meta in frozen.items():
        setting, key = meta['train_setting'], meta['key']
        triples = trained_triples(key) if setting == 'cia_only_70b' else zero_shot_triples(key) if setting == 'zero_shot_70b' else small_triples(key)
        for source in SOURCES:
            if source not in triples:
                continue
            s = triple_stats(*triples[source])
            print(f'{label:28s} {source:9s} {fmt(s)}')
            if a.check:
                f = meta['sources'][source]['all']
                compare(f'{label}/{source}/n', s['n'], f['n_E'])
                compare(f'{label}/{source}/baseline_correct', s['correct'][0], f['baseline_correct'])
                compare(f'{label}/{source}/P', s['P'], f['protected_correct_to_U']['attack']['den'])
                compare(f'{label}/{source}/protected_to_U', s['protected_to_U'], [f['protected_correct_to_U']['attack']['n'], f['protected_correct_to_U']['control']['n']])
                compare(f'{label}/{source}/protected_missed', s['protected_missed'], [f['protected_missed'][k]['n'] for k in ('base', 'attack', 'control')])
                compare(f'{label}/{source}/overprotect', s['overprotect'], [f['overprotect'][k]['n'] for k in ('base', 'attack', 'control')])
                compare(f'{label}/{source}/new_errors', s['new_errors'], [f['new_errors']['attack'], f['new_errors']['control']])
                compare(f'{label}/{source}/fixed_errors', s['fixed_errors'], [f['fixed_errors']['attack'], f['fixed_errors']['control']])

    norm_path = ROOT / 'data/summaries/normalization_same_documents.json'
    norm = json.loads(norm_path.read_text()) if norm_path.exists() else None
    print('\n== Other trained-model families released as small_models/*.csv.gz ==')
    for path in sorted((PRED / 'small_models').glob('*.csv.gz')):
        tag = path.name[:-7]
        by = load_small(tag)
        for source in SOURCES:
            if f'{source}_combo_ctrl' in by:
                print(f'{tag:44s} {source:9s} {fmt(triple_stats(by[f"{source}_base"], by[f"{source}_combo"], by[f"{source}_combo_ctrl"]))}')
        for source in SOURCES:
            p = position_stats(by, source)
            if p:
                print(f'{tag:44s} {source:9s} header-block move: end {p["end_new_errors"]} vs middle {p["middle_new_errors"]} new errors of {p["baseline_correct"]} (pairs {p["n"]})')
    if a.check and norm:
        names = {'Llama-3.1-8B': 'llama8b', 'Qwen2.5-7B': 'qwen7b', 'Mistral-7B': 'mistral7b', 'Gemma-2-9B': 'gemma9b', 'Phi-3.5-mini': 'phi35'}
        tags = lambda k: [f'{k}_cia_ext_seed7701', f'{k}_cia_ext_N1_inferonly_seed7701', f'{k}_cia_ext_N2_inferonly_seed7701', f'{k}_cia_ext_N2_seed7701']
        for label, by_cond in norm['models'].items():
            if label not in names:
                continue
            for tag, (cond, by_src) in zip(tags(names[label]), by_cond.items()):
                data = load_small(tag)
                if data is None:
                    continue
                for source, f in by_src.items():
                    s = triple_stats(data[f'{source}_base'], data[f'{source}_combo'], data[f'{source}_combo_ctrl'])
                    compare(f'norm/{tag}/{source}/correct', s['correct'], [f['correct'][k] for k in ('base', 'attack', 'control')])
                    compare(f'norm/{tag}/{source}/protected_missed', s['protected_missed'], [f['protected_missed'][k] for k in ('base', 'attack', 'control')])
                    compare(f'norm/{tag}/{source}/overprotect', s['overprotect'], [f['overprotect'][k] for k in ('base', 'attack', 'control')])
                    compare(f'norm/{tag}/{source}/protected_to_U', s['protected_to_U'] + [s['P']], [f['protected_to_U']['attack'], f['protected_to_U']['control'], f['P']])
        for label, by_cond in norm['position'].items():
            for tag, by_src in zip((f'{names[label]}_cia_ext_seed7701', f'{names[label]}_cia_ext_N2_seed7701'), by_cond.values()):
                data = load_small(tag)
                for source, f in by_src.items():
                    if f and data is not None:
                        p = position_stats(data, source)
                        compare(f'position/{tag}/{source}', [p['baseline_correct'], p['end_new_errors'], p['middle_new_errors']], [f['baseline_correct'], f['end_new_errors'], f['middle_new_errors']])

    print('\n== Zero-shot accuracy, original text with markings vs marker-removed text (CIA, 37,862 documents) ==')
    for name, cond in (('zs1_cia_raw', 'raw'), ('zs1_cia_masked', 'masked')):
        acc = defaultdict(lambda: [0, 0])
        for r in rows(PRED / 'zero_shot' / f'{name}.csv.gz'):
            acc[r['model']][0] += r['prediction'] == r['gold']
            acc[r['model']][1] += 1
        print(f'{cond:7s} ' + ' | '.join(f'{m} {c}/{n} = {100 * c / n:.2f}%' for m, (c, n) in sorted(acc.items())))
    if a.check:
        print(f'\nreaggregation checks: {ok}/{checks} passed')
        for label, got, want in mismatches[:20]:
            print('  MISMATCH', label, 'recomputed', got, 'frozen', want)
        sys.exit(0 if ok == checks and checks else 1)


if __name__ == '__main__':
    main()
