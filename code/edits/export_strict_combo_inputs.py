#!/usr/bin/env python3
"""(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
from __future__ import annotations
import datetime, glob, hashlib, json, os
from pathlib import Path

ROOT = Path('/path/to/workspace')
RUN = Path(__file__).resolve().parents[1]
SRC = ROOT / 'experiments/H_format_edit_screen/runs/run_20261102_r5_fullpool_firstseed/inputs'
COMBO = 'selected_combo:blank_collapse_native+capital_title_case'
SINGLE = 'single:blank_collapse_native'
LABELS = ['UNCLASSIFIED', 'CONFIDENTIAL', 'SECRET', 'TOP SECRET']
MODELS = {'llama33_70b': 'llama3.3:70b', 'qwen25_72b': 'qwen2.5:72b', 'nemotron_70b': 'nemotron:70b'}

LIMIT = {'llama': 8192 - 512 - 350, 'nemotron': 8192 - 512 - 350, 'qwen': 5120 - 512 - 350}


def sha(s: str | bytes) -> str:
    return hashlib.sha256(s.encode() if isinstance(s, str) else s).hexdigest()


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def apply_edits(original: str, variant: dict) -> str:
    edits = variant['edits']
    assert edits == sorted(edits, key=lambda e: (e['start'], e['end']))
    pieces, cursor = [], 0
    for e in edits:
        s, t = e['start'], e['end']
        assert 0 <= cursor <= s <= t <= len(original)
        assert original[s:t] == e['before']
        pieces.extend((original[cursor:s], e['after'])); cursor = t
    pieces.append(original[cursor:])
    edited = ''.join(pieces)
    assert sha(edited) == variant['text_sha256'], 'reconstructed variant hash mismatch'
    return edited


def main() -> None:
    docs = []  # (source, record)
    for phase, pattern in (('external', 'external/*.json'), ('cia', 'cia/*.json')):
        for f in sorted(glob.glob(str(SRC / pattern))):
            d = json.load(open(f, encoding='utf-8'))
            a = {x['name']: x for x in d.get('attacks', [])}; c = {x['name']: x for x in d.get('controls', [])}
            if not (a.get(COMBO, {}).get('status') == 'valid' and c.get('random:' + COMBO, {}).get('status') == 'valid'):
                continue
            assert sha(d['text']) == d['text_sha256'], 'base text hash'
            rec = dict(identifier=d['identifier'], source=d.get('dataset', 'cia'), gold=d['gold'], label=LABELS[d['gold']], h_input=f, h_input_sha256=sha(open(f, 'rb').read()),
                       base_sha256=d['text_sha256'], lengths=d['lengths'], base=d['text'])
            rec['combo'] = apply_edits(d['text'], a[COMBO]); rec['combo_sha256'] = a[COMBO]['text_sha256']; rec['combo_lengths'] = a[COMBO].get('token_lengths', {})
            rec['combo_ctrl'] = apply_edits(d['text'], c['random:' + COMBO]); rec['combo_ctrl_sha256'] = c['random:' + COMBO]['text_sha256']; rec['combo_ctrl_lengths'] = c['random:' + COMBO].get('token_lengths', {})
            if phase == 'external' and a.get(SINGLE, {}).get('status') == 'valid' and c.get('random:' + SINGLE, {}).get('status') == 'valid':
                rec['single'] = apply_edits(d['text'], a[SINGLE]); rec['single_sha256'] = a[SINGLE]['text_sha256']; rec['single_lengths'] = a[SINGLE].get('token_lengths', {})
                rec['single_ctrl'] = apply_edits(d['text'], c['random:' + SINGLE]); rec['single_ctrl_sha256'] = c['random:' + SINGLE]['text_sha256']; rec['single_ctrl_lengths'] = c['random:' + SINGLE].get('token_lengths', {})
            docs.append(rec)
    by_src = {}
    for r in docs: by_src.setdefault(r['source'], []).append(r)
    counts = {s: len(v) for s, v in by_src.items()}
    assert counts == {'wikileaks': 2528, 'frus': 431, 'cia': 597}, counts

    excluded = {}
    for r in docs:
        for cond in ('base', 'combo', 'combo_ctrl', 'single', 'single_ctrl'):
            if cond not in r: continue
            L = r['lengths'] if cond == 'base' else r.get(cond + '_lengths') or {}
            for m, lim in LIMIT.items():
                if L.get(m, 0) > lim: excluded.setdefault(m, []).append((r['identifier'], cond, L[m]))
    inputs = RUN / 'inputs'; manifests = RUN / 'manifests'
    files = {}

    def write(name, rows, text_key):
        p = inputs / f'{name}.jsonl'
        if not p.exists():
            with p.open('w', encoding='utf-8') as h:
                for r in rows: h.write(json.dumps({'identifier': r['identifier'], 'label': r['label'], 'marker_masked_text': r[text_key]}, ensure_ascii=False) + '\n')
        files[name] = dict(path=str(p), docs=len(rows), sha256=sha(p.read_bytes()), text_key=text_key)
        return p

    for src, rows in by_src.items():
        write(f'{src}_base', rows, 'base'); write(f'{src}_combo', rows, 'combo'); write(f'{src}_combo_ctrl', rows, 'combo_ctrl')
        sub = [r for r in rows if 'single' in r]
        if sub:
            write(f'{src}_single', sub, 'single'); write(f'{src}_single_ctrl', sub, 'single_ctrl')
    wl = by_src['wikileaks']
    longest = sorted(wl, key=lambda r: -r['lengths']['llama'])[:8]
    rest = sorted((r for r in wl if r not in longest), key=lambda r: sha('7701:' + r['identifier']))[:8]
    smoke = longest + rest
    write('smoke16_base', smoke, 'base'); write('smoke16_combo', smoke, 'combo'); write('smoke16_combo_ctrl', smoke, 'combo_ctrl')
    dm = manifests / 'documents.jsonl'
    if not dm.exists():
        with dm.open('w', encoding='utf-8') as h:
            for r in docs:
                h.write(json.dumps({k: v for k, v in r.items() if k not in ('base', 'combo', 'combo_ctrl', 'single', 'single_ctrl')}, ensure_ascii=False) + '\n')
    cond = manifests / 'conditions.tsv'
    if not cond.exists():
        rows = []
        def add(slug, cname, fname, mode=''):
            rows.append((slug, MODELS[slug], cname, str(inputs / f'{fname}.jsonl'), '', mode))
        add('llama33_70b', 'smoke16_base', 'smoke16_base'); add('llama33_70b', 'smoke16_combo', 'smoke16_combo'); add('llama33_70b', 'smoke16_combo_ctrl', 'smoke16_combo_ctrl')
        for slug in ('llama33_70b', 'qwen25_72b', 'nemotron_70b'):
            for src in ('wikileaks', 'frus', 'cia'):
                for c in ('base', 'combo', 'combo_ctrl'): add(slug, f'{src}_{c}', f'{src}_{c}')
            for src in ('wikileaks', 'frus'):
                for c in ('single', 'single_ctrl'): add(slug, f'{src}_{c}', f'{src}_{c}')
            if slug == 'llama33_70b':
                for c in ('base', 'combo', 'combo_ctrl'): add(slug, f'orig_wikileaks_{c}', f'wikileaks_{c}', 'orig')
        cond.write_text(''.join('\t'.join(r) + '\n' for r in rows), encoding='utf-8')
    man = dict(created_utc=now(), source_inputs=str(SRC), combo_condition=COMBO, single_condition=SINGLE, counts=counts,
               single_pair_counts={s: sum('single' in r for r in v) for s, v in by_src.items()}, files=files,
               token_limits=LIMIT, excluded_over_limit={m: len(v) for m, v in excluded.items()}, excluded_detail=excluded,
               label_map=dict(enumerate(LABELS)), reconstruction='attacks/controls edits x00b8 x0097 text_sha256 x01d1 x01d2(x0098 x0099 x000f)',
               planned_rows=sum(int(open(r[3]).read().count('\n')) for r in [l.split('\t') for l in cond.read_text().splitlines()]))
    mp = manifests / 'inputs_manifest.json'
    if not mp.exists(): mp.write_text(json.dumps(man, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: man[k] for k in ('counts', 'single_pair_counts', 'excluded_over_limit', 'planned_rows')}, ensure_ascii=False))
    print({k: v['docs'] for k, v in files.items()})


if __name__ == '__main__':
    main()
