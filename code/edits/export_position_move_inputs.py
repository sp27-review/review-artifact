#!/usr/bin/env python3
"""(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
from __future__ import annotations
import datetime, glob, hashlib, json
from pathlib import Path

ROOT = Path('/path/to/workspace')
RUN = Path(__file__).resolve().parents[1]
SRC = ROOT / 'experiments/H_format_edit_screen/runs/run_20261102_r5_fullpool_firstseed/inputs'
END, MID, FRONT = 'position:move:end', 'position:move:middle', 'position:move:front'
LABELS = ['UNCLASSIFIED', 'CONFIDENTIAL', 'SECRET', 'TOP SECRET']
MODELS = {'qwen25_72b': 'qwen2.5:72b', 'llama33_70b': 'llama3.3:70b', 'nemotron_70b': 'nemotron:70b'}
LIMIT = {'llama': 8192 - 512 - 350, 'nemotron': 8192 - 512 - 350, 'qwen': 5120 - 512 - 350}


def sha(s): return hashlib.sha256(s.encode() if isinstance(s, str) else s).hexdigest()
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()


def apply_edits(original: str, variant: dict) -> str:
    edits = variant['edits']; assert edits == sorted(edits, key=lambda e: (e['start'], e['end']))
    pieces, cursor = [], 0
    for e in edits:
        s, t = e['start'], e['end']; assert 0 <= cursor <= s <= t <= len(original); assert original[s:t] == e['before']
        pieces.extend((original[cursor:s], e['after'])); cursor = t
    pieces.append(original[cursor:]); edited = ''.join(pieces)
    assert sha(edited) == variant['text_sha256'], 'reconstructed variant hash mismatch'
    return edited


def main() -> None:
    docs = []
    for f in sorted(glob.glob(str(SRC / 'external/*.json'))):
        d = json.load(open(f, encoding='utf-8'))
        a = {x['name']: x for x in d.get('attacks', [])}; p = {x['name']: x for x in d.get('positions', [])}
        e, m, fr = a.get(END, {}), p.get(MID, {}), p.get(FRONT, {})
        if not (e.get('status') == 'valid' and m.get('status') == 'valid' and fr.get('status') == 'valid'): continue
        assert sha(d['text']) == d['text_sha256']
        rec = dict(identifier=d['identifier'], source=d['dataset'], gold=d['gold'], label=LABELS[d['gold']], h_input=f, h_input_sha256=sha(open(f, 'rb').read()),
                   base_sha256=d['text_sha256'], lengths=d['lengths'], block=e.get('block'), base=d['text'],
                   move_end=apply_edits(d['text'], e), move_end_sha256=e['text_sha256'], move_end_lengths=e.get('token_lengths', {}),
                   move_middle=apply_edits(d['text'], m), move_middle_sha256=m['text_sha256'], move_middle_lengths=m.get('token_lengths', {}))
        docs.append(rec)
    by = {}
    for r in docs: by.setdefault(r['source'], []).append(r)
    counts = {s: len(v) for s, v in by.items()}; assert counts == {'frus': 2694, 'wikileaks': 645}, counts
    excluded = {}
    for r in docs:
        for cond in ('base', 'move_end', 'move_middle'):
            L = r['lengths'] if cond == 'base' else r[cond + '_lengths']
            for mdl, lim in LIMIT.items():
                if L.get(mdl, 0) > lim: excluded.setdefault(mdl, []).append((r['identifier'], cond, L[mdl]))
    inputs, manifests = RUN / 'inputs', RUN / 'manifests'; files = {}
    def write(name, rows, key):
        pth = inputs / f'{name}.jsonl'
        if not pth.exists():
            with pth.open('w', encoding='utf-8') as h:
                for r in rows: h.write(json.dumps({'identifier': r['identifier'], 'label': r['label'], 'marker_masked_text': r[key]}, ensure_ascii=False) + '\n')
        files[name] = dict(path=str(pth), docs=len(rows), sha256=sha(pth.read_bytes()), text_key=key)
    for src, rows in by.items():
        for cond in ('base', 'move_end', 'move_middle'): write(f'{src}_{cond}', rows, cond)
    fr = by['frus']; longest = sorted(fr, key=lambda r: -r['lengths']['qwen'])[:8]
    rest = sorted((r for r in fr if r not in longest), key=lambda r: sha('7701:' + r['identifier']))[:8]; smoke = longest + rest
    write('smoke16_base', smoke, 'base'); write('smoke16_move_end', smoke, 'move_end')
    dm = manifests / 'documents.jsonl'
    if not dm.exists():
        with dm.open('w', encoding='utf-8') as h:
            for r in docs: h.write(json.dumps({k: v for k, v in r.items() if k not in ('base', 'move_end', 'move_middle')}, ensure_ascii=False) + '\n')
    cond = manifests / 'conditions.tsv'
    if not cond.exists():
        rows = [('qwen25_72b', MODELS['qwen25_72b'], 'smoke16_base', str(inputs / 'smoke16_base.jsonl'), '', ''), ('qwen25_72b', MODELS['qwen25_72b'], 'smoke16_move_end', str(inputs / 'smoke16_move_end.jsonl'), '', '')]
        for slug in ('qwen25_72b', 'llama33_70b', 'nemotron_70b'):
            for src in ('frus', 'wikileaks'):
                for c in ('base', 'move_end', 'move_middle'): rows.append((slug, MODELS[slug], f'{src}_{c}', str(inputs / f'{src}_{c}.jsonl'), '', ''))
        cond.write_text(''.join('\t'.join(r) + '\n' for r in rows), encoding='utf-8')
    man = dict(created_utc=now(), source_inputs=str(SRC), conditions=dict(edit=END, comparison=MID, excluded=FRONT), counts=counts, files=files, token_limits=LIMIT,
               excluded_over_limit={k: len(v) for k, v in excluded.items()}, excluded_detail=excluded, label_map=dict(enumerate(LABELS)),
               reconstruction='attacks(end)·positions(middle) edits x00b8 x0097 text_sha256 x01d1 x01d2(x0098 x0099 x000f)',
               planned_rows=sum(int(open(l.split('\t')[3]).read().count('\n')) for l in cond.read_text().splitlines() if l.strip()))
    mp = manifests / 'inputs_manifest.json'
    if not mp.exists(): mp.write_text(json.dumps(man, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: man[k] for k in ('counts', 'excluded_over_limit', 'planned_rows')}, ensure_ascii=False)); print({k: v['docs'] for k, v in files.items()})


if __name__ == '__main__':
    main()
