#!/usr/bin/env python3
"""(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
from __future__ import annotations
import collections, glob, hashlib, json, random
from pathlib import Path

ROOT = Path('/path/to/workspace')
RUN = Path(__file__).resolve().parents[1]
H_EXT = ROOT / 'experiments/H_format_edit_screen/runs/run_20261102_r5_fullpool_firstseed/inputs/external'
CIA_SPLIT = ROOT / 'experiments/X_nonllm_classifiers/runs/run_20261021_r5_noprompt_llama33_70b_training_smoke/manifests/split.jsonl'
LABELS = ['UNCLASSIFIED', 'CONFIDENTIAL', 'SECRET', 'TOP SECRET']
COMBO = 'selected_combo:blank_collapse_native+capital_title_case'


def sha(b): return hashlib.sha256(b.encode() if isinstance(b, str) else b).hexdigest()


def main() -> None:
    assert sha(CIA_SPLIT.read_bytes()) == '18a87ca7f750adb4accefead1f86e0d424ec2ebee8aae83f90b42ab47b263ca3'
    docs, held = [], {}
    for f in sorted(glob.glob(str(H_EXT / '*.json'))):
        d = json.load(open(f, encoding='utf-8'))
        a = {x['name']: x for x in d['attacks']}; p = {x['name']: x for x in d.get('positions', [])}; c = {x['name']: x for x in d['controls']}
        combo = a.get(COMBO, {}).get('status') == 'valid' and c.get('random:' + COMBO, {}).get('status') == 'valid'
        pos = a.get('position:move:end', {}).get('status') == 'valid' and p.get('position:move:middle', {}).get('status') == 'valid' and p.get('position:move:front', {}).get('status') == 'valid'
        rec = dict(identifier=d['identifier'], source=d['dataset'], label=LABELS[d['gold']], text_sha256=d['text_sha256'], text=d['text'])
        if combo or pos:
            held[d['identifier']] = dict(source=d['dataset'], combo_pair=combo, position_cohort=pos)
        else:
            docs.append(rec)
    assert len(held) == 2894 + 2986, len(held)
    rng = random.Random(7701); split = {}
    groups = collections.defaultdict(list)
    for r in docs: groups[(r['source'], r['label'])].append(r['identifier'])
    for key, ids in sorted(groups.items()):
        ids = sorted(ids, key=lambda i: sha('7701:' + i)); rng.shuffle(ids)
        n_dev = max(1, round(len(ids) * 0.1)) if len(ids) >= 2 else 0
        for i, ident in enumerate(ids): split[ident] = 'dev' if i < n_dev else 'train'
    man = RUN / 'manifests'; inp = RUN / 'inputs'
    ext_manifest = man / 'external_split.jsonl'
    if not ext_manifest.exists():
        with ext_manifest.open('w', encoding='utf-8') as h:
            for r in docs: h.write(json.dumps(dict(identifier=r['identifier'], source=r['source'], label=r['label'], split=split[r['identifier']], text_sha256=r['text_sha256']), ensure_ascii=False) + '\n')
    texts = inp / 'external_texts.jsonl'
    if not texts.exists():
        with texts.open('w', encoding='utf-8') as h:
            for r in docs: h.write(json.dumps(dict(identifier=r['identifier'], source=r['source'], label=r['label'], split=split[r['identifier']], marker_masked_text=r['text']), ensure_ascii=False) + '\n')
    hp = man / 'held_out_eval_ids.json'
    if not hp.exists(): hp.write_text(json.dumps(dict(count=len(held), by_source=dict(collections.Counter(v['source'] for v in held.values())), ids=held), ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    cia = collections.Counter(json.loads(l)['split'] for l in open(CIA_SPLIT))
    summary = dict(cia_split=dict(cia), cia_split_sha256='18a87ca7f750adb4accefead1f86e0d424ec2ebee8aae83f90b42ab47b263ca3',
                   external_pool=len(docs) + len(held), held_out_eval={s: sum(1 for v in held.values() if v['source'] == s) for s in ('wikileaks', 'frus')},
                   external_split={s: dict(collections.Counter(split[r['identifier']] for r in docs if r['source'] == s)) for s in ('wikileaks', 'frus')},
                   external_split_by_label={f'{s}/{sp}': dict(collections.Counter(r['label'] for r in docs if r['source'] == s and split[r['identifier']] == sp)) for s in ('wikileaks', 'frus') for sp in ('train', 'dev')},
                   training_set='cia train 28,359 + external train', dev_set='cia dev 3,530 + external dev', seed=7701, note='x0175 x0176 x0177 x0025 x0178 x0179 x017a x0099 x017b x017c x0149',
                   external_manifest_sha256=sha(ext_manifest.read_bytes()), external_texts_sha256=sha(texts.read_bytes()))
    sp = man / 'split_summary.json'
    if not sp.exists(): sp.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ('held_out_eval', 'external_split', 'external_split_by_label')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
