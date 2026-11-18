"""Materialize the authorized strict first-seed evaluation, CPU only."""
from __future__ import annotations
import argparse
import collections
import concurrent.futures
import json
import os
from pathlib import Path
import time

import external_attack_common as e
from prepare_relaxed_capital_combo import roundtrip

c = e.c
RUN = c.BASE / 'runs/run_20261102_r5_fullpool_firstseed'
CENSUS = c.BASE / 'runs/run_20261102_r5_strict_combo_fullpool'
TOKENIZERS = {}
SELECTION = None


def initialize(paths, selection):
    from tokenizers import Tokenizer
    global TOKENIZERS, SELECTION
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    TOKENIZERS = {name: Tokenizer.from_file(path) for name, path in paths.items()}
    SELECTION = selection


def build(job):
    raw, old, census, output = job
    if raw['dataset'] == 'cia':
        units, _, _, _ = e.build_units(raw)
        raw = dict(raw, units=units)
        attack = e.compose(raw, SELECTION['condition'], SELECTION['operators'], 'dev_selected_combo')
        doc = dict(raw, attacks=[attack], controls=[e.control(raw, attack)] if attack['status'] == 'valid' else [], positions=[])
    else:
        doc = e.add_selected(e.build_doc(raw), SELECTION)
    if old:
        assert c.source.digest(old['path']) == old['sha256']
        saved = c.source.read_json(old['path'])
        assert saved['text'] == doc['text']
        for group in ('attacks', 'controls', 'positions'):
            assert doc[group] == saved[group], (doc['identifier'], group)
    e.audit_doc(doc)
    valid = {v['name']: v for group in ('attacks', 'controls', 'positions') for v in doc[group] if v['status'] == 'valid'}
    if census:
        assert census['text_sha256'] == doc['text_sha256']
        for name, expected in census['conditions'].items():
            assert (name in valid) == expected['applicable']
            if name in valid:
                assert valid[name]['text_sha256'] == expected['attack_sha256']
            paired = name in valid and expected['control'] in valid
            assert paired == expected['paired']
            if paired:
                assert valid[expected['control']]['text_sha256'] == expected['control_sha256']
    texts = {doc['text_sha256']: doc['text']}
    for v in valid.values():
        roundtrip(doc, v)
        texts[v['text_sha256']] = c.apply_edits(doc, v['edits'])[0]
    lengths = {h: {} for h in texts}
    for name, tok in TOKENIZERS.items():
        for h, encoded in zip(texts, tok.encode_batch(list(texts.values()), add_special_tokens=True)):
            lengths[h][name] = len(encoded.ids)
    for row in lengths.values():
        row['nemotron'] = row['llama']
    if doc.get('lengths'):
        assert lengths[doc['text_sha256']] == doc['lengths'], doc['identifier']
    assert max(lengths[doc['text_sha256']].values()) <= 4096
    doc['lengths'] = lengths[doc['text_sha256']]
    over = collections.Counter()
    for v in valid.values():
        v['token_lengths'] = lengths[v['text_sha256']]
        v['length_excluded_models'] = [m for m, n in v['token_lengths'].items() if n > 4096]
        over.update(v['length_excluded_models'])
    lean = {k: doc[k] for k in ('identifier', 'dataset', 'gold', 'text', 'text_sha256', 'protected', 'sensitive', 'lengths', 'attacks', 'controls', 'positions')}
    lean['source'] = raw['source']
    lean['original_focus_sample'] = bool(old)
    if 'cluster_id' in doc:
        lean['cluster_id'] = doc['cluster_id']
    phase = 'cia' if doc['dataset'] == 'cia' else 'external'
    path = Path(output) / 'inputs' / phase / (c.key(doc['identifier']) + '.json')
    if path.exists():
        assert c.source.read_json(path) == lean, ('Prepared input changed', str(path))
    else:
        c.source.write_json(path, lean)
    return dict(identifier=doc['identifier'], dataset=doc['dataset'], phase=phase, gold=doc['gold'], path=str(path),
                sha256=c.source.digest(path), original_sample_verified=bool(old),
                valid_conditions=len(valid), unique_texts=len(texts), length_exclusions=dict(over),
                condition_counts={g + '|' + v['name'] + '|' + v['status']: 1 for g in ('attacks', 'controls', 'positions') for v in doc[g]})


def prepare(run=RUN, workers=4):
    run = run.resolve()
    if (run / 'prepared.json').exists():
        raise FileExistsError('Prepared manifest is frozen; verify and reuse it.')
    for part in ('analysis', 'queue/logs', 'guard', 'inputs/cia', 'inputs/external'):
        (run / part).mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    selection = c.source.read_json(e.RUN / 'selection.json')
    assert selection['operators'] == ['blank_collapse_native', 'capital_title_case']
    frozen = c.source.read_json(e.RUN / 'prepared.json')
    for path, digest in frozen['source_hashes'].items():
        assert c.source.digest(path) == digest
    census = {r['identifier']: r for r in c.source.records(CENSUS / 'analysis/all_pilots_rows.jsonl')}
    originals = {r['identifier']: r for r in c.source.read_json(e.RUN / 'external_manifest.json')['documents']}
    jobs = []
    source_hashes = dict(frozen['source_hashes'])
    cia_manifest = c.source.read_json(c.RUN / 'manifest.json')
    for r in cia_manifest['documents']:
        path = c.RUN / 'inputs' / (c.key(r['identifier']) + '.json')
        assert c.source.digest(path) == r['input_sha256']
        old = c.source.read_json(path)
        raw = {k: old[k] for k in ('identifier', 'gold', 'text', 'text_sha256', 'protected', 'sensitive', 'cluster_id')}
        raw.update(dataset='cia', source=dict(path=str(path), sha256=r['input_sha256']))
        jobs.append((raw, None, None, str(run)))
    for ds, root in e.m.EXTERNAL.items():
        src = root / 'evaluation.jsonl'
        digest = c.source.digest(src)
        assert digest == c.source.read_json(root / 'length_summary.json')['evaluation_sha256']
        source_hashes[str(src)] = digest
        lengths = {r['doc_id']: r['tokens'] for r in c.source.records(root / 'token_lengths.jsonl')}
        for number, source in enumerate(c.source.records(src), 1):
            ident = source['doc_id']
            if ident not in census:
                continue
            raw = dict(identifier=ident, dataset=ds, gold=source['label_id'], text=source['text'], text_sha256=source['text_sha256'],
                       protected=[], sensitive=[], lengths=lengths[ident], source=dict(path=str(src), sha256=digest, line=number))
            jobs.append((raw, originals.get(ident), census[ident], str(run)))
    assert len(jobs) == 2299 + 22023
    token_paths = {m: str(c.ROOT / 'checkpoints/hf' / c.source.BASE_MODELS[m] / 'tokenizer.json') for m in ('llama', 'qwen')}
    state = c.source.read_json(c.MODEL_RUNS['longformer'] / 'results/longformer/training_state.json')
    token_paths['longformer'] = str(Path(state['best_checkpoint']) / 'tokenizer.json')
    source_hashes.update({str(p): c.source.digest(p) for p in [Path(__file__), Path(__file__).with_name('prepare_relaxed_capital_combo.py'), e.RUN / 'selection.json', CENSUS / 'analysis/all_pilots_rows.jsonl', c.RUN / 'manifest.json']})
    source_hashes.update({p: c.source.digest(p) for p in token_paths.values()})
    checkpoints = {}
    for model in c.MODELS:
        p = c.MODEL_RUNS[model] / 'results' / model / 'training_state.json'
        s = c.source.read_json(p)
        assert s['status'] == 'complete'
        checkpoints[model] = dict(path=s['best_checkpoint'], state_path=str(p), state_sha256=c.source.digest(p))
    rows = []; counts = collections.defaultdict(collections.Counter); over = collections.Counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, initializer=initialize, initargs=(token_paths, selection)) as pool:
        for i, row in enumerate(pool.map(build, jobs, chunksize=8), 1):
            counts[row['dataset']].update(row.pop('condition_counts'))
            over.update(row['length_exclusions']); rows.append(row)
            if i % 250 == 0:
                progress = dict(status='cpu_preparing', completed=i, expected=len(jobs), seconds=time.monotonic()-start)
                c.source.write_json(run / 'analysis/preparation_progress.json', progress)
                print(json.dumps(progress), flush=True)
    assert len({r['identifier'] for r in rows}) == len(rows)
    assert sum(r['original_sample_verified'] for r in rows) == 2000
    assert counts['cia']['attacks|' + selection['condition'] + '|valid'] == 615
    assert counts['cia']['controls|random:' + selection['condition'] + '|valid'] == 597
    manifest = dict(created=c.source.now(), status='prepared_cpu_verified', seed=7701, models=c.MODELS, documents=rows,
                    counts={k: dict(v) for k, v in counts.items()}, source_hashes=source_hashes, checkpoints=checkpoints,
                    selection=selection, primary=e.PRIMARY, singles=e.SINGLES,
                    policy=dict(no_copy=True, no_relaxation=True, no_training=True, no_attention_gradient=True,
                                no_additional_seeds=True, position_controls='front/middle/end; no random destination',
                                cia_clean='Reuse verified legacy baseline; explicit original seed unknown and not claimed as three-seed evidence',
                                external_clean='Same-seed verified cache or new forward', length='No truncation; per-model edited-length exclusions remain explicit'))
    c.source.write_json(run / 'prepared.json', manifest)
    verification = dict(status='passed', documents=len(rows), counts_by_phase=dict(collections.Counter(r['phase'] for r in rows)),
                        old_external_documents_reproduced=2000, valid_conditions=sum(r['valid_conditions'] for r in rows),
                        unique_texts_per_document=sum(r['unique_texts'] for r in rows), length_exclusions=dict(over),
                        prepared_sha256=c.source.digest(run / 'prepared.json'), seconds=time.monotonic()-start, gpu_inference_started=False)
    c.source.write_json(run / 'analysis/input_verification.json', verification)
    c.source.write_json(run / 'analysis/preparation_progress.json', dict(verification, status='complete'))
    print(json.dumps(verification, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, default=RUN); p.add_argument('--workers', type=int, default=4)
    a = p.parse_args(); prepare(a.run, a.workers)
