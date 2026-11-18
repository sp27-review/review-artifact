"""Resumable, first-seed-only CIA combo and external full-pool prediction."""
from __future__ import annotations
import argparse
import fcntl
import json
from pathlib import Path
import random
import sqlite3
import time
import traceback
import numpy as np

import external_attack_common as e
from run_format_extension import Predictor
from prepare_fullpool_firstseed import RUN
c = e.c


def import_cache(predictor, path, label):
    key = 'fullpool_import:' + label
    prior = predictor.db.execute('SELECT value FROM meta WHERE name=?', (key,)).fetchone()
    if prior:
        return json.loads(prior[0])
    source = sqlite3.connect('file:' + str(path) + '?mode=ro', uri=True)
    try:
        identity = source.execute('SELECT value FROM meta WHERE name=?', ('identity',)).fetchone()
        own = predictor.db.execute('SELECT value FROM meta WHERE name=?', ('identity',)).fetchone()
        assert identity and json.loads(identity[0]) == json.loads(own[0]), ('Cache checkpoint mismatch', str(path))
        if label == 'external_attack_7701':
            protocol = source.execute('SELECT value FROM meta WHERE name=?', ('external_attack_protocol',)).fetchone()
            assert protocol and json.loads(protocol[0])['seed'] == 7701
        count = 0
        for row in source.execute('SELECT input_hash,logits,tokens,seconds FROM inference'):
            old = predictor.db.execute('SELECT logits,tokens FROM inference WHERE input_hash=?', (row[0],)).fetchone()
            if old:
                assert old[1] == row[2] and np.allclose(json.loads(old[0]), json.loads(row[1]), rtol=0, atol=.01)
            else:
                predictor.db.execute('INSERT INTO inference VALUES (?,?,?,?)', row)
                count += 1
        record = dict(path=str(path), sha256=c.source.digest(path), inserted=count, seed=7701,
                      provenance=label, checkpoint_identity_verified=True)
        predictor.db.execute('INSERT INTO meta VALUES (?,?)', (key, json.dumps(record, sort_keys=True)))
        predictor.db.commit()
        return record
    finally:
        source.close()


def import_external_clean(predictor, model):
    """The focus job saved logits in documents, not in its otherwise empty cache."""
    key = 'fullpool_import:external_clean_documents_7701'
    prior = predictor.db.execute('SELECT value FROM meta WHERE name=?', (key,)).fetchone()
    if prior:
        return json.loads(prior[0])
    manifest_path = e.RUN / 'external_manifest.json'
    manifest_hash = c.source.digest(manifest_path)
    rows = c.source.read_json(manifest_path)['documents']
    own = json.loads(predictor.db.execute('SELECT value FROM meta WHERE name=?', ('identity',)).fetchone()[0])
    inserted = 0; sources = {}
    for row in rows:
        path = e.RUN / 'results' / model / 'external/documents' / (c.key(row['identifier']) + '.json')
        saved = c.source.read_json(path)
        assert saved['seed'] == 7701 and saved['model'] == model and saved['phase'] == 'external'
        assert saved['manifest_sha256'] == manifest_hash and saved['input_sha256'] == row['sha256']
        assert saved['checkpoint_identity'] == own
        clean = saved['clean']; assert clean['status'] == 'valid'
        old = predictor.db.execute('SELECT logits,tokens FROM inference WHERE input_hash=?', (clean['input_hash'],)).fetchone()
        if old:
            assert old[1] == clean['tokens'] and np.allclose(json.loads(old[0]), clean['logits'], rtol=0, atol=.01)
        else:
            predictor.db.execute('INSERT INTO inference VALUES (?,?,?,?)',
                                 (clean['input_hash'], json.dumps(clean['logits']), clean['tokens'], 0.0))
            inserted += 1
        sources[str(path)] = c.source.digest(path)
    record = dict(manifest_path=str(manifest_path), manifest_sha256=manifest_hash, documents=len(rows), inserted=inserted,
                  seed=7701, provenance='saved_external_clean_checkpoint_input_verified', source_hashes=sources)
    predictor.db.execute('INSERT INTO meta VALUES (?,?)', (key, json.dumps(record, sort_keys=True)))
    predictor.db.commit()
    return record


def infer_document(doc, row, manifest_hash, predictor, phase, model, check_clean=False):
    e.audit_doc(doc)
    if phase == 'cia':
        path = c.RUN / 'results' / model / 'main/documents' / (c.key(doc['identifier']) + '.json')
        saved = c.source.read_json(path)
        assert saved['identifier'] == doc['identifier'] and saved['gold'] == doc['gold']
        assert saved['input_manifest_sha256'] == doc['source']['sha256']
        sig, _, tokens, error = predictor.encode(doc['text'])
        clean = saved['clean']
        assert not error and sig == clean['input_hash'] and tokens == clean['tokens']
        if check_clean:
            fresh = predictor.predict(doc['text'])
            assert fresh['prediction'] == clean['prediction']
            assert np.max(np.abs(np.array(fresh['logits']) - clean['logits'])) <= .01
        clean = dict(clean, provenance='verified_legacy_cia_baseline_original_seed_unspecified', source=str(path),
                     source_sha256=c.source.digest(path))
    else:
        clean = dict(predictor.predict(doc['text']), provenance='explicit7701_actual_forward_or_verified_same_seed_cache')
    assert clean['status'] == 'valid', ('Baseline length invalid', doc['identifier'])
    values = []
    for group in ('attacks', 'controls', 'positions'):
        for v in doc[group]:
            assert 'repeat' not in v['name']
            meta = {k: value for k, value in v.items() if k != 'edits'}
            if v['status'] != 'valid':
                values.append(dict(meta, group=group)); continue
            text, _ = c.apply_edits(doc, v['edits'])
            assert c.h(text) == v['text_sha256']
            prediction = predictor.predict(text)
            if model in v['token_lengths']:
                assert prediction['tokens'] == v['token_lengths'][model], ('Tokenizer drift', doc['identifier'], v['name'])
                assert (prediction['status'] == 'valid') == (prediction['tokens'] <= 4096)
            values.append(dict(meta, **{k: x for k, x in prediction.items() if k != 'status'}, status=prediction['status'],
                               group=group, input_changed=prediction.get('input_hash') != clean['input_hash']))
    return dict(identifier=doc['identifier'], dataset=doc['dataset'], gold=doc['gold'], model=model, phase=phase,
                seed=7701, text_sha256=doc['text_sha256'], input_sha256=row['sha256'], manifest_sha256=manifest_hash,
                clean=clean, conditions=values, finished=c.source.now(), worker_sha256=c.source.digest(Path(__file__)))


def worker(run, model, phase, gpus):
    out = run / 'results' / model / phase
    (out / 'documents').mkdir(parents=True, exist_ok=True)
    with (out.parent / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = c.source.read_json(run / 'prepared.json'); mh = c.source.digest(run / 'prepared.json')
        verify = c.source.read_json(run / 'analysis/input_verification.json')
        assert verify['status'] == 'passed' and verify['prepared_sha256'] == mh
        assert manifest['seed'] == 7701 and manifest['primary'] == e.PRIMARY
        checkpoint = manifest['checkpoints'][model]
        assert c.source.digest(checkpoint['state_path']) == checkpoint['state_sha256']
        rows = [r for r in manifest['documents'] if r['phase'] == phase]
        pending = []
        for row in rows:
            path = out / 'documents' / (c.key(row['identifier']) + '.json')
            if path.exists():
                old = c.source.read_json(path)
                assert old['input_sha256'] == row['sha256'] and old['manifest_sha256'] == mh and old['seed'] == 7701
                assert old['model'] == model and old['phase'] == phase
            else:
                pending.append(row)
        if not pending:
            previous = c.source.read_json(out / 'status.json') if (out / 'status.json').exists() else {}
            c.source.write_json(out / 'status.json', dict(previous, status='complete', completed=len(rows), expected=len(rows), resumed_complete=True))
            return
        started = time.monotonic(); completed = len(rows) - len(pending)
        c.source.write_json(out / 'status.json', dict(status='loading', model=model, phase=phase, completed=completed, expected=len(rows), heartbeat=c.source.now()))
        random.seed(7701); np.random.seed(7701)
        import torch
        torch.manual_seed(7701)
        predictor = Predictor(model, run, gpus)
        identity = json.loads(predictor.db.execute('SELECT value FROM meta WHERE name=?', ('identity',)).fetchone()[0])
        assert identity['checkpoint'] == checkpoint['path'] and identity['training_state_sha256'] == checkpoint['state_sha256']
        protocol = dict(seed=7701, prepared_sha256=mh, no_cross_seed_reuse=True)
        prior = predictor.db.execute('SELECT value FROM meta WHERE name=?', ('fullpool_protocol',)).fetchone()
        assert not prior or json.loads(prior[0]) == protocol
        predictor.db.execute('INSERT OR IGNORE INTO meta VALUES (?,?)', ('fullpool_protocol', json.dumps(protocol, sort_keys=True)))
        predictor.db.commit()
        imports = [import_cache(predictor, e.RUN / 'results' / model / 'inference_cache.sqlite', 'external_attack_7701'),
                   import_cache(predictor, e.m.RUN / 'external_focus_runtime/results' / model / 'inference_cache.sqlite', 'external_focus_clean_7701'),
                   import_external_clean(predictor, model)]
        c.source.write_json(out.parent / 'cache_imports.json', imports)
        check_ids = {r['identifier'] for r in rows[:2]}
        for row in pending:
            if (run / 'queue/STOP').exists():
                raise RuntimeError('Explicit STOP requested')
            assert c.source.digest(row['path']) == row['sha256']
            doc = c.source.read_json(row['path'])
            result = infer_document(doc, row, mh, predictor, phase, model, row['identifier'] in check_ids)
            result['checkpoint_identity'] = identity
            predictor.db.commit()
            c.source.write_json(out / 'documents' / (c.key(doc['identifier']) + '.json'), result)
            completed += 1
            state = dict(status='running', model=model, phase=phase, completed=completed, expected=len(rows),
                         seconds=time.monotonic()-started, heartbeat=c.source.now(), forward_calls=predictor.calls,
                         cache_hits=predictor.hits, inference_seconds=predictor.infer_seconds, load_seconds=predictor.load_seconds,
                         invocation_started_with=len(rows)-len(pending), seed=7701)
            c.source.write_json(out / 'status.json', state)
            if completed % 100 == 0 or completed == 1:
                print(json.dumps(state), flush=True)
        predictor.db.close()
        c.source.write_json(out / 'status.json', dict(state, status='complete', finished=c.source.now()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, default=RUN)
    p.add_argument('--model', choices=c.MODELS, required=True); p.add_argument('--phase', choices=['cia', 'external'], required=True)
    p.add_argument('--gpus', default='0,1'); a = p.parse_args()
    try:
        worker(a.run.resolve(), a.model, a.phase, a.gpus)
    except BaseException:
        c.source.write_json(a.run / 'results' / a.model / a.phase / 'error.json', dict(time=c.source.now(), traceback=traceback.format_exc()))
        raise
