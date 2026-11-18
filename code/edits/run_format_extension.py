"""Resumable batch-one inference, with exact-input caching and no truncation."""
from __future__ import annotations

import argparse
import collections
import contextlib
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import time
import traceback

import numpy as np
import format_extension_common as c

BACKEND_VERSION='batch-one-bf16-autocast-fulltext-v1'


class Predictor:
    def __init__(self,model,run,gpus):
        self.name=model;self.run=run;self.calls=0;self.hits=0;self.infer_seconds=0.;self.timings=[]
        state=c.source.read_json(c.MODEL_RUNS[model]/'results'/model/'training_state.json')
        assert state['status']=='complete'
        self.checkpoint=Path(state['best_checkpoint'])
        self.cache_path=run/'results'/model/'inference_cache.sqlite';self.cache_path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(self.cache_path,timeout=120)
        self.db.execute('PRAGMA journal_mode=DELETE');self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS meta (name TEXT PRIMARY KEY,value TEXT NOT NULL)')
        identity=dict(backend=BACKEND_VERSION,model=model,checkpoint=str(self.checkpoint),
                      training_state_sha256=c.source.digest(c.MODEL_RUNS[model]/'results'/model/'training_state.json'),
                      loader_sha256=c.source.digest(Path(c.source.__file__)))
        encoded=json.dumps(identity,sort_keys=True);row=self.db.execute('SELECT value FROM meta WHERE name=?',('identity',)).fetchone()
        if row and row[0]!=encoded:raise RuntimeError('Inference cache identity changed')
        self.db.execute('INSERT OR IGNORE INTO meta VALUES (?,?)',('identity',encoded))
        self.db.execute('CREATE TABLE IF NOT EXISTS inference (input_hash TEXT PRIMARY KEY, logits TEXT NOT NULL, tokens INTEGER, seconds REAL)')
        self.db.commit()
        started=time.monotonic()
        if model=='tfidf':
            import joblib
            saved=joblib.load(self.checkpoint);self.vec=saved['vectorizer'];self.classifier=saved['classifier']
            assert self.classifier.classes_.tolist()==[0,1,2,3]
        else:
            c.source.ensure_idle(gpus)
            import torch
            self.torch=torch;torch.set_num_threads(8)
            self.model,self.tok=c.source.load_classifier(model,self.checkpoint,'balanced' if model in c.source.BASE_MODELS else 'cuda:0')
        self.load_seconds=time.monotonic()-started

    def encode(self,text,offsets=False):
        if self.name=='tfidf':
            x=self.vec.transform([text]);x.sort_indices()
            signature=c.h(x.indices.astype('<i4').tobytes()+x.data.astype('<f4').tobytes())
            return signature,x,int(x.nnz),None
        encoded=self.tok(text,add_special_tokens=True,truncation=False,return_offsets_mapping=offsets)
        ids=encoded['input_ids'];signature=c.h(np.asarray(ids,dtype='<i4').tobytes())
        if self.name!='deberta' and len(ids)>4096:return signature,None,len(ids),'length_exceeds_4096_fulltext_preserved_by_skipping'
        return signature,encoded,len(ids),None

    def predict_encoded(self,signature,encoded,tokens):
        saved=self.db.execute('SELECT logits FROM inference WHERE input_hash=?',(signature,)).fetchone()
        if saved:
            self.hits+=1;return np.array(json.loads(saved[0])),True
        started=time.monotonic()
        if self.name=='tfidf':logits=self.classifier.decision_function(encoded)[0].astype(float)
        else:
            torch=self.torch;batch=c.source.make_batch(self.name,self.model,self.tok,encoded)
            kwargs={'use_cache':False} if self.name in c.source.BASE_MODELS else {}
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                logits=self.model(**batch,**kwargs).logits.float()[0].cpu().numpy()
            del batch
        seconds=time.monotonic()-started
        if logits.shape!=(4,) or not np.isfinite(logits).all():raise ValueError('Invalid classifier logits')
        self.db.execute('INSERT INTO inference VALUES (?,?,?,?)',(signature,json.dumps(logits.tolist()),tokens,seconds))
        self.calls+=1;self.infer_seconds+=seconds;self.timings.append(dict(tokens=tokens,seconds=seconds))
        if self.calls%16==0:self.db.commit()
        return logits,False

    def predict(self,text):
        signature,encoded,tokens,error=self.encode(text)
        if error:return dict(status='unavailable',reason=error,input_hash=signature,tokens=tokens)
        logits,cached=self.predict_encoded(signature,encoded,tokens)
        return dict(status='valid',input_hash=signature,tokens=tokens,logits=logits.tolist(),prediction=int(logits.argmax()),cached=cached)


def smoke_variants(doc,model,coverage):
    valid=[v for v in doc['variants'] if v['status']=='valid' and c.variant_for_model(v,model)]
    selected=[]
    for family in ('atomic','atomic','atomic','position','guided','combo'):
        candidates=[v for v in valid if v not in selected and v['family']==family]
        if not candidates:continue
        chosen=min(candidates,key=lambda v:(coverage[v['name']],c.h(doc['identifier']+v['name'])))
        selected.append(chosen);coverage[chosen['name']]+=1
    return selected


def summarize_model(run,model,phase):
    directory=run/'results'/model/phase/'documents';by_condition=collections.defaultdict(list)
    originals=[]
    for p in sorted(directory.glob('*.json')):
        doc=c.source.read_json(p);clean=doc['clean'];gold=doc['gold'];originals.append(clean['prediction']==gold)
        for r in doc['conditions']:
            if r['status']!='valid':continue
            pred=r['prediction'];before=clean['prediction']
            by_condition[r['name']].append(dict(flip=pred!=before,correct=pred==gold,clean_correct=before==gold,
                broken=before==gold and pred!=gold,recovered=before!=gold and pred==gold,down=pred<before,up=pred>before,
                input_changed=r['input_changed']))
    rows=[]
    for name,values in sorted(by_condition.items()):
        n=len(values);correct=sum(v['clean_correct'] for v in values)
        rows.append(dict(condition=name,documents=n,input_changed=sum(v['input_changed'] for v in values),
            flip_rate=sum(v['flip'] for v in values)/n,clean_accuracy=sum(v['clean_correct'] for v in values)/n,
            accuracy=sum(v['correct'] for v in values)/n,clean_correct=correct,broken=sum(v['broken'] for v in values),
            attack_success_rate=sum(v['broken'] for v in values)/correct if correct else None,
            recovered=sum(v['recovered'] for v in values),down=sum(v['down'] for v in values),up=sum(v['up'] for v in values)))
    summary=dict(model=model,phase=phase,documents=len(originals),clean_accuracy=float(np.mean(originals)) if originals else None,conditions=rows)
    c.source.write_json(run/'results'/model/phase/'summary.json',summary)
    return summary


def run(args):
    run=args.output.resolve();manifest=c.source.read_json(run/'manifest.json');model=args.model;phase=args.phase
    ids=set(manifest['smoke_identifiers']) if phase=='smoke' else None
    selected=[r for r in manifest['documents'] if ids is None or r['identifier'] in ids]
    out=run/'results'/model/phase;(out/'documents').mkdir(parents=True,exist_ok=True)
    status_path=out/'status.json';started=time.monotonic();coverage=collections.Counter()
    with (run/'results'/model/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pending=[]
        for row in selected:
            path=out/'documents'/(c.key(row['identifier'])+'.json')
            if path.exists():
                r=c.source.read_json(path)
                if r['input_manifest_sha256']!=row['input_sha256']:raise RuntimeError('Completed document input changed')
                for v in r['conditions']:coverage[v['name']]+=1
            else:pending.append(row)
        if not pending:
            summary=summarize_model(run,model,phase)
            prior=c.source.read_json(status_path) if status_path.exists() else {}
            timings=[]
            if phase=='smoke':
                signatures=set()
                for path in (out/'documents').glob('*.json'):
                    r=c.source.read_json(path);signatures.add(r['clean']['input_hash'])
                    signatures.update(v['input_hash'] for v in r['conditions'] if v['status']=='valid')
                db=sqlite3.connect(run/'results'/model/'inference_cache.sqlite')
                for signature in sorted(signatures):
                    row=db.execute('SELECT tokens,seconds FROM inference WHERE input_hash=?',(signature,)).fetchone()
                    if row:timings.append(dict(tokens=row[0],seconds=row[1]))
                db.close()
            c.source.write_json(status_path,dict(prior,status='complete',model=model,phase=phase,completed=len(selected),expected=len(selected),
                resumed_complete=True,timings=timings,clean_accuracy=summary['clean_accuracy'],finished=c.source.now()))
            return
        predictor=Predictor(model,run,args.gpus)
        whitebox={r['identifier']:r for r in c.source.records(c.MODEL_RUNS[model]/'results'/model/'whitebox.jsonl') if r['split']=='portion_eval'}
        completed=len(selected)-len(pending);maxdiff=0.;disagreements=0;rows_written=0;sample_hashes=set()
        for row in pending:
            ident=row['identifier'];path=run/'inputs'/(c.key(ident)+'.json')
            if c.source.digest(path)!=row['input_sha256']:raise RuntimeError('Prepared input manifest changed')
            doc=c.source.read_json(path);clean=predictor.predict(doc['text'])
            if clean['status']!='valid':raise RuntimeError('Clean fulltext cannot be evaluated')
            sample_hashes.add(clean['input_hash'])
            expected=whitebox[ident];diff=float(np.max(np.abs(np.array(clean['logits'])-np.array(expected['logits']))))
            maxdiff=max(maxdiff,diff);disagreements+=clean['prediction']!=expected['prediction']
            if clean['prediction']!=expected['prediction'] or diff>0.01:
                raise RuntimeError(f'Clean checkpoint readout mismatch: {model} {ident} diff={diff}')
            variants=smoke_variants(doc,model,coverage) if phase=='smoke' else [v for v in doc['variants'] if c.variant_for_model(v,model)]
            results=[]
            for index,v in enumerate(variants):
                common={k:v[k] for k in ['name','family','operator','policy','dose','sites','characters_changed','metadata_removed','paired_variant','unit_ids'] if k in v}
                if v['status']!='valid':
                    results.append(dict(common,status=v['status'],reason=v.get('reason')));continue
                text,edits=c.apply_edits(doc,v['edits'])
                if c.h(text)!=v['text_sha256']:raise ValueError('Edited input hash mismatch')
                pred=predictor.predict(text)
                if pred['status']=='valid':sample_hashes.add(pred['input_hash'])
                results.append(dict(common,**pred,text_sha256=v['text_sha256'],input_changed=pred['input_hash']!=clean['input_hash']))
                if index%16==0:
                    c.source.write_json(status_path,dict(status='running',model=model,phase=phase,completed=completed,expected=len(selected),
                        current_identifier=ident,current_condition=index,forward_calls=predictor.calls,cache_hits=predictor.hits,
                        seconds=time.monotonic()-started,heartbeat=c.source.now()))
            predictor.db.commit()
            saved=dict(identifier=ident,gold=doc['gold'],cluster_id=doc['cluster_id'],fold=doc['fold'],portion_max_grade=doc['portion_max_grade'],
                       model=model,phase=phase,input_manifest_sha256=row['input_sha256'],clean=clean,conditions=results,
                       whitebox_logit_max_difference=diff,completed=c.source.now(),worker_sha256=c.source.digest(Path(__file__)))
            c.source.write_json(out/'documents'/(c.key(ident)+'.json'),saved)
            completed+=1;rows_written+=len(results)
            c.source.write_json(status_path,dict(status='running',model=model,phase=phase,completed=completed,expected=len(selected),
                forward_calls=predictor.calls,cache_hits=predictor.hits,seconds=time.monotonic()-started,heartbeat=c.source.now()))
            if completed%8==0 or phase=='smoke':c.emit(stage=phase,model=model,completed=completed,expected=len(selected),forwards=predictor.calls,seconds=round(time.monotonic()-started,1))
        summary=summarize_model(run,model,phase)
        if phase=='smoke':
            for p in (out/'documents').glob('*.json'):
                r=c.source.read_json(p);sample_hashes.add(r['clean']['input_hash'])
                sample_hashes.update(v['input_hash'] for v in r['conditions'] if v['status']=='valid')
        measured=[dict(tokens=r[0],seconds=r[1]) for signature in sorted(sample_hashes)
                  if (r:=predictor.db.execute('SELECT tokens,seconds FROM inference WHERE input_hash=?',(signature,)).fetchone())] if phase=='smoke' else []
        status=dict(status='complete',model=model,phase=phase,completed=completed,expected=len(selected),finished=c.source.now(),
            seconds=time.monotonic()-started,load_seconds=predictor.load_seconds,forward_calls=predictor.calls,cache_hits=predictor.hits,
            inference_seconds=predictor.infer_seconds,timings=measured,timing_source='persisted measurements for smoke inputs, including exact cache reuse',
            clean_max_logit_difference=maxdiff,clean_prediction_disagreements=disagreements,
            coverage=dict(coverage) if phase=='smoke' else None,clean_accuracy=summary['clean_accuracy'])
        c.source.write_json(status_path,status);c.emit(stage=phase,model=model,status='complete',documents=completed,seconds=round(status['seconds'],1))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=c.RUN);p.add_argument('--model',choices=c.MODELS,required=True)
    p.add_argument('--phase',choices=['smoke','main'],required=True);p.add_argument('--gpus',default='0,1')
    a=p.parse_args()
    try:run(a)
    except BaseException:
        c.source.write_json(a.output/'results'/a.model/a.phase/'error.json',dict(time=c.source.now(),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
