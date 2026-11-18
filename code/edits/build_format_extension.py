"""Prepare the complete, frozen six-model format intervention manifest on CPU."""
import argparse
import collections
import concurrent.futures
import json
import multiprocessing
import os
from pathlib import Path
import time

import format_extension_common as c

_TAGGER=None
_WHITEBOX=None
_DISCOVERY=None


def build(row):
    return c.build_document(row,_TAGGER,_WHITEBOX,_DISCOVERY)


def smoke_selection(rows):
    chosen=[]
    for label in range(4):
        group=sorted([r for r in rows if r['gold']==label],key=lambda r:(len(r['text']),r['identifier']))
        for index in range(8):
            left=len(group)*index//8;right=len(group)*(index+1)//8
            chosen.append(min(group[left:right],key=lambda r:c.h('smoke-7701:'+r['identifier']))['identifier'])
    assert len(set(chosen))==32
    return chosen


def main():
    global _TAGGER,_WHITEBOX,_DISCOVERY
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=c.RUN);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();run=a.output.resolve();run.mkdir(parents=True,exist_ok=True);(run/'inputs').mkdir(exist_ok=True)
    start=time.monotonic();script=Path(__file__).resolve()
    source_hashes={str(x):c.source.digest(x) for x in [script,Path(c.__file__),c.PLAN,c.source.TAGGER]}
    setup=dict(status='preparing',created=c.source.now(),source_hashes=source_hashes,models=c.MODELS,
               input_field='marker_masked_text',protected='all analysis_eligible portions including U',
               focus_protected='eligible C/S/TS portions',basic_operators=23,
               guided_family_cap=2,random_seed=7701,cluster_folds=5,
               inference_policy='one document per forward, full input, exact input cache',
               stage='full factorial is not used; named contrasts are frozen before response evaluation')
    previous=c.source.read_json(run/'preparation.json') if (run/'preparation.json').exists() else None
    if previous and previous['source_hashes']!=source_hashes:
        raise RuntimeError('Preparation code changed; use a new output or explicitly rebuild unstarted preparation')
    c.source.write_json(run/'preparation.json',setup)
    rows=c.load_dataset();_WHITEBOX=c.load_whiteboxes();_TAGGER=c.source.import_file('extension_tagger',c.source.TAGGER)
    discovery=[r for r in c.source.read_json(c.ATTENTION/'focus_analysis/discovery_candidates.json') if r['split']=='portion_eval'][:20]
    _DISCOVERY=[r['signature'] for r in discovery]
    c.source.write_json(run/'discovery_review.json',dict(status='reviewed_for_content_preserving_boundary_edits',
        policy='Only whole-line occurrences outside eligible portions. Existing characters remain unchanged; insert one newline. Case edit limited to Attachment/Attachments/Report.',
        candidates=[dict(signature=r['signature'],selected_documents=r['selected_documents'],examples=r['examples'][:2]) for r in discovery]))
    pending=[r for r in rows if not (run/'inputs'/(c.key(r['identifier'])+'.json')).exists()]
    done=len(rows)-len(pending)
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('fork')) as pool:
        for r in pool.map(build,pending,chunksize=1):
            c.source.write_json(run/'inputs'/(c.key(r['identifier'])+'.json'),r);done+=1
            if done%16==0 or done==len(rows):
                c.source.write_json(run/'preparation_status.json',dict(status='running',completed=done,total=len(rows),seconds=time.monotonic()-start))
                c.emit(stage='prepare',completed=done,total=len(rows),seconds=round(time.monotonic()-start,1))
    counts=collections.Counter();by_model={m:collections.Counter() for m in c.MODELS};documents=[]
    for row in rows:
        path=run/'inputs'/(c.key(row['identifier'])+'.json');doc=c.source.read_json(path)
        if doc['text_sha256']!=row['text_sha256']:raise ValueError('Input changed')
        seen=set()
        for v in doc['variants']:
            if v['name'] in seen:raise ValueError('Duplicate condition name')
            seen.add(v['name']);counts[v['family']+':'+v['status']]+=1
            if v['status']=='valid':
                edited,edits=c.apply_edits(doc,v['edits'])
                if c.h(edited)!=v['text_sha256']:raise ValueError('Edit hash mismatch')
        for m in c.MODELS:
            variants=[v for v in doc['variants'] if c.variant_for_model(v,m)]
            by_model[m]['condition_rows']+=len(variants)
            by_model[m]['valid_conditions']+=sum(v['status']=='valid' for v in variants)
            by_model[m]['unique_texts_including_clean']+=len({doc['text_sha256']}|{v['text_sha256'] for v in variants if v['status']=='valid'})
        documents.append(dict(identifier=row['identifier'],gold=row['gold'],cluster_id=row['cluster_id'],fold=row['fold'],
                              text_sha256=row['text_sha256'],input_sha256=c.source.digest(path)))
    manifest=dict(setup,status='complete',completed=c.source.now(),seconds=time.monotonic()-start,
                  documents=documents,smoke_identifiers=smoke_selection(rows),condition_counts=dict(counts),
                  expected_work={m:dict(v) for m,v in by_model.items()},
                  note='Work counts are pre-tokenization upper bounds; model-identical inputs are deduplicated during inference.')
    c.source.write_json(run/'manifest.json',manifest);c.source.write_json(run/'execution_plan.json',c.source.read_json(c.PLAN))
    c.source.write_json(run/'preparation_status.json',dict(status='complete',completed=len(rows),total=len(rows),seconds=time.monotonic()-start))
    c.emit(stage='prepare',status='complete',documents=len(rows),conditions=dict(counts),work=manifest['expected_work'])


if __name__=='__main__':main()
