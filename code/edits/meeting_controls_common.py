"""Frozen, response-independent controls and external meeting inputs (2026-10-30)."""
from __future__ import annotations
import collections
import json
import math
from pathlib import Path
import random
import re

import format_extension_common as c

RUN = c.BASE / 'runs/run_20261030_r5_controls_external_seed3'
ORDER = ['position:move:end', 'combo:date_header_blanks', 'combo:doctype_blanks',
         'position:repeat1:end', 'position:repeat3:end']
SEEDS = [7701, 7702, 7703]
SAFE_OPS = ['date_iso', 'date_month_spelling', 'header_case', 'header_alias',
            'header_colon_delete', 'header_value_newline', 'horizontal_space_insert',
            'section_case', 'section_alias', 'doctype_case', 'doctype_alias',
            'agency_case', 'blank_collapse', 'blank_expand', 'separator_style',
            'list_marker_style', 'copy_page_label_style']
EXTERNAL = {
    'wikileaks': c.ROOT/'data/datasets/wikileaks_cable_classifier/masked_20261030_r2',
    'frus': c.ROOT/'data/datasets/frus_sachn/masked_20261030',
}


def rng(ident, condition):
    return random.Random(int(c.h('controls-7701:'+ident+':'+condition), 16))


def size(e):
    return max(len(e['before']), len(e['after']))


def edit_type(e):
    return 'insert' if not e['before'] else 'delete' if not e['after'] else 'replace'


def position_control(doc, attack):
    """Same block/copy count; uniform admissible original line boundary, excluding end."""
    text = doc['text']; block = attack['block']; a,b = block['start'],block['end']
    candidates = sorted({0} | {m.end() for m in re.finditer('\n',text)})
    candidates = [p for p in candidates if p < len(text) and not a <= p <= b
                  and not c.protected_overlap(p,p,doc['protected'])]
    randomizer = rng(doc['identifier'],attack['name']); randomizer.shuffle(candidates)
    for pos in candidates:
        copies = 3 if 'repeat3' in attack['name'] else 1
        # Match end insertion's total separator character count exactly. At an
        # internal line boundary these separators trail the block instead.
        end_insert = next(e['after'] for e in attack['edits'] if e['start']==e['end'])
        separators = len(end_insert) - copies*len(block['text'])
        minimum = copies  # between copies plus trailing boundary
        if separators < minimum:
            continue
        inserted = '\n'.join([block['text']]*copies) + '\n'*(separators-(copies-1))
        edits = [c.make_edit(text,pos,pos,inserted)]
        if ':move:' in attack['name']:
            edits.append(c.make_edit(text,a,b,''))
        result = c.variant(doc,'random:'+attack['name'],[dict(edits=[e]) for e in edits],
                           family='random_position',paired_variant=attack['name'])
        if result['status']=='valid' and result['text_sha256']!=attack['text_sha256']:
            result['matching'] = dict(block=block, destination=pos, admissible_boundaries=len(candidates),
                source_chars=sum(len(e['before']) for e in edits),
                inserted_chars=sum(len(e['after']) for e in edits),
                target_characters_changed=attack['characters_changed'],
                character_ratio=result['characters_changed']/attack['characters_changed'],
                randomized='destination only; edit distance deliberately not matched')
            assert result['characters_changed']==attack['characters_changed']
            return result
    return dict(name='random:'+attack['name'],paired_variant=attack['name'],status='unavailable',
                reason='no_nonend_boundary_with_same_payload_and_separator_budget')


def combination_control(doc, attack):
    """Random benign format units matched to atomic edit count, type and size.

    Identity candidates are retained: all-occurrence attacks sometimes exhaust
    the alternatives. No prediction or focus score participates in selection.
    """
    targets = attack['edits']; pool=[]
    for op in SAFE_OPS:
        for unit in doc['units'].get(op,[]):
            if len(unit['edits'])==1:
                pool.append(dict(op=op,unit=unit,edit=unit['edits'][0]))
    matches=[]
    for target in targets:
        bucket=min(2,int((target['start']+target['end'])/2/max(1,len(doc['text']))*3))
        options=[i for i,p in enumerate(pool) if edit_type(target)==edit_type(p['edit'])
                 and .5 <= size(p['edit'])/max(size(target),1) <= 2
                 and min(2,int(sum(p['unit']['anchor'])/2/max(1,len(doc['text']))*3))==bucket]
        matches.append(options)
    if any(not v for v in matches):
        return dict(name='random:'+attack['name'],paired_variant=attack['name'],status='unavailable',reason='no_type_size_position_matched_unit')
    randomizer=rng(doc['identifier'],attack['name'])
    target_size=sum(size(e) for e in targets)
    for attempt in range(32):
        picked={}; occupied=[]
        # Scarce slots first; randomize ties and alternative unit choices.
        order=sorted(range(len(targets)),key=lambda i:(len(matches[i]),randomizer.random()))
        for i in order:
            choices=matches[i].copy();randomizer.shuffle(choices)
            for index in choices:
                candidate=pool[index]; e=candidate['edit']
                try:
                    normalized=c.normalize_edits(doc['text'],occupied+[e])
                    if len(normalized)!=len(occupied)+1:continue
                except ValueError:continue
                occupied.append(e);picked[i]=candidate;break
            else:break
        if len(picked)!=len(targets):continue
        ratio=sum(size(e) for e in occupied)/max(target_size,1)
        if not .8 <= ratio <= 1.25:continue
        chosen=[picked[i] for i in range(len(targets))]
        result=c.variant(doc,'random:'+attack['name'],[p['unit'] for p in chosen],
                         family='random_format',paired_variant=attack['name'])
        if result['status']!='valid':continue
        result['matching']=dict(target_edits=len(targets),control_edits=len(result['edits']),
            character_ratio=ratio,target_characters_changed=target_size,attempt=attempt,
            source_position='same document third per edit',per_edit_size_ratio=[.5,2],
            aggregate_size_ratio=[.8,1.25],edit_types_matched=True,
            operators=[p['op'] for p in chosen],
            identical_to_attack=result['text_sha256']==attack['text_sha256'],
            randomized='operator and eligible unit within matched type/size/position; operator families not identical')
        assert len(result['edits'])==len(targets)
        return result
    return dict(name='random:'+attack['name'],paired_variant=attack['name'],status='unavailable',reason='no_valid_nonoverlapping_size_matched_assignment_32_attempts')


def allocation(groups,total):
    sizes={k:len(v) for k,v in groups.items()}; n=sum(sizes.values())
    exact={k:total*v/n for k,v in sizes.items()}
    counts={k:min(sizes[k],max(1,int(v))) for k,v in exact.items()}
    while sum(counts.values())>total:
        key=max((k for k in counts if counts[k]>1),key=lambda k:counts[k]-exact[k]);counts[key]-=1
    while sum(counts.values())<total:
        key=max((k for k in counts if counts[k]<sizes[k]),key=lambda k:exact[k]-counts[k]);counts[key]+=1
    return counts


def external_sample(dataset,path,n=1000):
    summary=c.source.read_json(path/'length_summary.json')
    assert c.source.digest(path/'evaluation.jsonl')==summary['evaluation_sha256']
    eligible=set(c.source.read_json(path/'common_4096_ids.json'))
    lengths={r['doc_id']:r for r in c.source.records(path/'token_lengths.jsonl')}
    rows=[r for r in c.source.records(path/'evaluation.jsonl') if r['doc_id'] in eligible]
    groups={}
    for label in sorted({r['label_id'] for r in rows}):
        ordered=sorted((r for r in rows if r['label_id']==label),key=lambda r:(max(lengths[r['doc_id']]['tokens'].values()),r['doc_id']))
        for q in range(4):groups[(label,q)]=ordered[len(ordered)*q//4:len(ordered)*(q+1)//4]
    counts=allocation(groups,min(n,len(rows)));selected=[];strata=[];smoke=[]
    for key,group in sorted(groups.items()):
        group=sorted(group,key=lambda r:c.h('external-sample-7701:'+r['doc_id']))
        chosen=group[:counts[key]]
        strata.append(dict(label=key[0],length_quartile=key[1],population=len(group),sample=len(chosen)))
        for r in chosen:
            selected.append(dict(identifier=r['doc_id'],dataset=dataset,gold=r['label_id'],text=r['text'],
                text_sha256=r['text_sha256'],lengths=lengths[r['doc_id']]['tokens'],
                stratum=list(key),sampling_weight=len(group)/len(chosen)))
        if key[1] in (0,3):smoke.append(chosen[0]['doc_id'])
    selected.sort(key=lambda r:c.h('external-order-7701:'+r['identifier']))
    return selected,dict(dataset=dataset,source=str(path),source_sha256=summary['evaluation_sha256'],
        full_documents=summary['documents'],eligible_documents=len(rows),sample_documents=len(selected),
        strata=strata,smoke_identifiers=smoke,sampling='proportional label x within-label length quartile; deterministic hash without predictions')


def prepare(run=RUN):
    if (run/'manifest.json').exists():raise RuntimeError('Frozen manifest already exists; reuse it')
    (run/'inputs').mkdir(parents=True,exist_ok=True);counts=collections.Counter();rows=[]
    for doc in c.iter_documents():
        by_name={v['name']:v for v in doc['variants'] if v['status']=='valid'}
        attacks=[by_name[name] for name in ORDER if name in by_name]
        # Include combo-only documents for the full fixed-rule paired comparison.
        if not attacks:continue
        controls=[position_control(doc,v) if v['name'].startswith('position:') else combination_control(doc,v) for v in attacks]
        record={k:doc[k] for k in ['identifier','gold','text','text_sha256','cluster_id','protected','sensitive','fold']}
        record.update(dataset='cia',attacks=attacks,controls=controls,
                      search_eligible='position:move:end' in by_name,
                      original_input_sha256=c.source.digest(c.RUN/'inputs'/(c.key(doc['identifier'])+'.json')))
        for v in controls:
            counts[v['name']+':'+v['status']]+=1
            if v['status']=='valid':
                edited,_=c.apply_edits(doc,v['edits']);assert c.h(edited)==v['text_sha256']
                counts['control_valid']+=1
                if v.get('matching',{}).get('identical_to_attack'):counts['identity_controls']+=1
        p=run/'inputs'/(c.key(doc['identifier'])+'.json');c.source.write_json(p,record)
        rows.append(dict(identifier=doc['identifier'],dataset='cia',gold=doc['gold'],length=len(doc['text']),input_sha256=c.source.digest(p)))
    smoke=[]
    for label in range(4):
        group=sorted([r for r in rows if r['gold']==label],key=lambda r:(r['length'],r['identifier']))
        for q in range(4):
            part=group[len(group)*q//4:len(group)*(q+1)//4]
            smoke.append(min(part,key=lambda r:c.h('controls-smoke:'+r['identifier']))['identifier'])
    datasets={}
    for dataset,path in EXTERNAL.items():
        selected,meta=external_sample(dataset,path);datasets[dataset]=meta;smoke+=meta['smoke_identifiers']
        for r in selected:
            p=run/'inputs'/(c.key(r['identifier'])+'.json');c.source.write_json(p,r)
            rows.append(dict(identifier=r['identifier'],dataset=dataset,gold=r['gold'],input_sha256=c.source.digest(p)))
    manifest=dict(created=c.source.now(),status='complete',original_run=str(c.RUN),models=c.MODELS,
        inference_seeds=SEEDS,candidate_seed=7701,candidate_order=ORDER,documents=rows,
        smoke_identifiers=smoke,external=datasets,control_counts=dict(counts),
        definitions=dict(search='both full available pools and equal-actual-query paired pools; minimum original-prediction-class margin; gold excluded from selection',
            asr='wrong after edit among applicable originally correct; prediction change among applicable all',
            main_single_seed='reuse old attack/clean after fresh smoke consistency checks; all controls/external computed in separate seed cache',
            seed_repetition='same frozen texts; separate actual-forward caches per inference seed; cached old predictions are not seed repetitions'),
        source_hashes={str(p):c.source.digest(p) for p in [Path(__file__),Path(c.__file__),Path(c.source.__file__)]})
    c.source.write_json(run/'manifest.json',manifest)
    c.emit(stage='prepared',run=str(run),counts=dict(counts),documents=len(rows),smoke=len(smoke))


if __name__=='__main__':prepare()
