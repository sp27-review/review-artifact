"""External transfer of frozen no-copy attacks and five format diagnostics.

Never modifies the completed CIA controls or integrated external focus runs.
"""
from __future__ import annotations
import collections
import contextlib
import re
from pathlib import Path

import meeting_controls_common as m
import external_focus_line_endings as le

c = m.c
RUN = c.BASE / 'runs/run_20261031_r5_external_nocopy_diagnostics'
PRIMARY = m.ORDER[:3]
SINGLES = ['blank_collapse_native', 'capital_title_case', 'section_case', 'date_iso', 'header_colon_delete']
ACRONYMS = set('CIA FBI DIA NSA NPIC DC USIB NSC JCS NATO UNESCO UNICEF UNHCR UNDP USAID OECD ASEAN OPEC USSR NATO CENTO SEATO IMF IBRD IAEA WTO WHO UNCTAD UNSC EU GDP GNP USA UK US DDR FRG GDR USSR KGB FARC HAMAS HIV AIDS'.split())
ONE_WORD_TITLES = set('SUMMARY BACKGROUND INTRODUCTION CONCLUSION CONCLUSIONS COMMENT COMMENTS DISCUSSION ANALYSIS ASSESSMENT OVERVIEW RECOMMENDATIONS CONTENTS ANNEX APPENDIX'.split())


def annotations(doc):
    view, boundaries = le.annotation_view(doc['text'])
    parsed = c.focus.annotations(view, [], le.tagger())
    return {name: [[boundaries[a], boundaries[b]] for a, b in pairs]
            for name, pairs in parsed.items() if 'protected' not in name}


def build_units(doc):
    spans = annotations(doc)
    # The original operator implementation uses source coordinates. Supply the
    # CRLF-aware spans without changing its frozen module on disk or the input.
    original = c.focus.annotations
    try:
        c.focus.annotations = lambda *_: spans
        units, metadata, rejected = c.operator_units(doc, le.tagger())
    finally:
        c.focus.annotations = original
    capital = []
    for a, b in spans.get('layout:capital_line', []):
        value = doc['text'][a:b]
        words = re.findall('[A-Z]+', value)
        line_start=doc['text'].rfind('\n',0,a)+1
        prev_start=doc['text'].rfind('\n',0,max(0,line_start-1))+1
        next_start=doc['text'].find('\n',b)
        next_end=doc['text'].find('\n',next_start+1) if next_start>=0 else -1
        before=doc['text'][prev_start:line_start].strip()
        after_line=doc['text'][next_start+1:next_end if next_end>=0 else len(doc['text'])].strip() if next_start>=0 else ''
        boundary=lambda s: not s or bool(re.fullmatch(r'[-_=*~. ]+',s))
        isolated=(line_start==0 or boundary(before)) and (next_start<0 or boundary(after_line))
        # Keep lexical words, digits, identifiers and common acronyms intact.
        # Single-word signatories and page/cable identifiers are not headings.
        if (len(value.strip()) > 120 or re.search(r'[0-9/]', value)
                or not words or len(words) > 14
                or len(words) == 1 and words[0] not in ONE_WORD_TITLES
                or not isolated and not (len(words)==1 and words[0] in ONE_WORD_TITLES)):
            rejected['capital_title_case:nonheading_or_identifier'] = rejected.get('capital_title_case:nonheading_or_identifier', 0) + 1
            continue
        after = re.sub('[A-Z]+', lambda x: x.group() if len(x.group()) <= 3 or x.group() in ACRONYMS
                       else x.group().capitalize(), value)
        if value == after or c.protected_overlap(a, b, doc['protected']):
            continue
        assert value.casefold() == after.casefold()
        capital.append(dict(id=c.h(f'capital:{a}:{b}:{after}')[:16], anchor=[a,b],
                            edits=[c.make_edit(doc['text'], a, b, after)], metadata_removed=False))
    units['capital_title_case'] = capital
    units['blank_collapse_native']=[]
    for u in units.get('blank_collapse',[]):
        old=u['edits'][0]
        replacement='\r\n' if '\r\n' in old['before'] else '\r' if '\r' in old['before'] else '\n'
        edit=c.make_edit(doc['text'],old['start'],old['end'],replacement)
        units['blank_collapse_native'].append(dict(u,id=c.h('native:'+u['id'])[:16],edits=[edit]))
    # Benign, content-preserving control sites outside protected CIA portions.
    # Sampling is deterministic and independent of all model predictions.
    gaps=list(re.finditer(r'(?<=[A-Za-z0-9,.;:)])[ \t]+(?=[A-Za-z0-9(])',doc['text']))
    gaps=sorted(gaps,key=lambda x:c.h('space-control:'+doc['identifier']+':'+str(x.start())))[:256]
    units['horizontal_gap_expand']=[];units['horizontal_gap_delete_one']=[]
    for gap in gaps:
        a,b=gap.span()
        if c.protected_overlap(a,b,doc['protected']):continue
        units['horizontal_gap_expand'].append(dict(id=c.h('gap-expand:'+str(a))[:16],anchor=[a,b],
            edits=[c.make_edit(doc['text'],a,b,gap.group()+' ')],metadata_removed=False))
        if b-a>1:
            units['horizontal_gap_delete_one'].append(dict(id=c.h('gap-delete:'+str(a))[:16],anchor=[a,b],
                edits=[c.make_edit(doc['text'],b-1,b,'')],metadata_removed=False))
    return units, metadata, rejected, spans


def control(doc, attack):
    # Same predeclared type/size/document-third matcher as the completed study.
    # Identity controls remain explicit; never manufacture a weak baseline.
    old = m.SAFE_OPS
    try:
        if attack['name'] not in PRIMARY:
            m.SAFE_OPS = [op for op in old if op!='blank_collapse'] + ['blank_collapse_native','capital_title_case','horizontal_gap_expand','horizontal_gap_delete_one']
        return m.combination_control(doc, attack)
    finally:
        m.SAFE_OPS = old


def compose(doc, name, ops, family):
    if not all(doc['units'].get(op) for op in ops):
        return dict(name=name, status='not_applicable', reason='all_components_required', operators=ops, family=family)
    v = c.variant(doc, name, [u for op in ops for u in doc['units'][op]], family=family, operators=ops)
    if len(ops) > 1 and v['status'] == 'valid':
        components = [c.variant(doc, 'component', doc['units'][op]) for op in ops]
        if any(v['text_sha256'] == x.get('text_sha256') for x in components):
            return dict(name=name, status='not_applicable', reason='combination_collapses_to_single', operators=ops, family=family)
    return v


def build_doc(doc):
    assert c.h(doc['text']) == doc['text_sha256']
    units, metadata, rejected, spans = build_units(doc)
    doc = dict(doc, units=units, rejected=rejected, detected_counts={k:len(v) for k,v in spans.items()})
    spatial = {v['name']:v for v in c.spatial_variants(doc, metadata) if ':move:' in v['name']}
    attacks = [spatial.get(PRIMARY[0], dict(name=PRIMARY[0], status='not_applicable'))]
    for name in PRIMARY[1:]:
        attacks.append(compose(doc, name, c.COMBINATIONS[name.split(':',1)[1]], 'primary_combo'))
    for op in SINGLES:
        attacks.append(compose(doc, 'single:'+op, [op], 'single'))
    controls = [control(doc, v) for v in attacks if v['status']=='valid' and not v['name'].startswith('position:')]
    # Same existing front/middle/end rule; not a new random-location experiment.
    positions = [spatial[name] for name in ['position:move:front','position:move:middle'] if name in spatial]
    result = dict(doc, attacks=attacks, controls=controls, positions=positions)
    audit_doc(result)
    return result


def audit_doc(doc):
    assert c.h(doc['text']) == doc['text_sha256']
    for group in ['attacks','controls','positions']:
        for v in doc[group]:
            assert 'repeat' not in v['name']
            if v['status'] != 'valid': continue
            text, edits = c.apply_edits(doc, v['edits'])
            assert c.h(text) == v['text_sha256'] and text != doc['text']
            assert edits == v['edits']
            # All untouched spans are copied exactly by apply_edits; all marked
            # CIA portions are additionally checked there after coordinate mapping.
            if v.get('family') == 'single':
                op = v['operators'][0]
                for e in edits:
                    if op in ['capital_title_case','section_case']:
                        assert e['before'].casefold() == e['after'].casefold()
                    elif op == 'blank_collapse_native':
                        assert not e['before'].strip() and e['after'] in ['\n','\r\n','\r']
                        assert ('\r\n' in e['before']) == (e['after']=='\r\n')
                    elif op == 'header_colon_delete':
                        assert e['before']==':' and e['after']==''
                    elif op == 'date_iso':
                        assert c.date_iso(e['before']) == e['after']


def add_selected(doc, selection):
    v = compose(doc, selection['condition'], selection['operators'], 'dev_selected_combo')
    doc = dict(doc, attacks=doc['attacks']+[v], controls=list(doc['controls']))
    if v['status']=='valid': doc['controls'].append(control(doc,v))
    audit_doc(doc)
    return doc


def load_dev():
    selected = {r['identifier']:r for r in c.source.read_json(c.source.RUN_A/'manifests/whitebox_selection.json')['rows'] if r['split']=='dev'}
    assert len(selected)==256
    split_path = c.source.read_json(c.source.RUN_A/'config.json')['split_manifest']
    splits = {r['identifier']:r for r in c.source.records(split_path) if r['identifier'] in selected}
    protected=collections.defaultdict(list); sensitive=collections.defaultdict(list)
    for r in c.source.records(c.source.PORTIONS):
        if r['doc_id'] in selected and r.get('analysis_eligible'):
            protected[r['doc_id']] += r['masked_portion_spans']
            if r['portion_label']!='U': sensitive[r['doc_id']] += r['masked_portion_spans']
    rows=[]
    for r in c.source.records(c.source.BENCHMARK):
        ident=r['identifier']
        if ident not in selected:continue
        assert splits[ident]['split']=='dev' and r['label']==selected[ident]['label']
        text=r['marker_masked_text']
        rows.append(dict(identifier=ident,dataset='cia_dev',gold=c.LABELS.index(r['label']),text=text,
            text_sha256=c.h(text),cluster_id=splits[ident]['cluster_id'],protected=c.source.union(protected[ident]),
            sensitive=c.source.union(sensitive[ident])))
    assert len(rows)==256
    return sorted(rows,key=lambda d:c.h('dev-7701:'+d['identifier']))


def prepare(run=RUN):
    manifest_path=run/'prepared.json'
    if manifest_path.exists():return c.source.read_json(manifest_path)
    for part in ['inputs/dev','inputs/external','queue/logs','analysis']:(run/part).mkdir(parents=True,exist_ok=True)
    source_manifest=c.source.read_json(m.RUN/'manifest.json')
    docs=load_dev()
    for r in source_manifest['documents']:
        if r['dataset']=='cia':continue
        path=m.RUN/'inputs'/(c.key(r['identifier'])+'.json')
        assert c.source.digest(path)==r['input_sha256']
        d=c.source.read_json(path)
        docs.append(dict(d,protected=[],sensitive=[],source_input_path=str(path),source_input_sha256=r['input_sha256']))
    rows=[];counts=collections.Counter();examples=collections.defaultdict(list)
    for i,raw in enumerate(docs):
        phase='dev' if raw['dataset']=='cia_dev' else 'external'
        doc=build_doc(raw)
        if phase=='dev':
            doc['attacks']=[v for v in doc['attacks'] if v['name'].startswith('single:')]
            doc['controls']=[v for v in doc['controls'] if v['paired_variant'].startswith('single:')]
            doc['positions']=[]
        p=run/'inputs'/phase/(c.key(doc['identifier'])+'.json');c.source.write_json(p,doc)
        rows.append(dict(identifier=doc['identifier'],dataset=doc['dataset'],phase=phase,path=str(p),sha256=c.source.digest(p)))
        for group in ['attacks','controls','positions']:
            for v in doc[group]:
                counts[doc['dataset']+'|'+v['name']+'|'+v['status']]+=1
                if v.get('matching',{}).get('identical_to_attack'):counts[doc['dataset']+'|'+v['name']+'|identity']+=1
                if v['status']=='valid' and len(examples[doc['dataset']+'|'+v['name']])<4:
                    examples[doc['dataset']+'|'+v['name']].append(dict(identifier=doc['identifier'],edits=v['edits']))
        if (i+1)%128==0:c.emit(stage='prepare_external_attack',completed=i+1,total=len(docs))
    hashes={str(p):c.source.digest(p) for p in [Path(__file__),Path(m.__file__),Path(c.__file__),Path(le.__file__)]}
    result=dict(created=c.source.now(),status='prepared',documents=rows,models=c.MODELS,seed=7701,source_hashes=hashes,
        counts=dict(counts),primary=PRIMARY,singles=SINGLES,source_run=str(m.RUN),source_manifest_sha256=c.source.digest(m.RUN/'manifest.json'),
        policy=dict(no_training=True,no_copy=True,no_external_selection=True,dev='frozen CIA dev256, not portion_eval',
            controls='primary combinations retain old safe-operator pool; new diagnostics add lexical-preserving horizontal whitespace and capital/native-blank units; same atomic edit type/document third and size 0.5-2; total size 0.8-1.25; identities explicitly reported',
            position='same metadata block moved to front/middle/end using existing rule; separator counts may differ by one',
            selection='rank eligible single families by mean model paired ASR gain on dev256, minimum 20 correct model-document pairs; top two distinct families; fall back to greatest dev coverage if insufficient support',
            external_protection='No trusted portion boundaries; exact unchanged-region preservation and edit-type invariants, with sampled manual diff audit; no independent semantic oracle'))
    c.source.write_json(run/'analysis/input_examples.json',dict(examples));c.source.write_json(manifest_path,result)
    return result
