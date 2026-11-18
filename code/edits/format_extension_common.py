"""Frozen portion2299 inputs and auditable, composable format edits (CPU only)."""
from __future__ import annotations

import collections
import datetime
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'experiments/H_format_edit_screen'
X = ROOT / 'experiments/X_nonllm_classifiers'
sys.path.insert(0, str(X / 'scripts'))
import extract_classifier_attention as source
import analyze_classifier_focus as focus

RUN = BASE / 'runs/run_20261028_r5_portion2299_sixmodel_format_extension'
ATTENTION = X / 'runs/run_20261028_r5_final_classifier_attention'
PLAN = BASE / 'analysis/extension_plan_20261028/plan.json'
MODELS = ['llama', 'qwen', 'nemotron', 'longformer', 'deberta', 'tfidf']
MODEL_RUNS = dict(source.MODEL_RUNS, tfidf=source.RUN_C)
LABELS = source.LABELS
MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
          'August', 'September', 'October', 'November', 'December']
MONTH_RX = r'(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?'
DATE_RX = re.compile(r'(?<!\w)(\d{1,2})\s+' + MONTH_RX + r',?\s+(\d{4})(?!\d)|(?<!\w)' + MONTH_RX + r'\s+(\d{1,2}),?\s+(\d{4})(?!\d)', re.I)
HEADER_ALIASES = {'SUBJECT':'Topic', 'SUBJ':'Topic', 'FROM':'Sender', 'TO':'Recipient',
                  'DATE':'Date', 'REF':'Reference', 'REFS':'References', 'ATTN':'Attention',
                  'ENCL':'Enclosures', 'COPIES':'Copies', 'SOURCE':'Source'}
SECTION_ALIASES = {'SUMMARY':'Overview', 'INTRODUCTION':'Introduction',
                   'TABLE OF CONTENTS':'Contents', 'CONTENTS':'Table of contents'}
DOCTYPE_ALIASES = {'MEMORANDUM':'Memo', 'STAFF MEMORANDUM':'Staff memo',
                   'INTELLIGENCE MEMORANDUM':'Intelligence memo'}
GUIDED_OPERATORS = ['date_iso', 'header_alias', 'section_alias', 'doctype_alias',
                    'blank_collapse', 'header_colon_delete', 'agency_case']


def h(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def key(ident):
    return h(ident)[:20]


def emit(**values):
    print(json.dumps(dict(time=source.now(), **values), ensure_ascii=False), flush=True)


def load_dataset():
    selection = source.read_json(source.RUN_A / 'manifests/whitebox_selection.json')['rows']
    selected = [r for r in selection if r['split'] == 'portion_eval']
    wanted = {r['identifier']:r for r in selected}
    assert len(wanted) == len(selected) == 2299
    cfg = source.read_json(source.RUN_A / 'config.json')
    split = {r['identifier']:r for r in source.records(cfg['split_manifest'])}
    portions, sensitive = collections.defaultdict(list), collections.defaultdict(list)
    portion_grade = collections.defaultdict(lambda:-1)
    for r in source.records(source.PORTIONS):
        if r['doc_id'] in wanted and r.get('analysis_eligible'):
            portions[r['doc_id']] += r['masked_portion_spans']
            grade = {'U':0, 'C':1, 'S':2, 'TS':3}[r['portion_label']]
            portion_grade[r['doc_id']] = max(portion_grade[r['doc_id']], grade)
            if grade:
                sensitive[r['doc_id']] += r['masked_portion_spans']
    result = []
    for r in source.records(source.BENCHMARK):
        ident = r['identifier']
        if ident not in wanted:
            continue
        assert r['label'] == wanted[ident]['label']
        text = r['marker_masked_text']
        cluster = split[ident]['cluster_id']
        result.append(dict(identifier=ident, gold=LABELS.index(r['label']), text=text,
                           text_sha256=h(text), cluster_id=cluster,
                           fold=int(h('format-fold-7701:'+cluster)[:8],16)%5,
                           protected=source.union(portions[ident]),
                           sensitive=source.union(sensitive[ident]),
                           portion_max_grade=portion_grade[ident]))
    assert len(result) == 2299
    return sorted(result, key=lambda r:r['identifier'])


def protected_overlap(a, b, spans):
    return any((c < a < d) if a == b else (a < d and c < b) for c,d in spans)


def make_edit(text, a, b, after):
    assert 0 <= a <= b <= len(text)
    return dict(start=int(a), end=int(b), before=text[a:b], after=after)


def normalize_edits(text, edits):
    unique = {(e['start'],e['end'],e['after']):e for e in edits if e['before'] != e['after']}
    rows = sorted(unique.values(), key=lambda e:(e['start'], e['end']))
    for i,e in enumerate(rows):
        if text[e['start']:e['end']] != e['before']:
            raise ValueError('Source edit text mismatch')
        if i and (rows[i-1]['end'] > e['start'] or rows[i-1]['start'] == rows[i-1]['end'] == e['start']):
            raise ValueError('Overlapping edit components')
    return rows


def map_spans(spans, edits):
    output = []
    for a,b in spans:
        left = sum(len(e['after'])-(e['end']-e['start']) for e in edits if e['end'] <= a)
        right = sum(len(e['after'])-(e['end']-e['start']) for e in edits
                    if e['end'] <= b and not e['start'] == e['end'] == b)
        output.append([a+left,b+right])
    return output


def apply_edits(doc, edits):
    text = doc['text']; edits = normalize_edits(text, edits)
    for e in edits:
        if protected_overlap(e['start'],e['end'],doc['protected']):
            raise ValueError('Edit touches an eligible portion')
    out = []; previous = 0
    for e in edits:
        out += [text[previous:e['start']],e['after']];previous=e['end']
    out.append(text[previous:]); edited = ''.join(out)
    mapped = map_spans(doc['protected'],edits)
    if any(text[a:b] != edited[c:d] for (a,b),(c,d) in zip(doc['protected'],mapped)):
        raise ValueError('Protected text was not preserved after coordinate mapping')
    return edited, edits


def core_span(text, a, b):
    while a < b and not text[a].isalnum(): a += 1
    while b > a and not text[b-1].isalnum(): b -= 1
    return a,b


def title_case(value):
    return re.sub(r'[A-Za-z]+', lambda m:m.group().capitalize() if m.group() not in
                  {'CIA','FBI','DIA','NSA','NPIC','DC','USIB','NSC','JCS'} else m.group(), value)


def date_iso(value):
    def convert(m):
        if m.group(1): day,month,year = m.group(1),m.group(2),m.group(3)
        else: month,day,year = m.group(4),m.group(5),m.group(6)
        month = next(i+1 for i,n in enumerate(MONTHS) if n.lower().startswith(month[:3].lower()))
        return datetime.date(int(year),month,int(day)).isoformat()
    return DATE_RX.sub(convert,value)


def operator_units(doc, tagger, discovery_signatures=()):
    text=doc['text']; spans=focus.annotations(text,doc['sensitive'],tagger)
    units=collections.defaultdict(list); rejected=collections.Counter(); metadata=[]
    def add(op,a,b,after,anchor=None,removed=False):
        if text[a:b] == after:return
        if protected_overlap(a,b,doc['protected']):
            rejected[op+':protected']+=1;return
        e=make_edit(text,a,b,after)
        units[op].append(dict(id=h(op+':'+str(a)+':'+str(b)+':'+after)[:16],
                             edits=[e],anchor=list(anchor or (a,b)),metadata_removed=removed))
    def cue(name):return spans.get('cue:'+name,[])
    for a,b in cue('docid:standalone_date'):
        a,b=core_span(text,a,b); value=text[a:b]
        try:add('date_iso',a,b,date_iso(value))
        except ValueError:rejected['date_iso:invalid_calendar_date']+=1
        month=re.compile(r'\b'+MONTH_RX+r'\b',re.I)
        def swap(m):
            value=m.group().rstrip('.'); full=next(n for n in MONTHS if n[:3].lower()==value[:3].lower())
            return full[:3] if len(value)>3 else full
        add('date_month_spelling',a,b,month.sub(swap,value))
        add('date_delete',a,b,'',removed=True)
        metadata.append((a,b,'date'))
    for a,b in cue('header:field_colon'):
        value=text[a:b]
        m=re.search(r'(?P<name>[A-Za-z][A-Za-z .]*?)\s*(?P<sep>[:\-–])(?P<gap>\s*)$',value)
        if not m:rejected['header:parse']+=1;continue
        na,nb=a+m.start('name'),a+m.end('name'); sa,sb=a+m.start('sep'),a+m.end('sep')
        name=text[na:nb]; add('header_case',na,nb,title_case(name))
        alias=HEADER_ALIASES.get(name.strip().upper())
        if alias:add('header_alias',na,nb,alias)
        if text[sa:sb]==':':add('header_colon_delete',sa,sb,'',anchor=(na,nb))
        ga,gb=a+m.start('gap'),a+m.end('gap')
        if '\n' not in text[ga:gb]:add('header_value_newline',ga,gb,'\n',anchor=(na,nb))
        add('header_label_delete',na,nb,'',removed=True)
        if gb>ga and '\n' not in text[ga:gb]:add('horizontal_space_insert',ga,gb,text[ga:gb]+' ',anchor=(na,nb))
        end=text.find('\n',sb);end=len(text) if end<0 else end
        if end-na<=240 and not protected_overlap(na,end,doc['protected']):metadata.append((na,end,'header'))
    for a,b in cue('docid:section_header'):
        a,b=core_span(text,a,b); value=text[a:b]
        add('section_case',a,b,title_case(value))
        alias=SECTION_ALIASES.get(re.sub(r'\s+',' ',value).upper())
        if alias:add('section_alias',a,b,alias)
        add('section_delete',a,b,'',removed=True)
        m=re.match(r'[ \t]*\n[ \t\n]*',text[b:])
        if m:add('section_boundary_join',b,b+m.end(),' ',anchor=(a,b))
    for a,b in source.union(cue('docid:doc_type_caps')+cue('docid:doc_type_title_case')):
        a,b=core_span(text,a,b); value=text[a:b]
        add('doctype_case',a,b,title_case(value));alias=DOCTYPE_ALIASES.get(value.upper())
        if alias:add('doctype_alias',a,b,alias)
        add('doctype_delete',a,b,'',removed=True)
        left=text.rfind('\n',0,a)+1; right=text.find('\n',b);right=len(text) if right<0 else right
        if not text[left:a].strip() and not text[b:right].strip():metadata.append((a,b,'doctype'))
    for a,b in cue('docid:agency_caps'):
        add('agency_case',a,b,title_case(text[a:b]))
    for a,b in spans.get('layout:blank_line',[]):
        add('blank_collapse',a,b,'\n');add('blank_expand',b,b,'\n',anchor=(a,b))
    for m in re.finditer(r'(?m)^[ \t]*([-_=*~.]{3,})[ \t]*$',text):
        a,b=m.span(1);add('separator_delete',a,b,'')
        add('separator_style',a,b,('=' if text[a]!='=' else '-')*(b-a))
    for m in re.finditer(r'(?m)^[ \t]*(\d+|[A-Za-z])([.)])(?=[ \t])',text):
        a,b=m.span(2);add('list_marker_style',a,b,')' if m.group(2)=='.' else '.',anchor=m.span())
    # Sentence splitting can cut at the abbreviation "No.". Recover the same
    # copy/page pattern on the whole text before splitting its label and value.
    copy_rx=next(rx for _,rules in tagger.FORMAT_RULE_GROUPS for label,rx in rules if label=='docid:copy_page')
    for a,b in source.union(cue('docid:copy_page')+[list(m.span()) for m in copy_rx.finditer(text)]):
        value=text[a:b];m=re.search(r'\bcopy\s*(?:no\.?|number|#)?(?=\s*\d)',value,re.I)
        if m:
            aa,bb=a+m.start(),a+m.end()
            # A bare "copy " match may consume the only separator before the value.
            suffix=' ' if bb<len(text) and text[bb].isdigit() else ''
            add('copy_page_label_style',aa,bb,'Copy number'+suffix,anchor=(a,b))
        else:
            m=re.search(r'\bpage(?=\s+\d)',value,re.I)
            if m:add('copy_page_label_style',a+m.start(),a+m.end(),'p.',anchor=(a,b))
    for label,op in [('control:redaction_code','redaction_layout'),('docid:rdp_number','identifier_layout'),
                     ('control:release_stamp','release_layout'),('control:handling','handling_layout')]:
        for a,b in cue(label):
            aa,bb=core_span(text,a,b)
            left=text.rfind('\n',0,aa)+1; right=text.find('\n',bb);right=len(text) if right<0 else right
            if not text[left:aa].strip() and not text[bb:right].strip():add(op,right,right,'\n',anchor=(aa,bb))
    sigs=set(discovery_signatures)
    for m in re.finditer(r'(?m)^([^\n]+)$',text):
        value=m.group().strip();sig=re.sub(r'\d+','<N>',re.sub(r'\s+',' ',value))
        if sig not in sigs or len(value)>100:continue
        a=m.start()+len(m.group())-len(m.group().lstrip());b=a+len(value)
        if protected_overlap(a,b,doc['protected']):continue
        add('discovery_boundary_blank',m.end(),m.end(),'\n',anchor=(a,b))
        if value in {'Attachment','Attachments','Report'}:
            add('discovery_label_case',a,b,value.upper())
    # Remove duplicate matches without changing original-coordinate identities.
    units={op:list({u['id']:u for u in us}.values()) for op,us in units.items()}
    metadata=[dict(start=a,end=b,kind=k,text=text[a:b]) for a,b,k in sorted(set(metadata))
              if a<b and not protected_overlap(a,b,doc['protected'])]
    return units,metadata,dict(rejected)


def variant(doc,name,units,**metadata):
    try:
        text,edits=apply_edits(doc,[e for u in units for e in u['edits']])
    except ValueError as error:
        return dict(name=name,status='unavailable',reason=str(error),**metadata)
    if not edits:return dict(name=name,status='not_applicable',**metadata)
    return dict(name=name,status='valid',text_sha256=h(text),edits=edits,
                sites=len(units),characters_changed=sum(max(len(e['before']),len(e['after'])) for e in edits),
                metadata_removed=any(u.get('metadata_removed',False) for u in units),**metadata)


def score_units(doc,units,whiteboxes):
    all_units={u['id']:u for us in units.values() for u in us}; scores=collections.defaultdict(dict)
    for model in MODELS:
        row=whiteboxes[model][doc['identifier']]
        with np.load(row['token_artifact'],allow_pickle=False) as z:
            offsets=z['occurrence_offsets' if model=='tfidf' else 'offsets'].copy()
            signed=z['signed_contributions' if model=='tfidf' else 'signed_scores'].astype(float)
        evidences=[('attribution',offsets,np.abs(signed))]
        if model!='tfidf':
            p=ATTENTION/model/'documents'/(key(doc['identifier'])+'.json')
            r=source.read_json(p)
            with np.load(r['artifact'],allow_pickle=False) as z:
                offsets=z['offsets'].copy();mass=z['attention_mean'].astype(float)
            valid=offsets[:,1]>offsets[:,0]; total=mass[:,valid].sum(1)
            evidences.append(('attention',offsets,(mass/total[:,None]).mean(0)))
        for method,offsets,mass in evidences:
            valid=offsets[:,1]>offsets[:,0];total=mass[valid].sum()
            density=focus.integral(offsets,mass/max(total,1e-30),len(doc['text']))
            count=focus.integral(offsets,np.ones(len(offsets)),len(doc['text']))
            n=valid.sum()
            for ident,u in all_units.items():
                a,b=u['anchor'];share=(count[b]-count[a])/max(n,1)
                scores[ident][model+':'+method]=float((density[b]-density[a])/share) if share>0 else None
    for op,us in units.items():
        rankings=collections.defaultdict(list)
        for method in [m+':'+v for m in MODELS for v in ('attention','attribution') if not(m=='tfidf' and v=='attention')]:
            valid=[u for u in us if scores[u['id']].get(method) is not None]
            values=np.array([scores[u['id']][method] for u in valid])
            for u in valid:
                value=scores[u['id']][method]
                # Percentile midrank preserves ties instead of rewarding source order.
                rankings[u['id']].append(float(((values<value).sum()+.5*(values==value).sum())/len(values)))
        for u in us:
            scores[u['id']]['consensus']=float(np.mean(rankings[u['id']])) if rankings[u['id']] else 0.
    return dict(scores)


def load_whiteboxes():
    return {m:{r['identifier']:r for r in source.records(p/'results'/m/'whitebox.jsonl')
               if r['split']=='portion_eval'} for m,p in MODEL_RUNS.items()}


def spatial_variants(doc,metadata):
    if not metadata:return []
    block=min(metadata,key=lambda b:h(doc['identifier']+str(b['start'])+'metadata'))
    a,b=block['start'],block['end'];text=doc['text'];payload=block['text']
    candidates=[m.end() for m in re.finditer('\n',text) if not protected_overlap(m.end(),m.end(),doc['protected']) and not a<=m.end()<=b]
    positions={'front':0,'end':len(text)}
    if candidates:positions['middle']=min(candidates,key=lambda p:abs(p-len(text)/2))
    rows=[]
    for place,pos in positions.items():
        if a<=pos<=b:continue
        prefix='\n' if pos and text[pos-1]!='\n' else ''
        suffix='\n' if pos<len(text) else ''
        for copies in (1,3):
            atom=dict(edits=[make_edit(text,pos,pos,prefix+'\n'.join([payload]*copies)+suffix)],anchor=[a,b])
            rows.append(variant(doc,f'position:repeat{copies}:{place}',[atom],family='position',block=block))
        atoms=[dict(edits=[make_edit(text,a,b,'')],anchor=[a,b]),
               dict(edits=[make_edit(text,pos,pos,prefix+payload+suffix)],anchor=[a,b])]
        rows.append(variant(doc,f'position:move:{place}',atoms,family='position',block=block))
    return rows


def guided_variants(doc,units,scores):
    available=[op for op in GUIDED_OPERATORS if len(units.get(op,[]))>=2]
    # At most two operator families per document; selection does not use effects or gold labels.
    chosen=sorted(available,key=lambda op:h('guided-7701:'+doc['identifier']+op))[:2]
    rows=[]
    for op in chosen:
        us=units[op]; policies=['consensus','random','low_focus']+[m+':'+v for m in MODELS for v in ('attention','attribution') if not(m=='tfidf' and v=='attention')]
        for policy in policies:
            if policy=='random':ordered=sorted(us,key=lambda u:h('7701:'+doc['identifier']+u['id']))
            elif policy=='low_focus':ordered=sorted(us,key=lambda u:(scores[u['id']]['consensus'],u['id']))
            else:
                field=policy
                ordered=sorted([u for u in us if scores[u['id']].get(field) is not None],
                               key=lambda u:(-scores[u['id']][field],u['id']))
            for n in (1,3):
                if len(ordered)<n:continue
                picked=ordered[:n]
                metadata=dict(family='guided',operator=op,policy=policy,dose=n,unit_ids=[u['id'] for u in picked],
                              focus_scores=[scores[u['id']] for u in picked])
                rows.append(variant(doc,f'guided:{op}:{policy}:{n}',picked,**metadata))
                if policy in ('random','low_focus','consensus'):continue
                # Matched random and low-focus controls for the exact high-focus set.
                used={u['id'] for u in picked}; controls=[]
                for target in picked:
                    aa,bb=target['anchor']; size=max(bb-aa,1); bucket=min(2,int(((aa+bb)/2)/max(1,len(doc['text']))*3))
                    pool=[u for u in us if u['id'] not in used and
                          min(2,int(sum(u['anchor'])/2/max(1,len(doc['text']))*3))==bucket and
                          .5<=max(u['anchor'][1]-u['anchor'][0],1)/size<=2]
                    if not pool:break
                    control=min(pool,key=lambda u:h('matched-7701:'+target['id']+u['id']))
                    controls.append(control);used.add(control['id'])
                if len(controls)==n:
                    rows.append(variant(doc,f'matched:{op}:{policy}:{n}',controls,family='matched',
                                        operator=op,policy=policy,dose=n,paired_variant=f'guided:{op}:{policy}:{n}',
                                        unit_ids=[u['id'] for u in controls]))
    return rows


COMBINATIONS={
    'header_case_colon':['header_case','header_colon_delete'],
    'section_case_blanks':['section_case','blank_collapse'],
    'date_header':['date_iso','header_alias'],
    'doctype_blanks':['doctype_alias','blank_collapse'],
    'date_header_blanks':['date_iso','header_alias','blank_collapse'],
    'section_doctype_blanks':['section_alias','doctype_alias','blank_collapse'],
}


def build_document(doc,tagger,whiteboxes,discovery):
    units,metadata,rejected=operator_units(doc,tagger,discovery)
    basic=[r['id'] for r in source.read_json(PLAN)['atomic_operator_catalog']]
    variants=[variant(doc,'atomic:'+op,units.get(op,[]),family='atomic',operator=op) for op in basic]
    for op in sorted(set(units)-set(basic)):
        variants.append(variant(doc,'extended:'+op,units[op],family='extended',operator=op))
    scores=score_units(doc,units,whiteboxes)
    variants+=guided_variants(doc,units,scores)
    spatial=spatial_variants(doc,metadata);variants+=spatial
    for name,ops in COMBINATIONS.items():
        if all(units.get(op) for op in ops):
            variants.append(variant(doc,'combo:'+name,[u for op in ops for u in units[op]],family='combo',operators=ops))
    date_units=units.get('date_iso',[])
    move=next((r for r in spatial if r['name']=='position:move:front' and r['status']=='valid'),None)
    if date_units and move:
        block=move['block'];a,b=block['start'],block['end'];composed=[]
        # Rewrite dates inside the moving block in the inserted payload, never also
        # edit the original span that the move removes.
        for e in move['edits']:
            e=dict(e)
            if e['start']==e['end']:
                try:e['after']=date_iso(e['after'])
                except ValueError:pass
            composed.append(dict(edits=[e]))
        composed += [u for u in date_units if all(not(e['start']<b and a<e['end']) for e in u['edits'])]
        variants.append(variant(doc,'combo:date_move',composed,family='combo',operators=['date_iso','position:move:front']))
    # One text shared across condition aliases is inferred only once per model.
    for v in variants:
        if v['status']=='valid':
            v['mapped_sensitive']=map_spans(doc['sensitive'],v['edits'])
    return dict(**doc,variants=variants,unit_counts={k:len(v) for k,v in units.items()},
                rejected=rejected,unit_scores=scores,units=units)


def variant_for_model(v,model):
    if v['family'] not in ('guided','matched'):return True
    policy=v['policy']
    return policy in ('consensus','random','low_focus') or policy.startswith(model+':')


def iter_documents(run=RUN,selection=None):
    manifest=source.read_json(run/'manifest.json')
    for r in manifest['documents']:
        if selection is None or r['identifier'] in selection:
            yield source.read_json(run/'inputs'/(key(r['identifier'])+'.json'))
