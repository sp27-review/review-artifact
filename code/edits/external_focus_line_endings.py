"""CPU-only line-ending-aware annotations on immutable external focus artifacts."""
from functools import lru_cache
from pathlib import Path
import re

import numpy as np
import meeting_controls_common as m

c=m.c
SCRIPT=Path(__file__).resolve()


def annotation_view(text):
    """Normalize line endings for the parser, mapping every boundary to source."""
    chars=[];boundaries=[0];i=0
    while i<len(text):
        if text[i]=='\r':
            i+=2 if text[i:i+2]=='\r\n' else 1
            chars.append('\n')
        else:
            chars.append(text[i]);i+=1
        boundaries.append(i)
    return ''.join(chars),boundaries


@lru_cache(maxsize=1)
def parser_identity():
    return dict(correction_sha256=c.source.digest(SCRIPT),
                focus_sha256=c.source.digest(Path(c.focus.__file__)),
                tagger_sha256=c.source.digest(c.source.TAGGER))


@lru_cache(maxsize=1)
def tagger():
    return c.source.import_file('external_focus_line_ending_tagger',c.source.TAGGER)


def corrected_spans(run,doc,text):
    path=run/'external_focus_corrected/annotations'/(c.key(doc['identifier'])+'.json')
    identity=dict(**parser_identity(),text_sha256=doc['text_sha256'])
    if path.exists():
        cached=c.source.read_json(path)
        if cached['identity']==identity:return cached['spans']
    view,boundaries=annotation_view(text)
    parsed=c.focus.annotations(view,[],tagger());spans={}
    for name,locations in parsed.items():
        if 'protected' in name:continue
        spans[name]=[]
        for a,b in locations:
            aa,bb=boundaries[a],boundaries[b]
            assert text[aa:bb].replace('\r\n','\n').replace('\r','\n')==view[a:b]
            spans[name].append([aa,bb])
    # Independent direct CRLF/LF matcher verifies the main missed category.
    if '\r' not in text.replace('\r\n',''):
        expected=[list(match.span()) for match in re.finditer(r'(?:\r\n|\n)[ \t]*(?:\r\n|\n)(?:[ \t]*(?:\r\n|\n))*',text)]
        assert spans['layout:blank_line']==expected
    path.parent.mkdir(parents=True,exist_ok=True)
    c.source.write_json(path,dict(identity=identity,spans=spans,
        note='Parser-only LF view, spans mapped to unchanged original input coordinates.'))
    return spans


def corrected_document(run,source_path,doc):
    """Return corrected derived metrics; never mutate predictions or extraction."""
    input_path=run/'inputs'/(c.key(doc['identifier'])+'.json')
    source_input=c.source.read_json(input_path);text=source_input['text']
    assert c.h(text)==doc['text_sha256']
    if '\r' not in text:return doc
    path=run/'external_focus_corrected'/doc['model']/'documents'/source_path.name
    identity=dict(**parser_identity(),source_document_sha256=c.source.digest(source_path),
                  input_manifest_sha256=doc['input_manifest_sha256'],artifact_sha256=doc['artifact_sha256'])
    if path.exists():
        cached=c.source.read_json(path)
        if cached.get('analysis_correction')==identity:return cached
    assert c.source.digest(input_path)==doc['input_manifest_sha256']
    artifact=Path(doc['artifact'])
    assert c.source.digest(artifact)==doc['artifact_sha256']
    spans=corrected_spans(run,doc,text)
    with np.load(artifact,allow_pickle=False) as arrays:
        offsets=arrays['offsets'];scores=arrays['signed_scores'].astype(np.float64)
        masses={'attribution':np.abs(scores)[None,:]}
        if 'attention_mean' in arrays:masses['attention']=arrays['attention_mean'].astype(np.float64)
        groups={}
        for method,mass in masses.items():
            c.focus.validate_arrays(offsets,mass,text,scores if method=='attribution' else None)
            values,_=c.focus.measure(offsets,mass,spans,len(text),scores if method=='attribution' else None)
            for value in values.values():
                value['layer_concentration']=[float(x) if np.isfinite(x) else None for x in value['layer_concentration']]
            groups[method]=values
    derived=dict(doc,spans=spans,groups=groups,analysis_correction=identity)
    assert derived['clean']==doc['clean']
    path.parent.mkdir(parents=True,exist_ok=True)
    c.source.write_json(path,derived)
    return derived
