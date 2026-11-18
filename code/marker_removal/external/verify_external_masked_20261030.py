#!/usr/bin/env python3
"""Re-read source, outputs and audits; verify without importing the remover."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from itertools import zip_longest
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();out=args.out
    manifest=json.loads((out/'manifest.json').read_text())
    source=ROOT/manifest['source']
    assert sha(source)==manifest['source_sha256']
    for path,h in manifest['code_sha256'].items():
        assert sha(ROOT/path)==h,('code_hash',path)
    for name,info in manifest['outputs'].items():
        assert (out/name).stat().st_size==info['bytes'] and sha(out/name)==info['sha256'],name
    decisions_path=ROOT/'scripts/external_marking_review_20261030.json'
    assert sha(decisions_path)==manifest['review_decisions_sha256']
    decisions=defaultdict(list)
    for d in json.loads(decisions_path.read_text())['decisions']:
        decisions[d['doc_id']].append(d)
    rows={};stats=Counter()
    for raw,row,audit in zip_longest(read(source),read(out/'all.jsonl'),read(out/'audit.jsonl')):
        assert raw is not None and row is not None and audit is not None
        did=raw['doc_id'];assert did==row['doc_id']==audit['doc_id'] and did not in rows
        for key in ['label','label_id','source_id','dataset','language','source_split']:
            assert raw[key]==row[key],(did,key)
        assert row['label_id']=={'U':0,'C':1,'S':2,'TS':3}[row['label']]
        old,new=raw['text'],row['text']
        assert new.strip()
        assert hashlib.sha256(old.encode()).hexdigest()==raw['text_sha256']==row['source_text_sha256']==audit['source_text_sha256']
        assert hashlib.sha256(new.encode()).hexdigest()==row['text_sha256']==audit['masked_text_sha256']
        assert not ({'source_metadata','source_label','portion_mark_candidates','deletions'} & row.keys())
        last=0;chunks=[]
        for e in audit['deletions']:
            a,b=e['start'],e['end'];assert last<=a<b<=len(old)
            assert old[a:b]==e['text'];chunks.append(old[last:a]);last=b
        chunks.append(old[last:]);assert ''.join(chunks)==new,did
        def project(pos):
            return pos-sum(max(0,min(pos,e['end'])-e['start']) for e in audit['deletions'])
        for mark in audit['portion_mark_candidates']:
            assert old[mark['start']:mark['end']]==mark['mark']
            assert project(mark['start'])==mark['masked_anchor']
        for d in decisions[did]:
            assert raw['text_sha256']==d['source_text_sha256'] and old[d['start']:d['end']]==d['text']
            a,b=project(d['start']),project(d['end'])
            if d['action']=='remove':
                assert a==b,('manual_removal',did,d)
            else:
                assert new[a:b]==d['text'],('manual_retention',did,d)
        rows[did]=row
        stats.update(documents=1,changed_documents=old!=new,removed_spans=len(audit['deletions']),
                     removed_characters=len(old)-len(new),portion_marks=len(audit['portion_mark_candidates']))
    for k,v in stats.items():
        assert manifest['stats'][k]==v,(k,v)
    excluded=json.loads((out/'exclusions.json').read_text())
    expected_source=set(rows)-{e['doc_id'] for e in excluded if e['stage']=='source'}
    unique=list(read(out/'unique.jsonl'))
    assert len(unique)==len(expected_source)==manifest['source_members']
    assert {r['doc_id'] for r in unique}==expected_source
    assert all(r==rows[r['doc_id']] for r in unique)
    # Independently reproduce FRUS's wide overlap membership.
    overlap_path=ROOT/'audits/overlap_20261030/candidates.jsonl'
    assert sha(overlap_path)==manifest['overlap_audit_sha256']
    if manifest['dataset']=='frus_sachn':
        overlap={r['external']['id'] for r in read(overlap_path) if r['external']['dataset']=='frus_sachn' and r['cosine']>=.15}
        assert not expected_source & overlap
        assert len(set(rows)&overlap)==262
        assert len(expected_source)==21864
        source_groups=defaultdict(list)
        for r in read(source):
            if r['doc_id'] not in overlap:
                source_groups[hashlib.sha256(r['text'].strip().encode()).hexdigest()].append((r['doc_id'],r['label']))
        independently_kept={rs[0][0] for rs in source_groups.values() if len({r[1] for r in rs})==1}
        assert independently_kept==expected_source
    else:
        assert expected_source=={r['doc_id'] for r in read(source.parent/'unique.jsonl')}
        old_manifest=json.loads((source.parent.parent/'masked_20261030/manifest.json').read_text())
        assert sha(source.parent/'unique.jsonl')==old_manifest['source_files']['unique.jsonl']
        assert sha(source)==old_manifest['source_files']['all.jsonl']
    groups=defaultdict(list)
    for r in unique:
        groups[r['text'].strip()].append(r)
    keep={rs[0]['doc_id'] for rs in groups.values() if len({r['label'] for r in rs})==1}
    dedup=list(read(out/'deduplicated.jsonl'))
    assert len(dedup)==len(keep) and {r['doc_id'] for r in dedup}==keep
    holds={r['doc_id'] for r in read(out/'residual_audit.jsonl') if any(c['status']=='review' for c in r['candidates'])}
    evaluation=list(read(out/'evaluation.jsonl'))
    assert len(evaluation)==len(keep-holds) and {r['doc_id'] for r in evaluation}==keep-holds
    assert all(r==rows[r['doc_id']] for r in dedup+evaluation)
    labels={name:dict(Counter(r['label'] for r in records)) for name,records in [('deduplicated',dedup),('evaluation',evaluation)]}
    assert labels==manifest['label_counts']
    report=dict(status='passed',verified_utc=datetime.now(timezone.utc).isoformat(),
        verifier_sha256=sha(Path(__file__)),verifier=str(Path(__file__).relative_to(ROOT)),
        stats=dict(stats),source_members=len(expected_source),deduplicated_documents=len(dedup),evaluation_documents=len(evaluation),
        held_after_dedup=len(keep&holds),labels=labels,original_source_hash_unchanged=True,
        ids_and_labels_preserved=True,deletion_log_reconstructs_every_document=True,other_characters_exactly_preserved=True,
        anchors_reprojected=True,reviewed_retained_spans_preserved=True,reviewed_removals_applied=True,
        regression_cases_passed=json.loads((out/'verification.json').read_text())['regression_cases_passed'],
        limitation='Mechanical integrity and recorded contextual review verified; this is not proof of no unknown markings or validated protected paragraph boundaries.')
    (out/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    manifest['outputs']['verification.json']=dict(bytes=(out/'verification.json').stat().st_size,sha256=sha(out/'verification.json'))
    manifest['independent_verification']='passed'
    if manifest['dataset']=='wikileaks_cable_classifier':
        manifest['membership_reference']=dict(path=str((source.parent/'unique.jsonl').relative_to(ROOT)),sha256=sha(source.parent/'unique.jsonl'))
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
