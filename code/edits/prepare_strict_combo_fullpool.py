"""Find the unchanged strict combo in the entire masked external evaluation pool.

CPU-only screening. No prediction, attention or attribution is used to select
documents. The original length gate and source exclusions remain in force.
"""
from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

import external_attack_common as e
from prepare_relaxed_capital_combo import capital_units, roundtrip

c = e.c
DEFAULT_RUN = c.BASE / 'runs/run_20261102_r5_strict_combo_fullpool'


def layout_spans(text):
    view, boundaries = e.le.annotation_view(text)
    return {name: [[boundaries[m.start()], boundaries[m.end()]]
                   for m in c.focus.STRUCTURE[name].finditer(view)]
            for name in ['layout:capital_line', 'layout:blank_line']}


def prepare(output=DEFAULT_RUN):
    started = time.monotonic()
    output = output.resolve()
    if (output / 'prepared.json').exists():
        raise FileExistsError('Full-pool manifest already exists; keep this census immutable.')
    (output / 'analysis').mkdir(parents=True, exist_ok=True)
    selected = c.source.read_json(e.RUN / 'selection.json')
    assert selected['operators'] == ['blank_collapse_native', 'capital_title_case']
    old = c.source.read_json(e.RUN / 'external_manifest.json')
    old_by_id = {row['identifier']: row for row in old['documents']}
    expected_sources = c.source.read_json(e.RUN / 'prepared.json')['source_hashes']
    for path, digest in expected_sources.items():
        assert c.source.digest(path) == digest
    counts = collections.defaultdict(collections.Counter)
    label_counts = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    rows = []
    sources = {}
    original_sample_verified = 0
    valid_conditions = 0
    seen = set()
    with (output / 'analysis/screening.jsonl').open('w') as stream:
        for dataset, root in e.m.EXTERNAL.items():
            lengths = {row['doc_id']: row for row in c.source.records(root / 'token_lengths.jsonl')}
            eligible = set(c.source.read_json(root / 'common_4096_ids.json'))
            summary = c.source.read_json(root / 'length_summary.json')
            digest = c.source.digest(root / 'evaluation.jsonl')
            assert digest == summary['evaluation_sha256']
            sources[dataset] = dict(path=str(root / 'evaluation.jsonl'), sha256=digest,
                                    length_gate='Unchanged baseline common_4096_ids, edited inputs still need predictor length validation',
                                    length_summary_sha256=c.source.digest(root / 'length_summary.json'))
            for index, source in enumerate(c.source.records(root / 'evaluation.jsonl'), 1):
                ident = source['doc_id']
                assert ident not in seen
                seen.add(ident)
                assert c.h(source['text']) == source['text_sha256'] == lengths[ident]['text_sha256']
                assert (ident in eligible) == lengths[ident]['common_4096']
                doc = dict(identifier=ident, dataset=dataset, gold=source['label_id'],
                           text=source['text'], text_sha256=source['text_sha256'],
                           protected=[], sensitive=[], lengths=lengths[ident]['tokens'])
                spans = layout_spans(doc['text'])
                capitals = capital_units(doc, spans, require_isolated=True)
                applicable = bool(capitals and spans['layout:blank_line'])
                stats = counts[dataset]
                stats['evaluation_documents'] += 1
                stats['blank_documents_all_lengths'] += bool(spans['layout:blank_line'])
                stats['capital_documents_all_lengths'] += bool(spans['layout:capital_line'])
                stats['strict_capital_documents_all_lengths'] += bool(capitals)
                stats['strict_combo_documents_all_lengths'] += applicable
                stats['baseline_length_eligible'] += ident in eligible
                stats['strict_combo_over_length'] += applicable and ident not in eligible
                label_counts[dataset]['evaluation'][source['label']] += 1
                screen = dict(identifier=ident, text_sha256=doc['text_sha256'],
                              baseline_length_eligible=ident in eligible,
                              strict_combo_layout_applicable=applicable,
                              original_focus_sample=ident in old_by_id)
                original = None
                if ident in old_by_id:
                    old_row = old_by_id[ident]
                    assert c.source.digest(old_row['path']) == old_row['sha256']
                    original = c.source.read_json(old_row['path'])
                    assert original['text'] == doc['text']
                    assert capitals == original['units']['capital_title_case']
                    saved_attack = next(v for v in original['attacks'] if v['name'] == selected['condition'])
                    assert applicable == (saved_attack['status'] == 'valid')
                    original_sample_verified += 1
                if applicable and ident in eligible:
                    units, _, _, full_spans = e.build_units(doc)
                    assert units['capital_title_case'] == capitals
                    assert full_spans['layout:blank_line'] == spans['layout:blank_line']
                    doc['units'] = units
                    attack = e.compose(doc, selected['condition'], selected['operators'], 'dev_selected_combo')
                    assert attack['status'] == 'valid'
                    control = e.control(doc, attack)
                    if original is not None:
                        assert attack['text_sha256'] == saved_attack['text_sha256']
                        saved_control = next(v for v in original['controls'] if v['paired_variant'] == selected['condition'])
                        assert control == saved_control
                    for variant in [attack, control]:
                        if variant['status'] == 'valid':
                            roundtrip(doc, variant)
                            valid_conditions += 1
                    stats['strict_combo_length_eligible'] += 1
                    stats['paired_control_valid'] += control['status'] == 'valid'
                    stats['paired_control_unavailable'] += control['status'] != 'valid'
                    stats['paired_in_original_sample'] += control['status'] == 'valid' and original is not None
                    stats['paired_outside_original_sample'] += control['status'] == 'valid' and original is None
                    label_counts[dataset]['attack_eligible'][source['label']] += 1
                    if control['status'] == 'valid':
                        label_counts[dataset]['paired_eligible'][source['label']] += 1
                        assert .8 <= control['matching']['character_ratio'] <= 1.25
                        assert control['matching']['target_edits'] == control['matching']['control_edits']
                    screen['control_status'] = control['status']
                    lean = {key: doc[key] for key in ['identifier', 'dataset', 'gold', 'text', 'text_sha256', 'protected', 'sensitive', 'lengths']}
                    lean.update(source_evaluation_path=str(root / 'evaluation.jsonl'),
                                source_evaluation_sha256=digest, source_line=index,
                                attacks=[attack], controls=[control], positions=[],
                                original_focus_sample=original is not None)
                    e.audit_doc(lean)
                    path = output / 'inputs' / (c.key(ident) + '.json')
                    c.source.write_json(path, lean)
                    rows.append(dict(identifier=ident, dataset=dataset, path=str(path), sha256=c.source.digest(path),
                                     paired=control['status'] == 'valid', original_focus_sample=original is not None,
                                     max_baseline_tokens=max(doc['lengths'].values()), gold=doc['gold']))
                stream.write(json.dumps(screen, ensure_ascii=False) + '\n')
                if index % 1000 == 0:
                    progress = dict(status='cpu_screening', dataset=dataset, scanned=index,
                                    counts={key: dict(value) for key, value in counts.items()}, seconds=time.monotonic()-started)
                    c.source.write_json(output / 'analysis/progress.json', progress)
                    print(f"{dataset} CPU screening {index}/{summary['documents']}", flush=True)
            assert stats['evaluation_documents'] == summary['documents']
            assert stats['baseline_length_eligible'] == len(eligible)
    assert original_sample_verified == 2000
    assert counts['wikileaks']['paired_in_original_sample'] == 307
    assert counts['frus']['paired_in_original_sample'] == 35
    rows.sort(key=lambda row: c.h('strict-fullpool-7701:' + row['identifier']))
    hashes = dict(expected_sources)
    for path in [Path(__file__), Path(__file__).with_name('prepare_relaxed_capital_combo.py')]:
        hashes[str(path.resolve())] = c.source.digest(path)
    manifest = dict(created=c.source.now(), status='fullpool_screened_no_gpu_inference',
                    condition=selected['condition'], operators=selected['operators'],
                    source_selection_sha256=c.source.digest(e.RUN / 'selection.json'),
                    sources=sources, source_hashes=hashes, documents=rows,
                    counts={key: dict(value) for key, value in counts.items()},
                    label_counts={ds: {kind: dict(values) for kind, values in groups.items()} for ds, groups in label_counts.items()},
                    policy=dict(rule='Original strict isolated-capital rule unchanged',
                                population='Full masked evaluation files after existing exclusions; same baseline common length gate',
                                selection='Only format applicability, original length eligibility, and matched-control availability; no model predictions or focus vectors',
                                sampling='This is a complete candidate census; no final GPU subsample selected here',
                                inference='New candidates need clean and edited inference; full attention/gradient extraction is not required for this fixed rule',
                                interpretation='Attack-eligible conditional population, separate from the original representative 1000-document accuracy/focus sample',
                                protection='External strict heading heuristics and lexical invariants; not validated portion boundaries'))
    c.source.write_json(output / 'prepared.json', manifest)
    c.source.write_json(output / 'analysis/verification.json', dict(status='passed',
        original_sample_documents_verified=original_sample_verified,
        verified_valid_conditions=valid_conditions, seconds=time.monotonic()-started,
        fast_layout_matches_saved_strict_sample=True, unchanged_legacy_attacks_and_controls=True,
        source_and_length_hashes_verified=True, gpu_inference_started=False))
    c.source.write_json(output / 'analysis/progress.json', dict(status='complete', counts=manifest['counts']))
    print(json.dumps(manifest['counts'], ensure_ascii=False, indent=2), flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT_RUN)
    prepare(parser.parse_args().output)
