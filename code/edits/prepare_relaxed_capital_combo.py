"""Prepare a versioned, CPU-only relaxation of the capital-line boundary rule.

The completed experiment stays immutable. Only the requirement for blank or
separator lines around a capital line changes; the other edit guards remain.
"""
from __future__ import annotations

import argparse
import collections
import re
import time
from pathlib import Path

import external_attack_common as legacy

c = legacy.c
VERSION = 'capital_line_no_blank_boundary_v1'
CONDITION = 'selected_combo_relaxed:blank_collapse_native+capital_title_case'
DEFAULT_RUN = c.BASE / 'runs/run_20261102_r5_relaxed_capital_combo'


def capital_units(doc, spans, require_isolated=False):
    """Keep legacy lexical/identifier/protection guards, optionally its boundary."""
    text = doc['text']
    units = []
    for a, b in spans.get('layout:capital_line', []):
        value = text[a:b]
        words = re.findall('[A-Z]+', value)
        if (len(value.strip()) > 120 or re.search(r'[0-9/]', value)
                or not words or len(words) > 14
                or len(words) == 1 and words[0] not in legacy.ONE_WORD_TITLES):
            continue
        if require_isolated:
            line_start = text.rfind('\n', 0, a) + 1
            prev_start = text.rfind('\n', 0, max(0, line_start - 1)) + 1
            next_start = text.find('\n', b)
            next_end = text.find('\n', next_start + 1) if next_start >= 0 else -1
            before = text[prev_start:line_start].strip()
            after_line = (text[next_start + 1:next_end if next_end >= 0 else len(text)].strip()
                          if next_start >= 0 else '')
            def boundary(s):
                return not s or bool(re.fullmatch(r'[-_=*~. ]+', s))
            isolated = ((line_start == 0 or boundary(before))
                        and (next_start < 0 or boundary(after_line)))
            if not isolated and not (len(words) == 1 and words[0] in legacy.ONE_WORD_TITLES):
                continue
        after = re.sub('[A-Z]+', lambda m: m.group() if len(m.group()) <= 3
                       or m.group() in legacy.ACRONYMS else m.group().capitalize(), value)
        if value == after or c.protected_overlap(a, b, doc['protected']):
            continue
        assert value.casefold() == after.casefold()
        units.append(dict(id=c.h(f'capital:{a}:{b}:{after}')[:16], anchor=[a, b],
                          edits=[c.make_edit(text, a, b, after)], metadata_removed=False))
    return units


def roundtrip(doc, variant):
    if variant['status'] != 'valid':
        return
    edited, edits = c.apply_edits(doc, variant['edits'])
    cursor = shift = 0
    pieces = []
    for edit in edits:
        start = edit['start'] + shift
        end = start + len(edit['after'])
        assert edited[start:end] == edit['after']
        pieces.extend([edited[cursor:start], edit['before']])
        cursor = end
        shift += len(edit['after']) - len(edit['before'])
    pieces.append(edited[cursor:])
    assert ''.join(pieces) == doc['text']
    assert c.h(edited) == variant['text_sha256']


def prepare(output=DEFAULT_RUN):
    started = time.monotonic()
    output = output.resolve()
    if (output / 'prepared.json').exists():
        raise FileExistsError('This prepared rule version is frozen; choose another output path.')
    source_manifest = legacy.c.source.read_json(legacy.RUN / 'external_manifest.json')
    selected = source_manifest['selection']
    assert selected['operators'] == ['blank_collapse_native', 'capital_title_case']
    old_prepared = c.source.read_json(legacy.RUN / 'prepared.json')
    for path, digest in old_prepared['source_hashes'].items():
        assert c.source.digest(path) == digest, (path, 'legacy source changed')
    cia_paths = sorted((c.RUN / 'inputs').glob('*.json'))
    assert len(cia_paths) == 2299
    sources = [('cia', path, None) for path in cia_paths]
    sources += [(r['dataset'], Path(r['path']), r['sha256']) for r in source_manifest['documents']]
    assert len(sources) == 4299
    totals = collections.defaultdict(collections.Counter)
    unique_texts = collections.defaultdict(set)
    strict_texts = collections.defaultdict(set)
    rows = []
    examples = collections.defaultdict(list)
    seen = set()
    verified_conditions = 0
    for index, (dataset, path, expected_digest) in enumerate(sources, 1):
        digest = c.source.digest(path)
        if expected_digest is not None:
            assert digest == expected_digest
        original = c.source.read_json(path)
        ident = original['identifier']
        assert ident not in seen
        seen.add(ident)
        assert c.h(original['text']) == original['text_sha256']
        doc = dict(original, dataset=dataset)
        units, _, rejected, spans = legacy.build_units(doc)
        strict_units = capital_units(doc, spans, require_isolated=True)
        assert strict_units == units['capital_title_case'], (ident, 'strict rule drift')
        relaxed_units = capital_units(doc, spans)
        assert {u['id'] for u in strict_units} <= {u['id'] for u in relaxed_units}
        doc['units'] = units
        strict = legacy.compose(doc, selected['condition'], selected['operators'], 'dev_selected_combo')
        strict_control = legacy.control(doc, strict) if strict['status'] == 'valid' else None
        if dataset != 'cia':
            saved = next(v for v in original['attacks'] if v['name'] == selected['condition'])
            assert strict['status'] == saved['status']
            if strict['status'] == 'valid':
                assert strict['text_sha256'] == saved['text_sha256']
                saved_control = next(v for v in original['controls'] if v['paired_variant'] == selected['condition'])
                assert strict_control == saved_control, (ident, 'strict control drift')
        doc['units'] = dict(units, capital_title_case=relaxed_units)
        attack = legacy.compose(doc, CONDITION, selected['operators'], 'user_revised_combo')
        # Keep the legacy control RNG key: only the eligible capital units change.
        # A new condition name alone must not randomly change unaffected controls.
        control = None
        if attack['status'] == 'valid':
            control = legacy.control(doc, dict(attack, name=selected['condition']))
            control = dict(control, name='random:' + CONDITION, paired_variant=CONDITION)
        stats = totals[dataset]
        stats['documents'] += 1
        stats['capital_detected_documents'] += bool(spans.get('layout:capital_line'))
        stats['strict_capital_documents'] += bool(strict_units)
        stats['relaxed_capital_documents'] += bool(relaxed_units)
        stats['strict_capital_sites'] += len(strict_units)
        stats['relaxed_capital_sites'] += len(relaxed_units)
        stats['strict_attack_valid'] += strict['status'] == 'valid'
        stats['strict_control_valid'] += strict_control is not None and strict_control['status'] == 'valid'
        stats['relaxed_attack_valid'] += attack['status'] == 'valid'
        stats['relaxed_control_valid'] += control is not None and control['status'] == 'valid'
        stats['newly_applicable_documents'] += strict['status'] != 'valid' and attack['status'] == 'valid'
        if strict['status'] == 'valid':
            assert attack['status'] == 'valid', (ident, 'relaxation lost an existing attack')
            stats['changed_preexisting_attack'] += strict['text_sha256'] != attack['text_sha256']
        for variant in [strict, strict_control]:
            if variant and variant['status'] == 'valid':
                strict_texts[dataset].add(variant['text_sha256'])
        for variant in [attack, control]:
            if variant and variant['status'] == 'valid':
                roundtrip(doc, variant)
                verified_conditions += 1
                unique_texts[dataset].add(variant['text_sha256'])
        if control and control['status'] == 'valid':
            assert .8 <= control['matching']['character_ratio'] <= 1.25
            assert control['matching']['target_edits'] == control['matching']['control_edits']
        if attack['status'] == 'valid':
            edited, _ = c.apply_edits(doc, attack['edits'])
            assert re.sub(r'\s+', '', edited).casefold() == re.sub(r'\s+', '', doc['text']).casefold()
        new_sites = [u for u in relaxed_units if u['id'] not in {s['id'] for s in strict_units}]
        if new_sites and len(examples[dataset]) < 5:
            examples[dataset].append(dict(identifier=ident, newly_allowed_edits=[u['edits'][0] for u in new_sites[:3]]))
        lean = {k: doc[k] for k in ['identifier', 'gold', 'text', 'text_sha256', 'protected', 'sensitive']}
        lean.update(dataset=dataset, rule_version=VERSION, source_input_path=str(path),
                    source_input_sha256=digest, attacks=[attack], controls=[control] if control else [], positions=[],
                    strict_attack_sha256=strict.get('text_sha256'), strict_status=strict['status'])
        legacy.audit_doc(lean)
        target = output / 'inputs' / (c.key(ident) + '.json')
        c.source.write_json(target, lean)
        rows.append(dict(identifier=ident, dataset=dataset, path=str(target), sha256=c.source.digest(target)))
        if index % 256 == 0:
            print(f'CPU preparation {index}/{len(sources)}', flush=True)
    for dataset, stats in totals.items():
        stats['strict_unique_edited_texts'] = len(strict_texts[dataset])
        stats['relaxed_unique_edited_texts'] = len(unique_texts[dataset])
        stats['new_texts_relative_to_strict_combo'] = len(unique_texts[dataset] - strict_texts[dataset])
    assert {ds: v['documents'] for ds, v in totals.items()} == {'cia': 2299, 'wikileaks': 1000, 'frus': 1000}
    assert totals['cia']['strict_attack_valid'] == 615 and totals['cia']['strict_control_valid'] == 597
    sources_hash = dict(old_prepared['source_hashes'])
    sources_hash[str(Path(__file__).resolve())] = c.source.digest(Path(__file__))
    manifest = dict(created=c.source.now(), status='cpu_prepared_no_inference', rule_version=VERSION,
                    condition=CONDITION, operators=selected['operators'], models=c.MODELS, seeds=[7701, 7702, 7703],
                    documents=rows, counts={k: dict(v) for k, v in totals.items()}, source_hashes=sources_hash,
                    source_selection_sha256=c.source.digest(legacy.RUN / 'selection.json'),
                    policy=dict(change='Remove adjacent blank/separator boundary requirement only',
                                retained='Capital-line detector, identifier/length/single-word/acronym guards, native line endings, CIA protected spans, matched control constraints',
                                control_rng_key=selected['condition'],
                                selection='Same two operator families; no reranking or new dev selection',
                                timing_of_revision='User-requested protocol revision after strict external results were observed; not an untouched external confirmation',
                                inference='New version requires revised external predictions and CIA evaluation; no cross-seed cache reuse',
                                protection='CIA verified portion boundaries; external lacks equivalent validated boundaries'))
    c.source.write_json(output / 'analysis/coverage.json', dict(created=c.source.now(), counts=manifest['counts'],
        seconds=time.monotonic() - started, examples=dict(examples), verified_valid_conditions=verified_conditions,
        validation='Strict rule reproduced, existing external strict attacks/controls reproduced, relaxed capital sites are a superset, all edits roundtrip, CIA protected text preserved'))
    c.source.write_json(output / 'prepared.json', manifest)
    print(manifest['counts'], flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    prepare(args.output)
