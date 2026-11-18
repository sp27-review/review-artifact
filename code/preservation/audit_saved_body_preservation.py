"""CPU-only audit of frozen edits; never imports models or changes inference inputs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def check_variant(doc, variant):
    original = doc['text']
    edits = variant['edits']
    assert edits == sorted(edits, key=lambda e: (e['start'], e['end']))
    pieces = []
    cursor = 0
    for index, edit in enumerate(edits):
        start, end = edit['start'], edit['end']
        assert 0 <= cursor <= start <= end <= len(original)
        if index:
            prior = edits[index - 1]
            assert not prior['start'] == prior['end'] == start
        assert original[start:end] == edit['before']
        pieces.extend((original[cursor:start], edit['after']))
        cursor = end
    pieces.append(original[cursor:])
    edited = ''.join(pieces)
    assert sha(edited.encode()) == variant['text_sha256']

    # Check every untouched interval against its shifted location, then reverse
    # the logged operations. Reversibility alone is not semantic equivalence.
    shift = cursor = original_cursor = 0
    restored = []
    for edit in edits:
        start = edit['start'] + shift
        end = start + len(edit['after'])
        assert edited[cursor:start] == original[original_cursor:edit['start']]
        assert edited[start:end] == edit['after']
        restored.extend((edited[cursor:start], edit['before']))
        cursor, original_cursor = end, edit['end']
        shift += len(edit['after']) - len(edit['before'])
    assert edited[cursor:] == original[original_cursor:]
    restored.append(edited[cursor:])
    assert ''.join(restored) == original

    for left, right in doc['protected']:
        for edit in edits:
            a, b = edit['start'], edit['end']
            assert not ((left < a < right) if a == b else (a < right and left < b))
        new_left = left + sum(len(e['after']) - (e['end'] - e['start'])
                              for e in edits if e['end'] <= left)
        new_right = right + sum(len(e['after']) - (e['end'] - e['start'])
                                for e in edits if e['end'] <= right
                                and not e['start'] == e['end'] == right)
        assert original[left:right] == edited[new_left:new_right]

    # This keeps whitespace-delimited token boundaries and their order, unlike
    # simply deleting all whitespace before comparing strings.
    tokens_equal = [s.casefold() for s in original.split()] == [s.casefold() for s in edited.split()]
    format_only = all((e['before'].casefold() == e['after'].casefold()) or
                      (not e['before'].strip() and not e['after'].strip()) for e in edits)
    if variant['name'] in {'single:blank_collapse_native', 'single:capital_title_case',
                            'single:section_case', 'selected_combo:blank_collapse_native+capital_title_case'}:
        assert format_only and tokens_equal

    move = variant['name'].startswith('position:move:')
    if move:
        deleted = [e for e in edits if e['start'] < e['end'] and not e['after']]
        inserted = [e for e in edits if e['start'] == e['end'] and e['after']]
        assert len(edits) == 2 and len(deleted) == len(inserted) == 1
        payload = deleted[0]['before']
        assert inserted[0]['after'] in {payload, '\n' + payload, payload + '\n', '\n' + payload + '\n'}
    return dict(untouched_intervals_exact=True, roundtrip_exact=True,
                protected_spans_exact=bool(doc['protected']),
                full_whitespace_tokens_casefold_equal=tokens_equal,
                only_case_or_whitespace_edits=format_only, moved_payload_exact=move)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    totals = Counter()
    by_condition = defaultdict(Counter)
    datasets = Counter()
    failures = []
    digest = hashlib.sha256()
    for number, ref in enumerate(manifest['documents'], 1):
        ident = ref['identifier']
        try:
            raw = Path(ref['path']).read_bytes()
            assert sha(raw) == ref['sha256'], 'input_file_hash'
            doc = json.loads(raw)
            assert doc['identifier'] == ident and doc['dataset'] == ref['dataset']
            assert sha(doc['text'].encode()) == doc['text_sha256'], 'original_text_hash'
            digest.update((ident + '\0' + sha(raw) + '\n').encode())
        except Exception as error:
            failures.append(dict(identifier=ident, stage='document', error=repr(error)))
            continue
        datasets[doc['dataset']] += 1
        totals['documents'] += 1
        totals['documents_with_protected_spans'] += bool(doc['protected'])
        for group in ('attacks', 'controls', 'positions'):
            for variant in doc[group]:
                if variant['status'] != 'valid':
                    continue
                key = doc['dataset'] + '|' + group + '|' + variant['name']
                totals['valid_variants'] += 1
                by_condition[key]['valid_variants'] += 1
                try:
                    checks = check_variant(doc, variant)
                    for name, passed in checks.items():
                        totals[name] += int(passed)
                        by_condition[key][name] += int(passed)
                except Exception as error:
                    failures.append(dict(identifier=ident, condition=key, error=repr(error)))
        if number % 5000 == 0:
            print(json.dumps(dict(documents=number, seconds=round(time.monotonic() - started, 2))), flush=True)
    report = dict(time_utc=datetime.now(timezone.utc).isoformat(),
                  status='passed' if not failures else 'failed', scope='frozen input strings and edit logs; no inference',
                  manifest=str(args.manifest.resolve()), manifest_sha256=sha(manifest_bytes),
                  script_sha256=sha(Path(__file__).read_bytes()), ordered_input_digest=digest.hexdigest(),
                  datasets=dict(datasets), totals=dict(totals),
                  by_condition={key: dict(value) for key, value in sorted(by_condition.items())},
                  failures=failures, seconds=time.monotonic() - started,
                  interpretation=['All valid variants are counted before model-specific length exclusions or paired-control availability.',
                                  'Unedited intervals and roundtrip verify edit locality/integrity; any reversible replacement can pass roundtrip.',
                                  'Casefold token equality checks spelling/order after ignoring case and whitespace; it is not a human meaning or naturalness judgment.',
                                  'Exact moved payload plus unchanged remainder does not establish unchanged contextual attachment.'])
    assert sha(args.manifest.read_bytes()) == report['manifest_sha256']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('status', 'datasets', 'totals', 'failures', 'seconds')}, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
