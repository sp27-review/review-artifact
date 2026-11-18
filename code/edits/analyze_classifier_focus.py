"""CPU-only, repeatable focus analysis of saved classifier attention/attribution.

No checkpoint loading and no inference. Inputs are snapshots; an incomplete
attention run can be analysed and rerun as more documents/models arrive.
"""
from __future__ import annotations

import argparse
import collections
import csv
import fcntl
import hashlib
import html
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

import extract_classifier_attention as source

MODELS = ['llama', 'qwen', 'nemotron', 'longformer', 'deberta', 'tfidf']
RUNS = dict(source.MODEL_RUNS, tfidf=source.RUN_C)
STRUCTURE = {
    'layout:newline': re.compile(r'\n+'),
    'layout:blank_line': re.compile(r'\n[ \t]*\n(?:[ \t]*\n)*'),
    'layout:repeated_space': re.compile(r'[ \t]{2,}'),
    'layout:separator': re.compile(r'[-_=*~.]{3,}'),
    'layout:brackets': re.compile(r'[\[\](){}]'),
    'layout:colon': re.compile(r':'),
    'layout:enumerator': re.compile(r'(?m)^[ \t]*(?:\d+[.)]|[A-Za-z][.)]|[-*•])[ \t]+'),
    'layout:capital_line': re.compile(r'(?m)^[ \t]*[A-Z][A-Z0-9 /&,:;()\-]{4,}[ \t]*$'),
}
POSITIONS = ('beginning', 'middle', 'end')
SCOPES = ('native', 'shared_prefix')


def snapshot_jsonl(path, audit):
    """Only a final unfinished line may be ignored in a live producer's file."""
    path = Path(path)
    if not path.exists():
        return []
    data = path.read_bytes()
    rows = []
    lines = data.splitlines(keepends=True)
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines)-1 and not line.endswith(b'\n'):
                audit['unfinished_lines'].append(str(path))
                break
            raise
    audit['source_snapshots'][str(path)] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    return rows


def clip(spans, end, start=0):
    return source.union([(max(start, a), min(end, b)) for a, b in spans if a < end and b > start])


def merge(spans):
    return source.union([tuple(s) for s in spans])


def difference(left, right):
    out = []
    right = merge(right)
    for a, b in merge(left):
        cursor = a
        for c, d in right:
            if d <= cursor:
                continue
            if c >= b:
                break
            if c > cursor:
                out.append((cursor, c))
            cursor = max(cursor, d)
        if cursor < b:
            out.append((cursor, b))
    return out


def intersect(left, right):
    return source.union([(max(a, c), min(b, d)) for a, b in left for c, d in right if max(a, c) < min(b, d)])


def annotations(text, protected, tagger):
    # Character coordinates remain independent of each model's tokenizer.
    _, _, spans, _ = source.annotate(text, [(0, len(text))], protected, tagger)
    spans.pop('text')
    for name, rule in STRUCTURE.items():
        spans[name] = [(m.start(), m.end()) for m in rule.finditer(text)]
    spans['format_and_protected'] = intersect(spans['format'], spans['protected'])
    spans['format_without_protected'] = difference(spans['format'], spans['protected'])
    spans['protected_without_format'] = difference(spans['protected'], spans['format'])
    spans['neither_format_nor_protected'] = difference([(0, len(text))], spans['format']+spans['protected'])
    return spans


def validate_arrays(offsets, mass, text, signed=None):
    offsets = np.asarray(offsets)
    if offsets.ndim != 2 or offsets.shape[1] != 2 or not np.issubdtype(offsets.dtype, np.integer):
        raise ValueError('Invalid character offsets')
    if np.any(offsets < 0) or np.any(offsets[:, 1] < offsets[:, 0]) or np.any(offsets > len(text)):
        raise ValueError('Offsets lie outside the input text')
    if mass.ndim != 2 or mass.shape[1] != len(offsets) or not np.isfinite(mass).all() or np.any(mass < 0):
        raise ValueError('Invalid nonnegative token/feature mass')
    if signed is not None and (signed.shape != (len(offsets),) or not np.isfinite(signed).all()):
        raise ValueError('Invalid signed attribution')


def measure(offsets, mass, spans, end, signed=None):
    """Normalize each layer on the same visible text, then average layers.

    Boundary tokens are allocated by character overlap; special tokens have
    zero length. Zero mass/zero coverage is missing evidence, never zero lift.
    """
    names = list(spans)
    visible = source.overlap_weights(offsets, [(0, end)])
    weights = np.stack([source.overlap_weights(offsets, clip(spans[n], end)) for n in names])
    units = float(visible.sum())
    totals = mass @ visible
    valid = totals > 0
    if units <= 0 or not valid.any():
        return {}, None
    normalized = mass[valid] / totals[valid, None]
    fractions = normalized @ weights.T
    shares = weights.sum(1) / units
    signed_values = None if signed is None else (signed @ weights.T) / float(totals[0])
    result = {}
    for i, name in enumerate(names):
        if shares[i] <= 0:
            continue
        values = fractions[:, i] / shares[i]
        full_layers = np.full(len(mass), np.nan)
        full_layers[valid] = values
        result[name] = dict(token_fraction=float(shares[i]), mass_fraction=float(fractions[:, i].mean()),
            concentration=float(values.mean()), layer_concentration=full_layers,
            signed_mass=None if signed_values is None else float(signed_values[i]))
    return result, dict(normalized=normalized.mean(0), units=units, visible=visible)


def integral(offsets, values, end):
    """Exact character-proportional span sums, including overlapping features."""
    lengths = offsets[:, 1]-offsets[:, 0]
    good = lengths > 0
    density = np.asarray(values)[good] / lengths[good]
    starts = np.minimum(offsets[good, 0], end)
    stops = np.minimum(offsets[good, 1], end)
    delta = np.zeros(end+1, dtype=np.float64)
    np.add.at(delta, starts, density)
    np.add.at(delta, stops, -density)
    return np.concatenate(([0.0], np.cumsum(np.cumsum(delta)[:end])))


def span_sum(prefix, spans):
    return sum(float(prefix[b]-prefix[a]) for a, b in spans)


def positional_measures(offsets, cache, spans, end):
    mass = integral(offsets, cache['normalized'], end)
    units = integral(offsets, np.ones(len(offsets)), end)
    output = []
    for i, position in enumerate(POSITIONS):
        a, b = end*i//3, end*(i+1)//3
        position_mass = mass[b]-mass[a]
        position_units = units[b]-units[a]
        if position_mass <= 0 or position_units <= 0:
            continue
        for name, locations in spans.items():
            here = clip(locations, b, a)
            token_share = span_sum(units, here)/position_units
            if token_share <= 0:
                continue
            fraction = span_sum(mass, here)/position_mass
            output.append((name, position, dict(token_fraction=float(token_share), mass_fraction=float(fraction),
                concentration=float(max(0, fraction)/token_share), signed_mass=None)))
    return output


def discover(text, offsets, cache, spans, end, count):
    """Inspect unlabeled windows, retaining content as well as format candidates."""
    if count <= 0:
        return []
    mass = integral(offsets, cache['normalized'], end)
    units = integral(offsets, np.ones(len(offsets)), end)
    known = merge([s for name, ss in spans.items() if name.startswith(('cue:', 'layout:')) for s in ss])
    covered = np.zeros(end+1)
    for a, b in clip(known, end):
        covered[a:b] = 1
    covered = np.concatenate(([0.], covered[:end].cumsum()))
    windows = []
    for line in re.finditer(r'[^\n]+', text[:end]):
        words = list(re.finditer(r'\S+', line.group()))
        for i in range(0, len(words), 4):
            a = line.start()+words[i].start()
            b = line.start()+words[min(i+7, len(words)-1)].end()
            if b-a < 3 or (covered[b]-covered[a])/(b-a) >= .5:
                continue
            share = (units[b]-units[a])/cache['units']
            fraction = mass[b]-mass[a]
            if share <= 0 or fraction <= 0:
                continue
            windows.append((float(fraction/share), float(fraction), a, b))
    selected = []
    for concentration, fraction, a, b in sorted(windows, reverse=True):
        if any(max(a, r['start']) < min(b, r['end']) for r in selected):
            continue
        quote = text[a:b]
        selected.append(dict(start=a, end=b, text=quote, context=text[max(0, a-70):min(end, b+70)],
            signature=re.sub(r'\d+', '<N>', re.sub(r'\s+', ' ', quote.strip())),
            concentration=concentration, mass_fraction=fraction,
            relative_position=(a+b)/2/end,
            protected_character_fraction=sum(d-c for c, d in intersect([(a,b)], spans['protected']))/(b-a)))
        if len(selected) == count:
            break
    return selected


def statistics(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(n=0, mean=None, median=None, p10=None, p90=None, fraction_above_one=None)
    return dict(n=len(values), mean=float(values.mean()), median=float(np.median(values)),
        p10=float(np.quantile(values, .1)), p90=float(np.quantile(values, .9)),
        fraction_above_one=float(np.mean(values > 1+1e-6)))


def write_csv(path, rows, fields=None):
    rows = list(rows)
    if fields is None:
        fields = list(rows[0]) if rows else ['no_rows']
    temp = path.with_suffix('.tmp.csv')
    with temp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


class Collector:
    def __init__(self):
        self.cells = collections.defaultdict(list)
        self.layers = {}
        self.positions = collections.defaultdict(list)
        self.eligible = collections.defaultdict(set)
        self.paired = collections.defaultdict(dict)
        self.examples = collections.defaultdict(list)
        self.heads = {}
        self.predictions = {}

    def add(self, model, method, scope, row, prediction, metrics, text, spans, end):
        ident, split = row['identifier'], row['split']
        self.predictions[split, model, method, ident] = prediction
        strata = ['all', 'correct' if prediction == source.LABELS.index(row['label']) else 'incorrect', 'gold:'+row['label']]
        for stratum in strata:
            self.eligible[model, method, scope, split, stratum].add(ident)
        for name, metric in metrics.items():
            value = (metric['concentration'], metric['mass_fraction'], metric['token_fraction'], metric['signed_mass'])
            for stratum in strata:
                self.cells[model, method, scope, split, name, stratum].append(value)
            key = model, method, scope, split, name
            layer = metric['layer_concentration']
            finite = np.isfinite(layer)
            if key not in self.layers:
                self.layers[key] = [np.zeros(len(layer)), np.zeros(len(layer), dtype=int)]
            self.layers[key][0] += np.where(finite, layer, 0)
            self.layers[key][1] += finite
            if scope == 'shared_prefix':
                self.paired[split, model, method, name][ident] = metric['concentration']
                locations = clip(spans[name], end)
                if locations:
                    a, b = locations[0]
                    example = dict(identifier=ident, concentration=metric['concentration'],
                        gold=row['label'], prediction=source.LABELS[prediction], start=a, end=b,
                        text=text[a:min(b, a+200)], context=text[max(0,a-60):min(end, b+60)],
                        occurrence_count=len(locations), example_span='first occurrence; score covers all candidate occurrences')
                    examples = self.examples[split, model, method, name]
                    examples.append(example)
                    examples.sort(key=lambda x:(x['concentration'], x['identifier']))
                    # Keep both low-focus and high-focus examples with distinct IDs.
                    if len(examples) > 4:
                        del examples[1]

    def matrix(self):
        output = []
        for key, values in sorted(self.cells.items()):
            model, method, scope, split, cue, stratum = key
            n = len(self.eligible[model, method, scope, split, stratum])
            stats = statistics([v[0] for v in values])
            signed = [v[3] for v in values if v[3] is not None]
            output.append(dict(model=model, method=method, scope=scope, split=split, cue=cue, stratum=stratum,
                eligible_documents=n, present_documents=len(values), prevalence=len(values)/n,
                mean_token_fraction_when_present=float(np.mean([v[2] for v in values])),
                mean_mass_fraction_when_present=float(np.mean([v[1] for v in values])),
                mean_mass_fraction_all_documents=float(sum(v[1] for v in values)/n),
                mean_concentration=stats['mean'], median_concentration=stats['median'],
                p10_concentration=stats['p10'], p90_concentration=stats['p90'], fraction_above_one=stats['fraction_above_one'],
                mean_signed_mass_over_absolute_total=float(np.mean(signed)) if signed else None,
                unit='feature_occurrence' if method == 'coefficient' else 'token'))
        return output

    def common(self, expected_models, min_documents):
        result = []
        splits = sorted({k[0] for k in self.paired})
        for split in splits:
            cues = sorted({k[3] for k in self.paired if k[0] == split})
            for family in ('attention', 'attribution'):
                expected = [m for m in expected_models if family != 'attention' or m != 'tfidf']
                methods = {m:('attention' if family == 'attention' else 'coefficient' if m == 'tfidf' else 'gradient') for m in expected}
                available = [m for m in expected if self.eligible[m, methods[m], 'shared_prefix', split, 'all']]
                if not available:
                    continue
                paired_ids = set.intersection(*(self.eligible[m, methods[m], 'shared_prefix', split, 'all'] for m in available))
                for cue in cues:
                    by_model = {}
                    for model in available:
                        values = self.paired[split, model, methods[model], cue]
                        by_model[model] = dict(method=methods[model], **statistics([v for ident,v in values.items() if ident in paired_ids]))
                    if not any(v['n'] for v in by_model.values()):
                        continue
                    focused = [m for m, v in by_model.items() if v['n'] >= min_documents and v['median'] > 1+1e-6]
                    examples = {m:[e for e in self.examples[split,m,methods[m],cue] if e['identifier'] in paired_ids] for m in available}
                    result.append(dict(split=split, family=family, cue=cue,
                        expected_models=expected, available_models=available, missing_models=[m for m in expected if m not in available],
                        paired_documents=len(paired_ids), paired_identifiers_sha256=hashlib.sha256('\n'.join(sorted(paired_ids)).encode()).hexdigest(),
                        focused_models=focused, focused_model_count=len(focused), models=by_model, examples=examples))
        return sorted(result, key=lambda r:(r['split'], r['family'], -r['focused_model_count'], r['cue']))


def load_sources(args, selected, audit):
    wanted = {r['identifier']:r for r in selected}
    gradients, attentions = {}, {}
    for model in args.models:
        path = RUNS[model] / 'results' / model / 'whitebox.jsonl'
        rows = snapshot_jsonl(path, audit)
        gradients[model] = {}
        for row in rows:
            ident = row['identifier']
            if ident not in wanted:
                continue
            if ident in gradients[model]:
                raise ValueError(f'Duplicate whitebox document: {model}/{ident}')
            target = wanted[ident]
            if row['split'] != target['split'] or row['gold'] != source.LABELS.index(target['label']):
                raise ValueError('Whitebox label/split mismatch')
            gradients[model][ident] = row
        attentions[model] = {}
        for root in args.attention_root:
            directory = root / model
            meta_path = directory / 'manifest.json'
            if not meta_path.exists():
                continue
            manifest = source.read_json(meta_path)
            audit['attention_manifests'][str(meta_path)] = dict(sha256=source.digest(meta_path),
                source_run=manifest.get('source_run'), checkpoint=manifest.get('checkpoint'), method=manifest.get('method'))
            if manifest['model'] != model or manifest.get('input_field') != 'marker_masked_text':
                raise ValueError('Unsupported attention manifest')
            if manifest.get('source_run') and Path(manifest['source_run']).resolve() != RUNS[model].resolve():
                raise ValueError('Attention and attribution come from different training runs')
            state_path = RUNS[model] / 'results' / model / 'training_state.json'
            if manifest.get('checkpoint') and state_path.exists():
                state = source.read_json(state_path)
                if Path(state['best_checkpoint']).resolve() != Path(manifest['checkpoint']).resolve():
                    raise ValueError('Attention does not use the selected classifier checkpoint')
            manifest_rows = {r['identifier']:r for r in manifest['documents']}
            for ident, target in wanted.items():
                filename = hashlib.sha256(ident.encode()).hexdigest()[:20]+'.json'
                path = directory / 'documents' / filename
                if not path.exists():
                    continue
                row = source.read_json(path)
                if row['identifier'] != ident or row['split'] != target['split'] or row['gold'] != source.LABELS.index(target['label']):
                    raise ValueError('Attention label/split mismatch')
                if ident not in manifest_rows or row['text_sha256'] != manifest_rows[ident]['text_sha256']:
                    raise ValueError('Attention document does not match its manifest')
                if ident in attentions[model]:
                    raise ValueError(f'Overlapping attention inputs: {model}/{ident}; choose one source per document')
                attentions[model][ident] = row
    return gradients, attentions


def load_evidence(model, wb, attention, text, audit):
    result = []
    gradient_offsets = gradient_ids = None
    if wb is not None:
        path = Path(wb['token_artifact'])
        with np.load(path, allow_pickle=False) as artifact:
            if model == 'tfidf':
                offsets = artifact['occurrence_offsets'].copy()
                signed = artifact['signed_contributions'].astype(np.float64)
                method = 'coefficient'
            else:
                offsets = artifact['offsets'].copy()
                signed = artifact['signed_scores'].astype(np.float64)
                gradient_offsets = offsets
                gradient_ids = artifact['input_ids'].copy()
                method = 'gradient'
        mass = np.abs(signed)[None, :]
        validate_arrays(offsets, mass, text, signed)
        if model != 'tfidf' and int(offsets[:,1].max()) != int(wb['visible_chars']):
            raise ValueError('Gradient visible range differs from saved token offsets')
        result.append(dict(method=method, offsets=offsets, mass=mass, signed=signed, prediction=wb['prediction'],
            visible=int(wb['visible_chars']), artifact=str(path), artifact_sha256=source.digest(path),
            heads=None, head_names=None, head_token_fractions=None))
    if attention is not None:
        if hashlib.sha256(text.encode()).hexdigest() != attention['text_sha256']:
            raise ValueError('Attention text has changed')
        path = Path(attention['artifact'])
        if source.digest(path) != attention['artifact_sha256']:
            raise ValueError(f'Attention artifact has changed: {path}')
        with np.load(path, allow_pickle=False) as artifact:
            offsets = artifact['offsets'].copy()
            ids = artifact['input_ids'].copy()
            mass = artifact['attention_mean'].astype(np.float64)
            heads = artifact['group_attention'].astype(np.float64)
            head_names = artifact['group_names'].tolist()
            head_shares = artifact['group_token_fractions'].copy()
            audit['head_token_rows_present'][model] += int('attention_heads' in artifact.files)
        validate_arrays(offsets, mass, text)
        if gradient_ids is not None and (not np.array_equal(ids, gradient_ids) or not np.array_equal(offsets, gradient_offsets)):
            raise ValueError('Saved attention and gradients use different input tokens')
        result.append(dict(method='attention', offsets=offsets, mass=mass, signed=None, prediction=attention['prediction'],
            visible=int(offsets[:,1].max()), artifact=str(path), artifact_sha256=attention['artifact_sha256'],
            heads=heads, head_names=head_names, head_token_fractions=head_shares))
    return result


def head_summary(collector, model, split, evidence):
    heads = evidence['heads']
    if heads is None:
        return
    names = evidence['head_names']
    totals = heads[:, :, names.index('text')]
    for i, name in enumerate(names):
        share = evidence['head_token_fractions'][i]
        if name == 'text' or share <= 0:
            continue
        valid = totals > 0
        values = np.divide(heads[:,:,i], totals*share, out=np.zeros_like(totals), where=valid)
        key = model, split, name
        if key not in collector.heads:
            collector.heads[key] = [np.zeros_like(values), np.zeros_like(values, dtype=int)]
        collector.heads[key][0] += values
        collector.heads[key][1] += valid


def agreement_rows(collector):
    rows = []
    for split, model, method, cue in sorted(collector.paired):
        if method != 'attention':
            continue
        a = collector.paired[split, model, method, cue]
        b = collector.paired.get((split, model, 'gradient', cue), {})
        all_ids = sorted(set(a) & set(b))
        ids = [i for i in all_ids if collector.predictions[split,model,'attention',i] == collector.predictions[split,model,'gradient',i]]
        if not ids:
            rows.append(dict(split=split, model=model, cue=cue, paired_present_documents=0,
                different_prediction_documents=len(all_ids), pearson_concentration=None,
                both_above_one=None, attention_only_above_one=None, gradient_only_above_one=None))
            continue
        x, y = np.array([a[i] for i in ids]), np.array([b[i] for i in ids])
        correlation = None
        if len(ids) >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0:
            correlation = float(np.corrcoef(x, y)[0,1])
        rows.append(dict(split=split, model=model, cue=cue, paired_present_documents=len(ids),
            different_prediction_documents=len(all_ids)-len(ids),
            pearson_concentration=correlation, both_above_one=float(np.mean((x>1+1e-6)&(y>1+1e-6))),
            attention_only_above_one=float(np.mean((x>1+1e-6)&(y<=1+1e-6))),
            gradient_only_above_one=float(np.mean((y>1+1e-6)&(x<=1+1e-6)))))
    return rows


def discovery_summary(rows):
    by_signature = collections.defaultdict(list)
    for row in rows:
        by_signature[row['split'], row['signature']].append(row)
    output = []
    for (split, signature), group in by_signature.items():
        output.append(dict(split=split, signature=signature,
            selected_documents=len({r['identifier'] for r in group}),
            models=sorted({r['model'] for r in group}), methods=sorted({r['method'] for r in group}),
            selected_occurrences=len(group), examples=sorted(group, key=lambda r:-r['concentration'])[:4]))
    return sorted(output, key=lambda r:(-len(r['models']), -r['selected_documents'], r['split'], r['signature']))


def render_report(path, audit, common, discoveries, matrix):
    esc = lambda value: html.escape(str(value), quote=True)
    coverage = {m:{k:(len(v) if isinstance(v,list) else v) for k,v in values.items()} for m,values in audit['coverage'].items()}
    lines = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Classifier focus analysis</title>',
        '<style>body{font:15px system-ui;max-width:1400px;margin:32px auto;padding:0 20px;color:#152235}table{border-collapse:collapse;width:100%;margin:16px 0}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left;vertical-align:top}th{background:#eef2f7;position:sticky;top:0}pre{white-space:pre-wrap}input{padding:10px;width:420px}small{color:#536174}details{margin:10px 0}</style>',
        '<h1>x010b x010c/x010d x010e x010f</h1>',
        f'<p>x0110 x0099 {audit["selected_documents"]}. x0106 x0111: {esc(audit["data_status"])}. GPU x0112 x0113 x0114 x0115 x0116 x0117. x0118: {esc(audit["finished"])}</p>',
        '<p>x0119 x011a x011b x011c x011d x011e x011f x0120 x0121 x0122. Attention x0123 x0124 x0125 x0126, x0127/TF-IDF x0128 x0129 x012a. TF-IDF x012b x012c x012d x012e.</p>',
        '<p>x012f x0130 x0131 x0132 x0046 x0133 x008a x0134 x0135 x0136. x009d x00ae x002e, x0137 x0138 x0139 x013a x013b x013c. x013d x010d x013e x013f x002e x0140. x0082 x0141 x0142 x0143 x0144 x0145 x0146 x0147.</p>',
        '<details open><summary>x0106 x0148</summary><pre>'+esc(json.dumps(coverage, ensure_ascii=False, indent=2))+'</pre></details>',
        '<input id="search" placeholder="x00c7/x00ae/x0149 x0014" aria-label="x0117 x0014">',
        '<h2>x012f x00c7: x00c9 x0099/x00c9 x014a x0041</h2><table><thead><tr><th>x0149 / x014b</th><th>x00c7</th><th>x012f x0099</th><th>x014c x00ae</th><th>x010b x014d (x014e x0099 x002e)</th><th>x008b x00ae</th></tr></thead><tbody>']
    for row in common:
        if row['cue'].startswith('primary:') or row['cue'] in ('neither_format_nor_protected',):
            continue
        values = '; '.join(f'{m}: {v["median"]:.2f}× (n={v["n"]})' if v['median'] is not None else f'{m}: x0096' for m,v in row['models'].items())
        lines.append('<tr class="filter">'+''.join('<td>'+esc(x)+'</td>' for x in (row['split']+' / '+row['family'], row['cue'],row['paired_documents'], ', '.join(row['focused_models']) or 'x00b2 x014f x0096',values, ', '.join(row['missing_models'])))+'</tr>')
    lines.append('</tbody></table><h2>x0150 x0151: x0152 x0153 x0154 x010f</h2><p>x011a x0155 x0156 x014e x0157 x0158. x0119 x0159 x011b x015a x015b x015c x015d x015e x015f x0122.</p>')
    for row in common:
        if not row['cue'].startswith(('cue:', 'layout:')):
            continue
        lines.append('<details class="filter"><summary>'+esc(row['split']+' / '+row['family']+' / '+row['cue'])+'</summary>')
        for model, examples in row['examples'].items():
            for example in examples:
                lines.append('<p>'+esc(f'{model} · {example["identifier"]} · {example["concentration"]:.2f}× · x0160 {example["gold"]} / x0161 {example["prediction"]}')+'</p><pre>'+esc(example['context'])+'</pre>')
        lines.append('</details>')
    lines.append('<h2>x0162 x0163 x0164 x0165 x0144 x0141</h2><p>x009d x0166 discovery_windows.jsonl x0140. x0167 x0049 x00c7 x0168 100 x0158.</p>')
    for row in discoveries[:100]:
        lines.append('<details class="filter"><summary>'+esc(f'{row["signature"]} · {row["selected_documents"]} · '+', '.join(row['models']))+'</summary>')
        for ex in row['examples']:
            lines.append('<p>'+esc(f'{ex["model"]} / {ex["method"]} / {ex["identifier"]} / {ex["concentration"]:.2f}×')+'</p><pre>'+esc(ex['context'])+'</pre>')
        lines.append('</details>')
    lines.append('<p>x0169 x016a: candidate_matrix.csv, layer_profiles.csv, position_profiles.csv, head_profiles.csv, method_agreement.csv. x016b x016c document_candidates.jsonl, x0041/x008b/x0161 x016d visibility_audit.jsonl x016e.</p>')
    lines.append('<script>document.getElementById("search").addEventListener("input",e=>{const q=e.target.value.toLowerCase();document.querySelectorAll(".filter").forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q));});</script></html>')
    path.write_text('\n'.join(lines))


def run(args):
    started = time.monotonic()
    out = args.output.resolve()
    forbidden = list(RUNS.values()) + [r / m for r in args.attention_root for m in args.models]
    if any(out == p.resolve() or p.resolve() in out.parents for p in forbidden):
        raise ValueError('Analysis output must be separate from model/source directories')
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            source.write_json(out / 'status.json', dict(status='running', started=source.now()))
            with threadpool_limits(limits=1):
                result = analyze(args, started)
            source.write_json(out / 'status.json', dict(status='complete', finished=source.now(), seconds=time.monotonic()-started))
            return result
        except BaseException as error:
            source.write_json(out / 'status.json', dict(status='failed', error=f'{type(error).__name__}: {error}'))
            raise


def analyze(args, started):
    out = args.output
    selection_path = source.RUN_A / 'manifests/whitebox_selection.json'
    selected = [r for r in source.read_json(selection_path)['rows'] if args.split == 'all' or r['split'] == args.split]
    if args.limit:
        selected = selected[:args.limit]
    audit = dict(version=1, selected_documents=len(selected), requested_models=args.models,
        selection_sha256=source.digest(selection_path), tagger_sha256=source.digest(source.TAGGER),
        analysis_sha256=source.digest(Path(__file__)), helper_sha256=source.digest(Path(source.__file__)),
        split=args.split, limit=args.limit, min_documents=args.min_documents,
        source_snapshots={}, attention_manifests={}, unfinished_lines=[],
        head_token_rows_present=collections.Counter(), coverage={}, zero_mass_documents=[],
        definitions=dict(attention='Classification-query attention; normalize text mass within each layer, then average layers',
            attribution='Absolute gradient-times-embedding or coefficient-times-feature occurrence contributions; signed contributions separate',
            common='Same document intersection within each evidence family; prefix ends at minimum visible character across available requested-model evidence',
            strata='Correct/incorrect according to that evidence forward; gold label separately; dev and portion_eval never pooled',
            positions='Within each character third: candidate mass / third mass divided by candidate units / third units',
            focus_rule='Descriptive screen: present documents >= min_documents and median concentration > 1 + 1e-6',
            discovery='Top nonoverlapping, mostly untagged 8-word windows per document/method. Counts are selected hits, not corpus prevalence',
            heads='Native-scope per-head concentration for originally saved candidate groups, averaged across present documents',
            output='Partial snapshots are explicit. No altered-input predictions or intervention effects are inferred'))
    gradients, attentions = load_sources(args, selected, audit)
    wanted = {r['identifier'] for r in selected}
    texts = {r['identifier']:r['marker_masked_text'] for r in source.records(source.BENCHMARK) if r['identifier'] in wanted}
    if set(texts) != wanted:
        raise ValueError('Benchmark text missing')
    protected = collections.defaultdict(list)
    for row in source.records(source.PORTIONS):
        if row['doc_id'] in wanted and row.get('analysis_eligible') and row['portion_label'] != 'U':
            protected[row['doc_id']] += row['masked_portion_spans']
    collector = Collector()
    tagger = source.import_file('automatic_focus_tagger', source.TAGGER)
    discovery_rows = []
    for model in args.models:
        audit['coverage'][model] = dict(expected=len(selected), attribution=len(gradients[model]),
            attention=len(attentions[model]), attention_applicable=model != 'tfidf',
            attribution_missing=sorted(wanted-set(gradients[model])),
            attention_missing=sorted(wanted-set(attentions[model])) if model != 'tfidf' else [])
    collected = sum(len(gradients[m])+len(attentions[m]) for m in args.models)
    expected = len(selected)*(len(args.models)+sum(m != 'tfidf' for m in args.models))
    audit['data_status'] = 'complete' if collected == expected else 'partial' if collected else 'empty'
    temporary_docs = out / 'document_candidates.tmp.jsonl'
    temporary_visibility = out / 'visibility_audit.tmp.jsonl'
    with temporary_docs.open('w') as docs_stream, temporary_visibility.open('w') as visible_stream:
        for index, row in enumerate(selected):
            ident = row['identifier']; text = texts[ident]
            spans = annotations(text, protected[ident], tagger)
            evidence = {m:load_evidence(m, gradients[m].get(ident), attentions[m].get(ident), text, audit) for m in args.models}
            ends = [e['visible'] for es in evidence.values() for e in es]
            if not ends:
                continue
            common_end = min(ends)
            if common_end <= 0 or max(ends) > len(text):
                raise ValueError('Invalid visibility range')
            visibility = dict(identifier=ident, split=row['split'], text_characters=len(text),
                text_sha256=hashlib.sha256(text.encode()).hexdigest(), common_visible_chars=common_end,
                missing_models=[m for m in args.models if not evidence[m]], models={})
            for model, entries in evidence.items():
                visibility['models'][model] = {e['method']:dict(visible_chars=e['visible'], prediction=e['prediction'],
                    artifact=e['artifact'], artifact_sha256=e['artifact_sha256']) for e in entries}
                for e in entries:
                    head_summary(collector, model, row['split'], e)
                    for scope in SCOPES:
                        end = e['visible'] if scope == 'native' else common_end
                        metrics, cache = measure(e['offsets'], e['mass'], spans, end, e['signed'])
                        if cache is None:
                            audit['zero_mass_documents'].append(dict(model=model, identifier=ident, method=e['method'], scope=scope))
                            continue
                        collector.add(model, e['method'], scope, row, e['prediction'], metrics, text, spans, end)
                        for name, position, metric in positional_measures(e['offsets'], cache, spans, end):
                            collector.positions[model,e['method'],scope,row['split'],name,position].append(metric['concentration'])
                        compact = {name:{k:v for k,v in metric.items() if k != 'layer_concentration'} for name,metric in metrics.items()}
                        docs_stream.write(json.dumps(dict(identifier=ident, split=row['split'], gold=row['label'],
                            model=model, method=e['method'], scope=scope, visible_chars=end,
                            prediction=source.LABELS[e['prediction']], candidates=compact), allow_nan=False)+'\n')
                        if scope == 'shared_prefix':
                            discovery_rows.extend(dict(identifier=ident, split=row['split'], model=model, method=e['method'], **r)
                                for r in discover(text,e['offsets'],cache,spans,end,args.discovery_per_document))
            visible_stream.write(json.dumps(visibility, allow_nan=False)+'\n')
            if (index+1) % 32 == 0 or index+1 == len(selected):
                print(json.dumps(dict(processed=index+1, selected=len(selected), seconds=round(time.monotonic()-started,1))), flush=True)
    os.replace(temporary_docs, out / 'document_candidates.jsonl')
    os.replace(temporary_visibility, out / 'visibility_audit.jsonl')
    matrix = collector.matrix()
    common = collector.common(args.models, args.min_documents)
    write_csv(out / 'candidate_matrix.csv', matrix)
    layer_rows = []
    for (model,method,scope,split,cue), (total,counts) in sorted(collector.layers.items()):
        if method != 'attention':
            continue
        for i, (value,n) in enumerate(zip(total,counts)):
            layer_rows.append(dict(model=model, method=method, scope=scope, split=split, cue=cue,
                layer=i+1, layer_depth=(i+1)/len(total), present_documents=int(n), mean_concentration=float(value/n) if n else None))
    write_csv(out / 'layer_profiles.csv', layer_rows)
    write_csv(out / 'position_profiles.csv', [dict(zip(['model','method','scope','split','cue','position'],key), **statistics(values)) for key,values in sorted(collector.positions.items())])
    head_rows = []
    for (model,split,cue), (sums,counts) in sorted(collector.heads.items()):
        means = np.divide(sums,counts,out=np.zeros_like(sums),where=counts>0)
        for flat in np.argsort(-means, axis=None)[:5]:
            layer,head = np.unravel_index(flat,means.shape)
            if counts[layer,head]:
                head_rows.append(dict(model=model,split=split,scope='native',cue=cue,layer=int(layer)+1,head=int(head),
                    present_documents=int(counts[layer,head]),mean_concentration=float(means[layer,head])))
    write_csv(out / 'head_profiles.csv',head_rows)
    write_csv(out / 'method_agreement.csv', agreement_rows(collector))
    source.write_json(out / 'common_candidates.json', common)
    write_csv(out / 'common_candidates.csv', [dict(split=r['split'],family=r['family'],cue=r['cue'],
        paired_documents=r['paired_documents'],expected_models='|'.join(r['expected_models']), available_models='|'.join(r['available_models']),
        focused_models='|'.join(r['focused_models']),focused_model_count=r['focused_model_count'],missing_models='|'.join(r['missing_models']),
        model_statistics=json.dumps(r['models'],allow_nan=False)) for r in common])
    with (out / 'discovery_windows.jsonl').open('w') as stream:
        for row in discovery_rows:
            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    discoveries = discovery_summary(discovery_rows)
    source.write_json(out / 'discovery_candidates.json', discoveries)
    # Reviewable templates only; extraction never launches edit inference.
    source.write_json(out / 'intervention_review.json', dict(status='candidate_review',
        candidate_selection_splits=sorted({r['split'] for r in selected}), final_evaluation_split='test',
        operations=['delete_exact_span','change_typography_or_delimiter','move_span','insert_span'],
        control='Match edited character length and relative position; preserve protected content when testing pure format',
        measurements=['prediction_flip','accuracy_delta','original_class_logit_margin_delta'],
        candidates=[dict(split=r['split'],family=r['family'],cue=r['cue'],focused_models=r['focused_models'],
            evidence_documents=r['paired_documents'], review_status='pending',
            examples=r['examples']) for r in common if r['focused_model_count'] >= 2 and r['cue'].startswith(('cue:','layout:'))]))
    audit['finished'] = source.now()
    audit['seconds'] = time.monotonic()-started
    audit['tables'] = dict(candidate_rows=len(matrix),common_rows=len(common),discovery_windows=len(discovery_rows))
    source.write_json(out / 'analysis_manifest.json',audit)
    render_report(out / 'report.html',audit,common,discoveries,matrix)
    print(json.dumps(dict(output=str(out), seconds=round(audit['seconds'],1), tables=audit['tables'])),flush=True)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attention-root', action='append', type=Path, help='Repeat for non-overlapping dev/portion output roots')
    parser.add_argument('--output', type=Path, default=source.BASE / 'analysis/final_classifier_focus')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=MODELS)
    parser.add_argument('--split', choices=['dev','portion_eval','all'], default='all')
    parser.add_argument('--limit', type=int, default=0, help='First N fixed manifest rows, for verification only')
    parser.add_argument('--min-documents', type=int, default=20, help='Support threshold for descriptive shared-candidate screen')
    parser.add_argument('--discovery-per-document', type=int, default=3)
    args = parser.parse_args()
    if args.limit < 0 or args.min_documents < 1 or args.discovery_per_document < 0 or len(set(args.models)) != len(args.models):
        parser.error('Invalid limits or duplicate models')
    if args.attention_root is None:
        args.attention_root = [source.RUNS / 'run_20261027_r5_final_classifier_attention']
    run(args)


if __name__ == '__main__':
    main()
