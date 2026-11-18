"""Post-training attention readout on the frozen dev256 / portion2299 manifest.

Use .venv-train. This is an independent analysis job, never a training stage.
The four existing classifiers use a read-only Q/K probe. DeBERTa's selected
dev attention is imported from trajectory; portion attention uses native CLS capture.
TF-IDF has coefficient attribution, and is deliberately not an attention job.
"""
from __future__ import annotations

import argparse
import collections
import csv
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'experiments/X_nonllm_classifiers'
RUNS = BASE / 'runs'
RUN_A = RUNS / 'run_20261022_r5_noprompt_llama70b_longformer_head_pipeline'
RUN_B = RUNS / 'run_20261023_r5_noprompt_qwen72b_nemotron70b_head_pipeline'
RUN_C = RUNS / 'run_20261027_r5_fulltext_deberta_tfidf_pipeline'
MODEL_RUNS = dict(llama=RUN_A, longformer=RUN_A, qwen=RUN_B, nemotron=RUN_B, deberta=RUN_C)
BASE_MODELS = dict(llama='llama3.3-70b-Q4KM-dequant-bf16', qwen='qwen2.5-72b-Q4KM-dequant-bf16',
                   nemotron='nemotron-70b-Q4KM-dequant-bf16')
BENCHMARK = ROOT / 'data/datasets/cia_benchmark/r5/benchmark.jsonl'
PORTIONS = ROOT / 'data/derived/portion/portion_v2_layout/high_confidence_portions_v5full_r5.jsonl'
TAGGER = ROOT / 'experiments/U_instruction_cue_priority/scripts/evidence_type_tagger.py'
LABELS = ['UNCLASSIFIED', 'CONFIDENTIAL', 'SECRET', 'TOP SECRET']
PRIMARY = ['grade_word', 'control_warning', 'header_field', 'doc_identity', 'mixed',
           'fragment_noise', 'title_headline', 'content']
CUES = ['docid:standalone_date', 'docid:agency_caps', 'docid:doc_type_caps',
        'docid:doc_type_title_case', 'header:field_colon', 'docid:copy_page',
        'docid:section_header', 'control:redaction_code', 'control:handling',
        'control:espionage_warning', 'control:release_stamp', 'docid:rdp_number']


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def records(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    os.replace(temp, path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def union(spans):
    out = []
    for a, b in sorted(spans):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1][1] = max(b, out[-1][1])
        else:
            out.append([int(a), int(b)])
    return out


def overlap_weights(offsets, spans):
    offsets = np.asarray(offsets)
    lengths = offsets[:, 1]-offsets[:, 0]
    covered = np.zeros(len(offsets), dtype=np.float64)
    for a, b in union(spans):
        covered += np.maximum(0, np.minimum(b, offsets[:, 1])-np.maximum(a, offsets[:, 0]))
    return np.divide(covered, lengths, out=np.zeros_like(covered), where=lengths > 0)


def annotate(text, offsets, protected, tagger):
    sys.path.insert(0, str(ROOT / 'src'))
    from corelib.quotematch import sentence_candidate_spans
    visible = int(np.asarray(offsets)[:, 1].max())
    spans = collections.defaultdict(list)
    spans['text'] = [(0, visible)]
    spans['protected'] = [(max(0, a), min(b, visible)) for a, b in protected if a < visible]
    rules = {label: rx for _, rr in tagger.FORMAT_RULE_GROUPS for label, rx in rr if label in CUES}
    sentences = []
    for a, b in sentence_candidate_spans(text):
        b = min(b, visible)
        if a >= b:
            continue
        value = text[a:b]
        tag = tagger.tag_quote(value)
        sentences.append(dict(start=a, end=b, primary=tag.primary))
        spans['primary:'+tag.primary].append((a, b))
        if tag.format_family:
            spans['format'].append((a, b))
        # Exact candidate regex spans, matching the exploratory fine_cue readout.
        for label, rx in rules.items():
            spans['cue:'+label] += [(a+m.start(), a+m.end()) for m in rx.finditer(value) if m.end() > m.start()]
    names = ['text', 'format', 'protected'] + ['primary:'+x for x in PRIMARY] + ['cue:'+x for x in CUES]
    spans = {name: union(spans[name]) for name in names}
    matrix = np.stack([overlap_weights(offsets, spans[name]) for name in names]).astype(np.float32)
    if not np.all((matrix >= 0) & (matrix <= 1.00001)):
        raise ValueError('Invalid span/token overlap')
    return names, matrix, spans, sentences


def prepare(args):
    run = MODEL_RUNS[args.model]
    state_path = run / 'results' / args.model / 'training_state.json'
    state = read_json(state_path)
    if state['status'] != 'complete':
        raise RuntimeError(f'{args.model} training is not complete; no checkpoint selected yet')
    checkpoint = Path(state['best_checkpoint'])
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    selection_path = run / 'manifests/whitebox_selection.json'
    selection = read_json(selection_path)['rows']
    canonical = read_json(RUN_A / 'manifests/whitebox_selection.json')['rows']
    if selection != canonical:
        raise ValueError('Whitebox document selection differs from the original four-model selection')
    selected = [r for r in selection if args.split == 'all' or r['split'] == args.split]
    if args.limit:
        selected = selected[:args.limit]
    wanted = {r['identifier']: r for r in selected}
    if len(wanted) != len(selected) or not selected:
        raise ValueError('Empty or duplicate document selection')
    texts = {}
    for row in records(BENCHMARK):
        if row['identifier'] in wanted:
            entry = wanted[row['identifier']]
            if row['label'] != entry['label']:
                raise ValueError('Gold label mismatch')
            texts[row['identifier']] = row['marker_masked_text']
    if set(texts) != set(wanted):
        raise ValueError('Selected text is missing')
    protected = collections.defaultdict(list)
    for row in records(PORTIONS):
        if row['doc_id'] in wanted and row.get('analysis_eligible') and row['portion_label'] != 'U':
            protected[row['doc_id']] += row['masked_portion_spans']
    whitebox = {r['identifier']: r for r in records(run / 'results' / args.model / 'whitebox.jsonl') if r['identifier'] in wanted}
    if set(whitebox) != set(wanted):
        raise ValueError('Existing whitebox baselines are missing')
    for row in selected:
        wb = whitebox[row['identifier']]
        if wb['gold'] != LABELS.index(row['label']) or wb['split'] != row['split']:
            raise ValueError('Whitebox label/split mismatch')
    manifest = dict(model=args.model, source_run=str(run), checkpoint=str(checkpoint),
        split=args.split, limit=args.limit, selection_sha256=digest(selection_path),
        training_state_sha256=digest(state_path), tagger_sha256=digest(TAGGER),
        scripts_sha256={p.name: digest(p) for p in [Path(__file__), Path(__file__).with_name('attention_probe.py')]},
        input_field='marker_masked_text', prompt=None, method='actual_qk_single_query_row' if args.model != 'deberta' else 'dev_import_and_native_cls_attention',
        save_token_heads=args.save_token_heads, classification_query='last_nonpadding_token' if args.model in BASE_MODELS else 'CLS',
        documents=[dict(**r, text_sha256=hashlib.sha256(texts[r['identifier']].encode()).hexdigest()) for r in selected],
        group_definitions=dict(format='Union of sentences tagged grade/control/header/document identity',
            protected='Union of eligible C/S/TS portion spans; empty in dev by split design',
            cues='Exact regex candidate spans; overlapping groups, not a partition',
            attention_fraction='Group mass divided by text-token mass, after averaging heads at each layer',
            concentration='Within-document attention fraction / token fraction; summarize present documents only',
            special_tokens='Excluded from text-normalized fractions; their attention mass is saved separately'))
    if args.model in BASE_MODELS:
        base = ROOT / 'checkpoints/hf' / BASE_MODELS[args.model]
        manifest['base_model'] = str(base)
        manifest['base_config_sha256'] = digest(base / 'config.json')
        manifest['base_index_sha256'] = digest(base / 'model.safetensors.index.json')
    if args.model == 'deberta':
        manifest['scripts_sha256']['deberta_attention.py'] = digest(Path(__file__).with_name('deberta_attention.py'))
        manifest['native_tokenizer_sha256'] = digest(RUN_C / 'scripts/native_tokenizer.py')
    return checkpoint, selected, texts, protected, whitebox, manifest


def ensure_idle(gpus):
    indices = gpus.split(',')
    if not indices or any(not i.isdigit() for i in indices) or len(set(indices)) != len(indices):
        raise ValueError('--gpus must contain unique physical GPU indices, e.g. 0,1')
    lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'], text=True).splitlines()
    uuids = {line.split(',')[0].strip(): line.split(',')[1].strip() for line in lines}
    if not set(indices) <= set(uuids):
        raise ValueError('Requested GPU does not exist')
    wanted = {uuids[i] for i in indices}
    busy = []
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'], text=True)
    for line in processes.splitlines():
        fields = [s.strip() for s in line.split(',')]
        if fields and fields[0] in wanted and len(fields) > 1 and fields[1] != str(os.getpid()):
            busy.append(line)
    if busy:
        raise RuntimeError('Requested GPUs are occupied; analysis was not started: '+ '; '.join(busy))
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus


def load_classifier(kind, checkpoint, device):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from peft import PeftModel
    if kind in BASE_MODELS:
        base = ROOT / 'checkpoints/hf' / BASE_MODELS[kind]
        tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
        if kind in ('llama', 'nemotron'):
            tok.pad_token = '<|finetune_right_pad_id|>'
            assert tok.pad_token_id == 128004
        else:
            assert tok.pad_token_id == 151643
        if device != 'balanced':
            raise ValueError('70B production models require --device balanced over two idle GPUs')
        if torch.cuda.device_count() != 2:
            raise ValueError('70B production models require two visible GPUs')
        model = AutoModelForSequenceClassification.from_pretrained(base, local_files_only=True,
            num_labels=4, id2label=dict(enumerate(LABELS)), label2id={s:i for i,s in enumerate(LABELS)},
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map='balanced',
            max_memory={0:'105GiB', 1:'105GiB'}, attn_implementation='sdpa')
        if any(str(d) in ('cpu', 'disk') for d in model.hf_device_map.values()):
            raise RuntimeError('Unexpected CPU/disk offload')
        model.config.pad_token_id = tok.pad_token_id
        model.config.use_cache = False
        model = PeftModel.from_pretrained(model, checkpoint, is_trainable=False)
    else:
        if kind == 'deberta':
            native = import_file('final_attention_native_tokenizer', RUN_C / 'scripts/native_tokenizer.py')
            tok = native.NativeTokenizer(checkpoint)
        else:
            tok = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True,
            torch_dtype=torch.float32).to('cuda:0' if device == 'balanced' else device)
    tok.padding_side = 'right'
    if kind != 'deberta' and not tok.is_fast:
        raise ValueError('Exact character offsets require the original fast tokenizer')
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, tok


def make_batch(kind, model, tok, encoded):
    import torch
    batch = tok.pad(dict(input_ids=[encoded['input_ids']], attention_mask=[[1]*len(encoded['input_ids'])]),
                    padding=True, pad_to_multiple_of=512 if kind == 'longformer' else None, return_tensors='pt')
    if kind == 'longformer':
        batch['global_attention_mask'] = torch.zeros_like(batch['input_ids'])
        batch['global_attention_mask'][:, 0] = 1
    device = model.get_input_embeddings().weight.device
    return {k:v.to(device) for k,v in batch.items()}


def save_document(out, row, text, encoded, values, logits, baseline_logits, wb, protected, tagger, save_heads):
    names, masks, spans, sentences = annotate(text, encoded['offset_mapping'], protected, tagger)
    if values.ndim != 3 or values.shape[-1] != len(encoded['input_ids']):
        raise ValueError('Attention dimensions disagree with tokenization')
    if not np.isfinite(values).all() or not np.isfinite(logits).all():
        raise ValueError('Non-finite model outputs')
    mean_attention = values.mean(axis=1)
    group_mass = np.einsum('lht,gt->lhg', values, masks, optimize=True)
    layer_mass = group_mass.mean(axis=1)
    text_mass = layer_mass[:, 0]
    if np.any(text_mass <= 0):
        raise ValueError('Zero text attention mass')
    fractions = layer_mass / text_mass[:, None]
    token_fractions = masks.sum(-1) / masks[0].sum()
    concentration = np.divide(fractions, token_fractions[None, :],
                              out=np.zeros_like(fractions), where=token_fractions[None, :] > 0)
    key = hashlib.sha256(row['identifier'].encode()).hexdigest()[:20]
    artifact = out / 'artifacts' / (key+'.npz')
    data = dict(attention_mean=mean_attention, group_attention=group_mass,
                group_names=np.array(names), group_token_fractions=token_fractions,
                input_ids=np.array(encoded['input_ids'], dtype=np.int32),
                offsets=np.array(encoded['offset_mapping'], dtype=np.int32))
    if save_heads:
        data['attention_heads'] = values.astype(np.float16)
    temporary = artifact.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **data)
    os.replace(temporary, artifact)
    record = dict(identifier=row['identifier'], split=row['split'], gold=LABELS.index(row['label']),
        prediction=int(np.argmax(logits)), logits=np.asarray(logits).tolist(), tokens=len(encoded['input_ids']),
        original_logit_max_difference=float(np.max(np.abs(np.asarray(logits)-np.asarray(baseline_logits)))) if baseline_logits is not None else None,
        probe_check='paired_forward_exact_equality' if baseline_logits is not None else 'imported_attention_no_new_forward',
        whitebox_prediction=wb['prediction'], whitebox_prediction_agrees=int(np.argmax(logits)) == wb['prediction'],
        whitebox_logit_max_difference=float(np.max(np.abs(np.asarray(logits)-np.asarray(wb['logits'])))),
        text_sha256=hashlib.sha256(text.encode()).hexdigest(), artifact=str(artifact), artifact_sha256=digest(artifact),
        groups=names, spans=spans, sentences=sentences, token_fractions=token_fractions.tolist(),
        text_attention_mass=text_mass.tolist(), special_attention_mass=(values.sum(-1).mean(1)-text_mass).tolist(),
        layer_attention_fraction=fractions.tolist(), layer_concentration=concentration.tolist())
    write_json(out / 'documents' / (key+'.json'), record)
    return record


def summarize(out, selected):
    rows = []
    for entry in selected:
        key = hashlib.sha256(entry['identifier'].encode()).hexdigest()[:20]
        path = out / 'documents' / (key+'.json')
        if path.exists():
            row = read_json(path)
            if row['identifier'] != entry['identifier'] or row['split'] != entry['split'] or row['gold'] != LABELS.index(entry['label']):
                raise ValueError('Completed document identity mismatch')
            if digest(row['artifact']) != row['artifact_sha256']:
                raise ValueError('Completed attention artifact changed')
            rows.append(row)
    result = dict(documents=len(rows), expected_documents=len(selected), splits={}, updated=now())
    csv_rows = []
    for split in sorted({r['split'] for r in rows}):
        group = [r for r in rows if r['split'] == split]
        names = group[0]['groups']
        if not all(r['groups'] == names for r in group):
            raise ValueError('Inconsistent attention groups')
        baseline = np.array([r['token_fractions'] for r in group])
        fractions = np.array([r['layer_attention_fraction'] for r in group])
        concentration = np.array([r['layer_concentration'] for r in group])
        stats = {}
        for i, name in enumerate(names):
            present = baseline[:, i] > 0
            n = int(present.sum())
            stats[name] = dict(present_documents=n,
                mean_token_fraction=float(baseline[:, i].mean()),
                mean_attention_fraction=fractions[:, :, i].mean(0).tolist(),
                mean_attention_fraction_when_present=fractions[present, :, i].mean(0).tolist() if n else None,
                mean_concentration_when_present=concentration[present, :, i].mean(0).tolist() if n else None)
            for layer in range(fractions.shape[1]):
                csv_rows.append(dict(split=split, group=name, layer=layer+1, documents=len(group), present_documents=n,
                    mean_token_fraction=float(baseline[:, i].mean()), mean_attention_fraction=float(fractions[:, layer, i].mean()),
                    mean_concentration_when_present=float(concentration[present, layer, i].mean()) if n else None))
        result['splits'][split] = dict(documents=len(group), groups=stats,
            accuracy=sum(r['prediction']==r['gold'] for r in group)/len(group),
            checked_documents=sum(r['original_logit_max_difference'] is not None for r in group),
            maximum_probe_logit_difference=max((r['original_logit_max_difference'] for r in group if r['original_logit_max_difference'] is not None), default=None),
            whitebox_prediction_disagreements=sum(not r['whitebox_prediction_agrees'] for r in group),
            mean_special_attention_mass=np.array([r['special_attention_mass'] for r in group]).mean(0).tolist())
    write_json(out / 'summary.json', result)
    if csv_rows:
        with (out / 'layer_groups.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
            writer.writeheader(); writer.writerows(csv_rows)
    return result


def run(args):
    checkpoint, selected, texts, protected, whitebox, manifest = prepare(args)
    if args.inspect:
        print(json.dumps(dict(model=args.model, checkpoint=str(checkpoint), documents=len(selected), split=args.split,
            method=manifest['method'], output=str(args.output / args.model)), indent=2))
        return
    imported_rows = {}
    if args.model == 'deberta' and any(r['split'] == 'dev' for r in selected):
        trajectory = MODEL_RUNS['deberta'] / 'results/deberta/trajectory'
        if not (trajectory / 'status.json').exists() or read_json(trajectory / 'status.json')['status'] != 'complete':
            raise RuntimeError('DeBERTa trajectory is not complete; import it after the active pipeline finishes')
        source_file = trajectory / checkpoint.name / 'attention.jsonl'
        manifest['import_source_sha256'] = digest(source_file)
        imported_rows = {r['identifier']:r for r in records(source_file)}
        manifest['import_artifacts_sha256'] = {r['identifier']:digest(imported_rows[r['identifier']]['artifact'])
            for r in selected if r['split'] == 'dev'}
    # Hash selected trained weights, never the 130 GiB frozen base tensors.
    manifest['checkpoint_files_sha256'] = {p.name:digest(p) for p in sorted(checkpoint.iterdir())
        if p.is_file() and (p.suffix == '.safetensors' or p.name in ('config.json', 'adapter_config.json'))}
    manifest['device'] = args.device
    out = args.output / args.model
    if args.output.resolve() in {p.resolve() for p in MODEL_RUNS.values()} or any(p.resolve() in out.resolve().parents for p in MODEL_RUNS.values()):
        raise ValueError('Choose a new output directory outside the source runs')
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    meta = out / 'manifest.json'
    if meta.exists():
        if not args.resume:
            raise FileExistsError('Output exists; use --resume for an identical manifest')
        if read_json(meta) != manifest:
            raise ValueError('Source, checkpoint, code or selection changed; use a new output directory')
    else:
        write_json(meta, manifest)
    for name in ['documents', 'artifacts']:
        (out / name).mkdir(exist_ok=True)
    tagger = import_file('classifier_attention_tagger', TAGGER)
    start = time.monotonic()
    completed = summarize(out, selected)['documents']
    pending = [r for r in selected if not (out / 'documents' / (hashlib.sha256(r['identifier'].encode()).hexdigest()[:20]+'.json')).exists()]
    if not pending:
        status_path = out / 'status.json'
        if not status_path.exists() or read_json(status_path).get('status') != 'complete':
            write_json(status_path, dict(status='complete', documents=completed, finished=now()))
        lock.close()
        return
    write_json(out / 'status.json', dict(status='running', started=now(), documents=completed, expected=len(selected)))
    try:
        if args.model == 'deberta':
            for row in [r for r in pending if r['split'] == 'dev']:
                ident = row['identifier']; entry = imported_rows[ident]
                if entry['gold'] != LABELS.index(row['label']):
                    raise ValueError('DeBERTa source label mismatch')
                with np.load(entry['artifact']) as data:
                    values = data['attention'].astype(np.float32)
                    encoded = dict(input_ids=data['input_ids'].tolist(), offset_mapping=data['offsets'].tolist())
                with np.load(whitebox[ident]['token_artifact']) as old:
                    if not np.array_equal(old['input_ids'], encoded['input_ids']) or not np.array_equal(old['offsets'], encoded['offset_mapping']):
                        raise ValueError(f'Imported DeBERTa tokenization differs from whitebox: {ident}')
                save_document(out, row, texts[ident], encoded, values, entry['logits'], None,
                              whitebox[ident], protected[ident], tagger, args.save_token_heads)
                completed += 1
                if completed % 32 == 0:
                    write_json(out / 'status.json', dict(status='running', documents=completed, expected=len(selected), updated=now()))
            pending = [r for r in pending if r['split'] != 'dev']
        if pending:
            if args.device != 'cpu':
                ensure_idle(args.gpus)
            import torch
            from attention_probe import ClassificationAttentionProbe
            import contextlib
            torch.set_num_threads(8)
            model, tok = load_classifier(args.model, checkpoint, args.device)
            cuda = next(model.parameters()).is_cuda
            amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16) if cuda else contextlib.nullcontext()
            if args.model == 'deberta':
                from deberta_attention import collect as collect_deberta
            capture = contextlib.nullcontext() if args.model == 'deberta' else ClassificationAttentionProbe(model)
            with capture as probe:
                for row in pending:
                    ident = row['identifier']; text = texts[ident]
                    encoded = tok(text, add_special_tokens=True, return_offsets_mapping=True,
                                  **({'truncation':False} if args.model == 'deberta' else {'truncation':True, 'max_length':4096}))
                    with np.load(whitebox[ident]['token_artifact']) as old:
                        if not np.array_equal(old['input_ids'], encoded['input_ids']) or not np.array_equal(old['offsets'], encoded['offset_mapping']):
                            raise ValueError(f'Input tokenization changed: {ident}')
                    batch = make_batch(args.model, model, tok, encoded)
                    kwargs = dict(use_cache=False) if args.model in BASE_MODELS else {}
                    with torch.no_grad(), amp():
                        original = model(**batch, **kwargs).logits.float()[0].clone()
                        if args.model == 'deberta':
                            observed, values = collect_deberta(model, batch)
                        else:
                            probe.start(len(encoded['input_ids']))
                            observed = model(**batch, **kwargs).logits.float()[0].clone()
                            values = probe.finish()
                    if not torch.equal(original, observed):
                        raise RuntimeError(f'Probe changed logits: {ident}, max difference {float((original-observed).abs().max())}')
                    save_document(out, row, text, encoded, values, observed.cpu().numpy(), original.cpu().numpy(),
                                  whitebox[ident], protected[ident], tagger, args.save_token_heads)
                    completed += 1
                    write_json(out / 'status.json', dict(status='running', documents=completed, expected=len(selected),
                        updated=now(), seconds=time.monotonic()-start, last_identifier=ident))
                    if completed == 1 or completed % 16 == 0:
                        print(json.dumps(dict(model=args.model, completed=completed, total=len(selected), seconds=time.monotonic()-start)), flush=True)
        summary = summarize(out, selected)
        if summary['documents'] != len(selected):
            raise RuntimeError('Incomplete output set')
        write_json(out / 'status.json', dict(status='complete', documents=len(selected), finished=now(), seconds=time.monotonic()-start))
    except BaseException as error:
        write_json(out / 'status.json', dict(status='failed', error=f'{type(error).__name__}: {error}',
            documents=len(list((out / 'documents').glob('*.json'))), updated=now(), seconds=time.monotonic()-start))
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=list(MODEL_RUNS))
    parser.add_argument('--split', default='dev', choices=['dev', 'portion_eval', 'all'])
    parser.add_argument('--limit', type=int, default=0, help='First N fixed manifest documents; smoke use only')
    parser.add_argument('--output', type=Path, default=RUNS / 'run_20261027_r5_final_classifier_attention')
    parser.add_argument('--gpus', default='0,1', help='Physical GPU indices; busy devices cause an immediate stop')
    parser.add_argument('--device', default='balanced', choices=['balanced', 'cuda:0', 'cpu'])
    parser.add_argument('--inspect', action='store_true', help='CPU-only input/checkpoint audit; writes nothing')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--save-token-heads', action='store_true', help='Also save full L x H x T rows (larger files)')
    parser.add_argument('--skip-analysis', action='store_true', help='Skip the automatic CPU focus analysis after extraction')
    parser.add_argument('--analysis-output', type=Path, help='CPU report directory; default <output>/focus_analysis')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit must be nonnegative')
    run(args)
    if not args.inspect and not args.skip_analysis:
        # Replace this process so CUDA contexts/weights are released before CPU
        # analysis. A failed report never invalidates completed extraction files.
        os.execv(sys.executable, analysis_command(args))


def analysis_command(args):
    command = [sys.executable, str(Path(__file__).with_name('analyze_classifier_focus.py')),
               '--attention-root', str(args.output), '--split', args.split,
               '--output', str(args.analysis_output or args.output / 'focus_analysis')]
    if args.limit:
        command += ['--limit', str(args.limit)]
    return command


if __name__ == '__main__':
    main()
