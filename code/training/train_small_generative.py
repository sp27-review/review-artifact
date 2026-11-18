#!/usr/bin/env python3
"""(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, random, sys, time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/path/to/workspace')
RUN = Path(__file__).resolve().parents[1]
LABELS = ['UNCLASSIFIED', 'CONFIDENTIAL', 'SECRET', 'TOP SECRET']
DISPLAY = ['Unclassified', 'Confidential', 'Secret', 'Top Secret']
CIA_SPLIT = ROOT / 'experiments/X_nonllm_classifiers/runs/run_20261021_r5_noprompt_llama33_70b_training_smoke/manifests/split.jsonl'
BENCHMARK = ROOT / 'data/datasets/cia_benchmark/r5/benchmark.jsonl'
EXT_TEXTS = RUN / 'inputs/external_texts.jsonl'
SUFFIX_TEXT = '\n\nClassification:'
GEN = {}  # tok id -> dict(suffix, labels(list of token id lists), first(list))
GREEDY_VALID = {}


def gen_spec(tok):
    k = id(tok)
    if k not in GEN:
        suffix = tok(SUFFIX_TEXT, add_special_tokens=False)['input_ids']; labels = [tok(' ' + l, add_special_tokens=False)['input_ids'] + [tok.eos_token_id] for l in LABELS]; first = [x[0] for x in labels]
        while len(set(first)) == 1 and all(len(x) > 2 for x in labels): labels = [x[1:] for x in labels]; first = [x[0] for x in labels]
        assert len(set(first)) == 4 and tok.eos_token_id is not None, (first, [tok.convert_ids_to_tokens(x) for x in labels])
        GEN[k] = dict(suffix=suffix, labels=labels, first=first)
    return GEN[k]


def with_suffix(tok, ids):
    g = gen_spec(tok); keep = 4096 - len(g['suffix'])
    return list(ids[:keep]) + g['suffix']


def lm_parts(model):
    lm = model.base_model.model if hasattr(model, 'base_model') and hasattr(model.base_model, 'model') else model
    return lm.model, lm.lm_head


def label_logits(model, tok, group):
    import torch
    g = gen_spec(tok); batch = make_batch(group, model, tok); dec, head = lm_parts(model)
    hidden = dec(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], use_cache=False).last_hidden_state
    last = torch.tensor([len(r['input_ids']) - 1 for r in group], device=hidden.device); h = hidden[torch.arange(len(group), device=hidden.device), last]
    full = head(h).float(); top = full.argmax(-1).tolist()
    for r, t in zip(group, top): GREEDY_VALID[r['identifier']] = t in g['first']
    return full[:, torch.tensor(g['first'], device=full.device)]

MODELS = {
    'llama8b': dict(path='checkpoints/hf/meta-llama_Llama-3.1-8B-Instruct', pad='<|finetune_right_pad_id|>', pad_id=128004, targets=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], attn='sdpa'),
    'qwen7b': dict(path='checkpoints/hf/Qwen_Qwen2.5-7B-Instruct', pad=None, pad_id=151643, targets=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], attn='sdpa'),
    'mistral7b': dict(path='checkpoints/hf/mistralai_Mistral-7B-Instruct-v0.3', pad='</s>', pad_id=2, targets=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], attn='sdpa'),
    'gemma9b': dict(path='checkpoints/hf/google_gemma-2-9b-it', pad=None, pad_id=0, targets=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], attn='eager'),
    'phi35': dict(path='checkpoints/hf/microsoft_Phi-3.5-mini-instruct', pad=None, pad_id=32000, targets=['qkv_proj', 'o_proj', 'gate_up_proj', 'down_proj'], attn='sdpa'),
}


@dataclass
class EarlyStopping:
    patience: int = 2
    min_delta: float = 0.001
    best_score: float = -math.inf
    improvement_anchor: float = -math.inf
    bad_evaluations: int = 0

    def update(self, score):
        if not math.isfinite(score): raise ValueError('Development metric is non-finite')
        is_best = score > self.best_score
        if is_best: self.best_score = score
        if score > self.improvement_anchor + self.min_delta: self.improvement_anchor = score; self.bad_evaluations = 0
        else: self.bad_evaluations += 1
        return is_best, self.bad_evaluations >= self.patience


def now(): return datetime.now(timezone.utc).isoformat()
def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'); os.replace(tmp, path)
def emit(event, **fields):

    value = dict(time=now(), event=event); value.update(fields); print(json.dumps(value), flush=True)
def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
    return h.hexdigest()
def records(path):
    with Path(path).open(encoding='utf-8') as s:
        for line in s:
            if line.strip(): yield json.loads(line)


def get_config(kind, data, seed, max_epochs):
    return dict(seed=seed, max_length=4096, max_epochs=max_epochs, min_epochs=1, effective_batch=32, microbatch=2, learning_rate=1e-4, weight_decay=.01, warmup_ratio=.05,
                max_grad_norm=1., patience=2, min_delta=.001, evals_per_epoch=2, lora=dict(r=16, alpha=32, dropout=.05, targets=MODELS[kind]['targets'], modules_to_save=[]), objective='generative: LM head + LoRA, target = label name + EOS after suffix', suffix=SUFFIX_TEXT,
                model=MODELS[kind]['path'], attn_implementation=MODELS[kind]['attn'], pad_token_id=MODELS[kind]['pad_id'], data=data, input_field='marker_masked_text', prompt=None,
                provenance='official Hugging Face BF16 safetensors (no dequantization)')


def load_tokenizer(kind):
    from transformers import AutoTokenizer
    spec = MODELS[kind]; tok = AutoTokenizer.from_pretrained(ROOT / spec['path'], local_files_only=True); tok.padding_side = 'right'
    if spec['pad']: tok.pad_token = spec['pad']
    assert tok.pad_token_id == spec['pad_id'], (kind, tok.pad_token_id)
    assert tok.is_fast
    return tok


def load_data(kind, data_name, tok):
    import torch
    path = RUN / 'inputs' / f'{kind}_{data_name}_gen_tokenized.pt'
    if path.exists(): return torch.load(path, weights_only=False)
    split = {r['identifier']: r for r in records(CIA_SPLIT) if r['split'] in ('train', 'dev', 'test', 'portion_eval')}
    data = {s: [] for s in ('train', 'dev', 'test', 'portion_eval', 'ext_dev')}
    lengths = {}
    def add(bucket, rows):
        enc = tok([r['marker_masked_text'] for r in rows], add_special_tokens=True, truncation=False)['input_ids']
        for r, ids in zip(rows, enc):
            n = len(ids)
            if n > 4096: ids = tok(r['marker_masked_text'], add_special_tokens=True, truncation=True, max_length=4096)['input_ids']
            assert tok.pad_token_id not in ids, (kind, r['identifier'])
            data[bucket].append(dict(identifier=r['identifier'], source=r.get('source', 'cia'), label=LABELS.index(r['label']), input_ids=with_suffix(tok, ids), original_tokens=n))
    emit('tokenizing', model=kind, data=data_name)
    pending = {s: [] for s in data}
    for row in records(BENCHMARK):
        m = split.get(row['identifier'])
        if not m: continue
        assert row['label'] == m['label']
        pending[m['split']].append(row)
        if len(pending[m['split']]) == 256: add(m['split'], pending[m['split']]); pending[m['split']] = []
    for s, rows in pending.items():
        if rows: add(s, rows)
    assert {s: len(data[s]) for s in ('train', 'dev', 'test', 'portion_eval')} == dict(train=28359, dev=3530, test=3562, portion_eval=2299)
    n_cia_train, n_cia_dev = len(data['train']), len(data['dev'])
    if data_name == 'cia_ext':
        buf = {'train': [], 'dev': []}
        for r in records(EXT_TEXTS):
            buf[r['split']].append(r)
            if len(buf[r['split']]) == 256: add('train' if r['split'] == 'train' else 'ext_dev', buf[r['split']]); buf[r['split']] = []
        for s, rows in buf.items():
            if rows: add('train' if s == 'train' else 'ext_dev', rows)
        data['dev'] = data['dev'] + data['ext_dev']
    for s, rows in data.items():
        if rows: lengths[s] = dict(count=len(rows), mean=sum(r['original_tokens'] for r in rows) / len(rows), max=max(r['original_tokens'] for r in rows), truncated=sum(r['original_tokens'] > 4096 for r in rows))
    lengths['composition'] = dict(cia_train=n_cia_train, cia_dev=n_cia_dev, external_train=len(data['train']) - n_cia_train, external_dev=len(data['ext_dev']))
    tmp = path.with_suffix('.tmp'); torch.save(data, tmp); os.replace(tmp, path)
    write(RUN / 'manifests' / f'{kind}_{data_name}_gen_lengths.json', lengths)
    return data


def load_model(kind, checkpoint_path=None, seed=7701):
    import torch
    from transformers import AutoModelForCausalLM, set_seed
    from peft import LoraConfig, TaskType, get_peft_model, PeftModel
    set_seed(seed); torch.set_num_threads(8)
    spec = MODELS[kind]; tok = load_tokenizer(kind)
    start = time.monotonic(); emit('loading', model=kind, checkpoint=str(checkpoint_path))
    kwargs = dict(local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(ROOT / spec['path'], torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map={'': 0}, attn_implementation=spec['attn'], **kwargs)
    model.config.pad_token_id = tok.pad_token_id; model.config.use_cache = False
    if checkpoint_path: model = PeftModel.from_pretrained(model, checkpoint_path, is_trainable=False)
    else: model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=.05, target_modules=spec['targets']))
    model.enable_input_require_grads(); model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    assert model.config.pad_token_id == tok.pad_token_id; gen_spec(tok)
    if not checkpoint_path:
        names = [n for n, p in model.named_parameters() if p.requires_grad]
        assert names and all('lora_' in n for n in names), names[:5]
    emit('loaded', model=kind, seconds=time.monotonic() - start, trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model, tok


def make_batch(rows, model, tok):
    batch = tok.pad(dict(input_ids=[r['input_ids'] for r in rows], attention_mask=[[1] * len(r['input_ids']) for r in rows]), padding=True, return_tensors='pt')
    dev = model.get_input_embeddings().weight.device
    return {k: v.to(dev) for k, v in batch.items()}


def metrics(rows):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    gold, pred = [r['gold'] for r in rows], [r['prediction'] for r in rows]; m = confusion_matrix(gold, pred, labels=range(4)); n = len(rows); cl = sum(g > 0 for g in gold)
    return dict(documents=n, accuracy=float(accuracy_score(gold, pred)), macro_f1=float(f1_score(gold, pred, labels=list(range(4)), average='macro', zero_division=0)), loss=sum(r['loss'] for r in rows) / n,
                recall={s: float(m[i, i] / m[i].sum()) if m[i].sum() else None for i, s in enumerate(LABELS)}, confusion_matrix=m.tolist(), class_order=LABELS,
                underclassification_rate=sum(p < g for p, g in zip(pred, gold)) / n, overclassification_rate=sum(p > g for p, g in zip(pred, gold)) / n,
                classified_to_U_count=sum(p == 0 and g > 0 for p, g in zip(pred, gold)), classified_to_U_rate=sum(p == 0 and g > 0 for p, g in zip(pred, gold)) / cl if cl else None)


def predict(model, tok, rows, output=None, batch_size=8):
    import torch
    model.eval(); order = sorted(rows, key=lambda r: len(r['input_ids'])); results = []; start = time.monotonic()
    stream = Path(output).open('w') if output else None
    try:
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for off in range(0, len(order), batch_size):
                group = order[off:off + batch_size]; logits = label_logits(model, tok, group); assert torch.isfinite(logits).all().item()
                gold = torch.tensor([r['label'] for r in group], device=logits.device); losses = torch.nn.functional.cross_entropy(logits, gold, reduction='none').cpu().tolist(); probs = logits.softmax(-1).cpu().tolist()
                for row, sc, pr, lo in zip(group, logits.cpu().tolist(), probs, losses):
                    item = dict(identifier=row['identifier'], source=row.get('source', 'cia'), gold=row['label'], prediction=max(range(4), key=sc.__getitem__), logits=sc, probabilities=pr, loss=lo, tokens=len(row['input_ids']), greedy_valid=GREEDY_VALID.get(row['identifier']))
                    results.append(item)
                    if stream: stream.write(json.dumps(item) + '\n')
                if off == 0 or (off + batch_size) % 512 == 0: emit('inference_progress', completed=len(results), total=len(order), seconds=time.monotonic() - start)
    finally:
        if stream: stream.close()
    return metrics(results), results


def epoch_groups(rows, epoch, seed, batch_size=32):
    rng = random.Random(seed + epoch); order = list(range(len(rows))); rng.shuffle(order)
    return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]


def backward_group(model, tok, rows, micro):
    import torch
    total = 0.; g = gen_spec(tok); dec, head = lm_parts(model)
    for off in range(0, len(rows), micro):
        group = rows[off:off + micro]; full = [dict(input_ids=r['input_ids'] + g['labels'][r['label']][:-1]) for r in group]; batch = make_batch(full, model, tok)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            hidden = dec(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], use_cache=False).last_hidden_state; loss = 0.
            for b, r in enumerate(group):
                tgt = g['labels'][r['label']]; start = len(r['input_ids']) - 1
                lg = head(hidden[b, start:start + len(tgt)]).float(); loss = loss + torch.nn.functional.cross_entropy(lg, torch.tensor(tgt, device=lg.device), reduction='mean')
            loss = loss / len(rows)
        assert torch.isfinite(loss).item(), 'Non-finite loss'
        loss.backward(); total += loss.detach().item()
    return total


def save_model(model, tok, path):
    path.mkdir(parents=True, exist_ok=False); model.save_pretrained(path, safe_serialization=True); tok.save_pretrained(path)


def train_stage(kind, model, tok, data, out, config):
    import torch
    assert not (out / 'training_state.json').exists(), 'Refuse to restart completed/partial training silently'
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=config['learning_rate'], weight_decay=.01, foreach=False)
    per_epoch = math.ceil(len(data['train']) / 32); total = per_epoch * config['max_epochs']; warm = math.ceil(total * .05)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0., (total - s) / (total - warm)))
    stopper = EarlyStopping(config['patience'], config['min_delta']); history, step, best = [], 0, None; start = time.monotonic(); stopping = False
    log_path = out / 'train_log.jsonl'; write(out / 'training_state.json', dict(status='running', config=config, started=now(), last_step=0))
    state = {}
    for epoch in range(config['max_epochs']):
        groups = epoch_groups(data['train'], epoch, config['seed']); seen = 0
        for local, idx in enumerate(groups, 1):
            model.train(); ex = [data['train'][i] for i in idx]; opt.zero_grad(set_to_none=True); t0 = time.monotonic(); lr = opt.param_groups[0]['lr']
            loss = backward_group(model, tok, ex, config['microbatch']); norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1; seen += len(ex); frac = epoch + seen / len(data['train'])
            ev = dict(time=now(), step=step, epoch=frac, documents=len(ex), train_loss=loss, learning_rate=lr, gradient_norm=float(norm), seconds=time.monotonic() - t0)
            with log_path.open('a') as s: s.write(json.dumps(ev) + '\n')
            if step == 1 or step % 10 == 0: emit('train_update', **ev); write(out / 'progress.json', dict(stage='training', **ev, elapsed_seconds=time.monotonic() - start, total_steps=total))
            if local not in (math.ceil(per_epoch / 2), per_epoch): continue
            emit('dev_start', step=step, epoch=frac); dev, _ = predict(model, tok, data['dev']); is_best, plateau = stopper.update(dev['macro_f1'])
            if is_best: best = out / 'checkpoints' / ('step_%06d' % step); save_model(model, tok, best); write(best / 'selection.json', dict(step=step, epoch=frac, dev=dev))
            stopping = plateau and frac >= config['min_epochs']; ev.update(dev=dev, is_best=is_best, early_stop=stopping, stopper=asdict(stopper)); history.append(ev)
            state = dict(status='running', last_step=step, best_checkpoint=str(best), best_dev_macro_f1=stopper.best_score, history=history, config=config); write(out / 'training_state.json', state); emit('dev_complete', **ev)
            if stopping: break
        if stopping: break
    state.update(status='complete', stop_reason='early_stopping' if stopping else 'max_epochs', finished=now(), elapsed_seconds=time.monotonic() - start, steps=step)
    write(out / 'training_state.json', state); emit('training_complete', model=kind, best_checkpoint=str(best), seconds=state['elapsed_seconds'])


def preflight(kind, model, tok, data, out, config):
    import gc, torch
    from safetensors.torch import load_file
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    params = [p for p in model.parameters() if p.requires_grad]; opt = torch.optim.AdamW(params, lr=config['learning_rate'], weight_decay=.01, foreach=False)
    pool = list(data['train']); random.Random(7701).shuffle(pool); group = pool[:32]; model.train(); opt.zero_grad(set_to_none=True); t0 = time.monotonic()
    loss = backward_group(model, tok, group, config['microbatch']); norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True); assert float(norm) > 0; opt.step(); opt.zero_grad(set_to_none=True); train_seconds = time.monotonic() - t0
    emit('preflight_accumulation_passed', model=kind, loss=loss, norm=float(norm), seconds=train_seconds, docs=len(group))
    longest = sorted(data['train'], key=lambda r: len(r['input_ids']), reverse=True)[:config['microbatch']]; t1 = time.monotonic()
    stress_loss = backward_group(model, tok, longest, config['microbatch']); torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True); opt.zero_grad(set_to_none=True); stress_seconds = time.monotonic() - t1
    del opt; gc.collect(); torch.cuda.empty_cache(); ck = out / 'checkpoint'; save_model(model, tok, ck)
    dev = sorted(data['dev'], key=lambda r: r['identifier'])[:4]; _, before = predict(model, tok, dev)
    state = load_file(str(ck / 'adapter_model.safetensors')); cur = get_peft_model_state_dict(model); assert set(cur) == set(state)
    with torch.no_grad():
        for p in params: p.add_(.01)
    set_peft_model_state_dict(model, state); _, after = predict(model, tok, dev)
    disc = max(abs(a - b) for x, y in zip(before, after) for a, b in zip(x['logits'], y['logits'])); assert disc == 0., disc
    t2 = time.monotonic(); _, _ = predict(model, tok, sorted(data['dev'], key=lambda r: len(r['input_ids']))[-64:]); infer_seconds = time.monotonic() - t2
    per_epoch = math.ceil(len(data['train']) / 32)
    write(out / 'checks.json', dict(status='passed', model=kind, data=config['data'], train_batch=32, train_seconds=train_seconds, train_loss=loss, gradient_norm=float(norm), stress_tokens=max(len(r['input_ids']) for r in longest),
        stress_seconds=stress_seconds, stress_loss=stress_loss, reload_logit_max_difference=disc, inference_64_longest_dev_seconds=infer_seconds,
        estimate=dict(steps_per_epoch=per_epoch, train_hours_per_epoch=round(per_epoch * train_seconds / 3600, 2), dev_docs=len(data['dev'])),
        peak_allocated_gib=torch.cuda.max_memory_allocated(0) / 2**30, peak_reserved_gib=torch.cuda.max_memory_reserved(0) / 2**30))
    emit('preflight_passed', model=kind, train_hours_per_epoch=round(per_epoch * train_seconds / 3600, 2))


def predict_jsonl(kind, model, tok, inputs, out):
    for name, path in inputs:
        rows = []
        for r in records(path):
            ids = tok(r['marker_masked_text'], add_special_tokens=True, truncation=True, max_length=4096)['input_ids']; assert tok.pad_token_id not in ids
            rows.append(dict(identifier=r['identifier'], label=LABELS.index(r['label'].upper()), input_ids=with_suffix(tok, ids)))
        final = out / name / 'final'; final.mkdir(parents=True, exist_ok=True)
        m, res = predict(model, tok, rows, final / 'predictions.jsonl')
        with (final / 'baseline_predictions.csv').open('w', encoding='utf-8', newline='') as h:
            w = csv.DictWriter(h, fieldnames=['doc_id', 'gold_label', 'pred_label', 'logits', 'tokens', 'prediction_source']); w.writeheader()
            for r in sorted(res, key=lambda r: r['identifier']): w.writerow(dict(doc_id=r['identifier'], gold_label=DISPLAY[r['gold']], pred_label=DISPLAY[r['prediction']], logits=json.dumps(r['logits']), tokens=r['tokens'], prediction_source='primary'))
        write(final / 'merge_summary.json', dict(model=kind, condition=name, rows=len(res), empty_predictions=0, gate='passed', input=str(path), input_sha256=digest(path), accuracy=m['accuracy'], macro_f1=m['macro_f1']))
        emit('predicted', model=kind, condition=name, rows=len(res), accuracy=m['accuracy'])


def main():
    p = argparse.ArgumentParser(); p.add_argument('--model', choices=list(MODELS), required=True); p.add_argument('--data', choices=['cia_ext', 'cia_only'], default='cia_ext')
    p.add_argument('--seed', type=int, default=7701); p.add_argument('--max-epochs', type=int, default=2); p.add_argument('--stage', choices=['prepare', 'preflight', 'train', 'evaluate', 'predict_jsonl'], required=True)
    p.add_argument('--inputs', nargs='*', default=[], help='predict_jsonl: name=path ...'); p.add_argument('--tag-suffix', default='', help='x0117 TAG x0174(x0048: _ep2)'); a = p.parse_args()
    kind = a.model; tag = f'{kind}_{a.data}_gen{a.tag_suffix}_seed{a.seed}'; out = RUN / ('preflight' if a.stage == 'preflight' else 'results') / tag; out.mkdir(parents=True, exist_ok=True)
    config = get_config(kind, a.data, a.seed, a.max_epochs); write(out / 'config.json', config)
    if a.stage == 'prepare':
        tok = load_tokenizer(kind); data = load_data(kind, a.data, tok); emit('prepared', model=kind, data=a.data, sizes={s: len(v) for s, v in data.items()}); return
    ck = None
    if a.stage in ('evaluate', 'predict_jsonl'):
        st = json.loads((out / 'training_state.json').read_text()); assert st['status'] == 'complete'; ck = st['best_checkpoint']
    model, tok = load_model(kind, ck, a.seed); data = load_data(kind, a.data, tok)
    if a.stage == 'preflight': preflight(kind, model, tok, data, out, config)
    elif a.stage == 'train': train_stage(kind, model, tok, data, out, config)
    elif a.stage == 'evaluate':
        res = dict(status='running', checkpoint=ck, splits={})
        for split in ('test', 'portion_eval', 'ext_dev'):
            if data[split]: res['splits'][split], rows = predict(model, tok, data[split], out / (split + '_predictions.jsonl'))
            if split == 'ext_dev' and data[split]:
                for src in ('wikileaks', 'frus'): res['splits'][f'ext_dev_{src}'] = metrics([r for r in rows if r['source'] == src])
            write(out / 'evaluation.json', res)
        res.update(status='complete', finished=now()); write(out / 'evaluation.json', res)
    else:
        predict_jsonl(kind, model, tok, [tuple(x.split('=', 1)) for x in a.inputs], out)


if __name__ == '__main__':
    main()
