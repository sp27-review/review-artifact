#!/usr/bin/env python3
"""(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
from __future__ import annotations
import csv, datetime, importlib.util, json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path('/path/to/workspace'); RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments/X_nonllm_classifiers/scripts'))
import extract_classifier_attention as src  # noqa: E402
spec = importlib.util.spec_from_file_location('tsn', RUN / 'scripts/train_small_norm.py'); tsn = importlib.util.module_from_spec(spec); sys.modules['tsn'] = tsn; spec.loader.exec_module(tsn)
E1 = ROOT / 'experiments/H_format_edit_screen/runs/run_20261110_ft3pairs3556_3model_P0_zeroshot_format_edit/inputs'
LABELS = ['UNCLASSIFIED', 'CONFIDENTIAL', 'SECRET', 'TOP SECRET']; DISPLAY = ['Unclassified', 'Confidential', 'Secret', 'Top Secret']
CONDS = ['wikileaks_base', 'wikileaks_combo', 'wikileaks_combo_ctrl']


def main():
    src.ensure_idle('0,1')
    import torch
    torch.manual_seed(7701)
    st = json.load(open(src.MODEL_RUNS['llama'] / 'results/llama/training_state.json')); assert st['status'] == 'complete'; ck = st['best_checkpoint']
    print(json.dumps(dict(event='loading', checkpoint=ck, time=datetime.datetime.now(datetime.timezone.utc).isoformat())), flush=True); t0 = time.monotonic()
    model, tok = src.load_classifier('llama', ck, 'balanced'); print(json.dumps(dict(event='loaded', seconds=round(time.monotonic() - t0, 1), device_map=sorted(set(str(v) for v in model.hf_device_map.values())))), flush=True)
    for N in ('N1', 'N2'):
        tsn.NORM = N; out = RUN / 'results' / f'llama70b_cia_only_{N}_inferonly_seed7701'; out.mkdir(parents=True, exist_ok=True)
        tsn.write(out / 'SOURCE.json', dict(loaded_from=str(src.MODEL_RUNS['llama'] / 'results/llama'), best_checkpoint=ck, normalize=N, loader='X extract_classifier_attention.load_classifier balanced 2 GPU', note='x0170 x00a9; x0171 H FT3 Llama 70B'))
        for cond in CONDS:
            final = out / cond / 'final'
            if (final / 'baseline_predictions.csv').exists(): print(json.dumps(dict(event='skip', norm=N, condition=cond)), flush=True); continue
            final.mkdir(parents=True, exist_ok=True); rows = list(tsn.records(E1 / f'{cond}.jsonl')); res = []; start = time.monotonic(); unavailable = 0
            with (final / 'predictions.jsonl').open('w') as stream:
                for k, r in enumerate(rows):
                    text = tsn.normalize(r['marker_masked_text']); enc = tok(text, add_special_tokens=True, truncation=False); ids = enc['input_ids']; gold = LABELS.index(r['label'].upper())
                    if len(ids) > 4096:
                        item = dict(identifier=r['identifier'], gold=gold, prediction=None, logits=None, tokens=len(ids), status='unavailable_length_exceeds_4096'); unavailable += 1
                    else:
                        batch = src.make_batch('llama', model, tok, enc)
                        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16): logits = model(**batch, use_cache=False).logits.float()[0].cpu().numpy()
                        del batch; assert logits.shape == (4,) and np.isfinite(logits).all()
                        item = dict(identifier=r['identifier'], gold=gold, prediction=int(logits.argmax()), logits=logits.tolist(), tokens=len(ids), status='valid')
                    res.append(item); stream.write(json.dumps(item) + '\n')
                    if (k + 1) % 100 == 0 or k + 1 == len(rows): print(json.dumps(dict(event='progress', norm=N, condition=cond, completed=k + 1, total=len(rows), seconds=round(time.monotonic() - start, 1))), flush=True)
            with (final / 'baseline_predictions.csv').open('w', encoding='utf-8', newline='') as h:
                w = csv.DictWriter(h, fieldnames=['doc_id', 'gold_label', 'pred_label', 'logits', 'tokens', 'prediction_source']); w.writeheader()
                for it in sorted(res, key=lambda x: x['identifier']): w.writerow(dict(doc_id=it['identifier'], gold_label=DISPLAY[it['gold']], pred_label=DISPLAY[it['prediction']] if it['prediction'] is not None else '', logits=json.dumps(it['logits']), tokens=it['tokens'], prediction_source='primary' if it['status'] == 'valid' else it['status']))
            valid = [it for it in res if it['prediction'] is not None]; acc = sum(it['prediction'] == it['gold'] for it in valid) / max(1, len(valid))
            tsn.write(final / 'merge_summary.json', dict(model='llama70b_cia_only', normalize=N, condition=cond, rows=len(res), unavailable_length=unavailable, empty_predictions=unavailable, gate='passed', input=str(E1 / f'{cond}.jsonl'), input_sha256=tsn.digest(E1 / f'{cond}.jsonl'), accuracy=round(acc, 4), seconds=round(time.monotonic() - start, 1)))
            print(json.dumps(dict(event='predicted', norm=N, condition=cond, rows=len(res), unavailable=unavailable, accuracy=round(acc, 4))), flush=True)
    print(json.dumps(dict(event='complete')), flush=True)


if __name__ == '__main__':
    main()
