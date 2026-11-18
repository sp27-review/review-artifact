# Reproduction guide

Three levels, from cheapest to most expensive.

## 1. Recompute reported counts from released predictions (minutes, CPU)

```bash
python code/tools/verify_checksums.py
python code/tools/reaggregate.py --check      # ends with "reaggregation checks: 582/582 passed"
```

`reaggregate.py` prints, for every model family and source, the accuracy on
baseline / edited / control inputs and the counts with their denominators for
protected-to-U, protected missed, over-protection, and new and fixed errors;
the header-block move counts (end versus middle); and the zero-shot accuracy
on original versus marker-removed CIA text. With `--check` each count is
compared with the frozen value in `data/summaries/`.

Confidence intervals in the paper for these comparisons are paired document
bootstraps (10,000 resamples, seed 7701, percentile 95%) of the
attack-minus-control difference; `code/analysis/strict_combo_reaggregate.py`
and `code/analysis/normalization_same_documents.py` are the scripts that
produced them.

## 2. Restore the texts and rebuild every model input (hours, CPU, network)

No text is shipped. Restore it from the public collections, then verify.

**WikiLeaks and FRUS.** Download the two public Kaggle datasets named in
`DATA_CARD.md` and check the archive digests. Then run, from a workspace laid
out as the scripts expect (see `CODE_MAP.md`):

1. `code/marker_removal/external/prepare_published.py` — converts the archives
   to one JSON line per document with the agency label.
2. `code/marker_removal/external/prepare_external_masked_20261030.py --dataset
   {wikileaks_cable_classifier,frus_sachn} --out DIR` — removes classification
   markings (it imports the rule set in `prepare_wikileaks_masked_20261030.py`);
   `verify_external_masked_20261030.py` re-checks the deletion log and residual
   candidates. The 240 contextual review decisions the script applies are
   shipped text-free as `external_marking_review_textfree.json` (offsets,
   action, reason and a hash of the reviewed string); the script's string
   comparison has to be replaced by the hash comparison to use that file.
3. Verify against the manifest:

```bash
python code/tools/verify_text_hashes.py --manifest data/documents/wikileaks_documents.csv.gz \
    --column masked_text_sha256 restored_wikileaks.jsonl
```

   (`restored_*.jsonl` has one object per line with `document_id` and `text`.)

**CIA.** `python code/restore/download_cia_ocr.py` prints the download plan;
with `--download --accept-source-terms --output-dir DIR` it fetches each
`*_djvu.txt` and verifies its SHA-256. The corpus builder additionally reads
the `*_djvu.xml` layout file of the same Internet Archive item. Then run
`code/marker_removal/cia/pipeline/` (p2–p5) and `code/marker_removal/cia/r5/`
(s0, s1, s3) to obtain the marker-removed text, and verify with
`--manifest data/documents/cia_documents.csv.gz --column masked_text_sha256`.

**Edited, control and moved inputs.** With restored baseline texts:

```bash
python code/tools/apply_edits.py \
    --variant 'selected_combo:blank_collapse_native+capital_title_case' \
    --variant 'random:selected_combo:blank_collapse_native+capital_title_case' \
    --out edited.jsonl restored_wikileaks.jsonl
```

Every rebuilt text is checked against its released SHA-256. Variant names:
`selected_combo:blank_collapse_native+capital_title_case` (the strict combined
edit), `single:<operator>`, `position:move:{end,middle,front}`,
`combo:{date_header_blanks,doctype_blanks}`, and `random:<name>` for the
matched random control of `<name>`. `data/inputs/condition_inputs.csv.gz`
lists the hash of each input as fed to the models for the paired cohorts.

**Normalized inputs.** `code/tools/normalize.py {N1,N2}` applies the two
normalization rules; the normalization experiments apply them to the baseline,
edited and control inputs above.

**Preservation checks.** `code/preservation/audit_saved_body_preservation.py`
verifies, for every saved variant, that text outside the registered edits is
unchanged and that inverting the edits returns the original text.

**PORTIONLINK.** `code/portionlink/build_high_confidence_portions.py` rebuilds
the protected spans from the OCR layout; the released coordinates can be
checked against restored text through `visible_portion_sha256` (original
text) and `recoverable_text_sha256` (marker-removed text).

## 3. Re-run models (GPU)

- Trained classifiers: `code/training/train_small_head.py` (LoRA + 4-way
  score head), `train_small_generative.py` (LoRA on the language-model head,
  label name generated after the document) and `train_small_head_normalized.py`
  (training and inference on N2-normalized text). Hyper-parameters are in each
  file's `get_config`; all runs use seed 7701, at most 4,096 tokens, two
  epochs, checkpoint selection on development macro-F1.
- Zero-shot classification uses a locally served model through
  `code/zero_shot/run_classify.py`.

Model weights are not part of this artifact; the base models are public.
