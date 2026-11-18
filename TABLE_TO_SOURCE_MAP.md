# Reported result → released files

Results are listed in the order they appear in the paper. All paths are under
`data/predictions/` unless stated otherwise.

| Result | Files and conditions |
|---|---|
| Zero-shot accuracy on original vs. marker-removed text, CIA 37,862 documents, three models | `zero_shot/zs1_cia_raw.csv.gz`, `zero_shot/zs1_cia_masked.csv.gz` |
| Same comparison on WikiLeaks and FRUS documents whose text changed under marker removal (4,257 / 2,190 common documents) | `zero_shot/zs1_external_raw_masked.csv.gz`; an additional model that is not reported in the paper: `zero_shot/zs1_external_gemma4.csv.gz` |
| Instruction-priority experiment, 3,493 documents (format-first vs. content-first instruction) | `zero_shot/zs4_instruction_rule_{masked,raw}.csv.gz` (rule-preserving prompts, with P0), `zero_shot/zs4_instruction_initial_{masked,raw}.csv.gz` (initial prompts), `zero_shot/zs4_instruction_gemma4.csv.gz` (additional model, not reported in the paper) |
| Six classifiers trained on CIA: clean test and portion-evaluation accuracy | `classifiers_cia/test_and_portion_predictions.csv.gz` |
| Format edits against the six CIA-trained classifiers: single edits, combinations, block moves, matched controls (seed 7701) | `trained_classifiers_edits/{external,cia}_{llama,qwen,nemotron,longformer,deberta,tfidf}.csv.gz`; condition names as in `data/inputs/edit_variant_hashes.csv.gz` |
| Score-based choice among up to three candidate edits (block move to the end, `combo:date_header_blanks`, `combo:doctype_blanks`) | predictions and logits of the baseline and of every candidate in `trained_classifiers_edits/{cia,external}_<model>.csv.gz` |
| Strict combined edit, baseline / attack / control, all model families on the same documents | `derived/strict_combo_per_document.csv.gz` (`in_E` = 1 when baseline, edit and control are all valid for the model; `pilot` = 1 marks the documents used earlier for the concentration measurement, so the analysis that excludes them can be recomputed); frozen counts and intervals in `data/summaries/strict_combo_aggregates.json` |
| Four-edit table (blank-line collapse, title case, block move to the end, combined edit) × three sources × eleven classifiers | `data/summaries/four_edit_table.csv` (frozen counts, differences and 95% intervals). Row level: `trained_classifiers_edits/{external,cia}_<model>.csv.gz`, `trained_classifiers_edits/cia_singles_<model>.csv.gz` (CIA single edits and their controls) and the `s4_*` conditions of `small_models/<backbone>_cia_ext_seed7701.csv.gz` |
| Same-format small models (five backbones trained on CIA + WikiLeaks + FRUS): 34 edit conditions and clean evaluation | `small_models/<backbone>_cia_ext_seed7701.csv.gz` |
| Small models trained on CIA only | `small_models/{llama8b,qwen7b}_cia_only_seed7701.csv.gz` |
| Zero-shot classifiers on the same edited documents | `zero_shot/zsc_format_edit.csv.gz`, `zero_shot/zsc_position_move.csv.gz` |
| Input normalization: inference only (N1, N2) and training + inference (N2) | `small_models/<backbone>_cia_ext_{N1,N2}_inferonly_seed7701.csv.gz`, `small_models/<backbone>_cia_ext_N2_seed7701.csv.gz`, `small_models/llama70b_cia_only_{N1,N2}_inferonly_seed7701.csv.gz`; frozen counts in `data/summaries/normalization_same_documents.json` |
| Generatively fine-tuned classifiers (no added head) | `small_models/<backbone>_cia_ext_gen*_seed7701.csv.gz` (`qwen7b_cia_ext_gen_seed7701` was trained for one epoch; files with `_ep2` and the other backbones for two); free-generation check counts in `data/summaries/free_generation_checks.json` |
| Format-only logistic regression (layout features, no words) | `data/summaries/format_only_logreg_results.json`; code `code/analysis/format_only_logreg.py` |
| Protected spans | `data/portionlink/portionlink_r5_high_confidence.jsonl.gz` |

Small-model condition names: `{wikileaks,frus,cia}_{base,combo,combo_ctrl}`,
`{wikileaks,frus}_{single,single_ctrl}`,
`pos_{frus,wikileaks,cia}_{base,move_end,move_middle}`,
`s4_{wikileaks,frus,cia}_{blank,blank_ctrl,capital,capital_ctrl}` (single
edits; `data/inputs/condition_inputs.csv.gz`, input set `single_edits`), and
`eval_{test,portion_eval,ext_dev}` / `cia_test` / `ext_dev` for clean
evaluation.

Columns of `data/summaries/four_edit_table.csv`: `E` documents with a valid
baseline, edited and control prediction; `acc_base`, `acc_edit`, `acc_ctrl`
accuracy (%); `neg_edit`, `neg_ctrl` documents correct at baseline and wrong
after the edit / the control, out of `neg_denom` baseline-correct documents;
`up`, `down` those edit errors split by direction (predicted above / below the
gold level); `ptu_edit`, `ptu_ctrl` baseline-correct protected documents
(C/S/TS) predicted U, out of `ptu_denom`; `d_neg`, `d_ptu` edit − control
differences in percentage points with 95% intervals (`*_lo`, `*_hi`; paired
bootstrap over documents, 10,000 resamples, seed 7701). For the block move the
control is the same block moved to the middle.

Baselines of the single-edit rows. CIA-trained classifiers: the `clean`
condition. Same-format models on CIA: the `eval_portion_eval` rows. Same-format
models on WikiLeaks and FRUS: the union of `{source}_base` and
`pos_{source}_base`; when a document is in both, the `{source}_base` row is
used. These two baseline runs disagree for about ten document-model pairs
(batch-dependent numerical noise in half-precision inference), which can move
a count by one.

Reported in the paper as aggregates only (no row-level files in this release):

- quotation-level results of the zero-shot classifiers: overlap between
  quotations and protected spans (lift), the share of quotations that point at
  formatting, and the effect of deleting a model's own first quotation
  (quotations are text-bearing);
- the protected-span deletion comparison for the zero-shot classifiers;
- the attribution analysis of the CIA-trained classifiers: most influential
  sentences, their deletion, and attention / gradient concentration;
- the single-edit screen on 256 CIA development documents used to choose the
  combined edit, and the relocation subset restricted to standalone lines;
- the audits of the restored protected spans (the 200-span sample and the pass
  over the full resource);
- per-document outputs of the format-only logistic regression (its summary
  values and code are included).
