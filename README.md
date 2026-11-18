# Review artifact

Anonymous, review-oriented artifact for a submission to the IEEE Symposium on
Security and Privacy 2027.

It contains code for marker removal, protected-span restoration (PORTIONLINK),
edit and control generation, and preservation checks; the data splits; the
PORTIONLINK protected-span coordinates and input hashes; and per-document
predictions for the accuracy, format-edit, normalization and
generative-classifier experiments. `TABLE_TO_SOURCE_MAP.md` lists each reported
result with its files and names the analyses that are reported as aggregates
only.

**No document text is redistributed.** Source documents are referenced by
their identifiers in public collections, together with SHA-256 digests of the
exact texts used, and code that restores those texts. Predictions are released
without titles, quotations, rationales or raw model responses.

## Quick start

Python 3.9+; the checks below use only the standard library.

```bash
python code/tools/verify_checksums.py
python code/tools/reaggregate.py --check
```

A successful run ends with

```text
reaggregation checks: 582/582 passed
```

`reaggregate.py` recomputes, from the released per-document predictions, the
baseline / edited / control counts (accuracy, protected-to-U, protected
missed, over-protection, new and fixed errors) for all model families, the
header-block move counts, and the zero-shot accuracies on original versus
marker-removed text, and compares them with the frozen summaries
`strict_combo_aggregates.json` and `normalization_same_documents.json` in
`data/summaries/`. The four-edit table (`four_edit_table.csv`) is not part of
`--check`; its counts follow from the row-level files named in
`TABLE_TO_SOURCE_MAP.md`.

## Layout

| Path | Content |
|---|---|
| `data/documents/` | One row per document: public identifier or collection key, label, SHA-256 and length of the original and marker-removed text, split membership. CIA 37,862; WikiLeaks 8,939; FRUS 21,864. |
| `data/splits/` | CIA train/dev/test/portion-evaluation split with near-duplicate clusters; WikiLeaks/FRUS train/dev/held-out split used for the same-format models. |
| `data/portionlink/` | 7,773 high-confidence protected spans in 2,299 CIA documents: character coordinates in the original and marker-removed text, portion label, layout provenance and span hashes (no span text). |
| `data/inputs/` | SHA-256 of every model input for the edit conditions (`condition_inputs.csv.gz`) and of every generated edit/control/position variant (`edit_variant_hashes.csv.gz`). |
| `data/edits/` | Text-free edit operations for all valid variants (`edit_operations.jsonl.gz`); see `code/tools/edit_ops.py`. |
| `data/predictions/` | Per-document predictions; see `TABLE_TO_SOURCE_MAP.md`. |
| `data/summaries/` | Frozen aggregate values; `reaggregate.py --check` reproduces the strict-combination and normalization summaries. |
| `code/tools/` | Stand-alone tools written for this release (standard library only). |
| `code/marker_removal/`, `code/portionlink/`, `code/edits/`, `code/preservation/` | Research code for the four released components, as run. |
| `code/restore/`, `code/training/`, `code/analysis/`, `code/zero_shot/` | Source re-acquisition, classifier training, analysis and zero-shot inference code. |

See `DATA_CARD.md` (sources, labels, counts), `REPRODUCTION.md` (restoring
text, rebuilding inputs, recomputing results), `CODE_MAP.md` (what each code
file does and what it needs), `TABLE_TO_SOURCE_MAP.md` (reported result →
released file), `LICENSE_STATUS.md`, and `ARTIFACT_MANIFEST.md` (every file with
its row count and size; about 105 MB in total).

## Scope and limits

- Results are for one training seed (7701). Inference was repeated with seeds
  7702 and 7703 for the six CIA-trained classifiers with identical predictions
  and scores; only the seed-7701 rows are included.
- 446 of 21,822 random-control variants cannot be rebuilt from the text-free
  operations alone (flagged `reconstructible = 0`); they can be regenerated
  with `code/edits/`. All 51,588 attack and position variants rebuild exactly.
- The research code is released as run. It expects the original workspace
  layout and, for training and edit generation, GPUs and model weights that
  are not part of this artifact; `CODE_MAP.md` lists the requirements.
- Model weights, raw model responses and extracted quotations are not
  included.
