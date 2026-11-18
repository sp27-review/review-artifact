# Code map

`code/tools/` is written for this release, documented, and runs with the
Python standard library. Everything else is the research code as it was run.
For the review release, workspace paths and internal project names were
replaced by neutral ones, most inline comments and docstrings were removed, and
free-text string literals (report labels, log messages) were replaced by opaque
tokens of the form `x` + four hexadecimal digits; the logic is unchanged. These
files expect the original workspace layout (dataset and run directories under
the workspace root) and are provided so that every transformation can be
inspected. Several of them depend on modules or inputs that are not part of
this release: the zero-shot prompt templates and their driver, the training
code of the six CIA-trained classifiers, and parts of the `corelib` helper
package.

## `code/tools/` — stand-alone

| File | Purpose |
|---|---|
| `verify_checksums.py` | Verify all released files against `checksums.sha256`. |
| `reaggregate.py` | Recompute reported counts from released predictions; `--check` compares with `data/summaries/`. |
| `verify_text_hashes.py` | Compare locally restored texts with the manifest hashes. |
| `edit_ops.py`, `acronyms.txt` | Text-free edit encoding and the transforms it uses. |
| `apply_edits.py` | Rebuild edited / control / moved inputs from restored baseline texts and verify their hashes. |
| `normalize.py` | Normalization rules N1 and N2. |

## `code/restore/`

| File | Purpose |
|---|---|
| `download_cia_ocr.py` | Plan or perform download of CIA OCR text from the Internet Archive with hash verification (standard library). |

## `code/marker_removal/` — removal of classification markings

| Path | Purpose | Needs |
|---|---|---|
| `cia/pipeline/p1_download.py`, `p1b_repair_items.py` | Collect Internet Archive items of the CIA Reading Room. | network |
| `cia/pipeline/p2_filter_length.py`, `p3_filter_quality.py` | Length and OCR-quality filters. | OCR files |
| `cia/pipeline/p4_extract_labels.py` | Recover the document-level classification from banner lines. | OCR layout files |
| `cia/pipeline/p5_build_marker_dataset.py`, `p5_r4_build_marker_dataset.py`, `p5u_*.py` | Build original and marker-removed text for classified and unclassified CIA documents. | `corelib/`, `config.py` |
| `cia/r5/s0_build_eligible_manifest.py`, `s1_masking_rules_r5.py`, `s3_build_marker_dataset.py`, `s3b_verify_residual.py` | Final rule revision used in the paper (residual banner words and portion marks) and its residual check. | output of the previous step |
| `external/prepare_published.py` | Convert the two public archives to per-document records with agency labels. | Kaggle archives |
| `external/prepare_external_masked_20261030.py`, `prepare_wikileaks_masked_20261030.py`, `verify_external_masked_20261030.py` | Marker removal for WikiLeaks and FRUS, deletion log, residual audit, verification. | review decisions (`external_marking_review_textfree.json`) |
| `external/supplement_{wikileaks,frus}/` | Supplemental removal pass used for the original-versus-marker-removed comparison on these sources. | — |

## `code/portionlink/` — protected-span restoration

| File | Purpose | Needs |
|---|---|---|
| `build_high_confidence_portions.py` | Detect portion marks in the OCR layout, resolve each mark's span, project it onto the original and marker-removed text, assign confidence. | OCR layout XML, corpus text |
| `audit_high_confidence_portions.py` | Integrity audit of spans, hashes and overlaps. | builder output |

## `code/edits/` — edit and control generation

| File | Purpose |
|---|---|
| `format_extension_common.py` | Edit operators (dates, headers, section titles, document type, blank lines, separators, block moves), span protection, composition of variants. |
| `meeting_controls_common.py` | Matched random controls (same number, type, size range and position zone of edits). |
| `external_attack_common.py`, `external_focus_line_endings.py` | Transfer of the frozen operators to WikiLeaks and FRUS, including the capital-heading rule and CRLF handling. |
| `prepare_strict_combo_fullpool.py`, `prepare_relaxed_capital_combo.py`, `prepare_fullpool_firstseed.py`, `build_format_extension.py` | Build and freeze the per-document variant files. |
| `export_strict_combo_inputs.py`, `export_position_move_inputs.py` | Re-apply the frozen edits, check hashes, and export the paired model inputs. |
| `run_format_extension.py`, `run_fullpool_firstseed.py` | Inference over frozen variants with the trained classifiers (exact-input cache). |
| `extract_classifier_attention.py`, `analyze_classifier_focus.py`, `attention_probe.py`, `deberta_attention.py` | Dependencies of the modules above (model loading, layout cue annotation). |

Needs: restored corpus text; for the `run_*` scripts, GPUs and trained checkpoints.

## `code/preservation/`

`audit_saved_body_preservation.py` — for every saved variant, checks that text
outside the registered edits is identical, that inverting the edits returns
the original, and that input and output hashes match.

## `code/training/`, `code/zero_shot/`, `code/analysis/`

| File | Purpose |
|---|---|
| `training/build_training_split.py` | WikiLeaks/FRUS train/dev split with the evaluation cohorts held out. |
| `training/train_small_head.py` | LoRA + 4-way score-head classifier (prepare, preflight, train, evaluate, predict). |
| `training/train_small_generative.py` | LoRA on the language-model head; the label name is generated after the document. |
| `training/train_small_head_normalized.py`, `training/score_70b_normalized.py` | Normalization experiments. |
| `zero_shot/run_classify.py` | Zero-shot classification driver for a locally served model. |
| `analysis/strict_combo_reaggregate.py`, `analysis/normalization_same_documents.py`, `analysis/analyze_format_edit.py`, `analysis/analyze_position_move.py`, `analysis/format_only_logreg.py` | Scripts that produced the summaries and intervals reported in the paper. |

Needs: GPUs and public base-model weights (training), a model server (zero-shot).
