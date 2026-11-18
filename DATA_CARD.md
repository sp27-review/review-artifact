# Data card

All counts are documents. Labels are the four classification levels
UNCLASSIFIED (U), CONFIDENTIAL (C), SECRET (S), TOP SECRET (TS) assigned by the
originating agency; they are not annotations made by the authors.

## Sources and identifiers

| Source | Public collection | Identifier in this artifact | Documents |
|---|---|---|---:|
| CIA | CIA Reading Room (CREST) items mirrored on the Internet Archive | `cia-readingroom-document-<id>`; `official_catalog_url`, `acquisition_url` | 37,862 |
| WikiLeaks | Public Kaggle dataset `yilouislu/wikileaks` (U.S. diplomatic cables, Jan–Feb 2010), archive SHA-256 `71a1123e…d77f8` as retrieved 2026-10-30 | `wikileaks_cable_classifier:<cable id>` | 8,939 |
| FRUS | Public Kaggle dataset `mehmetturulsariek/frus-dataset` (Foreign Relations of the United States documents), archive SHA-256 `5ab3f12b…3fd85` as retrieved 2026-10-30 | `frus_sachn:<document id>` | 21,864 |

Full archive digests: WikiLeaks
`71a1123e58fd1181b2272c18d8f32bdd9867f27f582e8cfbd12423775e3d77f8`, FRUS
`5ab3f12b85335b37dce5ccab236cc2e14b0abcc941b67a3e41b8de453a13fd85`.

## Text versions and hashes

Two text versions exist for every document; neither is distributed.

- **Original (visible) text** — OCR or collection text with classification
  markings in place (`visible_text_sha256` / `raw_text_sha256`).
- **Marker-removed text** — the same text after removing banner markings,
  portion marks and other explicit level cues (`masked_text_sha256`). This is
  the baseline input of all experiments unless a condition says "raw".

Hashes are SHA-256 of the UTF-8 text exactly as given to the models. Character
coordinates elsewhere in the artifact index Python `str` positions of these
texts.

## Splits

CIA (`data/splits/cia_split.csv.gz`). Near-duplicate clusters never cross train, dev, test and portion_eval; near duplicates of portion_eval documents are quarantined.

| Split | Documents | Use |
|---|---:|---|
| train | 28,359 | classifier training |
| dev | 3,530 | checkpoint selection |
| test | 3,562 | clean accuracy |
| portion_eval | 2,299 | documents with restored protected spans; edit experiments |
| portion_neighbor_quarantine | 112 | near duplicates of portion_eval documents; unused |

WikiLeaks and FRUS (`data/splits/external_split.csv.gz`), used by the
same-format small models. The edit pool (`in_edit_pool`) contains the documents
whose marker-removed text fits within 4,096 tokens for every evaluated
tokenizer.

| Source | Edit pool | train | dev | held-out evaluation |
|---|---:|---:|---:|---:|
| WikiLeaks | 8,287 (U 4,731 / C 2,952 / S 604) | 4,854 | 539 | 2,894 |
| FRUS | 13,736 (U 142 / C 3,714 / S 8,089 / TS 1,791) | 9,676 | 1,074 | 2,986 |

Held-out documents are those in the paired edit cohorts below; they are never
used for training or model selection.

## Edit cohorts (`data/inputs/condition_inputs.csv.gz`)

| Input set | Conditions | WikiLeaks | FRUS | CIA |
|---|---|---:|---:|---:|
| strict_combo | base, combo (blank-line collapse + capital-heading title case), combo_ctrl (matched random control) | 2,528 | 431 | 597 |
| strict_combo | single (blank-line collapse only), single_ctrl | 1,519 | 92 | — |
| position_move | base, move_end, move_middle (same header block moved to the end / middle) | 645 | 2,694 | 1,638 |
| single_edits | blank, blank_ctrl (blank-line collapse alone and its matched random control) | 1,743 | 648 | 1,998 |
| single_edits | capital, capital_ctrl (capital-heading title case alone and its matched random control) | 2,536 | 495 | 615 |

A document enters a single-edit cohort only when the edit changes its text and
a matched control could be built for it, so these cohorts are smaller than the
set of documents the edit applies to. For WikiLeaks and FRUS the single-edit
cohorts are restricted to held-out evaluation documents. A matched control
whose text equals the edited text is kept (rows with the same `text_sha256`
for the edit and its control); this is frequent for the title-case edit.

Text-free operations (`data/edits/`) cover, for CIA documents, the strict
combined edit, the two single edits, the block moves and their controls. The
two auxiliary combinations used in the score-based selection experiment
(`combo:date_header_blanks`, `combo:doctype_blanks` and their controls, 2,952
CIA variants) are released as predictions together with the SHA-256 of each
input; they can be regenerated with `code/edits/`.

`data/inputs/edit_variant_hashes.csv.gz` lists 73,410 valid edited, control
and moved inputs: the 71,136 inputs of the preservation check reported in the
paper without the 2,952 auxiliary CIA variants above, plus the 5,226 CIA
single-edit inputs (edits and controls), which pass the same checks.

## PORTIONLINK (`data/portionlink/portionlink_r5_high_confidence.jsonl.gz`)

7,773 high-confidence portion-marked spans in 2,299 CIA documents
(U 2,755 / C 1,530 / S 3,375 / TS 113). Each record gives the portion mark as
printed (`marking_original`), its normalized level, page/paragraph/line
position and bounding boxes from the OCR layout, the span of the mark and of
the portion in the original text (`visible_mark_span`, `visible_portion_span`),
the corresponding spans in the marker-removed text (`masked_portion_spans`),
and SHA-256 digests of the span text. Use `masked_portion_spans` when comparing
with model inputs.

## Predictions

Every prediction file carries the document identifier, the gold label and the
predicted label. Trained-classifier files also carry the four logits and token
counts; zero-shot files carry the SHA-256 of the input given to the model, the
parse path and the number of attempts. An empty prediction means the model
produced no parseable label and is scored as an error; so is a label outside
the four grades (two `SBU` rows).

Field notes. `prediction_source` in the zero-shot files: `primary` (first
response parsed), `retry1` (parsed after one retry), `reuse_r4_reparsed`
(prediction taken from an earlier run on a byte-identical input) and
`fresh_v5full_r5` (re-run because the final marking-removal revision changed
the input). Conditions prefixed `orig_` repeat the WikiLeaks conditions with
the unmodified ClassifyIt classification call; `r5_` and `raw_` in the
additional-model file denote marker-removed and original text.
`source_generation` in the PORTIONLINK file is the version tag of the corpus
build. `greedy_first_token_is_label` (generative classifiers) is 1 when the
most likely first generated token is one of the four grade names.
