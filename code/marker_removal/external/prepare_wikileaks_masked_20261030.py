#!/usr/bin/env python3
"""Born-digital WikiLeaks adaptation of the frozen CIA v5full-r5 text rules.

No OCR reconstruction, fuzzy OCR deletion, model execution, or label inference.
Only recorded character spans are deleted; all other characters remain exact.
Mark offsets are retained for later portion-boundary annotation, not asserted gold.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/datasets/wikileaks_cable_classifier/v1"
R5_PATH = ROOT / "experiments/dataset_construction/pipeline/r5/s1_masking_rules_r5.py"
DEFAULT_OUT = ROOT / "data/datasets/wikileaks_cable_classifier/masked_20261030"


def load_rules():
    # Resolve the original machine's hardcoded ROOT in memory; frozen code unchanged.
    tree = ast.parse(R5_PATH.read_text(), filename=str(R5_PATH))
    replaced = 0
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ROOT" for t in node.targets):
            node.value = ast.Call(func=ast.Name(id="Path", ctx=ast.Load()),
                                  args=[ast.Constant(str(ROOT))], keywords=[])
            replaced += 1
    assert replaced == 1
    mod = types.ModuleType("wikileaks_cia_r5_rules")
    mod.__file__ = str(R5_PATH)
    sys.modules[mod.__name__] = mod
    exec(compile(ast.fix_missing_locations(tree), str(R5_PATH), "exec"), mod.__dict__)
    return mod


R5 = load_rules()
MARK = re.compile(r"\([ \t]*(?P<grade>TOP SECRET|CONFIDENTIAL|UNCLASSIFIED|SECRET|SBU|TS|[USC]|sbu|ts|[usc])"
                  r"(?=[ \t/)-])(?P<tail>[^()\n]{0,90})\)")
TAIL = re.compile(r"^(?:(?://|/|-)[ \t]*)?(?:NF|NOFORN|FOUO|LES|"
                  r"REL(?:[ \t]+TO)?(?:[ \t]+[A-Za-z., /-]+)?)"
                  r"(?:(?://|/)(?:NF|NOFORN|FOUO|LES|REL(?:[ \t]+TO)?[ \t]+[A-Za-z., /-]+))*$", re.I)
PARA_PREFIX = re.compile(r"^(?:[¶][ \t]*)?(?:\d{1,3}(?:[.,)>]{1,2})?|\(\d{1,3}\)|"
                         r"[A-Za-z][.)]{1,2}|\d{1,3}\.(?:B(?:[ \t]+B)?|C,B))[ \t]*$")
BOUNDARY = re.compile(r"^(?:[¶][ \t]*)?\d{1,3}[.,)][ \t]*(?:\(|[A-Za-z])")
FIELD = re.compile(r"^(?:SUBJECT|TAGS|REF|TO|FROM|SIPDIS|SUMMARY|COMMENT)\b", re.I)
CLASSIFIER = re.compile(r"^CLASSIFIED[ \t]+BY\b", re.I)
EO = re.compile(r"^E\.?[ \t]*O\.?[ \t]*(?:12958|13526)\b", re.I)
REASONS = re.compile(r"^(?:REASONS?|DECLASSIFY ON|DECLASSIFICATION DATE)[ \t]*:", re.I)
REASON_CODE = re.compile(r"^(?:REASONS?[ \t]*:[ \t]*)?1\.[45][ \t]*\([a-hA-H]\)")
SBU_NOTICE = re.compile(r"\bSENSITIVE[ \t]+BUT[ \t]+UNCLASSIFIED\b", re.I)
NOTICE_TAIL = re.compile(
    r"^[\s.,;:\-]*(?:(?:PLEASE\s+)?(?:PROTECT|HANDLE|TREAT)\s+ACCORDINGLY[.!;]?|"
    r"NOT\s+(?:FOR|INTENDED\s+FOR)\s+(?:PUBLIC\s+)?(?:INTERNET(?:\s+(?:DISTRIBUTION|DISSEMINATION))?|PUBLICATION)[.!;]?|"
    r"(?:AND\s+)?FOR\s+OFFICIAL\s+USE\s+ONLY[.!;]?|ENTIRE\s+TEXT[.!;]?|"
    r"NOT\s+FOR\s+TRANSMISSION\s+OUTSIDE\s+USG\s+CHANNELS[.!;]?)", re.I)
SELF_NOTICE = re.compile(r"\b(?:This\s+(?:cable|report)\s+is|(?:The\s+)?entire\s+(?:text\s+of\s+this\s+)?report\s+is)"
                         r"\s+Sensitive\s+But\s+Unclassified\s*\(SBU\)[.]?", re.I)
GRADE_BANNER = re.compile(
    r"^(?:TOP[ \t]+SECRET|SECRET|CONFIDENTIAL|UNCLASSIFIED|UNCLAS|"
    r"S E C R E T|C O N F I D E N T I A L|U N C L A S)(?:[ \t]*//[ \t]*(?:NF|NOFORN))?"
    r"(?:[ \t]+SECTION[ \t]+\d+[ \t]+OF[ \t]+\d+(?:[ \t]+[A-Z]+[ \t]+\d+)?)?[ \t.]*$")
HANDLING = re.compile(r"^(?:NOFORN|SIPDIS|EXDIS|NODIS|LIMDIS|FOR OFFICIAL USE ONLY|CONFIDENTIAL FOR TRANSMISSION ONLY)[ \t.]*$", re.I)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def line_positions(text):
    pos = 0
    for line in text.splitlines(keepends=True):
        end = pos + len(line.rstrip("\r\n"))
        yield pos, end, line.rstrip("\r\n")
        pos += len(line)


def valid_position(text, start, end):
    ls, le = R5._line_bounds(text, start)
    prefix = text[ls:start].strip()
    if re.match(r'^(?:REF|REFS|REFERENCES)[ \t]*:', prefix, re.I):
        return False
    prefix_without_marks = MARK.sub('', prefix).strip()
    # Extend r5's paragraph-number test to WikiLeaks' optional pilcrow.
    numbered = bool(PARA_PREFIX.fullmatch(prefix_without_marks))
    summary_prefix = bool(re.fullmatch(r'\d{1,3}[.)][ \t]*(?:SUMMARY|COMMENT)', prefix_without_marks, re.I))
    leading = numbered or summary_prefix or not prefix_without_marks or R5.at_leading(text, start)
    # Title-end marking is accepted only on a separate short block, not a
    # wrapped prose line ending in an acronym such as Technical Secretariat (TS).
    trailing = R5.at_trailing(text, start, end)
    before_line = text[:ls].rstrip("\r\n")
    prev_line = before_line.rsplit("\n", 1)[-1] if before_line else ""
    trailing = trailing and (ls == 0 or not prev_line.strip() or bool(re.match(r'^[-•]',prefix)))
    context = re.sub(r"\s+", " ", text[max(0, start - 100):start]).lower()
    if re.search(r"technical secretariat\s*$|(?:ukrainian security services?|security service of ukraine)\s*$", context):
        return False
    if leading and not prefix and ls > 0 and prev_line.strip():
        # A hard line wrap is not a paragraph start. Accept only when a marker
        # is followed by a capitalized sentence, not '(TS) to discuss ...'.
        after = text[end:le].lstrip()
        if not re.match(r'[A-Z0-9"“\[]', after):
            leading = False
    return leading or trailing


def detect(text):
    edits, marks, unresolved = [], [], []
    original_r5 = set(R5.portion_spans(text))
    for m in MARK.finditer(text):
        tail = m['tail'].strip()
        accepted_tail = not tail or bool(TAIL.fullmatch(tail))
        position = valid_position(text, m.start(), m.end())
        if m['grade'].islower():
            ls, _ = R5._line_bounds(text, m.start())
            prefix = text[ls:m.start()].strip()
            position = position and bool(PARA_PREFIX.fullmatch(prefix) or re.fullmatch('SUBJECT:',prefix,re.I))
            # Preserve legal references such as '1(c)(iii)' and lettered lists.
            position = position and bool(re.match(r'[ \t]+[A-Z]',text[m.end():]))
        if not accepted_tail or not position:
            unresolved.append(dict(start=m.start(), end=m.end(), text=m.group(),
                                   reason="nonmark_tail_or_acronym" if not accepted_tail else "nonmark_position"))
            continue
        end = m.end()
        suffix = re.match(r"[ \t]*//[ \t]*(?:NF|NOFORN|FOUO|LES)\b", text[end:])
        if suffix:
            end += suffix.end()
        rule = "r5_portion_position_guard" if (m.start(), m.end()) in original_r5 else "wikileaks_portion_extension"
        if not tail and m['grade'].upper() != 'SBU':
            assert R5.R4.PORTION_ABBR_RE.fullmatch(m.group().upper()) or R5.R4.PORTION_MARK_RE.fullmatch(m.group().upper())
            rule = "r4_portion_with_position_guard"
        if m['grade'].islower():
            rule = 'wikileaks_lowercase_numbered_portion'
        mark = dict(start=m.start(), end=end, grade=m['grade'].upper(), mark=text[m.start():end], rule=rule)
        marks.append(mark)
        edits.append((m.start(), end, rule))

    lines = list(line_positions(text))
    for i, (start, end, line) in enumerate(lines):
        s = line.strip()
        # A classification header can itself have a leading portion marker.
        s = re.sub(r'^\d{1,3}[.)][ \t]*(?=\()', '', s)
        s = MARK.sub('', s, count=1).strip() if MARK.match(s) else s
        is_classifier = CLASSIFIER.match(s) and (re.match(r'^CLASSIFIED[ \t]+BY[ \t]*:',s,re.I)
                         or re.search(r'\breasons?\b|\b1\.[45]\b',text[start:min(len(text),end+100)],re.I))
        if is_classifier:
            # Administrative field plus wrapped continuation; stop before any
            # blank line, numbered paragraph, or new body/header field.
            j = i + 1
            stop = end
            while j < len(lines) and j < i + 6:
                ns = lines[j][2].strip()
                if not ns or BOUNDARY.match(ns) or FIELD.match(ns) or ns.startswith('('):
                    break
                if CLASSIFIER.match(ns) or EO.match(ns):
                    break
                if re.match(r'^[/_=*-]{3,}|^PLS[ \t]+ZFR',ns):
                    break
                stop = lines[j][1]
                j += 1
            edits.append((start, stop, "wikileaks_classification_header"))
        elif EO.match(s) or (REASONS.match(s) and re.search(r'1\.[45]|\d{2}/\d{2}/\d{4}', s)):
            edits.append((start, end, "wikileaks_classification_header"))
        elif REASON_CODE.match(s):
            edits.append((start, end, "wikileaks_classification_reason"))
        elif GRADE_BANNER.fullmatch(s):
            # Reuse the r5 exact-grade detector for ordinary (unspaced) banners.
            assert R5.banner_grade_token_spans(s) or ' ' in s or s == 'UNCLAS'
            edits.append((start, end, "r5_exact_banner_with_full_line_guard"))
        elif HANDLING.fullmatch(s):
            edits.append((start, end, "wikileaks_handling_banner"))
        for m in SBU_NOTICE.finditer(line):
            prefix = line[:m.start()].strip()
            prefix = re.sub(r'^\d{1,3}[.)][ \t]*', '', prefix)
            prefix = MARK.sub('', prefix).strip()
            if prefix:
                continue
            finish = m.end()
            # Exact phrase whitelist also recognizes hard line wraps.
            tailtext = text[start + finish:]
            while True:
                mt = NOTICE_TAIL.match(tailtext)
                if not mt:
                    break
                finish += mt.end()
                tailtext = tailtext[mt.end():]
            edits.append((start + m.start(), start + finish, "wikileaks_sbu_notice"))
    for m in SELF_NOTICE.finditer(text):
        finish=m.end()
        while (tail:=NOTICE_TAIL.match(text[finish:])):
            finish+=tail.end()
        edits.append((m.start(),finish,'wikileaks_sbu_self_declaration'))
    merged = []
    for a, b, rule in sorted(set(edits)):
        if merged and a < merged[-1]['end']:
            merged[-1]['end'] = max(b, merged[-1]['end'])
            merged[-1]['rules'] = sorted(set(merged[-1]['rules'] + [rule]))
        else:
            merged.append(dict(start=a, end=b, rules=[rule]))
    unresolved = [m for m in unresolved if not any(e['start'] <= m['start'] < e['end'] for e in merged)]
    return merged, marks, unresolved


def apply_edits(text, edits):
    parts, last = [], 0
    for e in edits:
        a, b = e['start'], e['end']
        assert last <= a < b <= len(text)
        parts.append(text[last:a])
        last = b
    parts.append(text[last:])
    return ''.join(parts)


def project(pos, edits):
    return pos - sum(max(0, min(pos, e['end']) - e['start']) for e in edits if e['start'] < pos)


def self_test():
    cases = [
        ('1. (C) The meeting starts at nine.', '1.  The meeting starts at nine.'),
        ('1. (SBU) Keep this substantive sentence.', '1.  Keep this substantive sentence.'),
        ('2. (S/REL TO USA, AUS, CAN, GBR, NZL) Facts remain.', '2.  Facts remain.'),
        ('5 (C) Facts remain.', '5  Facts remain.'),
        ('A) (S/NF) Facts remain.', 'A)  Facts remain.'),
        ('(2) (U) Facts remain.', '(2)  Facts remain.'),
        ('3.) (C) Facts remain.', '3.)  Facts remain.'),
        ('2. (c) Facts remain.', '2.  Facts remain.'),
        ('(c) A list item.\n1(c)(iii) is a legal reference.', '(c) A list item.\n1(c)(iii) is a legal reference.'),
        ('1. (C) (C) Facts remain.', '1.   Facts remain.'),
        ('REF: (C) ANKARA 100', 'REF: (C) ANKARA 100'),
        ('The Technical Secretariat\n(TS) to discuss the report.', 'The Technical Secretariat\n(TS) to discuss the report.'),
        ('Office of the Coordinator (S/CT) discussed the issue.', 'Office of the Coordinator (S/CT) discussed the issue.'),
        ('Security Service of Ukraine (SBU)', 'Security Service of Ukraine (SBU)'),
        ('Secret Detention Study:\nThe Secret Service attended.', 'Secret Detention Study:\nThe Secret Service attended.'),
        ('Classified By: An official. R\neasons: 1.4 (b) and (d)\n\n1. (C) Facts.', '\n\n1.  Facts.'),
        ('reasons: in Turkey, Syria was dealing with a state.', 'reasons: in Turkey, Syria was dealing with a state.'),
        ('Channels were\nclassified by Conatel as national producers.\nThe same channels remain.', 'Channels were\nclassified by Conatel as national producers.\nThe same channels remain.'),
        ('(U) Sensitive but unclassified. NOT\nFOR INTERNET DISTRIBUTION.\n1. Facts.', ' \n1. Facts.'),
    ]
    for text, expected in cases:
        edits, _, _ = detect(text)
        assert apply_edits(text, edits) == expected, (text, apply_edits(text, edits), expected)
    return len(cases)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--limit', type=int)
    args = ap.parse_args()
    tests = self_test()
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise SystemExit(f'Refusing nonempty output directory: {args.out}')
    unique_ids = {json.loads(s)['doc_id'] for s in (SOURCE / 'unique.jsonl').open()}
    stats = Counter()
    labels = {'all': Counter(), 'unique': Counter()}
    edits_by_rule = Counter()
    samples = defaultdict(list)
    residuals = []
    groups = defaultdict(list)
    files = {n: (args.out / n).open('w') for n in ['all.jsonl', 'unique.jsonl', 'audit.jsonl']}
    for n, line in enumerate((SOURCE / 'all.jsonl').open()):
        if args.limit and n >= args.limit:
            break
        r = json.loads(line)
        text = r['text']
        edits, marks, unresolved = detect(text)
        masked = apply_edits(text, edits)
        assert sha(text.encode()) == r['text_sha256'], r['doc_id']
        assert masked.strip(), r['doc_id']
        # No rewriting or whitespace normalization: exact removal log reproduces input.
        assert len(text) - len(masked) == sum(e['end'] - e['start'] for e in edits)
        repeated, _, _ = detect(masked)
        if repeated:
            residuals.append(dict(doc_id=r['doc_id'], repeats=[dict(**e, text=masked[e['start']:e['end']]) for e in repeated]))
        row = {k: r[k] for k in ['doc_id', 'dataset', 'source_id', 'label', 'label_id', 'language', 'source_split']}
        row.update(text=masked, text_sha256=sha(masked.encode()), source_text_sha256=r['text_sha256'])
        payload = json.dumps(row, ensure_ascii=False) + '\n'
        files['all.jsonl'].write(payload)
        labels['all'][r['label']] += 1
        if r['doc_id'] in unique_ids:
            files['unique.jsonl'].write(payload)
            labels['unique'][r['label']] += 1
            groups[sha(masked.strip().encode())].append((r['doc_id'], r['label']))
        audit = dict(doc_id=r['doc_id'], source_text_sha256=r['text_sha256'], masked_text_sha256=row['text_sha256'],
                     deletions=[dict(**e, text=text[e['start']:e['end']]) for e in edits],
                     portion_mark_candidates=[dict(**m, masked_anchor=project(m['start'], edits)) for m in marks],
                     retained_marklike_candidates=unresolved)
        files['audit.jsonl'].write(json.dumps(audit, ensure_ascii=False) + '\n')
        stats.update(documents=1, changed_documents=bool(edits), removed_spans=len(edits),
                     removed_characters=len(text)-len(masked), original_characters=len(text),
                     portion_documents=bool(marks), portion_marks=len(marks),
                     classified_portion_documents=any(m['grade'] in ['C','S','TS','CONFIDENTIAL','SECRET','TOP SECRET'] for m in marks),
                     retained_marklike_candidates=len(unresolved))
        for e in edits:
            for rule in e['rules']:
                edits_by_rule[rule] += 1
                # Deterministic reservoir for checking every rule family.
                item = dict(doc_id=r['doc_id'], rule=rule, removed=text[e['start']:e['end']],
                            before=text[max(0,e['start']-90):e['start']], after=text[e['end']:e['end']+90])
                bucket=samples[rule]
                if len(bucket)<12:
                    bucket.append(item)
                else:
                    pick=random.Random(7701+edits_by_rule[rule]).randrange(edits_by_rule[rule])
                    if pick<12: bucket[pick]=item
    for f in files.values(): f.close()
    if not args.limit:
        assert sum(labels['all'].values()) == 9005
        assert sum(labels['unique'].values()) == 8940
    duplicates=[dict(masked_text_sha256=h, doc_ids=[p[0] for p in rr], labels=[p[1] for p in rr],
                     label_conflict=len({p[1] for p in rr})>1) for h,rr in groups.items() if len(rr)>1]
    dedup_ids={rr[0][0] for rr in groups.values() if len({p[1] for p in rr})==1}
    labels['deduplicated']=Counter()
    with (args.out/'deduplicated.jsonl').open('w') as out:
        for line in (args.out/'unique.jsonl').open():
            row=json.loads(line)
            if row['doc_id'] in dedup_ids:
                out.write(line)
                labels['deduplicated'][row['label']]+=1
    for name,obj in [('review_samples.json',dict(samples)),('residual_candidates.json',residuals),('masked_duplicates.json',duplicates)]:
        (args.out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    manifest=dict(date='2026-10-30',status='automated marking-removal build; portion boundaries are not validated gold',
        input_field='text',gold_field='label_id',label_map={'U':0,'C':1,'S':2,'TS':3},
        stats=dict(stats),label_counts={k:dict(v) for k,v in labels.items()},rule_counts=dict(edits_by_rule),
        remaining_rule_candidates=len(residuals),masked_duplicate_groups=len(duplicates),
        masked_conflicting_groups=sum(d['label_conflict'] for d in duplicates),regression_cases_passed=tests,
        source_files={name:sha((SOURCE/name).read_bytes()) for name in ['all.jsonl','unique.jsonl']},
        code_files={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in [Path(__file__),R5_PATH,Path(R5.R4.__file__),
                    R5_PATH.parent.parent/'config.py',ROOT/'src/corelib/textmatch.py']},
        processing='Delete accepted spans only; preserve other characters including whitespace; no sentinel; no truncation; labels unchanged.',
        adaptation='Reuse r4 portion patterns and r5 position/tail and exact banner logic. Add SBU/REL tails and cable admin fields. Apply stricter acronym/position and whole-banner guards. Skip OCR/layout/fuzzy rules.',
        audit_usage='audit.jsonl contains raw marking strings and grades; evaluator-only, never model input. Mark anchors are candidates, not protected-span boundaries.',
        deduplication='unique.jsonl retains v1 unique membership; deduplicated.jsonl additionally keeps one per masked text.strip() group and excludes label-conflicting groups. All groups reported.',
        outputs={p.name:dict(bytes=p.stat().st_size,sha256=sha(p.read_bytes())) for p in sorted(args.out.iterdir())})
    (args.out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ['stats','label_counts','rule_counts','remaining_rule_candidates','masked_duplicate_groups','masked_conflicting_groups','regression_cases_passed']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
