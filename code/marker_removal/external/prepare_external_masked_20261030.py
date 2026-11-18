#!/usr/bin/env python3
"""Auditable marking removal for WikiLeaks and FRUS, without model execution.

The frozen WikiLeaks builder is reused for its established rules. Extensions
operate on original character offsets. A separate, broader residual scanner
retains benign lexical matches and holds unresolved documents out of evaluation.
No whitespace normalization, truncation, label-dependent rule, or OCR fuzzy pass.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import re
import time

import prepare_wikileaks_masked_20261030 as BASE

ROOT = Path(__file__).resolve().parents[1]
OVERLAP = ROOT / 'audits/overlap_20261030/candidates.jsonl'
LABELS = {'U': 0, 'C': 1, 'S': 2, 'TS': 3}
GRADE = r'TOP\s+SECRET|LIMITED\s+OFFICIAL\s+USE|CONFIDENTIAL|UNCLASSIFIED|SECRET|SBU|FOUO|LOU|TS|[USC]'
MARK = re.compile(r'\(\s*(?P<grade>' + GRADE + r')(?=[\s)/—–-])(?P<tail>[^()]{0,180})\)', re.I)
ENUM = re.compile(r'(?:(?:[¶]\s*)?\d{1,3}[.,)]{0,2}|\(\d{1,3}\)|[A-Za-z][.)]?|[IVXivx]{1,5}[.)])')
FULL_NOTICE = re.compile(r'^[\s—–.,:;-]*(?:(?:entire|full)\s+(?:text|message|contents)|all\s+paras?)[\s.]*$', re.I)
CODE = re.compile(r'^(?:NF|NOFORN|FOUO|LES|NC|NOCONTRACT|OC|ORCON|RD|FRD|N|'
                  r'SI|TK|HCS|NFD|CD|WNINTEL|NODIS|EXDIS|LIMDIS|SPECAT|SPOKE|'
                  r'NO\s+FOREIGN\s+DISSEM|CONTROLLED\s+DISSEM|NO\s+CONTRACT|'
                  r'S|U|O|R|L|OWL|SEN|SENSITIVE|LD|EXCL)$', re.I)
SELF_NOTICE = re.compile(
    r'\b(?:(?:this|the\s+entire|entire)\s+(?:cable|message|document|report|telegram|text|information)'
    r'\s+(?:is|contains)|(?:the\s+)?entire\s+text\s+of\s+this\s+(?:cable|report)\s+is)'
    r'\s+(?:(?:business|company)\s+proprietary\s+information\s+and\s+is\s+)?'
    r'(?:sensitive\s*,?\s*but\s+unclassif(?:ied|ed)|unclassified\s+but\s+sensitive|SBU)\b(?:\s*\(SBU\))?(?:\s+information)?', re.I)
SBU_START = re.compile(r'(?im)^[ \t]*(?:(?:\d+[.)]|[A-Z][.)])[ \t]*)?'
                       r'(?:\((?:U|SBU)\)[ \t]*)?(?P<notice>SENSITIVE\s*,?\s*BUT\s+UNCLASSIF(?:IED|ED))\b')
IT_NOTICE = re.compile(r'\bIt\s+is\s+sensitive\s+but\s+unclassified\b', re.I)
NOTICE_CONT = re.compile(
    r'^[\s.,;:—–-]*(?:AND\s+)?(?:'
    r'(?:PLEASE\s+)?(?:PROTECT|HANDLE|TREAT)\s+ACCORDINGLY|'
    r'NOT\s+(?:FOR|INTENDED\s+FOR)\s+(?:(?:PUBLIC\s+)?INTERNET\s+(?:DISTRIBUTION|DISSEMINATION)|PUBLICATION)|'
    r'NOT\s+FOR\s+(?:RELEASE|DISTRIBUTION|DISSEMINATION|TRANSMISSION)\s+'
    r'(?:ON\s+THE\s+INTERNET|OUTSIDE\s+(?:OF\s+)?(?:U\.?S\.?\s*GOVERNMENT|USG|U\.?S\.?G\.?)\s*(?:CHANNELS)?)|'
    r'(?:IT\s+)?SHOULD\s+NOT\s+BE\s+(?:DISSEMINATED|RELEASED)\s+OUTSIDE\s+(?:OF\s+)?'
    r'(?:U\.?S\.?\s*GOVERNMENT|USG)\s*(?:CHANNELS)?'
    r'(?:\s+OR\s+IN\s+ANY\s+PUBLIC\s+FORUM)?'
    r'(?:\s+WITHOUT\s+THE\s+(?:WRITTEN\s+)?(?:CONSENT|PERMISSION|APPROVAL)\s+OF\s+'
    r'(?:THE\s+)?(?:ORIGINATOR|ORIGINATING\s+OFFICE))?|'
    r'FOR\s+OFFICIAL\s+USE\s+ONLY|ENTIRE\s+TEXT)(?:[.!;]|(?=\s|$))', re.I)
REASON = re.compile(r'\bREASONS?\s*:\s*1\.[45]\s*\([a-h]\)'
                    r'(?:[ \t,]*(?:AND[ \t]+)?(?:1\.[45]\s*)?\([a-h]\))*', re.I)
ADMIN_LINE = re.compile(r'^(?:CLASSIFIED\s+BY\s*:|E\.?\s*O\.?\s*(?:12958|13526)\b|'
                        r'(?:DECLASSIFY\s+ON|DECLASSIFICATION\s+DATE)\s*:)', re.I)
BANNER = re.compile(r'^(?:TOP[ \t]+SECRET|SECRET|CONFIDENTIAL|UNCLASSIFIED|UNCLAS|'
                    r'S E C R E T|C O N F I D E N T I A L|U N C L A S)'
                    r'(?:[ \t]*(?:[/;—–-]|//)[ \t]*(?:NOFORN|NF|NODIS|EXDIS|LIMDIS|CODE[ \t]+WORD))*[ \t.]*$', re.I)
HANDLING = re.compile(r'^(?:NOFORN|NODIS|EXDIS|LIMDIS|SIPDIS|FOR OFFICIAL USE ONLY)[ \t.]*$', re.I)
ACRONYM = re.compile(r'(?:technical\s+secretariat|(?:ukrainian\s+)?security\s+services?|'
                     r'security\s+service\s+of\s+ukraine|executive\s+secretariat|'
                     r'under\s+secretary|secretary\s+of\s+state)\s*$', re.I)
AGENCY_TAIL = re.compile(r'^/(?:MS|P|CPR|AS|AL|NM|PC|PM|CT|DEL|GWI|CRS|WCI|'
                         r'SRGIA|ES|FSNI|CIEA|SEMEP|S(?:\s+|\s*:).*)\b', re.I)
LEGACY_AUDIT_PATH = ROOT/'experiments/dataset_construction/audits/masking_pipeline/audit_marking_residual.py'
_spec = importlib.util.spec_from_file_location('external_legacy_residual_audit', LEGACY_AUDIT_PATH)
LEGACY_AUDIT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(LEGACY_AUDIT)
UNBRACKETED_NOTICE = re.compile(r'(?im)^[ \t]*(?:\d+[.)][ \t]*)?'
    r'(?P<notice>(?:TOP[ \t]+SECRET|CONFIDENTIAL|SECRET|TS|S|C|U)'
    r'(?:[ \t]*/[ \t]*(?:NOFORN|NF|NODIS|EXDIS|NC))*[ \t—–-]+'
    r'(?:ENTIRE|FULL)[ \t]+(?:TEXT|MESSAGE|CONTENTS)[.!]?)')
REVERSE_NOTICE = re.compile(r'\b(?:Entire|Full)\s+(?:text|message)\s+(?:is\s+)?'
    r'(?:Top\s+Secret|Secret|Confidential|Unclassified)\b[.]?',re.I)
INLINE_ENTIRE = re.compile(r'\b(?:Top\s+Secret|Secret|Confidential|Unclassified)'
    r'(?:\s*/\s*(?:Sensitive|NOFORN|Nodis))*[\s./—–-]+(?:Entire|Full)\s+(?:text|message)\b[.]?',re.I)
GRADE_CLAUSE = re.compile(r'[.;)]\s*(?P<grade>Top\s+Secret|Secret|Confidential|Unclassified)(?=[ \t]*[;./])')
DECISIONS_PATH = ROOT/'scripts/external_marking_review_20261030.json'
BROAD_MARKING = re.compile(r'\bsensitive\s*,?\s*but\s+unclassif(?:ied|ed)\b|\bunclassified\s+but\s+sensitive\b|'
    r'\bnot\s+(?:for|intended\s+for)\s+(?:internet|public|distribution|dissemination)|'
    r'\bshould\s+not\s+be\s+disseminated\b|\bclassified\s+by\b|'
    r'\breport\s+class\s+(?:secret|confidential|top\s+secret)\b|\bnot\s+releasable\b|'
    r'\b(?:do\s+not\s+release|for\s+official\s+use\s+only)\b',re.I)
HANDLING_SENTENCE = re.compile(r'\b(?:AND\s+)?(?:IT\s+IS\s+)?NOT(?:/NOT)?\s+(?:INTENDED\s+)?FOR\s+'
    r'(?:INTERNET\s+(?:DISTRIBUTION|DISSEMINATION|PUBLICATION)|'
    r'(?:DISTRIBUTION|DISSEMINATION|RELEASE|TRANSMISSION)\s+OUTSIDE\s+(?:OF\s+)?(?:THE\s+)?'
    r'(?:UNITED\s+STATES\s+GOVERNMENT|U\.?S\.?\s*GOVERNMENT|USG)(?:\s+CHANNELS)?)'
    r'[.]?(?:[\s—–-]*PLEASE\s+PROTECT[.]?)?|\bDO\s+NOT\s+RELEASE\s+TO\s+PUBLIC\s+INTERNET[.]?',re.I)
NUMBERED_CLASSIFIED = re.compile(r'(?im)^[ \t]*\d+[.)][ \t]*(?P<field>CLASSIFIED\s+BY\s*:[^\r\n]*)')
TREAT_NOTICE = re.compile(r'PLEASE\s+TREAT\s+AS\s+SENSITIVE\s*,?\s*BUT\s+UNCLASSIFIED\s+UNTIL\s+THE\s+REPORT\s+IS\s+FINALIZED\s+FOR\s+SUBMISSION\s+TO\s+CONGRESS[.]?',re.I)
PROPRIETARY_NOTICE = re.compile(r'\b(?:This\s+(?:message|cable|report)\s+contains\s+(?:company|business)\s+proprietary\s+information|CONTAINS\s+BUSINESS\s+SENSITIVE\s+INFORMATION)[.]?',re.I)


def read(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def compact(s):
    return re.sub(r'\s+', ' ', s).strip()


def merge(edits):
    result = []
    for e in sorted(edits, key=lambda x: (x['start'], x['end'])):
        if result and e['start'] < result[-1]['end']:
            result[-1]['end'] = max(result[-1]['end'], e['end'])
            result[-1]['rules'] = sorted(set(result[-1]['rules'] + e['rules']))
        else:
            result.append(dict(start=e['start'], end=e['end'], rules=list(e['rules'])))
    return result


def code_tail(tail):
    t = compact(tail).strip(' .')
    if not t:
        return True
    if FULL_NOTICE.fullmatch(t):
        return True
    t = re.sub(r'[—–-]?\s*(?:entire|full)\s+(?:text|message|contents)\b.*$', '', t, flags=re.I)
    t = re.sub(r'[—–-]?\s*all\s+paras?\b.*$', '', t, flags=re.I)
    t = t.strip(' /—–-.')
    if not t:
        return True
    if re.fullmatch(r'REL(?:\s+TO)?\s+[A-Z., /-]+', t):
        return True
    if re.fullmatch(r'(?:R\s*)?\[\s*(?:(?:handling restriction|distribution)|less than 1 line)\s+not\s+declassified\s*\]', t, re.I):
        return True
    return all(CODE.fullmatch(x.strip()) for x in re.split(r'\s*(?:/|—|–|-)\s*', t) if x) and bool(t)


def line_context(text, a, b):
    ls = text.rfind('\n', 0, a) + 1
    le = text.find('\n', b)
    if le < 0:
        le = len(text)
    return ls, le, text[ls:a].strip(), text[b:le].strip()


def list_letters(text):
    return {m.group(1).lower() for m in re.finditer(r'(?m)^[ \t]*(?:\d+[.)][ \t]*)?\(([a-z])\)[ \t]+', text, re.I)}


def alphabetic_list_at(text, m):
    # Require neighboring alphabetic items, not just other letters somewhere
    # in a document. This distinguishes a true (C) portion in a mixed document.
    if not re.fullmatch(r'\([A-Za-z]\)', m.group()):
        return False
    start=max(0,m.start()-12000)
    items=list(re.finditer(r'(?<![A-Za-z])\(([A-Za-z])\)(?![A-Za-z])',text[start:m.end()+12000]))
    k=next((i for i,x in enumerate(items) if x.start()+start==m.start()),None)
    if k is None:
        return False
    value=ord(m['grade'].lower())
    prev=ord(items[k-1][1].lower()) if k else None
    nxt=ord(items[k+1][1].lower()) if k+1<len(items) else None
    prev2=ord(items[k-2][1].lower()) if k>1 else None
    return (prev==value-1 and nxt==value+1) or (prev==value-1 and prev2==value-2)


def portion_position(text, m, dataset, letters):
    ls, le, before, after = line_context(text, m.start(), m.end())
    g = m['grade']
    tail = compact(m['tail'])
    context = compact(text[max(0, m.start() - 120):m.start()])
    if ACRONYM.search(context):
        return False
    if alphabetic_list_at(text,m):
        return False
    if re.search(r'(?:sections?|paragraphs?|questions?|reference|ref)\s*(?:\d+\s*)?$', context, re.I):
        return False
    if m.start() and text[m.start()-1].isalpha() and not before.endswith(':'):
        return False
    prefix = MARK.sub('', before).strip()
    numbered = bool(ENUM.fullmatch(prefix))
    # A sequence (a), (b), ..., (u) is a list, even if an isolated (u) looks like a grade.
    if g.islower() and len(g) == 1:
        return bool(numbered and re.match(r'\s+[A-Z]', text[m.end():])) or bool(
            re.fullmatch(r'(?:SUBJECT|REFS?)\s*:', prefix, re.I) and g.lower() == 'u')
    if dataset == 'wikileaks_cable_classifier':
        if BASE.valid_position(text, m.start(), m.end()):
            return True
        return numbered or bool(re.fullmatch(r'(?:SUBJECT|SUMMARY|COMMENT)\s*:', prefix, re.I))
    if re.search(r'\b(?:SUBJECT|SUBJ|SUMMARY|COMMENT)\s*:\s*$',prefix,re.I):
        return True
    if re.search(r'\b(?:SUBJECT|SUBJ)\s*:',prefix,re.I):
        return True
    if re.fullmatch(r'[—–•-]+',prefix) or re.search(r'\bSUMMARY\s*$',prefix,re.I):
        return True
    if re.match(r'[—–-]\s*(?:ACTION|INFORMATION)\s+MEMORANDUM',after,re.I):
        return True
    if numbered or not prefix:
        return True
    # FRUS places classification anchors at paragraph ends, including wrapped text.
    paragraph_end = bool(re.match(r'[ \t\r.,;:—–-]*(?:\n[ \t\r]*\n|$)',text[m.end():]))
    sentence_end = bool(re.search(r'[.!?][”"’\]]?\s*$', before))
    if paragraph_end or sentence_end:
        return True
    return bool(tail and code_tail(tail)) or (len(g) > 2 and g.upper() != 'SBU')


def notice_end(text, end):
    while (m := NOTICE_CONT.match(text[end:])):
        end += m.end()
    p = re.match(r'[.!;]', text[end:])
    return end + (p.end() if p else 0)


def isolated_line(text, start, end):
    # Check actual adjacent physical lines, preserving blank lines with spaces.
    blank_before = start == 0 or bool(re.search(r'\n[ \t\r]*\n$', text[:start]))
    blank_after = not text[end:].strip() or bool(re.match(r'\r?\n[ \t\r]*\n', text[end:]))
    return blank_before and blank_after


def detect(text, dataset):
    edits, marks = [], []
    if dataset == 'wikileaks_cable_classifier':
        edits, marks, _ = BASE.detect(text)
    letters = list_letters(text)
    def add(a, b, rule):
        edits.append(dict(start=a, end=b, rules=[rule]))
    anchors = {(x['start'], x['end']) for x in marks}
    for m in MARK.finditer(text):
        fouo_request=m['grade'].upper()=='FOUO' and compact(m['tail']).lower()=='please'
        if not fouo_request and (not code_tail(m['tail']) or not portion_position(text, m, dataset, letters)):
            continue
        end = m.end()
        suffix = re.match(r'[ \t]*(?://|/)[ \t]*(?:NF|NOFORN|FOUO|LES)\b', text[end:], re.I)
        if suffix:
            end += suffix.end()
        entire = re.match(r'[ \t—–-]*(?:entire|full)[ \t]+(?:text|message|contents)[.!]?', text[end:], re.I)
        if entire:
            end += entire.end()
        rule = 'external_portion_context'
        add(m.start(), end, rule)
        if not any(a <= m.start() and b >= m.end() for a, b in anchors):
            marks.append(dict(start=m.start(), end=end, grade=compact(m['grade']).upper(),
                              mark=text[m.start():end], rule=rule))
            anchors.add((m.start(), end))
    for m in SELF_NOTICE.finditer(text):
        add(m.start(), notice_end(text, m.end()), 'sbu_self_declaration_extended')
    for m in SBU_START.finditer(text):
        add(m.start('notice'), notice_end(text, m.end()), 'sbu_standalone_notice_extended')
    for m in UNBRACKETED_NOTICE.finditer(text):
        add(m.start('notice'), m.end('notice'), 'unbracketed_entire_text_notice')
    for pattern in [REVERSE_NOTICE,INLINE_ENTIRE]:
        for m in pattern.finditer(text):
            add(m.start(),m.end(),'entire_text_declaration')
    for m in GRADE_CLAUSE.finditer(text):
        end=m.end()
        tail=re.match(r'[; /]*(?:(?:Sensitive|Eyes Only|Code word)[;. /]*)+',text[end:],re.I)
        if tail:
            end+=tail.end()
        add(m.start('grade'),end,'standalone_grade_clause')
    for m in HANDLING_SENTENCE.finditer(text):
        # These are exact distribution instructions; ordinary references to a
        # confidential conversation or a study "not for public" do not match.
        add(m.start(),m.end(),'explicit_distribution_instruction')
    for m in NUMBERED_CLASSIFIED.finditer(text):
        add(m.start('field'),m.end('field'),'numbered_classification_field')
    for m in TREAT_NOTICE.finditer(text):
        add(m.start(),m.end(),'explicit_report_handling_instruction')
    for m in PROPRIETARY_NOTICE.finditer(text):
        # Exact self-description only, not substantive business information.
        add(m.start(),m.end(),'proprietary_self_declaration')
    if dataset == 'wikileaks_cable_classifier':
        for m in IT_NOTICE.finditer(text):
            if re.search(r'\bthis\s+(?:cable|report|message)\b', text[max(0, m.start()-500):m.start()], re.I):
                add(m.start(), notice_end(text, m.end()), 'sbu_pronoun_self_declaration')
    for a, b, line in BASE.line_positions(text):
        stripped = line.strip()
        if BANNER.fullmatch(stripped) and isolated_line(text, a, b):
            add(a, b, 'exact_grade_banner')
        elif HANDLING.fullmatch(stripped) and isolated_line(text, a, b):
            add(a, b, 'exact_handling_banner')
        # Remove just the reason field; preserve the institution text before it.
        for m in REASON.finditer(line):
            add(a+m.start(), a+m.end(), 'classification_reason_inline')
    # Complete an administrative field whose reason-code list wraps to the
    # next physical line. Never consume arbitrary following body paragraphs.
    for e in list(edits):
        if 'wikileaks_classification_header' in e['rules']:
            m=re.match(r'\r?\n[ \t]*\([a-hA-H]\)(?:[ \t,]*(?:and[ \t]+)?\([a-hA-H]\))*[.]?[ \t]*(?=\r?\n|$)',text[e['end']:])
            if m:
                add(e['end'],e['end']+m.end(),'classification_reason_wrapped')
    return merge(edits), sorted(marks, key=lambda x: x['start'])


def benign_mark_reason(text, m, letters):
    g, tail = m['grade'], compact(m['tail'])
    ls, le, before, after = line_context(text, m.start(), m.end())
    prior = compact(text[max(0, m.start()-140):m.start()])
    if alphabetic_list_at(text,m):
        return 'alphabetic_list_sequence'
    if len(g) == 1 and g.islower():
        if m.start() and text[m.start()-1].isalnum():
            return 'word_suffix_or_legal_subsection'
        if not ENUM.fullmatch(before):
            return 'lowercase_list_or_prose_not_portion_position'
    if ACRONYM.search(prior):
        return 'expanded_organization_acronym'
    if re.match(r'^(?:[-—–]\d|\s*\d|/\d)', tail):
        return 'equipment_legislation_or_reference_number'
    if AGENCY_TAIL.match(re.sub(r'/\s+','/',tail)):
        return 'agency_office_abbreviation'
    if len(g) == 1 and m.start() and text[m.start()-1].isalnum():
        return 'word_suffix_or_legal_subsection'
    if re.search(r'(?:\([a-z0-9]+\))\s*$',prior,re.I) and re.match(r'\([a-z0-9]+\)',text[m.end():],re.I):
        return 'legal_subsection_chain'
    if re.search(r'\b(?:Questions?|Sections?|Paragraphs?)\b[^.]{0,90}\d+\s*$',prior,re.I):
        return 'legal_list_or_document_reference'
    if re.search(r'\bHaneflex\s*$',prior,re.I) and re.match(r'\s*Pte\.',text[m.end():],re.I):
        return 'company_name_singapore_abbreviation'
    if re.search(r"can.t\s+say\s*$",prior,re.I) and compact(m.group()).upper()=='(C/S)':
        return 'survey_answer_abbreviation'
    if g.lower()=='unclassified' and re.search(r'(?:email|e-mail|@)[^\n]*$',prior,re.I):
        return 'email_network_description'
    if re.search(r'(?:sections?|paragraphs?|questions?|accomplishment|reference|ref)\s*(?:\d+\s*)?$', prior, re.I):
        return 'legal_list_or_document_reference'
    if re.match(r'^(?:REFS?|REFERENCES)\b', before, re.I):
        return 'reference_list_label'
    if tail and not code_tail(tail):
        return 'nonmark_tail_review_candidate'
    return None


def residual_scan(text):
    """Broad scanner independent of detect's edit list and acceptance decisions."""
    candidates = []
    letters = list_letters(text)
    for m in MARK.finditer(text):
        reason = benign_mark_reason(text, m, letters)
        status = 'retained_nonmark' if reason and reason != 'nonmark_tail_review_candidate' else 'review'
        candidates.append(dict(start=m.start(), end=m.end(), text=m.group(), kind='marklike',
                               status=status, reason=reason,
                               context=text[max(0,m.start()-100):m.end()+100]))
    # The scanner deliberately checks all self-declarations, including variants
    # excluded by the remover's pronoun-context guard.
    for kind, pattern in [('sbu_self_declaration', SELF_NOTICE), ('sbu_pronoun', IT_NOTICE),
                          ('sbu_standalone', SBU_START), ('classification_reason', REASON), ('entire_text_notice',UNBRACKETED_NOTICE)]:
        for m in pattern.finditer(text):
            candidates.append(dict(start=m.start(), end=m.end(), text=m.group(), kind=kind,
                                   status='review', reason=None, context=text[max(0,m.start()-100):m.end()+100]))
    for kind,pattern in [('reverse_entire_text',REVERSE_NOTICE),('inline_entire_text',INLINE_ENTIRE),('grade_clause',GRADE_CLAUSE),('proprietary_self_declaration',PROPRIETARY_NOTICE)]:
        for m in pattern.finditer(text):
            candidates.append(dict(start=m.start(),end=m.end(),text=m.group(),kind=kind,status='review',reason=None,
                                   context=text[max(0,m.start()-100):m.end()+100]))
    for a, b, line in BASE.line_positions(text):
        if ADMIN_LINE.match(line.strip()) or ((BANNER.fullmatch(line.strip()) or HANDLING.fullmatch(line.strip())) and isolated_line(text,a,b)):
            candidates.append(dict(start=a, end=b, text=line, kind='admin_or_banner',
                                   status='review', reason=None, context=text[max(0,a-100):b+100]))
    for kind, pattern in LEGACY_AUDIT.MARKING_CHECKS.items():
        if kind == 'portion_mark':
            continue  # MARK additionally includes whitespace, case and long tails.
        for m in pattern.finditer(text):
            candidates.append(dict(start=m.start(), end=m.end(), text=m.group(), kind='legacy:'+kind,
                                   status='review', reason=None, context=text[max(0,m.start()-100):m.end()+100]))
    for m in BROAD_MARKING.finditer(text):
        candidates.append(dict(start=m.start(),end=m.end(),text=m.group(),kind='broad:explicit_notice',status='review',reason=None,
                               context=text[max(0,m.start()-100):m.end()+100]))
    return sorted(candidates, key=lambda x: x['start'])


def source_membership(dataset, rows):
    exclusions = []
    if dataset == 'wikileaks_cable_classifier':
        ids = {r['doc_id'] for r in read(ROOT/'data/datasets'/dataset/'v1/unique.jsonl')}
        exclusions = [dict(doc_id=r['doc_id'], stage='source', reason='v1_exact_duplicate') for r in rows if r['doc_id'] not in ids]
        return ids, exclusions
    overlap = defaultdict(list)
    for r in read(OVERLAP):
        if r['external']['dataset'] == 'frus_sachn' and r['cosine'] >= .15:
            overlap[r['external']['id']].append(dict(cia_id=r['cia']['id'], cosine=r['cosine'], split=r['cia']['split']))
    groups = defaultdict(list)
    for r in rows:
        if r['doc_id'] in overlap:
            exclusions.append(dict(doc_id=r['doc_id'], stage='source', reason='cia_overlap_candidate_ge_0_15', evidence=overlap[r['doc_id']]))
        else:
            groups[BASE.sha(r['text'].strip().encode())].append(r)
    ids = set()
    for rs in groups.values():
        conflict = len({r['label'] for r in rs}) > 1
        if not conflict:
            ids.add(rs[0]['doc_id'])
        for r in rs if conflict else rs[1:]:
            exclusions.append(dict(doc_id=r['doc_id'], stage='source', reason='source_label_conflict' if conflict else 'source_exact_duplicate', group=[x['doc_id'] for x in rs]))
    assert len(ids) == 21864, len(ids)
    return ids, exclusions


def regression():
    assert BASE.self_test() == 19
    cases = [
        ('wikileaks_cable_classifier', '1. (U) This message is Sensitive but Unclassified. Please\nhandle accordingly.\n\n2. Facts remain.', '1.  \n\n2. Facts remain.'),
        ('wikileaks_cable_classifier', 'k  (u) Human rights remain.', 'k   Human rights remain.'),
        ('wikileaks_cable_classifier', '(a) Inventory.\n(b) Guidance.\n(c) Consular tables.\n(s) Files.\n(u) Fraud programs.', '(a) Inventory.\n(b) Guidance.\n(c) Consular tables.\n(s) Files.\n(u) Fraud programs.'),
        ('wikileaks_cable_classifier', 'Ref: (u) Ref A UAE letter.', 'Ref:  Ref A UAE letter.'),
        ('wikileaks_cable_classifier', 'Department; REASON: 1.4(B), (C), (D)\n\n1. Facts.', 'Department; \n\n1. Facts.'),
        ('wikileaks_cable_classifier', 'Security Service of Ukraine (SBU)', 'Security Service of Ukraine (SBU)'),
        ('wikileaks_cable_classifier', 'Technical Secretariat\n(TS) to discuss facts.', 'Technical Secretariat\n(TS) to discuss facts.'),
        ('frus_sachn', 'Title\r\n\r\n1. ( TS ) Facts\r\n    remain. (C)\r\n\r\nMore facts.', 'Title\r\n\r\n1.  Facts\r\n    remain. \r\n\r\nMore facts.'),
        ('frus_sachn', '1. (S—Entire text.)\r\n\r\n2. Facts.', '1. \r\n\r\n2. Facts.'),
        ('frus_sachn', 'Facts. ( S/S )\r\n\r\nMore facts.', 'Facts. \r\n\r\nMore facts.'),
        ('frus_sachn', 'Executive Secretariat ( S/S ) sent a memo.', 'Executive Secretariat ( S/S ) sent a memo.'),
        ('frus_sachn', 'Robert Kelley (S/AS). The Secret Service attended.', 'Robert Kelley (S/AS). The Secret Service attended.'),
        ('frus_sachn', '(c) Think that the proposal is wrong.\r\n\r\nSection 1(c)(iii).', '(c) Think that the proposal is wrong.\r\n\r\nSection 1(c)(iii).'),
        ('frus_sachn', 'The U–2 (U–2 or SR–71) was discussed.', 'The U–2 (U–2 or SR–71) was discussed.'),
        ('frus_sachn', 'Facts. (S/[ handling restriction not declassified ])\n\nNext.', 'Facts. \n\nNext.'),
        ('frus_sachn', 'Facts. ( TS /S/OWL)\n\nNext.', 'Facts. \n\nNext.'),
        ('frus_sachn', '1. Entire text confidential.\r\n\r\n2. Facts.', '1. \r\n\r\n2. Facts.'),
        ('frus_sachn', 'Secret/Sensitive–Entire text.\n\nFacts.', '\n\nFacts.'),
        ('frus_sachn', '1. ( LOU ) Facts.\n\nMore.', '1.  Facts.\n\nMore.'),
        ('frus_sachn', 'Body. (FOUO)\n\nMore.', 'Body. \n\nMore.'),
        ('frus_sachn', 'The secret police discussed a top secret meeting.', 'The secret police discussed a top secret meeting.'),
        ('frus_sachn', 'Archive Box 1. Top Secret; Code word. More facts.', 'Archive Box 1. More facts.'),
        ('frus_sachn', 'We agreed to keep the talks\r\nsecret.\r\n\r\nMore facts.', 'We agreed to keep the talks\r\nsecret.\r\n\r\nMore facts.'),
        ('wikileaks_cable_classifier', "Linda (Lou's Police) sent a form.", "Linda (Lou's Police) sent a form."),
        ('frus_sachn', '(LOUD APPLAUSE) greeted the speaker.', '(LOUD APPLAUSE) greeted the speaker.'),
        ('wikileaks_cable_classifier', 'Sources in the Department (FOUO\nplease) told us facts.', 'Sources in the Department  told us facts.'),
        ('wikileaks_cable_classifier', 'This message is Sensitive, but Unclassified.\n\nFacts.', '\n\nFacts.'),
        ('wikileaks_cable_classifier', 'SENSITIVE BUT UNCLASSIFED.\n\nFacts.', '\n\nFacts.'),
        ('wikileaks_cable_classifier', 'Not for dissemination outside the USG.\n\nFacts.', '\n\nFacts.'),
        ('wikileaks_cable_classifier', '1. Classified By: An official.\n\n2. Facts.', '1. \n\n2. Facts.'),
        ('wikileaks_cable_classifier', 'This message contains company proprietary information.\n\nFacts.', '\n\nFacts.'),
    ]
    for ds, text, expected in cases:
        edits, _ = detect(text, ds)
        actual = BASE.apply_edits(text, edits)
        assert actual == expected, (ds, text, expected, actual)
    return 19 + len(cases)


def build(dataset, out, limit=None):
    t0 = time.monotonic()
    source = ROOT/'data/datasets'/dataset/'v1'/('all.jsonl' if dataset == 'wikileaks_cable_classifier' else 'explicit_labels.jsonl')
    source_hash = digest(source)
    codefiles=[Path(__file__),Path(BASE.__file__),BASE.R5_PATH,Path(BASE.R5.R4.__file__),ROOT/'src/corelib/textmatch.py',BASE.R5_PATH.parent.parent/'config.py',LEGACY_AUDIT_PATH]
    code_hashes={str(p.relative_to(ROOT)):digest(p) for p in codefiles}
    review_hash=digest(DECISIONS_PATH) if DECISIONS_PATH.exists() else None
    rows = list(read(source))
    members, exclusions = source_membership(dataset, rows)
    if limit:
        rows = rows[:limit]
        members &= {r['doc_id'] for r in rows}
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise RuntimeError(f'Refusing nonempty output directory: {out}')
    files = {n: (out/n).open('w') for n in ['all.jsonl','unique.jsonl','audit.jsonl','residual_audit.jsonl']}
    stats, rule_counts, residual_counts = Counter(), Counter(), Counter()
    groups = defaultdict(list)
    hold = set()
    samples = defaultdict(list)
    rng = random.Random(7701)
    decisions=defaultdict(list)
    if DECISIONS_PATH.exists():
        for d in json.loads(DECISIONS_PATH.read_text())['decisions']:
            decisions[d['doc_id']].append(d)
    applied_decisions=Counter()
    for i, r in enumerate(rows):
        text = r['text']
        assert BASE.sha(text.encode()) == r['text_sha256']
        assert LABELS[r['label']] == r['label_id']
        edits, marks = detect(text, dataset)
        for d in decisions[r['doc_id']]:
            assert d['source_text_sha256']==r['text_sha256']
            a,b=d['start'],d['end']
            assert text[a:b]==d['text'],d
            if d['action']=='remove':
                edits.append(dict(start=a,end=b,rules=['reviewed_marking_span']))
                if (match:=MARK.fullmatch(text[a:b])) and not any(x['start']==a and x['end']==b for x in marks):
                    marks.append(dict(start=a,end=b,grade=compact(match['grade']).upper(),mark=text[a:b],rule='reviewed_marking_span'))
            else:
                assert not any(e['start']<b and e['end']>a for e in edits),('retention_conflict',r['doc_id'],d)
            applied_decisions[d['action']]+=1
        edits=merge(edits)
        masked = BASE.apply_edits(text, edits)
        assert masked.strip(), r['doc_id']
        assert len(text)-len(masked) == sum(e['end']-e['start'] for e in edits)
        residuals = residual_scan(masked)
        for d in decisions[r['doc_id']]:
            if d['action']=='retain':
                a=BASE.project(d['start'],edits);b=BASE.project(d['end'],edits)
                assert masked[a:b]==d['text']
                for c in residuals:
                    if c.get('start')==a and c.get('end')==b:
                        c.update(status='reviewed_nonmark',reason=d['reason'])
        repeated, _ = detect(masked, dataset)
        if repeated:
            residuals.append(dict(kind='not_idempotent', status='review', deletions=repeated))
        if any(c['status']=='review' for c in residuals):
            hold.add(r['doc_id'])
        row = {k:r[k] for k in ['doc_id','dataset','source_id','label','label_id','language','source_split']}
        row.update(text=masked, text_sha256=BASE.sha(masked.encode()), source_text_sha256=r['text_sha256'])
        payload = json.dumps(row, ensure_ascii=False)+'\n'
        files['all.jsonl'].write(payload)
        if r['doc_id'] in members:
            files['unique.jsonl'].write(payload)
            groups[BASE.sha(masked.strip().encode())].append((r['doc_id'], r['label']))
        for m in marks:
            m['masked_anchor'] = BASE.project(m['start'], edits)
            assert text[m['start']:m['end']] == m['mark']
        audit = dict(doc_id=r['doc_id'], source_text_sha256=r['text_sha256'], masked_text_sha256=row['text_sha256'],
                     deletions=[dict(**e, text=text[e['start']:e['end']]) for e in edits], portion_mark_candidates=marks)
        files['audit.jsonl'].write(json.dumps(audit, ensure_ascii=False)+'\n')
        if residuals:
            files['residual_audit.jsonl'].write(json.dumps(dict(doc_id=r['doc_id'], candidates=residuals),ensure_ascii=False)+'\n')
        stats.update(documents=1, changed_documents=bool(edits), removed_spans=len(edits), removed_characters=len(text)-len(masked),
                     original_characters=len(text), portion_documents=bool(marks), portion_marks=len(marks),
                     review_documents=any(c['status']=='review' for c in residuals), idempotence_failures=bool(repeated))
        residual_counts.update((c['status']+':'+c['kind']+':'+str(c.get('reason'))) for c in residuals)
        for e in edits:
            for rule in e['rules']:
                rule_counts[rule]+=1
                item=dict(doc_id=r['doc_id'],removed=text[e['start']:e['end']],before=text[max(0,e['start']-100):e['start']],after=text[e['end']:e['end']+100])
                bucket=samples[rule]
                if len(bucket)<16:
                    bucket.append(item)
                else:
                    k=rng.randrange(rule_counts[rule])
                    if k<16:bucket[k]=item
        if (i+1)%2000==0:
            print(dataset, i+1, dict(stats), 'seconds',round(time.monotonic()-t0,1),flush=True)
    for f in files.values():
        f.close()
    keep = set()
    for h, rs in groups.items():
        conflict = len({r[1] for r in rs})>1
        if not conflict:
            keep.add(rs[0][0])
        for did, _ in rs if conflict else rs[1:]:
            exclusions.append(dict(doc_id=did,stage='masked',reason='masked_label_conflict' if conflict else 'masked_exact_duplicate',group=[x[0] for x in rs],masked_text_sha256=h))
    label_counts = {'deduplicated': Counter(), 'evaluation': Counter()}
    with (out/'deduplicated.jsonl').open('w') as dedup, (out/'evaluation.jsonl').open('w') as evaluation:
        for r in read(out/'unique.jsonl'):
            if r['doc_id'] not in keep:
                continue
            payload=json.dumps(r,ensure_ascii=False)+'\n'
            dedup.write(payload)
            label_counts['deduplicated'][r['label']]+=1
            if r['doc_id'] in hold:
                exclusions.append(dict(doc_id=r['doc_id'],stage='residual',reason='unresolved_residual_candidate'))
            else:
                evaluation.write(payload)
                label_counts['evaluation'][r['label']]+=1
    assert digest(source)==source_hash
    dump(out/'exclusions.json',exclusions)
    dump(out/'review_samples.json',dict(samples))
    dump(out/'verification.json',dict(mechanical_checks='passed during build; independent streaming verification still required',
         documents=len(rows),idempotence_failures=stats['idempotence_failures'],regression_cases_passed=regression()))
    assert all(digest(ROOT/p)==h for p,h in code_hashes.items()),'Code changed during build'
    assert (digest(DECISIONS_PATH) if DECISIONS_PATH.exists() else None)==review_hash,'Review decisions changed during build'
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),dataset=dataset,
        source=str(source.relative_to(ROOT)),source_sha256=source_hash,source_members=len(members),
        stats=dict(stats),rule_counts=dict(rule_counts),residual_counts=dict(residual_counts),
        label_counts={k:dict(v) for k,v in label_counts.items()},exclusion_counts=dict(Counter(e['reason'] for e in exclusions)),
        input='evaluation.jsonl:text',gold='label_id',label_map=LABELS,
        policies=dict(no_normalization=True,no_truncation=True,no_model_execution=True,raw_preserved=True,
                      frus_overlap='All saved cosine >=0.15 candidates against CIA benchmark excluded; not a guarantee against undiscovered overlap.',
                      residual='Broad candidates with unresolved status held from evaluation. Retained nonmark matches are not counted as markings.',
                      portion='Candidate anchors only, not validated protected paragraph boundaries.',
                      audit='Evaluator-only; do not feed audit, labels, source metadata, or residual records to models.'),
        code_sha256=code_hashes,
        review_decisions_sha256=review_hash,
        applied_review_decisions=dict(applied_decisions),
        overlap_audit_sha256=digest(OVERLAP),elapsed_seconds=time.monotonic()-t0,
        outputs={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(out.iterdir()) if p.is_file()})
    dump(out/'manifest.json',manifest)
    print('COMPLETE',dataset,json.dumps({k:manifest[k] for k in ['stats','label_counts','exclusion_counts','elapsed_seconds']},ensure_ascii=False),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',choices=['wikileaks_cable_classifier','frus_sachn'],required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--limit',type=int)
    args=ap.parse_args()
    regression()
    build(args.dataset,args.out,args.limit)


if __name__=='__main__':
    main()
