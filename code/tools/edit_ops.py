"""Text-free encoding of format edits.

Every released edit is a list ``[start, end, kind, *payload]`` over the
restored baseline text of a document.  No payload carries document content:
edits are expressed as deletions, whitespace/punctuation literals,
deterministic transforms of the replaced span, fixed alias strings defined
here, or copies of another span of the same baseline text.

``apply_ops`` rebuilds the edited text; callers compare its SHA-256 with the
released ``text_sha256``.
"""
import datetime
import hashlib
import re
from pathlib import Path

ACRONYMS = set((Path(__file__).with_name('acronyms.txt')).read_text().split())
ONE_SET = {'CIA', 'FBI', 'DIA', 'NSA', 'NPIC', 'DC', 'USIB', 'NSC', 'JCS'}
MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
          'August', 'September', 'October', 'November', 'December']
MONTH_RX = (r'(January|February|March|April|May|June|July|August|September|October|'
            r'November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?')
DATE_RX = re.compile(r'(?<!\w)(\d{1,2})\s+' + MONTH_RX + r',?\s+(\d{4})(?!\d)|(?<!\w)'
                     + MONTH_RX + r'\s+(\d{1,2}),?\s+(\d{4})(?!\d)', re.I)
ALIASES = {'Topic', 'Sender', 'Recipient', 'Date', 'Reference', 'References', 'Attention',
           'Enclosures', 'Copies', 'Source', 'Overview', 'Introduction', 'Contents',
           'Table of contents', 'Memo', 'Staff memo', 'Intelligence memo'}


def sha256(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def title_heading(value):
    """Title-case an all-capital heading; short words and listed acronyms stay."""
    return re.sub('[A-Z]+', lambda m: m.group() if len(m.group()) <= 3 or m.group() in ACRONYMS
                  else m.group().capitalize(), value)


def title_words(value):
    return re.sub(r'[A-Za-z]+', lambda m: m.group().capitalize() if m.group() not in ONE_SET
                  else m.group(), value)


def date_iso(value):
    def convert(m):
        if m.group(1):
            day, month, year = m.group(1), m.group(2), m.group(3)
        else:
            month, day, year = m.group(4), m.group(5), m.group(6)
        month = next(i + 1 for i, n in enumerate(MONTHS) if n.lower().startswith(month[:3].lower()))
        return datetime.date(int(year), month, int(day)).isoformat()
    try:
        return DATE_RX.sub(convert, value)
    except ValueError:
        return None


TRANSFORMS = {'title_heading': title_heading, 'title_words': title_words, 'date_iso': date_iso,
              'lower': str.lower, 'upper': str.upper, 'capitalize': str.capitalize}


def encode_edit(text, start, end, before, after):
    if after == '':
        return [start, end, 'del']
    if after.strip() == '':
        return [start, end, 'ws', after]
    for name, fn in TRANSFORMS.items():
        if fn(before) == after:
            return [start, end, 'fn', name]
    if not re.search(r'[A-Za-z0-9]', after):
        return [start, end, 'punct', after]
    core = after.strip()
    lead, trail = after[:len(after) - len(after.lstrip())], after[len(after.rstrip()):]
    if core in ALIASES:
        return [start, end, 'alias', after]
    k = text.find(core)
    if k >= 0:
        return [start, end, 'copy', lead, k, k + len(core), trail]
    return [start, end, 'opaque', len(after), sha256(after)]


def apply_ops(text, ops):
    """Return the edited text, or None if an op is opaque (not reconstructible)."""
    out, cur = [], 0
    for op in ops:
        start, end, kind = op[0], op[1], op[2]
        out.append(text[cur:start])
        before = text[start:end]
        if kind == 'del':
            pass
        elif kind in ('ws', 'punct', 'alias'):
            out.append(op[3])
        elif kind == 'fn':
            out.append(TRANSFORMS[op[3]](before))
        elif kind == 'copy':
            out.append(op[3] + text[op[4]:op[5]] + op[6])
        else:
            return None
        cur = end
    out.append(text[cur:])
    return ''.join(out)
