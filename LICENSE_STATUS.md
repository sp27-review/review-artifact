# License and source-status statement

This review artifact follows a conservative link-only policy. Including a
document identifier, URL, checksum, span coordinate, label or model prediction
does not assert that the underlying source text is freely redistributable.

| Source | Distributed here | Not distributed |
|---|---|---|
| CIA Reading Room documents | Identifiers, catalog and acquisition URLs, acquired-OCR hash, labels, text hashes and lengths, protected-span coordinates and hashes, predictions | OCR output, original text, marker-removed text, span text |
| WikiLeaks cable collection (public Kaggle dataset) | Cable identifiers, collection URL, labels, text hashes and lengths, edit operations, predictions | Cable text in any form |
| FRUS collection (public Kaggle dataset) | Document identifiers, collection URL, labels, text hashes and lengths, edit operations, predictions | Document text in any form |

Edit operations are text-free by construction: deletions, whitespace or
punctuation literals, deterministic transforms of the replaced span, fixed
alias strings defined in `code/tools/edit_ops.py`, or copies of another span of
the same document given by coordinates. Short original portion-marking strings
(for example `(S/NF)`, at most 40 characters) are kept in the PORTIONLINK file
solely as label provenance.

Evidence quotations, rationales, raw model responses, titles and any other
text-bearing fields were removed from all prediction files.

No blanket copyright or data license is asserted by this artifact. Code written
by the authors will be released under an open-source license with the final,
non-anonymous version.
