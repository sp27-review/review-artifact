"""Behavioral regressions for missing variants, semantic preservation and audit independence."""
import itertools
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from remover import apply, detect
from residual_scanner import scan


class SupplementTests(unittest.TestCase):
    def test_whole_document_variants(self):
        for grade, sep, scope in itertools.product(
            ["Confidential", "SECRET", "Top Secret", "(C)", "( TS )"],
            [" ", " — ", ": ", "\r\n    "],
            ["Entire text", "(Entire text)", "[FULL MESSAGE]"]):
            declaration = grade+sep+scope
            source = "1. "+declaration+"\r\n\r\n2. Payload unchanged.\r\n"
            with self.subTest(declaration=declaration):
                result = apply(source, detect(source))
                self.assertEqual(result, "1. \r\n\r\n2. Payload unchanged.\r\n")
                self.assertEqual(detect(result), [])

    def test_reverse_and_spaced_variants(self):
        for declaration in ["Entire text is Secret", "Full message: (C)", "C O N F I D E N T I A L (Entire text)"]:
            self.assertEqual(apply(declaration+"\n\nPayload.", detect(declaration+"\n\nPayload.")), "\n\nPayload.")

    def test_payload_and_quotations_retained(self):
        examples = [
            "The confidential talks concern the secret police.",
            "Technical Secretariat (TS) approved section (c).",
            "She quoted Confidential (Entire text) from another cable.",
            '1. "Confidential (Entire text)" was printed on that document.',
            "Quote:\nConfidential (Entire text)\n\nEnd quote.",
            "The entire text is secret according to the witness.",
            "No marking.\n\n1. (C) This is an enumerated alternative.",
            "The NODIS priority was discussed in the meeting.",
            "NODIS communications were discussed in the meeting.",
            "NOFORN is a distribution control.",
            "Confidential x0109 Entire text",
            "Confidential_Entire text",
            "12345. Subject: NODIS communications discussed yesterday.",
            "Quote. Top Secret Pretoria 4211. Subject: Negotiations.",
            "1. Confidential\n\nEntire text was missing from the archive.",
        ]
        for source in examples:
            with self.subTest(source=source):
                self.assertEqual(detect(source), [])
                self.assertEqual(apply(source, detect(source)), source)

    def test_routing_fields_preserve_content(self):
        source = "248372. Exdis ; USOECD; military\r\n addressees handle as Specat Exclusive. Subject: Meeting.\r\n\r\nFacts."
        self.assertEqual(apply(source, detect(source)), "248372.  ; USOECD; . Subject: Meeting.\r\n\r\nFacts.")
        source = "R 290824Z NOV 76\nFM AMEMBASSY MANILA\nTO SECSTATE WASHDC 3344\nNODIS\nE.O. 11652: GDS\nTAGS: MARR, RP\n\nPayload."
        self.assertEqual(apply(source, detect(source)), source.replace("NODIS", ""))
        source = "TELEGRAM Department of State MANILA 18586\r\n\r\nR 290824Z NOV 76\r\n\r\nFM AMEMBASSY MANILA TO SECSTATE WASHDC 3344\r\n\r\nNODIS\r\n E.O. 11652: GDS TAGS: MARR, RP\r\n\r\nSUBJECT: NEGOTIATIONS\r\n\r\nPayload."
        self.assertEqual(apply(source, detect(source)), source.replace("NODIS", ""))

    def test_scanner_does_not_depend_on_acceptance_grammar(self):
        source = "A confidential discussion of an E.O. 11652 instruction and (C)."
        self.assertEqual(detect(source), [])
        kinds = {x["kind"] for x in scan(source)}
        self.assertTrue({"grade_word", "administrative", "bracket_abbreviation"} <= kinds)
        self.assertTrue(all(x["status"] == "review" for x in scan(source)))

    def test_offsets_unicode_and_line_endings(self):
        source = "1. Confidential — (Entire text)\r\n\r\n café\t  x010a\r\n"
        edits = detect(source)
        self.assertEqual(edits[0]["text"], "Confidential — (Entire text)")
        self.assertEqual(source[edits[0]["start"]:edits[0]["end"]], edits[0]["text"])
        self.assertEqual(apply(source, edits), "1. \r\n\r\n café\t  x010a\r\n")

    def test_invalid_and_overlapping_edits_rejected(self):
        with self.assertRaises(ValueError):
            apply("abcd", [dict(start=1,end=3,text="bc"),dict(start=2,end=4,text="cd")])
        with self.assertRaises(ValueError):
            apply("abcd", [dict(start=1,end=3,text="WRONG")])

    def test_build_review_verification_and_no_overwrite(self):
        from verify import verify
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            texts=["1. Confidential (Entire text)\n\nProtected payload.", "NOFORN is a distribution control."]
            source=[]
            for i,text in enumerate(texts):
                source.append(dict(doc_id=str(i),label="C",label_id=1,text=text,text_sha256=hashlib.sha256(text.encode()).hexdigest()))
            src=folder/"source.jsonl"; g3=folder/"g3.jsonl"
            src.write_text("".join(json.dumps(x)+"\n" for x in source))
            g3.write_text("".join(json.dumps(dict(identifier=str(i),label="CONFIDENTIAL",marker_masked_text=t,text_source=str(src)))+"\n" for i,t in enumerate(texts)))
            original=src.read_bytes()
            cmd=[sys.executable,str(Path(__file__).with_name("build.py")),"--source",str(src),"--out",str(folder/"out"),"--g3-input",str(g3)]
            result=subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(verify(folder/"out")["stats"]["changed_documents"],1)
            queue=json.loads((folder/"out/review_queue.jsonl").read_text())
            self.assertEqual(queue["doc_id"],"1")
            decision=dict(doc_id="1",input_sha256=queue["input_sha256"],action="retain",reason="Substantive explanation of a handling code",**queue["candidates"][0]["input_span"])
            reviews=folder/"decisions.jsonl"; reviews.write_text(json.dumps(decision)+"\n")
            cmd_review=cmd.copy(); cmd_review[cmd_review.index("--out")+1]=str(folder/"reviewed")
            checked=subprocess.run(cmd_review+["--review-decisions",str(reviews)],capture_output=True,text=True)
            self.assertEqual(checked.returncode,0,checked.stderr)
            self.assertEqual(verify(folder/"reviewed")["review_documents"],0)
            self.assertEqual(src.read_bytes(),original)
            self.assertNotEqual(subprocess.run(cmd,capture_output=True).returncode,0)
            with (folder/"out/masked.jsonl").open("a") as f: f.write("{}\n")
            with self.assertRaises(ValueError): verify(folder/"out")


if __name__ == "__main__":
    unittest.main()
