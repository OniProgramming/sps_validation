import unittest
import xml.etree.ElementTree as ET

from sps_validation.docx_reader import Run
from sps_validation.ingest import _spans
from sps_validation.segment import _split_clauses, _variant_kind, _wh_text_reading


class SpsApparatusTest(unittest.TestCase):
    def test_span_types(self):
        runs = [
            Run("In ", False, False),
            Run("{be}", True, True),
            Run(" ", False, False),
            Run("reshit", True, False),
            Run(", and darkness* over the ", False, False),
            Run("tehom", True, False),
            Run(", and ", False, False),
            Run("[[a|the]]", True, True),
            Run(" ", False, False),
            Run("ruach", True, False),
            Run(" came {— *nakar* — the root: he knows them}.", False, False),
        ]
        text, spans = _spans(runs)
        got = [(s["type"], s["text"]) for s in spans]
        self.assertEqual(
            got,
            [
                ("source_form", "{be}"),
                ("translit", "reshit"),
                ("loss_mark", "*"),
                ("translit", "tehom"),
                ("alternatives", "[[a|the]]"),
                ("translit", "ruach"),
                ("note", "{— *nakar* — the root: he knows them}"),
            ],
        )
        self.assertEqual(spans[0]["form"], "be")
        self.assertEqual(spans[2]["word"], "darkness")
        self.assertEqual(spans[4]["options"], ["a", "the"])
        self.assertEqual(spans[6]["form"], "nakar")
        self.assertEqual(text[spans[1]["start"] : spans[1]["end"]], "reshit")


class HebrewSentenceTest(unittest.TestCase):
    def test_coordinated_clauses_split_with_leading_conjunction(self):
        root = ET.fromstring(
            "<wg>"
            '<w class="cj">וְ</w><wg class="cl"><w>a</w></wg>'
            '<w class="cj">וְ</w><wg class="cl"><w>b</w></wg>'
            '<wg class="np"><w>c</w></wg>'
            "</wg>"
        )
        groups = _split_clauses([root])
        self.assertEqual([[n.tag for n in g] for g in groups], [["w", "wg"], ["w", "wg", "wg"]])

    def test_single_clause_verse_is_one_sentence(self):
        root = ET.fromstring('<wg class="cl"><w>a</w><w>b</w></wg>')
        self.assertEqual(len(_split_clauses([root])), 1)


class GreekEditionsTest(unittest.TestCase):
    def test_wh_keeps_text_reading_and_drops_versification_notes(self):
        toks = "en 1722 {PREP} autw 846 (1:11) en | autw 848 | autw 846 | eiv".split()
        self.assertEqual(
            _wh_text_reading(toks), ["en", "1722", "{PREP}", "autw", "846", "en", "autw", "848", "eiv"]
        )

    def test_variant_kinds(self):
        a = [{"text": "εἴ", "strong": "1487"}, {"text": "γε", "strong": "1065"}]
        self.assertEqual(_variant_kind(a, [{"text": "ειγε"}]), "word-division")
        self.assertEqual(
            _variant_kind([{"text": "ἐγκακεῖν", "strong": "1573", "morph": "V-PAN"}],
                          [{"text": "ενκακειν", "strong": "1573", "morph": "V-PAN"}]),
            "orthographic",
        )
        self.assertEqual(_variant_kind([{"text": "φωτὸς", "strong": "5457"}], [{"text": "πνευματος", "strong": "4151"}]), "substantive")


if __name__ == "__main__":
    unittest.main()
