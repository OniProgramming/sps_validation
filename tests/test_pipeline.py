import unittest
import xml.etree.ElementTree as ET

from sps_validation.docx_reader import Run
from sps_validation.ingest import _spans, strip_spans
from sps_validation.align import skeleton
from sps_validation.features import carries_number, hebrew_features
from sps_validation.report import krippendorff_nominal, sentence_rows
from sps_validation.validate import perturb
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


class NoteRemovalTest(unittest.TestCase):
    def test_notes_are_cut_and_offsets_rebased(self):
        text = "under my yarekh {— thigh — the oath-gesture}, and ruach."
        spans = [
            {"type": "translit", "start": 9, "end": 15},
            {"type": "note", "start": 16, "end": 44},
            {"type": "translit", "start": 50, "end": 55},
        ]
        out, kept = strip_spans(text, spans, {"note"})
        self.assertEqual(out, "under my yarekh, and ruach.")
        self.assertEqual([out[s["start"]:s["end"]] for s in kept], ["yarekh", "ruach"])


class AlignKeysTest(unittest.TestCase):
    def test_popular_and_academic_transliterations_share_a_skeleton(self):
        self.assertEqual(skeleton("Yosef"), skeleton("yôsēp̄"))
        self.assertEqual(skeleton("reshit"), skeleton("rēʾšiyṯ"))
        self.assertEqual(skeleton("apolytrōsin"), skeleton("ἀπολύτρωσιν"))


class FeatureTest(unittest.TestCase):
    def test_hebrew_finite_verb(self):
        tok = {"class": "verb", "pos": "verb", "type": "wayyiqtol", "stem": "hiphil", "person": "third",
               "gender": "masculine", "number": "singular", "lemma": "x", "gloss": "y"}
        classes = [c for c, _ in hebrew_features(tok)]
        self.assertEqual(classes, ["LEX", "ASP", "STEM", "REF"])

    def test_plural_of_majesty_carries_no_number(self):
        self.assertFalse(carries_number({"lemma": "אֱלֹהִים", "sdbh": "000397001003000", "gloss": "gods"}))
        self.assertTrue(carries_number({"lemma": "אֱלֹהִים", "sdbh": "000397001001000", "gloss": "God"}))
        self.assertFalse(carries_number({"lemma": "מַיִם"}))


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


class StatisticsTest(unittest.TestCase):
    def test_krippendorff(self):
        self.assertEqual(krippendorff_nominal([("a", "a"), ("b", "b"), ("a", "a")]), 1.0)
        self.assertLess(krippendorff_nominal([("a", "b"), ("b", "a"), ("a", "b"), ("b", "a")]), 0)


class UnalignedEnglishTest(unittest.TestCase):
    def test_unaligned_english_is_kept_for_judging(self):
        from sps_validation.judge import merge_unaligned
        beads = [{"units": [], "english": "Intro."}, {"units": ["A"], "english": "One."},
                 {"units": [], "english": "Extra."}, {"units": ["B"], "english": "Two."}]
        out = merge_unaligned(beads)
        self.assertEqual([b["english"] for b in out], ["Intro. One. Extra.", "Two."])
        self.assertEqual(out[0]["unaligned"], ["Intro.", "Extra."])


class ResumeSafetyTest(unittest.TestCase):
    def test_fingerprint_changes_with_model_or_prompt(self):
        from sps_validation import judge as J

        class Fake:
            model = "m1"
            def params(self, prompt):
                return {"model": self.model, "p": prompt}

        a, b = Fake(), Fake()
        b.model = "m2"
        reqs = {"r1": {"prompt": "x"}}
        self.assertNotEqual(J.fingerprint(a, reqs), J.fingerprint(b, reqs))
        self.assertNotEqual(J.fingerprint(a, reqs), J.fingerprint(a, {"r1": {"prompt": "y"}}))
        self.assertEqual(J.fingerprint(a, reqs), J.fingerprint(a, {"r1": {"prompt": "x"}}))


class ScoringTest(unittest.TestCase):
    def test_a_refusing_judge_does_not_change_the_score(self):
        units = {"U": {"text": "x", "tokens": []}}
        feats = {"U": [{"fid": "U/1", "class": "LEX"}, {"fid": "U/2", "class": "LEX"}]}
        req = {"id": "r", "book": "GEN", "translation": "WEB", "units": ["U"], "english": "e", "fids": ["U/1", "U/2"]}
        ok = {"result": {"status": "ok", "additions": [], "features": [
            {"fid": "U/1", "outcome": "retained"}, {"fid": "U/2", "outcome": "distorted"}]}}
        one = sentence_rows([req], {"claude": {"r": ok}}, units, feats)[0]
        two = sentence_rows([req], {"claude": {"r": ok}, "gpt": {"r": {"result": {"status": "refusal"}}}}, units, feats)[0]
        self.assertEqual(one["fidelity"], two["fidelity"])
        self.assertEqual(two["distorted"], 1.0)


class ZeroDenominatorTest(unittest.TestCase):
    def test_everything_lost_gives_numbers_not_nan(self):
        import math
        from sps_validation.report import totals
        units = {"U": {"text": "x", "tokens": []}}
        feats = {"U": [{"fid": "U/1", "class": "LEX"}]}
        req = {"id": "r", "book": "GEN", "translation": "WEB", "units": ["U"], "english": "e", "fids": ["U/1"]}
        res = {"result": {"status": "ok", "additions": [], "features": [{"fid": "U/1", "outcome": "lost"}]}}
        rows = sentence_rows([req], {"claude": {"r": res}}, units, feats)
        t = totals(rows)["GEN.WEB"]
        for k in ("retention", "accuracy", "fidelity"):
            self.assertFalse(math.isnan(t[k]), k)
        self.assertEqual(t["fidelity"], 0.0)
        self.assertFalse(any(math.isnan(v) for v in t["fidelity_CI95"] + t["retention_CI95"]))


class PerturbationGrammarTest(unittest.TestCase):
    def test_inflection(self):
        from sps_validation.validate import inflect_number
        self.assertEqual(inflect_number("wives"), "wife")
        self.assertEqual(inflect_number("wife"), "wives")
        self.assertEqual(inflect_number("sons"), "son")
        self.assertIsNone(inflect_number("sheep"))
        self.assertIsNone(inflect_number("shelves"))
        self.assertIsNone(inflect_number("promised"))

    def test_tense_shift_skips_auxiliary_forms(self):
        import random
        units = {"U": {"tokens": [{"id": "v", "class": "verb", "type": "qatal", "english": "left", "gloss": "left"}]}}
        feats = {"U": [{"fid": "U/1", "token": "v", "class": "ASP", "value": "qatal"}]}
        self.assertIsNone(perturb({"units": ["U"], "english": "They had left."}, "tense_shift", units, feats,
                                  random.Random(0)))
        new, fid, _ = perturb({"units": ["U"], "english": "They left."}, "tense_shift", units, feats, random.Random(0))
        self.assertEqual(new, "They will leave.")


class PerturbationSafetyTest(unittest.TestCase):
    """The three cases from the second review must not be produced."""

    def _run(self, kind, english, tok, cls):
        import random
        units = {"U": {"tokens": [dict(tok, id="t")]}}
        feats = {"U": [{"fid": "U/1", "token": "t", "class": cls, "value": ""}]}
        return perturb({"units": ["U"], "english": english}, kind, units, feats, random.Random(0))

    def test_no_agreement_breaks(self):
        noun = {"class": "noun", "type": "common", "english": "children", "gloss": "children"}
        self.assertIsNone(self._run("wrong_sense", "The children are here.", noun, "LEX"))
        man = {"class": "noun", "type": "common", "english": "man", "gloss": "man"}
        self.assertIsNone(self._run("number_flip", "He spoke to the man who was here.", man, "NUM"))
        left = {"class": "verb", "type": "qatal", "english": "left", "gloss": "left"}
        self.assertIsNone(self._run("tense_shift", "They had already quietly left.", left, "ASP"))
        self.assertIsNone(self._run("tense_shift", "When they left, it rained.", left, "ASP"))

    def test_safe_edits_still_happen(self):
        man = {"class": "noun", "type": "common", "english": "man", "gloss": "man"}
        new, _, _ = self._run("number_flip", "He spoke to the man.", man, "NUM")
        self.assertEqual(new, "He spoke to the men.")


class PerturbationTest(unittest.TestCase):
    def test_negation_drop_targets_the_neg_feature(self):
        import random
        req = {"units": ["U"], "english": "and he did not eat."}
        units = {"U": {"tokens": [{"id": "t1", "class": "adv"}]}}
        feats = {"U": [{"fid": "U/1", "token": "t1", "class": "NEG", "value": "לֹא"}]}
        new, fid, _ = perturb(req, "negation_drop", units, feats, random.Random(0))
        self.assertEqual((new, fid), ("and he did eat.", "U/1"))


if __name__ == "__main__":
    unittest.main()
