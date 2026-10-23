"""Dataset integrity tests; run with python3 -m unittest discover -s tests."""

import copy
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "prepare_dataset", ROOT / "scripts/prepare_dataset.py"
)
dataset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset)


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = dataset.read_jsonl(ROOT / "data/pairs.jsonl")
        cls.splits = {
            name: dataset.read_jsonl(ROOT / f"data/{name}.jsonl")
            for name in ("train", "validation", "test")
        }

    def test_delivered_corpus_and_reproducibility(self):
        dataset.validate_pairs(self.rows)
        dataset.validate_splits(self.rows, self.splits)
        self.assertGreaterEqual(len(self.rows), 1000)
        self.assertGreaterEqual(len({r["topic"] for r in self.rows}), 8)
        seeds, families, constructions = dataset.load_inputs()
        self.assertEqual(dataset.build_pairs(seeds, families, constructions), self.rows)
        self.assertEqual(dataset.split_pairs(self.rows, 42), self.splits)
        for part in self.splits.values():
            self.assertEqual(
                {r["topic"] for r in part}, {r["topic"] for r in self.rows}
            )

    def test_references_are_represented_in_training(self):
        all_tags = {tag for row in self.rows for tag in row["reference_tags"]}
        train_tags = {tag for row in self.splits["train"] for tag in row["reference_tags"]}
        self.assertEqual(all_tags, train_tags)
        self.assertGreaterEqual(len(all_tags), 80)

    def test_whole_composition_frames_stay_together(self):
        seeds, families, _ = dataset.load_inputs()
        owner = {r["id"]: name for name, rows in self.splits.items() for r in rows}
        by_pair = {(dataset.canonical(r["standard"]), dataset.canonical(r["genz"])): owner[r["id"]] for r in self.rows}
        frame_splits = {}
        for (_, standard, genz), family in zip(seeds, families):
            if family.startswith("pattern-"):
                frame_splits.setdefault(family, set()).add(by_pair[(dataset.canonical(standard), dataset.canonical(genz))])
        self.assertEqual(len(frame_splits), 36)
        self.assertTrue(all(len(names) == 1 for names in frame_splits.values()))

    def test_no_universal_suffix_expansion(self):
        self.assertGreaterEqual(sum(r["construction"] == "authored" for r in self.rows), 700)
        self.assertFalse(any(r["standard"].endswith(" Je voulais te le dire.") for r in self.rows))
        self.assertTrue(all(r["review_status"] == "pending_human_review" for r in self.rows))

    def test_preserves_numeric_literals(self):
        row = copy.deepcopy(self.rows[0])
        row["standard"], row["genz"] = "Rendez-vous à 14 h.", "On se capte à 15 h."
        with self.assertRaisesRegex(ValueError, "Numeric literals"):
            dataset.validate_pairs([row])

    def test_detects_leakage(self):
        splits = copy.deepcopy(self.splits)
        counts = dataset.Counter(r["group_id"] for r in splits["train"])
        index = next(i for i, r in enumerate(splits["train"]) if counts[r["group_id"]] > 1)
        row = splits["train"].pop(index)
        splits["test"].append(row)
        with self.assertRaisesRegex(ValueError, "Group leakage"):
            dataset.validate_splits(self.rows, splits)

    def test_detects_corruption(self):
        for field, value in (("standard", ""), ("genz", "x" * 501), ("provenance", {})):
            with self.subTest(field=field):
                rows = copy.deepcopy(self.rows[:1])
                rows[0][field] = value
                with self.assertRaises(ValueError):
                    dataset.validate_pairs(rows)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            dataset.validate_pairs([self.rows[0], self.rows[0]])

    def test_near_variants_join_components(self):
        rows = dataset.build_pairs(
            [
                ("a", "Je ne comprends pas cet exercice.", "Je capte pas cet exo."),
                ("b", "Je ne comprends pas cet exercice !", "Je capte pas cet exo !"),
            ]
        )
        self.assertEqual(len({r["group_id"] for r in rows}), 1)

    def test_review_sample_is_training_only(self):
        review = dataset.read_jsonl(ROOT / "data/review_sample.jsonl")
        self.assertEqual(len(review), 100)
        self.assertEqual(len({r["id"] for r in review}), 100)
        self.assertEqual({r["topic"] for r in review}, {r["topic"] for r in self.rows})
        self.assertEqual({tag for r in review for tag in r["reference_tags"]}, {tag for r in self.rows for tag in r["reference_tags"]})
        self.assertTrue(
            {r["id"] for r in review} <= {r["id"] for r in self.splits["train"]}
        )


if __name__ == "__main__":
    unittest.main()
