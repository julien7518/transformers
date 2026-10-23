"""Dataset integrity tests; run with python3 -m unittest discover -s tests."""
import copy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_dataset", ROOT / "scripts/prepare_dataset.py")
dataset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset)


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = dataset.read_jsonl(ROOT / "data/pairs.jsonl")
        cls.splits = {name: dataset.read_jsonl(ROOT / f"data/{name}.jsonl") for name in ("train", "validation", "test")}

    def test_delivered_corpus_and_reproducibility(self):
        dataset.validate_pairs(self.rows)
        dataset.validate_splits(self.rows, self.splits)
        self.assertGreaterEqual(len(self.rows), 1000)
        self.assertGreaterEqual(len({r["topic"] for r in self.rows}), 8)
        seeds = dataset.load_seeds(ROOT / "data/seeds.txt")
        self.assertEqual(dataset.build_pairs(seeds), self.rows)
        self.assertEqual(dataset.split_pairs(self.rows, 42), self.splits)
        for part in self.splits.values():
            self.assertEqual({r["topic"] for r in part}, {r["topic"] for r in self.rows})

    def test_detects_leakage(self):
        splits = copy.deepcopy(self.splits)
        row = splits["train"].pop(0)
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
        rows = dataset.build_pairs([("a", "Je ne comprends pas cet exercice.", "Je capte pas cet exo."), ("b", "Je ne comprends pas cet exercice !", "Je capte pas cet exo !")])
        self.assertEqual(len({r["group_id"] for r in rows}), 1)

    def test_review_sample_is_training_only(self):
        review = dataset.read_jsonl(ROOT / "data/review_sample.jsonl")
        self.assertEqual(len(review), 50)
        self.assertEqual(len({r["id"] for r in review}), 50)
        self.assertTrue({r["id"] for r in review} <= {r["id"] for r in self.splits["train"]})


if __name__ == "__main__":
    unittest.main()
