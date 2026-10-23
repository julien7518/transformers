#!/usr/bin/env python3
"""Build the documented synthetic corpus and deterministic, group-safe splits.

Uses the Python standard library only. Run from any working directory.
"""
import argparse
from collections import Counter
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
SUFFIXES = (
    ("", ""),
    (" Je voulais te le dire.", " Je voulais te le dire."),
    (" Nous pourrons en reparler demain.", " On pourra en reparler demain."),
    (" Je préfère que tu le saches.", " Je préfère que tu le saches."),
)


def canonical(text):
    text = unicodedata.normalize("NFC", text).casefold().replace("’", "'")
    return re.sub(r"[^\w]+", " ", text).strip()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def load_seeds(path):
    seeds, topic = [], None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            topic = line[1:-1]
            continue
        if topic is None or line.count(" | ") != 1:
            raise ValueError(f"Invalid seed line: {line}")
        standard, genz = (unicodedata.normalize("NFC", part.strip()) for part in line.split(" | "))
        seeds.append((topic, standard, genz))
    return seeds


def build_pairs(seeds):
    # Conservative connected components: either side can identify a near variant.
    parent = list(range(len(seeds)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    texts = [(canonical(s), canonical(g)) for _, s, g in seeds]
    for i, (s, g) in enumerate(texts):
        for j in range(i):
            sj, gj = texts[j]
            if max(SequenceMatcher(None, s, sj).ratio(), SequenceMatcher(None, g, gj).ratio()) >= .88:
                parent[find(i)] = find(j)
    pairs = []
    for index, (topic, standard, genz) in enumerate(seeds):
        group = f"family-{find(index) + 1:04d}"
        for variant, (ss, gs) in enumerate(SUFFIXES):
            pairs.append({
                "id": f"pair-{index + 1:04d}-{variant}",
                "standard": standard + ss, "genz": genz + gs,
                "group_id": group, "topic": topic,
                "provenance": {
                    "kind": "synthetic", "source": "Original AI-authored educational seeds in data/seeds.txt; controlled pragmatic suffix expansion v1",
                    "usage_rights": "Project-generated synthetic text; dedicated under CC0-1.0; no third-party corpus used",
                },
            })
    return pairs


def validate_pairs(rows):
    ids, pairs, sources, targets = set(), set(), {}, {}
    for row in rows:
        for key in ("id", "standard", "genz", "group_id", "topic"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Missing or empty {key}")
        if row["id"] in ids:
            raise ValueError("Duplicate id")
        ids.add(row["id"])
        for key in ("standard", "genz"):
            if len(row[key]) > 500:
                raise ValueError("Text exceeds 500 Unicode code points")
            if row[key] != unicodedata.normalize("NFC", row[key]):
                raise ValueError("Text must use NFC")
        pair = (canonical(row["standard"]), canonical(row["genz"]))
        if pair in pairs:
            raise ValueError("Duplicate normalized pair")
        pairs.add(pair)
        for text, seen in ((pair[0], sources), (pair[1], targets)):
            if text in seen and seen[text] != row["group_id"]:
                raise ValueError("Identical text appears in different groups")
            seen[text] = row["group_id"]
        provenance = row.get("provenance", {})
        if provenance.get("kind") not in ("synthetic", "sourced"):
            raise ValueError("Invalid provenance kind")
        if any(not isinstance(provenance.get(k), str) or not provenance[k].strip() for k in ("source", "usage_rights")):
            raise ValueError("Incomplete provenance")


def split_pairs(rows, seed=42):
    topics = {}
    for row in rows:
        topics.setdefault(row["group_id"], row["topic"])
    rng = random.Random(seed)
    splits = {"train": [], "validation": [], "test": []}
    assignment = {}
    # Stratify by a component's first topic; never break cross-topic components.
    for topic_index, topic in enumerate(sorted(set(topics.values()))):
        groups = sorted(g for g, t in topics.items() if t == topic)
        rng.shuffle(groups)
        # Alternate rounding half-groups so small strata stay near 80/10/10.
        n = max(1, int(len(groups) * .1 + (.5 if topic_index % 2 == 0 else 0)))
        for index, group in enumerate(groups):
            assignment[group] = "validation" if index < n else "test" if index < 2 * n else "train"
    for row in rows:
        splits[assignment[row["group_id"]]].append(row)
    validate_splits(rows, splits)
    return splits


def validate_splits(rows, splits):
    expected = {r["id"]: r for r in rows}
    seen, groups = set(), {}
    for name, part in splits.items():
        validate_pairs(part)
        for row in part:
            if row["id"] in seen or expected.get(row["id"]) != row:
                raise ValueError("Split duplicates or changes a pair")
            seen.add(row["id"])
            group = row["group_id"]
            if group in groups and groups[group] != name:
                raise ValueError("Group leakage between splits")
            groups[group] = name
    if seen != set(expected):
        raise ValueError("Splits do not cover the corpus")


def report(rows, splits, seeds, seed):
    return {
        "seed": seed, "pairs": len(rows), "semantic_seeds": len(seeds),
        "context_variants_per_seed": len(SUFFIXES),
        "groups": len({r["group_id"] for r in rows}),
        "topics": dict(sorted(Counter(r["topic"] for r in rows).items())),
        "splits": {name: {"pairs": len(part), "groups": len({r["group_id"] for r in part}), "topics": dict(sorted(Counter(r["topic"] for r in part).items()))} for name, part in splits.items()},
        "max_chars": {key: max(len(r[key]) for r in rows) for key in ("standard", "genz")},
        "normalized_pair_duplicates": 0, "shared_groups_between_splits": 0,
        "near_variant_rule": "Connected components over semantic seeds, either side normalized SequenceMatcher ratio >= 0.88; all suffix variants inherit the component",
        "pairs_sha256": hashlib.sha256("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()).hexdigest(),
        "human_review_completed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    seeds = load_seeds(ROOT / "data/seeds.txt")
    rows = build_pairs(seeds)
    validate_pairs(rows)
    splits = split_pairs(rows, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "pairs.jsonl", rows)
    for name, part in splits.items():
        write_jsonl(args.output_dir / f"{name}.jsonl", part)
    # Review only training examples: human curation must not tune the held-out test.
    rng = random.Random(args.seed)
    review = []
    for topic in sorted({r["topic"] for r in rows}):
        candidates = [r for r in splits["train"] if r["topic"] == topic]
        review.extend(rng.sample(candidates, min(5, len(candidates))))
    write_jsonl(args.output_dir / "review_sample.jsonl", review)
    summary = report(rows, splits, seeds, args.seed)
    (args.output_dir / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
