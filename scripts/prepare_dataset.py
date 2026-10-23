#!/usr/bin/env python3
"""Build the documented synthetic corpus and deterministic, group-safe splits.

Uses the Python standard library only. Run from any working directory.
"""

import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS_VERSION = "v2-memes-context"



def canonical(text):
    text = unicodedata.normalize("NFC", text).casefold().replace("’", "'")
    return re.sub(r"[^\w]+", " ", text).strip()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    Path(path).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


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
        standard, genz = (
            unicodedata.normalize("NFC", part.strip()) for part in line.split(" | ")
        )
        seeds.append((topic, standard, genz))
    return seeds


def load_inputs():
    seeds, families, constructions = [], [], []
    for filename in ("seeds.txt", "meme_seeds.txt"):
        for i, seed in enumerate(load_seeds(ROOT / "data" / filename)):
            seeds.append(seed)
            families.append(f"{filename}:{i}")
            constructions.append("authored")
    frames = json.loads((ROOT / "data/compositions.json").read_text())["frames"]
    for frame in frames:
        for slot in frame["slots"]:
            seeds.append((frame["topic"], *(
                unicodedata.normalize("NFC", frame[key].format(x=slot[key]))
                for key in ("standard", "genz")
            )))
            families.append(frame["id"])
            constructions.append("compositional")
    return seeds, families, constructions


def load_references():
    references = []
    for line in (ROOT / "data/references.tsv").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        tag, forms, meaning = (part.strip() for part in line.split(" | "))
        references.append((tag, forms.split(";"), meaning))
    return references


def reference_tags(text, references):
    normalized = canonical(text)
    return [tag for tag, forms, _ in references if any(
        re.search(r"(?<!\w)" + re.escape(canonical(form)) + r"(?!\w)", normalized)
        for form in forms
    )]


def build_pairs(seeds, families=None, constructions=None):
    # Whole compositional frames and lexical neighbours stay in one split.
    parent = list(range(len(seeds)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def join(i, j):
        parent[find(i)] = find(j)
    known_families = {}
    for i, family in enumerate(families or []):
        if family in known_families:
            join(i, known_families[family])
        known_families[family] = i
    texts = [(canonical(s), canonical(g)) for _, s, g in seeds]
    for i, (s, g) in enumerate(texts):
        for j in range(i):
            if find(i) == find(j):
                continue
            for left, right in zip((s, g), texts[j]):
                matcher = SequenceMatcher(None, left, right, autojunk=False)
                if matcher.real_quick_ratio() >= .88 and matcher.quick_ratio() >= .88 and matcher.ratio() >= .88:
                    join(i, j)
                    break
    references = load_references()
    pairs, seen = [], set()
    for index, (topic, standard, genz) in enumerate(seeds):
        normalized = texts[index]
        if normalized in seen:
            continue
        seen.add(normalized)
        construction = constructions[index] if constructions else "authored"
        identity = hashlib.sha256((standard + "\0" + genz).encode()).hexdigest()[:16]
        pairs.append({
            "id": f"v2-{identity}", "standard": standard, "genz": genz,
            "group_id": f"family-{find(index)+1:04d}", "topic": topic,
            "construction": construction,
            "review_status": "pending_human_review",
            "reference_tags": reference_tags(genz, references),
            "provenance": {
                "kind": "synthetic",
                "source": "Project-authored data/seeds.txt and data/meme_seeds.txt" if construction == "authored" else "Controlled grammatical compositions in data/compositions.json; entire frame held in one split",
                "usage_rights": "Project-generated synthetic examples; CC0-1.0 to the extent applicable; cultural reference names are not original project works",
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
        if re.findall(r"\d+", row["standard"]) != re.findall(r"\d+", row["genz"]):
            raise ValueError("Numeric literals changed between source and target")
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
        if any(
            not isinstance(provenance.get(k), str) or not provenance[k].strip()
            for k in ("source", "usage_rights")
        ):
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
        n = max(1, int(len(groups) * 0.1 + (0.5 if topic_index % 2 == 0 else 0)))
        for index, group in enumerate(groups):
            assignment[group] = (
                "validation" if index < n else "test" if index < 2 * n else "train"
            )
    for row in rows:
        splits[assignment[row["group_id"]]].append(row)
    # A lexical transfer benchmark: ensure every represented reference is seen
    # in training, while holding whole contexts/frames out. Rare singletons
    # therefore belong to train, never pretend to measure unseen slang.
    all_tags = {tag for row in rows for tag in row.get("reference_tags", [])}
    for tag in sorted(all_tags):
        if any(tag in row.get("reference_tags", []) for row in splits["train"]):
            continue
        candidates = [row for name in ("validation", "test") for row in splits[name] if tag in row.get("reference_tags", [])]
        group = min((r["group_id"] for r in candidates), key=lambda g: (sum(r["group_id"] == g for r in rows), g))
        for name in ("validation", "test"):
            splits["train"].extend(row for row in splits[name] if row["group_id"] == group)
            splits[name] = [row for row in splits[name] if row["group_id"] != group]
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


def mappings(rows, source, target):
    result = {}
    for row in rows:
        result.setdefault(canonical(row[source]), set()).add(canonical(row[target]))
    return result


def report(rows, splits, seeds, seed):
    return {
        "seed": seed,
        "pairs": len(rows),
        "input_examples_before_deduplication": len(seeds),
        "version": CORPUS_VERSION,
        "universal_suffix_expansion": False,
        "construction_counts": dict(Counter(r["construction"] for r in rows)),
        "reference_coverage": dict(sorted(Counter(tag for r in rows for tag in r["reference_tags"]).items())),
        "references_in_inventory": len(load_references()),
        "pairs_with_reference_tags": sum(bool(r["reference_tags"]) for r in rows),
        "largest_group": max(Counter(r["group_id"] for r in rows).values()),
        "ambiguous_standard_texts": sum(len(v) > 1 for v in mappings(rows, "standard", "genz").values()),
        "ambiguous_genz_texts": sum(len(v) > 1 for v in mappings(rows, "genz", "standard").values()),
        "groups": len({r["group_id"] for r in rows}),
        "topics": dict(sorted(Counter(r["topic"] for r in rows).items())),
        "splits": {
            name: {
                "pairs": len(part),
                "groups": len({r["group_id"] for r in part}),
                "topics": dict(sorted(Counter(r["topic"] for r in part).items())),
                "reference_coverage": dict(sorted(Counter(tag for r in part for tag in r.get("reference_tags", [])).items())),
            }
            for name, part in splits.items()
        },
        "max_chars": {
            key: max(len(r[key]) for r in rows) for key in ("standard", "genz")
        },
        "normalized_pair_duplicates": 0,
        "exact_identity_pairs": sum(canonical(r["standard"]) == canonical(r["genz"]) for r in rows),
        "digit_preservation_errors": [r["id"] for r in rows if re.findall(r"\d+", r["standard"]) != re.findall(r"\d+", r["genz"])],
        "reference_split_policy": "All reference tags represented in train; evaluate held-out contexts, not unseen expressions",
        "shared_groups_between_splits": 0,
        "near_variant_rule": "Whole composition frames + connected components, either normalized side SequenceMatcher >= 0.88; no universal suffix variants",
        "pairs_sha256": hashlib.sha256(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()
        ).hexdigest(),
        "human_review_completed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    seeds, families, constructions = load_inputs()
    rows = build_pairs(seeds, families, constructions)
    validate_pairs(rows)
    splits = split_pairs(rows, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "pairs.jsonl", rows)
    for name, part in splits.items():
        write_jsonl(args.output_dir / f"{name}.jsonl", part)
    # Review only training examples: human curation must not tune the held-out test.
    rng = random.Random(args.seed)
    # Training-only sample: cover topics, then lexical references, then fill.
    train = list(splits["train"])
    rng.shuffle(train)
    review = []
    def features(row):
        return {"topic:" + row["topic"]} | {"ref:" + tag for tag in row["reference_tags"]}
    uncovered = set().union(*(features(row) for row in train))
    while uncovered:
        row = max(train, key=lambda r: len(features(r) & uncovered))
        review.append(row)
        uncovered -= features(row)
    if len(review) > 100:
        raise ValueError("Review coverage requires more than 100 examples")
    selected = {r["id"] for r in review}
    review.extend(r for r in train if r["id"] not in selected)
    write_jsonl(args.output_dir / "review_sample.jsonl", review[:100])
    summary = report(rows, splits, seeds, args.seed)
    (args.output_dir / "report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
