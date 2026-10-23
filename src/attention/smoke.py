"""Écrit quatre paires pour un test de mémorisation, sans prétention de généralisation.

python -m attention.smoke --output checkpoints/smoke-fixtures
python -m attention.train --train checkpoints/smoke-fixtures/train.jsonl --validation checkpoints/smoke-fixtures/validation.jsonl --output checkpoints/smoke --config configs/smoke.json --epochs 180 --batch-size 8 --lr .003 --patience 0 --vocab-size 400 --threads 2
python -m attention.evaluate --model-dir checkpoints/smoke --data checkpoints/smoke-fixtures/train.jsonl --output checkpoints/smoke/memorization.json

Les mêmes textes sont volontairement utilisés en validation : seule la capacité
à mémoriser est testée ; ce score ne mesure pas la qualité sur des textes inédits.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="checkpoints/smoke-fixtures")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    pairs = [
        ("Bonjour.", "Salut."),
        ("Je suis fatigué.", "Je suis KO."),
        ("Ce film est excellent.", "Ce film est une dinguerie."),
        ("Je ne viens pas.", "Je viens pas."),
    ]
    for split in ["train", "validation"]:
        rows = [
            {
                "id": f"{split}-{i}",
                "group_id": f"{split}-{i}",
                "standard": standard,
                "genz": genz,
            }
            for i, (standard, genz) in enumerate(pairs)
        ]
        (output / f"{split}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        )


if __name__ == "__main__":
    main()
