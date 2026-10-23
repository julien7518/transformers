"""Évaluation descriptive par direction ; aucun score de sens automatique prétendu."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import torch
from attention.data import load_pairs
from attention.inference import Translator


def evaluate(translator, pairs):
    result = {}
    for direction, source_key, target_key in [('to_genz', 'standard', 'genz'), ('to_standard', 'genz', 'standard')]:
        rows = []
        for pair in pairs:
            start = time.monotonic()
            output = translator.translate(pair[source_key], direction)
            rows.append({'id': pair['id'], 'source': pair[source_key], 'reference': pair[target_key], **asdict(output), 'seconds': time.monotonic() - start})
        count = len(rows)
        result[direction] = {'count': count, 'exact_match': sum(r['translation'] == r['reference'] for r in rows) / count, 'copy_rate': sum(r['translation'] == r['source'] for r in rows) / count, 'unchanged_source_exact_match': sum(r['source'] == r['reference'] for r in rows) / count, 'truncated_rate': sum(r['truncated'] for r in rows) / count, 'mean_seconds': sum(r['seconds'] for r in rows) / count, 'examples': rows}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    result = evaluate(Translator(args.model_dir), load_pairs(args.data))
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({d: {k: v for k, v in metrics.items() if k != 'examples'} for d, metrics in result.items()}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
