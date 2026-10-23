"""Audit lengths with a train-only BPE; never truncate the delivered corpus."""
import argparse
import hashlib
import json
from pathlib import Path

from attention.data import examples, load_pairs
from attention.tokenizer import TextTokenizer

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vocab-size', type=int, default=1024)
    parser.add_argument('--max-length', type=int, default=64)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/tokenization_report.json')
    args = parser.parse_args()
    train = load_pairs(ROOT / 'data/train.jsonl')
    tokenizer = TextTokenizer.train(train, args.vocab_size)
    report = {'vocab_size_requested': args.vocab_size, 'vocab_size_actual': len(tokenizer), 'max_length': args.max_length, 'tokenizer_trained_on': 'train only', 'special_tokens_counted': 2, 'splits': {}}
    for name in ('train', 'validation', 'test'):
        path = ROOT / f'data/{name}.jsonl'
        rows = load_pairs(path)
        lengths = sorted(len(tokenizer.encode(row[key])) + 2 for row in rows for key in ('standard', 'genz'))
        # Also exercise the exact training pipeline and reject overflow.
        examples(rows, tokenizer, args.max_length)
        roundtrips = all(tokenizer.decode(tokenizer.encode(row[key])) == row[key] for row in rows for key in ('standard', 'genz'))
        if not roundtrips:
            raise ValueError(f'Unicode roundtrip failed: {name}')
        report['splits'][name] = {'pairs': len(rows), 'max_tokens_with_control_tokens': max(lengths), 'p95_tokens': lengths[int(.95 * (len(lengths)-1))], 'overflow': 0, 'unicode_roundtrip': roundtrips, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
