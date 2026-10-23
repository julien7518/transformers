import json
from pathlib import Path
import torch
from torch.nn.utils.rnn import pad_sequence


def load_pairs(path):
    pairs = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not pairs:
        raise ValueError(f'Corpus vide : {path}')
    for pair in pairs:
        if not all(isinstance(pair.get(k), str) and pair[k].strip() for k in ('id', 'standard', 'genz', 'group_id')):
            raise ValueError('Paire invalide')
    return pairs


def examples(pairs, tokenizer, max_length=64):
    result = []
    for pair in pairs:
        for direction, source_key, target_key in [('to_genz', 'standard', 'genz'), ('to_standard', 'genz', 'standard')]:
            source = [tokenizer.ids[f'<{direction}>']] + tokenizer.encode(pair[source_key]) + [tokenizer.ids['<eos>']]
            target = [tokenizer.ids['<bos>']] + tokenizer.encode(pair[target_key]) + [tokenizer.ids['<eos>']]
            if max(len(source), len(target)) > max_length:
                raise ValueError(f"Paire {pair['id']} trop longue : {len(source)}/{len(target)} > {max_length}; aucune troncature")
            result.append((torch.tensor(source), torch.tensor(target)))
    return result


def collate_batch(batch, pad_id=0):
    source, target = zip(*batch)
    source = pad_sequence(source, batch_first=True, padding_value=pad_id)
    target = pad_sequence(target, batch_first=True, padding_value=pad_id)
    return source, target[:, :-1], target[:, 1:]
