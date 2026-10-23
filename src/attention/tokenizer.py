"""BPE ByteLevel partagé, entraîné exclusivement sur le train."""
import json
import unicodedata
from pathlib import Path
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers

SPECIAL = ['<pad>', '<bos>', '<eos>', '<unk>', '<to_genz>', '<to_standard>']

class TextTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.ids = {token: tokenizer.token_to_id(token) for token in SPECIAL}
        if any(value is None for value in self.ids.values()):
            raise ValueError('Tokens réservés absents')

    @classmethod
    def train(cls, pairs, vocab_size=2000):
        tokenizer = Tokenizer(models.BPE(unk_token='<unk>'))
        tokenizer.normalizer = normalizers.NFC()
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
        tokenizer.train_from_iterator((p[key] for p in pairs for key in ('standard', 'genz')), trainer)
        return cls(tokenizer)

    def encode(self, text):
        return self.tokenizer.encode(unicodedata.normalize('NFC', text), add_special_tokens=False).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(directory / 'tokenizer.json'))
        (directory / 'special_tokens.json').write_text(json.dumps(self.ids, indent=2), encoding='utf-8')

    @classmethod
    def load(cls, directory):
        result = cls(Tokenizer.from_file(str(Path(directory) / 'tokenizer.json')))
        recorded = json.loads((Path(directory) / 'special_tokens.json').read_text())
        if result.ids != recorded:
            raise ValueError('Identifiants réservés incompatibles')
        return result

    def __len__(self):
        return self.tokenizer.get_vocab_size()
