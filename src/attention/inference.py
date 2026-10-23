"""Inférence locale gloutonne ; aucun poids préentraîné ni réponse simulée."""
import argparse
from dataclasses import dataclass
import json
import hashlib
from pathlib import Path
import torch
from attention.tokenizer import TextTokenizer
from attention.transformers import Transformer, ModelConfig
from attention.train import select_device

class TranslationError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def validate_text(text, direction):
    if not isinstance(text, str) or direction not in ('to_genz', 'to_standard'):
        raise TranslationError('INVALID_REQUEST', 'Texte ou direction invalide.')
    text = text.strip()
    if not text:
        raise TranslationError('EMPTY_TEXT', 'Saisissez un texte à traduire.')
    if len(text) > 500:
        raise TranslationError('INPUT_TOO_LONG', 'Le texte dépasse 500 caractères.')
    return text

@dataclass
class Translation:
    translation: str
    direction: str
    truncated: bool
    model_id: str

class Translator:
    def __init__(self, directory, device='auto'):
        directory = Path(directory)
        self.device = select_device(device)
        self.tokenizer = TextTokenizer.load(directory)
        checkpoint = torch.load(directory / 'best.pt', map_location=self.device, weights_only=True)
        self.config = ModelConfig(**checkpoint['config'])
        if self.config.vocab_size != len(self.tokenizer) or self.config.pad_id != self.tokenizer.ids['<pad>']:
            raise ValueError('Tokenizer incompatible')
        # Verify exact tokenizer identity, beyond vocabulary size and reserved IDs.
        expected_hash = checkpoint.get('tokenizer_sha256')
        if expected_hash is not None and hashlib.sha256((directory / 'tokenizer.json').read_bytes()).hexdigest() != expected_hash:
            raise ValueError('Tokenizer différent de celui de l’entraînement')
        self.model = Transformer(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        self.model_id = checkpoint['model_id']
        self.max_new_tokens = self.config.max_length - 1

    @torch.inference_mode()
    def translate(self, text, direction):
        text = validate_text(text, direction)
        ids = self.tokenizer.ids
        source = [ids[f'<{direction}>']] + self.tokenizer.encode(text) + [ids['<eos>']]
        if len(source) > self.config.max_length:
            raise TranslationError('INPUT_TOO_LONG', f'Le texte dépasse {self.config.max_length} tokens, marqueurs compris.')
        memory, padding = self.model.encode(torch.tensor([source], device=self.device))
        generated = [ids['<bos>']]
        truncated = True
        for _ in range(self.max_new_tokens):
            logits = self.model.decode(torch.tensor([generated], device=self.device), memory, padding)[0, -1].clone()
            for token in ('<pad>', '<bos>', '<unk>', '<to_genz>', '<to_standard>'):
                logits[ids[token]] = float('-inf')
            token = logits.argmax().item()
            if token == ids['<eos>']:
                truncated = False
                break
            generated.append(token)
        return Translation(self.tokenizer.decode(generated[1:]), direction, truncated, self.model_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--direction', choices=['to_genz', 'to_standard'], required=True)
    parser.add_argument('text')
    args = parser.parse_args()
    print(json.dumps(vars(Translator(args.model_dir).translate(args.text, args.direction)), ensure_ascii=False))

if __name__ == '__main__':
    main()
