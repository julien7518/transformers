"""Transformer pédagogique post-norm ; attention calculée uniquement avec des primitives."""

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ModelConfig:
    vocab_size: int
    pad_id: int = 0
    d_model: int = 128
    heads: int = 4
    encoder_layers: int = 2
    decoder_layers: int = 2
    d_ff: int = 512
    dropout: float = 0.1
    max_length: int = 64

    def __post_init__(self):
        if self.d_model <= 0 or self.heads <= 0 or self.d_model % self.heads:
            raise ValueError("d_model doit être divisible par heads")
        if min(self.encoder_layers, self.decoder_layers, self.max_length) < 1:
            raise ValueError("Les profondeurs et longueurs doivent être positives")


class Embeddinger(nn.Module):
    def __init__(
        self, num_embeddings, embedding_dim, max_length=64, pad_id=0, dropout=0.1
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=pad_id)
        positions = torch.arange(max_length).float().unsqueeze(1)
        rates = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-math.log(10000) / embedding_dim)
        )
        pe = torch.zeros(max_length, embedding_dim)
        pe[:, 0::2] = torch.sin(positions * rates)
        pe[:, 1::2] = torch.cos(positions * rates[: embedding_dim // 2])
        self.register_buffer("positional_encoding", pe)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(embedding_dim)

    def forward(self, tokens):
        if tokens.size(1) > self.positional_encoding.size(0):
            raise ValueError("Séquence trop longue")
        return self.dropout(
            self.embedding(tokens) * self.scale
            + self.positional_encoding[: tokens.size(1)]
        )


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim_embedding, dim_out, mask=False, dropout=0.0):
        super().__init__()
        self.WQ = nn.Linear(dim_embedding, dim_out)
        self.WK = nn.Linear(dim_embedding, dim_out)
        self.WV = nn.Linear(dim_embedding, dim_out)
        self.mask = mask
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, query=None, key_padding_mask=None):
        q = self.WQ(input if query is None else query)
        k, v = self.WK(input), self.WV(input)
        scores = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))
        forbidden = torch.zeros_like(scores, dtype=torch.bool)
        if self.mask:
            forbidden |= torch.ones(
                q.size(1), k.size(1), device=q.device, dtype=torch.bool
            ).triu(1)
        if key_padding_mask is not None:
            forbidden |= key_padding_mask[:, None, :]
        scores = scores.masked_fill(forbidden, float("-inf"))
        # Explicitly handle fully masked rows without NaNs (including backward).
        all_masked = forbidden.all(-1, keepdim=True)
        scores = scores.masked_fill(all_masked, 0)
        weights = scores.softmax(-1).masked_fill(forbidden, 0)
        return self.dropout(weights) @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, h, dim_embedding, mask=False, dropout=0.0):
        super().__init__()
        if dim_embedding % h:
            raise ValueError("Dimension non divisible par le nombre de têtes")
        self.heads = nn.ModuleList(
            [
                ScaledDotProductAttention(
                    dim_embedding, dim_embedding // h, mask, dropout
                )
                for _ in range(h)
            ]
        )
        self.lin = nn.Linear(dim_embedding, dim_embedding)

    def forward(self, input, query=None, key_padding_mask=None):
        return self.lin(
            torch.cat(
                [head(input, query, key_padding_mask) for head in self.heads], dim=-1
            )
        )


class TransformersEncoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.attention = MultiHeadAttention(c.heads, c.d_model, dropout=c.dropout)
        self.ff = nn.Sequential(
            nn.Linear(c.d_model, c.d_ff),
            nn.ReLU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.d_ff, c.d_model),
        )
        self.norm1, self.norm2 = nn.LayerNorm(c.d_model), nn.LayerNorm(c.d_model)
        self.dropout = nn.Dropout(c.dropout)

    def forward(self, x, padding):
        x = self.norm1(x + self.dropout(self.attention(x, key_padding_mask=padding)))
        return self.norm2(x + self.dropout(self.ff(x)))


class TransformersDecoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.self_attention = MultiHeadAttention(c.heads, c.d_model, True, c.dropout)
        self.cross_attention = MultiHeadAttention(c.heads, c.d_model, dropout=c.dropout)
        self.ff = nn.Sequential(
            nn.Linear(c.d_model, c.d_ff),
            nn.ReLU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.d_ff, c.d_model),
        )
        self.norms = nn.ModuleList([nn.LayerNorm(c.d_model) for _ in range(3)])
        self.dropout = nn.Dropout(c.dropout)

    def forward(self, x, memory, target_padding, source_padding):
        x = self.norms[0](
            x + self.dropout(self.self_attention(x, key_padding_mask=target_padding))
        )
        x = self.norms[1](
            x
            + self.dropout(
                self.cross_attention(memory, query=x, key_padding_mask=source_padding)
            )
        )
        return self.norms[2](x + self.dropout(self.ff(x)))


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = Embeddinger(
            config.vocab_size,
            config.d_model,
            config.max_length,
            config.pad_id,
            config.dropout,
        )
        self.encoder = nn.ModuleList(
            [TransformersEncoder(config) for _ in range(config.encoder_layers)]
        )
        self.decoder = nn.ModuleList(
            [TransformersDecoder(config) for _ in range(config.decoder_layers)]
        )
        self.projection = nn.Linear(config.d_model, config.vocab_size)

    def encode(self, source):
        padding = source.eq(self.config.pad_id)
        x = self.embedding(source)
        for layer in self.encoder:
            x = layer(x, padding)
        return x, padding

    def decode(self, target, memory, source_padding):
        padding = target.eq(self.config.pad_id)
        x = self.embedding(target)
        for layer in self.decoder:
            x = layer(x, memory, padding, source_padding)
        return self.projection(x)

    def forward(self, source, target):
        memory, padding = self.encode(source)
        return self.decode(target, memory, padding)


Transformers = Transformer
