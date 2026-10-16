import math

import tiktoken
import torch
from torch import nn


class Tokenizer:
    def __init__(
        self, model_name: str | None = None, encoder_name: str = "o200k_base"
    ) -> None:
        self.encoder = (
            tiktoken.encoding_for_model(model_name)
            if model_name
            else tiktoken.get_encoding(encoder_name)
        )

    def encode(self, text: str):
        return self.encoder.encode(text)

    def decode(self, tokens: list[int]):
        return self.encoder.decode(tokens)


class Embeddinger(nn.Module):
    def __init__(self, num_embeddings, embedding_dim) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.positional_encoding = torch.tensor(
            [
                [
                    math.sin(pos / (10000 ** (2 * (i // 2) / embedding_dim)))
                    if i % 2 == 0
                    else math.cos(pos / (10000 ** (2 * (i // 2) / embedding_dim)))
                    for i in range(embedding_dim)
                ]
                for pos in range(num_embeddings)
            ]
        )

    def forward(self, input):
        emb = self.embedding(input)
        seq_len = input.shape[-1]
        return emb + self.positional_encoding[:seq_len]


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim_embedding, dim_out, mask: bool) -> None:
        super().__init__()
        self.WQ = nn.Linear(dim_embedding, dim_out)
        self.WK = nn.Linear(dim_embedding, dim_out)
        self.WV = nn.Linear(dim_embedding, dim_out)

    def forward(self, input, query=None):
        q = self.WQ(query) if query else self.WQ(input)
        k = self.WK(input)
        v = self.WV(input)

        matmul = torch.matmul(q, k.transpose(-2, -1))
        mask_matrix = torch.triu(
            torch.ones(matmul.size(1), matmul.size(2)), diagonal=1
        ).bool()
        if self.mask:
            matmul.masked_fill(mask_matrix, float("-inf"))
        scale = matmul / k.shape(2) ** 0.5
        softmax = torch.softmax(scale, 2)
        result = torch.matmul(softmax, v)

        return result


class MultiHeadAttention(nn.Module):
    def __init__(self, h, dim_embedding, mask: bool) -> None:
        super().__init__()
        self.number_of_head = h
        self.heads = [
            ScaledDotProductAttention(dim_embedding, dim_embedding / h, mask)
            for _ in range(h)
        ]
        self.lin = nn.Linear(dim_embedding, dim_embedding)

    def forward(self, input, query=None):
        results = [
            self.heads[i].forward(input, query) for i in range(self.number_of_head)
        ]
        concat = torch.cat(results, 2)
        linear = self.lin(concat)

        return linear


class FeedForward(nn.Module):
    def __init__(self, dim_embedding, dim_ff) -> None:
        super().__init__()
        self.firstLayer = nn.Linear(dim_embedding, dim_ff)
        self.relu = nn.ReLU()
        self.secondLayer = nn.Linear(dim_ff, dim_embedding)
        self.sequence = nn.Sequential(self.firstLayer, self.relu, self.secondLayer)

    def forward(self, input):
        return self.sequence(input)


def add_and_norm(input, multi_head_output):
    return torch.norm(multi_head_output + input)


class TransformersEncoder(nn.Module):
    def __init__(self, h, dim_embedding, dim_ff) -> None:
        super().__init__()
        self.multi_head = MultiHeadAttention(h, dim_embedding, False)
        self.ff = FeedForward(dim_embedding, dim_ff)

    def forward(self, input):
        multi_out = self.multi_head(input)
        input_ff = add_and_norm(input, multi_out)
        output_ff = self.ff(input_ff)
        return add_and_norm(input_ff, output_ff)


class TransformersDecoder(nn.Module):
    def __init__(self, masked_h, h, dim_embedding, dim_ff) -> None:
        super().__init__()
        self.multi_head = MultiHeadAttention(h, dim_embedding, False)
        self.masked_multi_head = MultiHeadAttention(h, dim_embedding, True)
        self.ff = FeedForward(dim_embedding, dim_ff)

    def forward(self, input, encoder_input):
        masked_multi_out = self.masked_multi_head(input)
        multi_in = add_and_norm(input, masked_multi_out)
        multi_out = self.multi_head(input=encoder_input, query=input)
        input_ff = add_and_norm(input, multi_out)
        output_ff = self.ff(input_ff)
        return add_and_norm(input_ff, output_ff)


class Transformers(nn.Module):
    def __init__(
        self,
        h_encoder,
        dim_embedding,
        dim_ff_encoder,
        h_masked_decoder,
        h_decoder,
        dim_ff_decoder,
    ) -> None:
        super().__init__()
        self.encoder = TransformersEncoder(h_encoder, dim_embedding, dim_ff_encoder)
        self.decode = TransformersDecoder(
            h_masked_decoder, h_decoder, dim_embedding, dim_ff_decoder
        )

    def forward(self, input_encoder, input_decoder):
        encoded = self.encoder(input_encoder)
        decoded = self.decode(input_decoder, encoded)

        return decoded
