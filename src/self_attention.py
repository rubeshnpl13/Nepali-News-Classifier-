from pathlib import Path
import json
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from news_dataset import NepaliNewsDataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_PATH = PROCESSED_DATA_DIR / "train_encoded.json"
VOCAB_PATH = PROCESSED_DATA_DIR / "character_vocab.json"

BATCH_SIZE = 2
MAX_LENGTH = 320
EMBEDDING_DIM = 64


class TokenAndPositionEmbedding(nn.Module):
    def __init__(
        self,
        vocabulary_size,
        max_length,
        embedding_dim,
        padding_idx=0,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        self.position_embedding = nn.Embedding(
            num_embeddings=max_length,
            embedding_dim=embedding_dim,
        )

    def forward(self, input_ids):
        batch_size, sequence_length = input_ids.shape

        positions = torch.arange(
            sequence_length,
            device=input_ids.device,
        )

        positions = positions.unsqueeze(0)

        token_vectors = self.token_embedding(input_ids)
        position_vectors = self.position_embedding(positions)

        return token_vectors + position_vectors


class SelfAttention(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.query_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        self.key_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        self.value_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        self.output_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

    def forward(self, x, attention_mask=None):
        queries = self.query_projection(x)
        keys = self.key_projection(x)
        values = self.value_projection(x)

        scores = torch.matmul(
            queries,
            keys.transpose(-2, -1),
        )

        scores = scores / math.sqrt(queries.size(-1))

        if attention_mask is not None:
            key_mask = attention_mask.unsqueeze(1)
            scores = scores.masked_fill(
                key_mask == 0,
                torch.finfo(scores.dtype).min,
            )

        attention_weights = torch.softmax(
            scores,
            dim=-1,
        )

        context = torch.matmul(
            attention_weights,
            values,
        )

        output = self.output_projection(context)

        if attention_mask is not None:
            output = output * attention_mask.unsqueeze(-1)

        return output, attention_weights


def load_vocabulary_size():
    with VOCAB_PATH.open("r", encoding="utf-8") as file:
        vocabulary_data = json.load(file)

    return len(vocabulary_data["char_to_id"])


def main():
    vocabulary_size = load_vocabulary_size()

    embedding_layer = TokenAndPositionEmbedding(
        vocabulary_size=vocabulary_size,
        max_length=MAX_LENGTH,
        embedding_dim=EMBEDDING_DIM,
    )

    attention_layer = SelfAttention(
        embedding_dim=EMBEDDING_DIM,
    )

    dataset = NepaliNewsDataset(TRAIN_PATH)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    batch = next(iter(loader))

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    x = embedding_layer(input_ids)

    output, attention_weights = attention_layer(
        x,
        attention_mask=attention_mask,
    )

    print("Input IDs shape:")
    print(input_ids.shape)

    print("\nEmbedding output shape:")
    print(x.shape)

    print("\nAttention output shape:")
    print(output.shape)

    print("\nAttention weights shape:")
    print(attention_weights.shape)

    print("\nAttention weights for the first query token:")
    print(attention_weights[0, 0, :10])

    print("\nAttention weights sum:")
    print(attention_weights[0, 0].sum())

    print("\nAttention weights over padding:")
    print(attention_weights[0, 0, 190:])

    print("\nAll output values are finite:")
    print(torch.isfinite(output).all())


if __name__ == "__main__":
    main()