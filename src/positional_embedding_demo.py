from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from news_dataset import NepaliNewsDataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_PATH = PROCESSED_DATA_DIR / "train_encoded.json"
VOCAB_PATH = PROCESSED_DATA_DIR / "character_vocab.json"

BATCH_SIZE = 8
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

        combined_vectors = token_vectors + position_vectors

        return combined_vectors


def load_vocabulary_size():
    with VOCAB_PATH.open("r", encoding="utf-8") as file:
        vocabulary_data = json.load(file)

    return len(vocabulary_data["char_to_id"])


def main():
    vocabulary_size = load_vocabulary_size()

    model = TokenAndPositionEmbedding(
        vocabulary_size=vocabulary_size,
        max_length=MAX_LENGTH,
        embedding_dim=EMBEDDING_DIM,
    )

    print(f"Vocabulary size: {vocabulary_size}")
    print(f"Maximum sequence length: {MAX_LENGTH}")
    print(f"Embedding dimension: {EMBEDDING_DIM}")

    print("\nToken embedding weight shape:")
    print(model.token_embedding.weight.shape)

    print("\nPosition embedding weight shape:")
    print(model.position_embedding.weight.shape)

    train_dataset = NepaliNewsDataset(TRAIN_PATH)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    batch = next(iter(train_loader))
    input_ids = batch["input_ids"]

    print("\nInput IDs shape:")
    print(input_ids.shape)

    output = model(input_ids)

    print("\nCombined output shape:")
    print(output.shape)

    print("\nPosition IDs:")
    positions = torch.arange(
        input_ids.shape[1],
        device=input_ids.device,
    )
    print(positions[:10])

    print("\nFirst token embedding:")
    print(model.token_embedding(input_ids)[0, 0, :5])

    print("\nFirst position embedding:")
    print(model.position_embedding(positions[:1])[0, 0, :5])

    print("\nCombined first vector:")
    print(output[0, 0, :5])

    print("\nChecking addition:")
    expected = (
        model.token_embedding(input_ids)[0, 0]
        + model.position_embedding(positions[:1])[0, 0]
    )

    print(torch.allclose(output[0, 0], expected))


if __name__ == "__main__":
    main()