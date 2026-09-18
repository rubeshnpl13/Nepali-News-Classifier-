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
EMBEDDING_DIM = 64


def load_vocabulary_size():
    with VOCAB_PATH.open("r", encoding="utf-8") as file:
        vocabulary_data = json.load(file)

    return len(vocabulary_data["char_to_id"])


def main():
    vocabulary_size = load_vocabulary_size()

    print(f"Vocabulary size: {vocabulary_size}")
    print(f"Embedding dimension: {EMBEDDING_DIM}")

    embedding = nn.Embedding(
        num_embeddings=vocabulary_size,
        embedding_dim=EMBEDDING_DIM,
        padding_idx=0,
    )

    print("\nEmbedding weight shape:")
    print(embedding.weight.shape)

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

    embedded = embedding(input_ids)

    print("\nEmbedded tensor shape:")
    print(embedded.shape)

    print("\nFirst token ID:")
    print(input_ids[0, 0])

    print("\nVector for the first token:")
    print(embedded[0, 0])

    print("\nFirst article, first five token IDs:")
    print(input_ids[0, :5])

    print("\nFirst article, first five embedding vectors:")
    print(embedded[0, :5])

    print("\nEmbedding tensor data type:")
    print(embedded.dtype)

    print("\nPadding vector:")
    print(embedding.weight[0])

    print("\nPadding vector is all zeros:")
    print(torch.all(embedding.weight[0] == 0))


if __name__ == "__main__":
    main()