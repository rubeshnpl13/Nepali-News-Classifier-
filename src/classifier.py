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

BATCH_SIZE = 8
MAX_LENGTH = 320
EMBEDDING_DIM = 64
NUM_HEADS = 4
FFN_DIM = 128
NUM_CLASSES = 3


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
        _, sequence_length = input_ids.shape

        positions = torch.arange(
            sequence_length,
            device=input_ids.device,
        ).unsqueeze(0)

        token_vectors = self.token_embedding(input_ids)
        position_vectors = self.position_embedding(positions)

        return token_vectors + position_vectors


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads"
            )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

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

    def split_into_heads(self, x):
        batch_size, sequence_length, _ = x.shape

        x = x.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )

        return x.transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, sequence_length, _ = x.shape

        x = x.transpose(1, 2)
        x = x.contiguous().view(
            batch_size,
            sequence_length,
            self.embedding_dim,
        )

        return x

    def forward(self, x, attention_mask=None):
        queries = self.query_projection(x)
        keys = self.key_projection(x)
        values = self.value_projection(x)

        queries = self.split_into_heads(queries)
        keys = self.split_into_heads(keys)
        values = self.split_into_heads(values)

        scores = torch.matmul(
            queries,
            keys.transpose(-2, -1),
        )

        scores = scores / math.sqrt(self.head_dim)

        if attention_mask is not None:
            key_mask = attention_mask[:, None, None, :]

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

        context = self.combine_heads(context)
        output = self.output_projection(context)

        if attention_mask is not None:
            output = output * attention_mask.unsqueeze(-1)

        return output, attention_weights


class FeedForwardNetwork(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()

        self.first_linear = nn.Linear(
            embedding_dim,
            hidden_dim,
        )

        self.activation = nn.GELU()

        self.second_linear = nn.Linear(
            hidden_dim,
            embedding_dim,
        )

    def forward(self, x):
        x = self.first_linear(x)
        x = self.activation(x)
        x = self.second_linear(x)

        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_heads,
        ffn_dim,
        dropout=0.1,
    ):
        super().__init__()

        self.attention = MultiHeadSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
        )

        self.feed_forward = FeedForwardNetwork(
            embedding_dim=embedding_dim,
            hidden_dim=ffn_dim,
        )

        self.attention_norm = nn.LayerNorm(embedding_dim)
        self.feed_forward_norm = nn.LayerNorm(embedding_dim)

        self.attention_dropout = nn.Dropout(dropout)
        self.feed_forward_dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        attention_output, attention_weights = self.attention(
            x,
            attention_mask=attention_mask,
        )

        x = x + self.attention_dropout(attention_output)
        x = self.attention_norm(x)

        feed_forward_output = self.feed_forward(x)

        x = x + self.feed_forward_dropout(
            feed_forward_output
        )
        x = self.feed_forward_norm(x)

        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)

        return x, attention_weights


class NepaliNewsClassifier(nn.Module):
    def __init__(
        self,
        vocabulary_size,
        max_length,
        embedding_dim,
        num_heads,
        ffn_dim,
        num_classes,
    ):
        super().__init__()

        self.embedding = TokenAndPositionEmbedding(
            vocabulary_size=vocabulary_size,
            max_length=max_length,
            embedding_dim=embedding_dim,
        )

        self.encoder = TransformerEncoderBlock(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
        )

        self.classifier = nn.Linear(
            embedding_dim,
            num_classes,
        )

    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids)

        x, attention_weights = self.encoder(
            x,
            attention_mask=attention_mask,
        )

        bos_representation = x[:, 0, :]

        logits = self.classifier(bos_representation)

        return logits, attention_weights


def load_vocabulary_size():
    with VOCAB_PATH.open("r", encoding="utf-8") as file:
        vocabulary_data = json.load(file)

    return len(vocabulary_data["char_to_id"])


def main():
    vocabulary_size = load_vocabulary_size()

    model = NepaliNewsClassifier(
        vocabulary_size=vocabulary_size,
        max_length=MAX_LENGTH,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        ffn_dim=FFN_DIM,
        num_classes=NUM_CLASSES,
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
    labels = batch["label"]

    logits, attention_weights = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    predictions = torch.argmax(logits, dim=-1)

    print("Configuration:")
    print(f"Vocabulary size: {vocabulary_size}")
    print(f"Maximum sequence length: {MAX_LENGTH}")
    print(f"Embedding dimension: {EMBEDDING_DIM}")
    print(f"Number of heads: {NUM_HEADS}")
    print(f"Feed-forward dimension: {FFN_DIM}")
    print(f"Number of classes: {NUM_CLASSES}")

    print("\nInput IDs shape:")
    print(input_ids.shape)

    print("\nLogits shape:")
    print(logits.shape)

    print("\nAttention weights shape:")
    print(attention_weights.shape)

    print("\nLogits:")
    print(logits)

    print("\nLabels:")
    print(labels)

    print("\nPredictions before training:")
    print(predictions)

    print("\nNumber of trainable parameters:")
    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(total_parameters)

    print("\nLogits are finite:")
    print(torch.isfinite(logits).all())


if __name__ == "__main__":
    main()