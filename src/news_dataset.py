from pathlib import Path
import json

import torch
from torch.utils.data import Dataset, DataLoader


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_PATH = PROCESSED_DATA_DIR / "train_encoded.json"
TEST_PATH = PROCESSED_DATA_DIR / "test_encoded.json"

BATCH_SIZE = 8


class NepaliNewsDataset(Dataset):
    def __init__(self, path):
        with path.open("r", encoding="utf-8") as file:
            examples = json.load(file)

        self.input_ids = torch.tensor(
            [example["input_ids"] for example in examples],
            dtype=torch.long,
        )

        self.attention_masks = torch.tensor(
            [example["attention_mask"] for example in examples],
            dtype=torch.long,
        )

        self.labels = torch.tensor(
            [example["label"] for example in examples],
            dtype=torch.long,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_masks[index],
            "label": self.labels[index],
        }


def main():
    print("Loading datasets...")

    train_dataset = NepaliNewsDataset(TRAIN_PATH)
    test_dataset = NepaliNewsDataset(TEST_PATH)

    print(f"Training examples: {len(train_dataset)}")
    print(f"Test examples: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    batch = next(iter(train_loader))

    print("\nOne batch:")
    print(f"input_ids shape: {batch['input_ids'].shape}")
    print(
        "attention_mask shape:",
        batch["attention_mask"].shape,
    )
    print(f"labels shape: {batch['label'].shape}")

    print("\nBatch data types:")
    print(f"input_ids: {batch['input_ids'].dtype}")
    print(
        f"attention_mask: {batch['attention_mask'].dtype}"
    )
    print(f"labels: {batch['label'].dtype}")

    print("\nFirst batch labels:")
    print(batch["label"])

    print("\nFirst example in batch:")
    print(batch["input_ids"][0])

    print("\nFirst example attention mask:")
    print(batch["attention_mask"][0])


if __name__ == "__main__":
    main()