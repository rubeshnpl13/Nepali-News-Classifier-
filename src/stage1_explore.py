from pathlib import Path
from collections import Counter

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def inspect_split(split_name, split_dataset):
    print(f"\n{'=' * 50}")
    print(f"{split_name.upper()} SPLIT")
    print(f"{'=' * 50}")

    print(f"Number of examples: {len(split_dataset)}")
    print(f"Columns: {split_dataset.column_names}")

    labels = split_dataset["label"]
    label_counts = Counter(labels)

    print("\nLabel counts:")
    for label, count in sorted(label_counts.items()):
        percentage = count / len(labels) * 100
        print(f"Label {label}: {count} examples ({percentage:.2f}%)")

    print("\nFirst three examples:")
    for index in range(min(3, len(split_dataset))):
        example = split_dataset[index]

        print(f"\nExample {index + 1}")
        print(f"Label: {example['label']}")
        print(f"Text: {example['text']}")
        print(f"Character count: {len(example['text'])}")
        print(f"Whitespace word count: {len(example['text'].split())}")


def main():
    print("Loading dataset...")

    dataset = load_dataset("mteb/NepaliNewsClassification")

    print("\nDataset structure:")
    print(dataset)

    for split_name, split_dataset in dataset.items():
        inspect_split(split_name, split_dataset)


if __name__ == "__main__":
    main()