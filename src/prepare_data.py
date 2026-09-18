from pathlib import Path
import json
import statistics

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_PATH = RAW_DATA_DIR / "train.parquet"
TEST_PATH = RAW_DATA_DIR / "test.parquet"
VOCAB_PATH = PROCESSED_DATA_DIR / "character_vocab.json"

TRAIN_OUTPUT_PATH = PROCESSED_DATA_DIR / "train_encoded.json"
TEST_OUTPUT_PATH = PROCESSED_DATA_DIR / "test_encoded.json"

MAX_LENGTH = 320

class CharacterTokenizer:
    def __init__(self, special_tokens, char_to_id):
        self.special_tokens = special_tokens
        self.char_to_id = char_to_id

    @classmethod
    def load(cls, path):
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        special_tokens = {
            token: int(token_id)
            for token, token_id in data["special_tokens"].items()
        }

        char_to_id = {
            character: int(token_id)
            for character, token_id in data["char_to_id"].items()
        }

        return cls(special_tokens, char_to_id)

    def encode(self, text, add_special_tokens=True):
        token_ids = []

        if add_special_tokens:
            token_ids.append(self.char_to_id["<BOS>"])

        for character in text:
            token_id = self.char_to_id.get(
                character,
                self.char_to_id["<UNK>"],
            )
            token_ids.append(token_id)

        if add_special_tokens:
            token_ids.append(self.char_to_id["<EOS>"])

        return token_ids

    def pad_or_truncate(self, token_ids, max_length):
        pad_id = self.char_to_id["<PAD>"]
        bos_id = self.char_to_id["<BOS>"]
        eos_id = self.char_to_id["<EOS>"]

        if len(token_ids) <= max_length:
            padding_length = max_length - len(token_ids)

            return token_ids + [pad_id] * padding_length

        truncated_ids = token_ids[:max_length]

        if truncated_ids[0] == bos_id:
            truncated_ids[-1] = eos_id

        return truncated_ids

    def attention_mask(self, token_ids):
        pad_id = self.char_to_id["<PAD>"]

        return [
            0 if token_id == pad_id else 1
            for token_id in token_ids
        ]

    @property
    def vocabulary_size(self):
        return len(self.char_to_id)


def load_local_split(path):
    return load_dataset(
        "parquet",
        data_files={"data": str(path)},
        split="data",
    )


def calculate_percentile(values, percentile):
    sorted_values = sorted(values)

    position = (len(sorted_values) - 1) * percentile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)

    weight = position - lower_index

    return (
        sorted_values[lower_index] * (1 - weight)
        + sorted_values[upper_index] * weight
    )


def print_length_statistics(split_name, lengths):
    print(f"\n{split_name.upper()} TOKEN LENGTHS")
    print(f"Number of examples: {len(lengths)}")
    print(f"Minimum: {min(lengths)}")
    print(f"Maximum: {max(lengths)}")
    print(f"Mean: {statistics.mean(lengths):.2f}")
    print(f"Median: {statistics.median(lengths):.2f}")
    print(f"90th percentile: {calculate_percentile(lengths, 0.90):.2f}")
    print(f"95th percentile: {calculate_percentile(lengths, 0.95):.2f}")
    print(f"99th percentile: {calculate_percentile(lengths, 0.99):.2f}")


def truncation_percentage(lengths, max_length):
    truncated_count = sum(
        length > max_length
        for length in lengths
    )

    return truncated_count / len(lengths) * 100

def encode_split(dataset, tokenizer, max_length):
    encoded_examples = []

    for text, label in zip(
        dataset["text"],
        dataset["label"],
    ):
        token_ids = tokenizer.encode(text)
        token_ids = tokenizer.pad_or_truncate(
            token_ids,
            max_length,
        )
        attention_mask = tokenizer.attention_mask(token_ids)

        encoded_examples.append(
            {
                "input_ids": token_ids,
                "attention_mask": attention_mask,
                "label": int(label),
            }
        )

    return encoded_examples

def save_json(data, path):
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file)

    print(f"Saved {len(data)} examples to {path}")

def main():
    print("Loading tokenizer...")
    tokenizer = CharacterTokenizer.load(VOCAB_PATH)

    print(f"Vocabulary size: {tokenizer.vocabulary_size}")
    print(f"Maximum sequence length: {MAX_LENGTH}")

    print("\nLoading local datasets...")
    train_dataset = load_local_split(TRAIN_PATH)
    test_dataset = load_local_split(TEST_PATH)

    train_lengths = [
        len(tokenizer.encode(text))
        for text in train_dataset["text"]
    ]

    test_lengths = [
        len(tokenizer.encode(text))
        for text in test_dataset["text"]
    ]

    print_length_statistics("Train", train_lengths)
    print_length_statistics("Test", test_lengths)

    print("\nEncoding training data...")
    encoded_train = encode_split(
        train_dataset,
        tokenizer,
        MAX_LENGTH,
    )

    print("Encoding test data...")
    encoded_test = encode_split(
        test_dataset,
        tokenizer,
        MAX_LENGTH,
    )

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    save_json(encoded_train, TRAIN_OUTPUT_PATH)
    save_json(encoded_test, TEST_OUTPUT_PATH)

    print("\nChecking one encoded training example:")
    print(encoded_train[0])

    print("\nChecking lengths:")
    print(f"input_ids: {len(encoded_train[0]['input_ids'])}")
    print(
        "attention_mask:",
        len(encoded_train[0]["attention_mask"]),
    )


if __name__ == "__main__":
    main()