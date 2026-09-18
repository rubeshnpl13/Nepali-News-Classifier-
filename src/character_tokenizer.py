from pathlib import Path
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "train.parquet"


class CharacterTokenizer:
    def __init__(self):
        self.special_tokens = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3,
        }

        self.char_to_id = {}
        self.id_to_char = {}

    def build_vocabulary(self, texts):
        character_counts = Counter()

        for text in texts:
            character_counts.update(text)

        sorted_characters = sorted(character_counts.keys())

        self.char_to_id = dict(self.special_tokens)

        next_id = len(self.char_to_id)

        for character in sorted_characters:
            self.char_to_id[character] = next_id
            next_id += 1

        self.id_to_char = {
            token_id: character
            for character, token_id in self.char_to_id.items()
        }

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

    def decode(self, token_ids, remove_special_tokens=True):
        characters = []

        special_token_ids = set(self.special_tokens.values())

        for token_id in token_ids:
            if remove_special_tokens and token_id in special_token_ids:
                continue

            character = self.id_to_char.get(token_id, "<UNK>")
            characters.append(character)

        return "".join(characters)

    def pad_or_truncate(self, token_ids, max_length):
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]

        padding_length = max_length - len(token_ids)

        if padding_length > 0:
            pad_id = self.char_to_id["<PAD>"]
            token_ids = token_ids + [pad_id] * padding_length

        return token_ids

    @property
    def vocabulary_size(self):
        return len(self.char_to_id)


def load_local_texts():
    from datasets import load_dataset

    dataset = load_dataset(
        "parquet",
        data_files={"train": str(TRAIN_PATH)},
        split="train",
    )

    return dataset["text"]


def main():
    print("Loading local training data...")
    texts = load_local_texts()

    print(f"Number of training texts: {len(texts)}")

    tokenizer = CharacterTokenizer()
    tokenizer.build_vocabulary(texts)

    print(f"Vocabulary size: {tokenizer.vocabulary_size}")

    print("\nSpecial tokens:")
    for token, token_id in tokenizer.special_tokens.items():
        print(f"{token}: {token_id}")

    example_text = texts[0]

    print("\nOriginal text:")
    print(example_text)

    encoded = tokenizer.encode(example_text)

    print("\nEncoded token IDs:")
    print(encoded)

    decoded = tokenizer.decode(encoded)

    print("\nDecoded text:")
    print(decoded)

    print("\nDecoded text matches original:")
    print(decoded == example_text)

    print("\nExample character mappings:")
    for character in sorted(set(example_text)):
        token_id = tokenizer.char_to_id[character]
        print(repr(character), "->", token_id)

    unknown_text = "यो पाठमा नयाँ अक्षर 😀 छ।"

    unknown_encoded = tokenizer.encode(unknown_text)

    print("\nUnknown-character test:")
    print(f"Text: {unknown_text}")
    print(f"Encoded: {unknown_encoded}")
    print(f"Decoded: {tokenizer.decode(unknown_encoded)}")

    print("\nPadding and truncation test:")

    short_ids = tokenizer.encode("नेपाल")
    padded_ids = tokenizer.pad_or_truncate(short_ids, max_length=12)

    print("Short text IDs:")
    print(short_ids)

    print("Padded IDs:")
    print(padded_ids)

    print("Length:")
    print(len(padded_ids))

    long_ids = tokenizer.encode("नेपाल खेलकुद समाचार " * 10)
    truncated_ids = tokenizer.pad_or_truncate(long_ids, max_length=12)

    print("\nLong text original length:")
    print(len(long_ids))

    print("Truncated length:")
    print(len(truncated_ids))

if __name__ == "__main__":
    main()