from pathlib import Path
from collections import Counter
import json

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

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "special_tokens": self.special_tokens,
            "char_to_id": self.char_to_id,
        }

        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        path = Path(path)

        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        tokenizer = cls()
        tokenizer.special_tokens = {
            token: int(token_id)
            for token, token_id in data["special_tokens"].items()
        }
        tokenizer.char_to_id = {
            character: int(token_id)
            for character, token_id in data["char_to_id"].items()
        }
        tokenizer.id_to_char = {
            token_id: character
            for character, token_id in tokenizer.char_to_id.items()
        }

        return tokenizer

    def attention_mask(self, token_ids):
        pad_id = self.char_to_id["<PAD>"]

        return [
            0 if token_id == pad_id else 1
            for token_id in token_ids
        ]

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

    vocabulary_path = (
        PROJECT_ROOT / "data" / "processed" / "character_vocab.json"
    )

    tokenizer.save(vocabulary_path)

    print(f"Vocabulary saved to: {vocabulary_path}")

    loaded_tokenizer = CharacterTokenizer.load(vocabulary_path)

    print(
        "Loaded vocabulary size:",
        loaded_tokenizer.vocabulary_size,
    )

    example_text = texts[0]

    encoded = loaded_tokenizer.encode(example_text)
    decoded = loaded_tokenizer.decode(encoded)

    print("\nOriginal text:")
    print(example_text)

    print("\nEncoded length:")
    print(len(encoded))

    print("\nDecoded text matches original:")
    print(decoded == example_text)

    print("\nPadding test:")
    padded_ids = loaded_tokenizer.pad_or_truncate(
        encoded,
        max_length=32,
    )
    print("Token IDs:", padded_ids)
    print("Length:", len(padded_ids))

    print("\nAttention-mask test:")
    mask = loaded_tokenizer.attention_mask(padded_ids)
    print("Attention mask:", mask)
    print("Mask length:", len(mask))

    print("\nReload test:")
    print(
        loaded_tokenizer.decode(
            loaded_tokenizer.encode("नेपाल खेलकुद")
        )
    )



if __name__ == "__main__":
    main()