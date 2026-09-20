from pathlib import Path
import json

import torch

from classifier import NepaliNewsClassifier


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

VOCAB_PATH = PROCESSED_DATA_DIR / "character_vocab.json"
CHECKPOINT_PATH = MODEL_DIR / "best_model.pt"

MAX_LENGTH = 320
EMBEDDING_DIM = 64
NUM_HEADS = 4
FFN_DIM = 128
NUM_CLASSES = 3

LABEL_NAMES = {
    0: "general_society",
    1: "culture_entertainment",
    2: "sports",
}


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

    def encode(self, text):
        token_ids = [
            self.char_to_id["<BOS>"]
        ]

        for character in text:
            token_id = self.char_to_id.get(
                character,
                self.char_to_id["<UNK>"],
            )
            token_ids.append(token_id)

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


def choose_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_vocabulary_size():
    with VOCAB_PATH.open("r", encoding="utf-8") as file:
        vocabulary_data = json.load(file)

    return len(vocabulary_data["char_to_id"])


def load_model(device):
    vocabulary_size = load_vocabulary_size()

    model = NepaliNewsClassifier(
        vocabulary_size=vocabulary_size,
        max_length=MAX_LENGTH,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        ffn_dim=FFN_DIM,
        num_classes=NUM_CLASSES,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def predict_text(text, model, tokenizer, device):
    token_ids = tokenizer.encode(text)
    token_ids = tokenizer.pad_or_truncate(
        token_ids,
        MAX_LENGTH,
    )

    attention_mask = tokenizer.attention_mask(token_ids)

    input_ids = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device,
    )

    attention_mask = torch.tensor(
        [attention_mask],
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        logits, _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )[0]

        predicted_id = int(torch.argmax(probabilities))

    return {
        "label_id": predicted_id,
        "label_name": LABEL_NAMES[predicted_id],
        "probabilities": {
            LABEL_NAMES[class_id]: float(
                probabilities[class_id]
            )
            for class_id in range(NUM_CLASSES)
        },
        "token_count": len(tokenizer.encode(text)),
        "unknown_character_count": sum(
            character not in tokenizer.char_to_id
            for character in text
        ),
    }


def main():
    device = choose_device()

    tokenizer = CharacterTokenizer.load(VOCAB_PATH)
    model = load_model(device)

    text = "नेपालले नयाँ खेलकुद प्रतियोगिता आयोजना गर्ने भएको छ।"

    result = predict_text(
        text=text,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    print("Text:")
    print(text)

    print("\nPrediction:")
    print(result["label_name"])

    print("\nProbabilities:")
    for label, probability in result["probabilities"].items():
        print(f"{label}: {probability:.4f}")

    print("\nToken count:")
    print(result["token_count"])

    print("\nUnknown characters:")
    print(result["unknown_character_count"])


if __name__ == "__main__":
    main()