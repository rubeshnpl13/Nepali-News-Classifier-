from pathlib import Path
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from classifier import NepaliNewsClassifier
from news_dataset import NepaliNewsDataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "models"

TRAIN_PATH = PROCESSED_DATA_DIR / "train_encoded.json"
TEST_PATH = PROCESSED_DATA_DIR / "test_encoded.json"
VOCAB_PATH = PROCESSED_DATA_DIR / "character_vocab.json"

BATCH_SIZE = 4
MAX_LENGTH = 320
EMBEDDING_DIM = 64
NUM_HEADS = 4
FFN_DIM = 128
NUM_CLASSES = 3

NUM_EPOCHS = 5
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001


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


def calculate_accuracy(logits, labels):
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == labels).sum().item()

    return correct, labels.size(0)


def train_one_epoch(
    model,
    data_loader,
    loss_function,
    optimizer,
    device,
    epoch_number,
):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch_number, batch in enumerate(data_loader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        logits, _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        loss = loss_function(logits, labels)

        loss.backward()
        optimizer.step()

        correct, number_of_examples = calculate_accuracy(
            logits,
            labels,
        )

        total_loss += loss.item() * number_of_examples
        total_correct += correct
        total_examples += number_of_examples

        if batch_number % 50 == 0:
            average_loss = total_loss / total_examples
            accuracy = total_correct / total_examples

            print(
                f"Epoch {epoch_number} | "
                f"Batch {batch_number}/{len(data_loader)} | "
                f"Loss: {average_loss:.4f} | "
                f"Accuracy: {accuracy:.4f}"
            )

    epoch_loss = total_loss / total_examples
    epoch_accuracy = total_correct / total_examples

    return epoch_loss, epoch_accuracy


def evaluate(
    model,
    data_loader,
    loss_function,
    device,
):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            loss = loss_function(logits, labels)

            correct, number_of_examples = calculate_accuracy(
                logits,
                labels,
            )

            total_loss += loss.item() * number_of_examples
            total_correct += correct
            total_examples += number_of_examples

    average_loss = total_loss / total_examples
    accuracy = total_correct / total_examples

    return average_loss, accuracy


def save_checkpoint(
    model,
    optimizer,
    epoch,
    validation_loss,
    validation_accuracy,
    path,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_loss": validation_loss,
        "validation_accuracy": validation_accuracy,
        "configuration": {
            "max_length": MAX_LENGTH,
            "embedding_dim": EMBEDDING_DIM,
            "num_heads": NUM_HEADS,
            "ffn_dim": FFN_DIM,
            "num_classes": NUM_CLASSES,
        },
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def main():
    device = choose_device()

    print(f"Using device: {device}")

    vocabulary_size = load_vocabulary_size()

    print(f"Vocabulary size: {vocabulary_size}")

    train_dataset = NepaliNewsDataset(TRAIN_PATH)
    test_dataset = NepaliNewsDataset(TEST_PATH)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = NepaliNewsClassifier(
        vocabulary_size=vocabulary_size,
        max_length=MAX_LENGTH,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        ffn_dim=FFN_DIM,
        num_classes=NUM_CLASSES,
    )

    model.to(device)

    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print(f"Training examples: {len(train_dataset)}")
    print(f"Test examples: {len(test_dataset)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Number of epochs: {NUM_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")

    best_accuracy = 0.0
    best_checkpoint_path = CHECKPOINT_DIR / "best_model.pt"

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            data_loader=train_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            device=device,
            epoch_number=epoch,
        )

        test_loss, test_accuracy = evaluate(
            model=model,
            data_loader=test_loader,
            loss_function=loss_function,
            device=device,
        )

        elapsed_time = time.time() - start_time

        print(f"\nEpoch {epoch}/{NUM_EPOCHS} complete")
        print(f"Training loss: {train_loss:.4f}")
        print(f"Training accuracy: {train_accuracy:.4f}")
        print(f"Test loss: {test_loss:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Time: {elapsed_time:.2f} seconds")

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                validation_loss=test_loss,
                validation_accuracy=test_accuracy,
                path=best_checkpoint_path,
            )

            print(f"Saved best model to: {best_checkpoint_path}")

    print("\nTraining finished.")
    print(f"Best test accuracy: {best_accuracy:.4f}")


if __name__ == "__main__":
    main()