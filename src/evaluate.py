from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from classifier import NepaliNewsClassifier
from news_dataset import NepaliNewsDataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

TEST_PATH = PROCESSED_DATA_DIR / "test_encoded.json"
VOCAB_PATH = PROCESSED_DATA_DIR / "character_vocab.json"
CHECKPOINT_PATH = MODEL_DIR / "best_model.pt"

BATCH_SIZE = 4
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


def build_model(device):
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

    return model, checkpoint


def create_confusion_matrix(labels, predictions, num_classes):
    matrix = [
        [0 for _ in range(num_classes)]
        for _ in range(num_classes)
    ]

    for actual, predicted in zip(labels, predictions):
        matrix[actual][predicted] += 1

    return matrix


def calculate_class_metrics(
    confusion_matrix,
    class_index,
):
    true_positive = confusion_matrix[class_index][class_index]

    false_positive = sum(
        confusion_matrix[actual][class_index]
        for actual in range(len(confusion_matrix))
        if actual != class_index
    )

    false_negative = sum(
        confusion_matrix[class_index][predicted]
        for predicted in range(len(confusion_matrix))
        if predicted != class_index
    )

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative

    if precision_denominator == 0:
        precision = 0.0
    else:
        precision = true_positive / precision_denominator

    if recall_denominator == 0:
        recall = 0.0
    else:
        recall = true_positive / recall_denominator

    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = (
            2 * precision * recall
            / (precision + recall)
        )

    support = sum(confusion_matrix[class_index])

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1_score,
        "support": support,
    }


def evaluate_model(model, data_loader, device):
    all_labels = []
    all_predictions = []
    all_probabilities = []
    misclassified_examples = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            probabilities = torch.softmax(
                logits,
                dim=-1,
            )

            predictions = torch.argmax(
                logits,
                dim=-1,
            )

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())
            all_probabilities.extend(
                probabilities.cpu().tolist()
            )

            for index in range(len(labels)):
                actual = labels[index].item()
                predicted = predictions[index].item()

                if actual != predicted:
                    misclassified_examples.append(
                        {
                            "actual": actual,
                            "predicted": predicted,
                            "probabilities": probabilities[
                                index
                            ].cpu().tolist(),
                        }
                    )

    return (
        all_labels,
        all_predictions,
        all_probabilities,
        misclassified_examples,
    )


def print_confusion_matrix(matrix):
    print("\nCONFUSION MATRIX")
    print("Rows = actual, columns = predicted\n")

    print(
        f"{'Actual/Pred':<24}"
        f"{'general':>12}"
        f"{'culture':>12}"
        f"{'sports':>12}"
    )

    for actual, row in enumerate(matrix):
        print(
            f"{LABEL_NAMES[actual]:<24}"
            f"{row[0]:>12}"
            f"{row[1]:>12}"
            f"{row[2]:>12}"
        )


def print_class_metrics(confusion_matrix):
    print("\nCLASS METRICS")
    print(
        f"{'Category':<24}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'Support':>12}"
    )

    f1_scores = []

    for class_index in range(NUM_CLASSES):
        metrics = calculate_class_metrics(
            confusion_matrix,
            class_index,
        )

        f1_scores.append(metrics["f1"])

        print(
            f"{LABEL_NAMES[class_index]:<24}"
            f"{metrics['precision']:>12.4f}"
            f"{metrics['recall']:>12.4f}"
            f"{metrics['f1']:>12.4f}"
            f"{metrics['support']:>12}"
        )

    macro_f1 = sum(f1_scores) / len(f1_scores)

    print(f"\nMacro-F1: {macro_f1:.4f}")


def print_misclassified_examples(
    misclassified_examples,
    test_dataset,
    limit=10,
):
    print("\nMISCLASSIFIED EXAMPLES")

    number_to_show = min(
        limit,
        len(misclassified_examples),
    )

    for index in range(number_to_show):
        error = misclassified_examples[index]

        print(f"\nError {index + 1}")
        print(
            f"Actual: "
            f"{LABEL_NAMES[error['actual']]}"
        )
        print(
            f"Predicted: "
            f"{LABEL_NAMES[error['predicted']]}"
        )
        print(
            "Probabilities: "
            f"{[round(value, 4) for value in error['probabilities']]}"
        )


def main():
    device = choose_device()

    print(f"Using device: {device}")

    model, checkpoint = build_model(device)

    print(
        f"Loaded checkpoint from epoch: "
        f"{checkpoint['epoch']}"
    )
    print(
        f"Checkpoint test accuracy: "
        f"{checkpoint['validation_accuracy']:.4f}"
    )

    test_dataset = NepaliNewsDataset(TEST_PATH)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    (
        labels,
        predictions,
        probabilities,
        misclassified_examples,
    ) = evaluate_model(
        model,
        test_loader,
        device,
    )

    correct = sum(
        actual == predicted
        for actual, predicted in zip(labels, predictions)
    )

    accuracy = correct / len(labels)

    print(f"\nTotal test examples: {len(labels)}")
    print(f"Correct predictions: {correct}")
    print(f"Overall accuracy: {accuracy:.4f}")

    confusion_matrix = create_confusion_matrix(
        labels,
        predictions,
        NUM_CLASSES,
    )

    print_confusion_matrix(confusion_matrix)
    print_class_metrics(confusion_matrix)

    print(
        f"\nTotal misclassified examples: "
        f"{len(misclassified_examples)}"
    )

    print_misclassified_examples(
        misclassified_examples,
        test_dataset,
        limit=10,
    )


if __name__ == "__main__":
    main()