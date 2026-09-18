from pathlib import Path
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

print("Loading dataset...")
dataset = load_dataset("mteb/NepaliNewsClassification")

print("\nDataset structure:")
print(dataset)

for split_name, split_dataset in dataset.items():
    print(f"\n{split_name.upper()} split")
    print(f"Number of examples: {len(split_dataset)}")
    print(f"Columns: {split_dataset.column_names}")

    print("\nFirst example:")
    print(split_dataset[0])

    output_path = RAW_DATA_DIR / f"{split_name}.parquet"
    split_dataset.to_parquet(str(output_path))
    print(f"Saved to: {output_path}")