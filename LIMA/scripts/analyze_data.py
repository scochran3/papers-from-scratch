"""Inspect the LIMA training dataset described by a config file."""


import argparse

from datasets import load_dataset
from dotenv import load_dotenv

from lima.config import load_config
from lima.paths import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")



def _create_lima_splits(dataset_name, test_size: float = 0.1, seed: int = 42) -> "DatasetDict":
    """
    Create train, validation, and test splits from the LIMA dataset.

    Args:
        dataset_name: The name of the LIMA dataset.
    Returns:
        DatasetDict: A dictionary containing the train, validation, and test splits.
    """
    # Split the dataset into train, validation, and test sets
    dataset = _load_lima_dataset(dataset_name)
    split_ds = dataset["train"].train_test_split(test_size=test_size, seed=seed)
    train_dataset = split_ds["train"]
    val_dataset = split_ds["test"]
    test_dataset = dataset["test"]

    return {"train": train_dataset, "validation": val_dataset, "test": test_dataset}

def find_token_length_distribution(model_name: str, dataset_name: str, split: str):
    """
    Find the distribution of token lengths in a dataset split.

    Args:
        dataset_name: The name of the dataset to analyze.
        split: The split of the dataset to analyze (e.g., "train", "validation", "test").

    Returns:
        List[int]: A list of token lengths for each example in the dataset split.
    """
    tokenizer = _load_tokenizer(model_name)
    dataset = _load_lima_dataset(dataset_name)[split]
    sequence_lengths = [len(tokenizer.encode(example["input"])) for example in dataset]

    metrics = {
        "min_length": min(sequence_lengths),
        "max_length": max(sequence_lengths),
        "mean_length": sum(sequence_lengths) / len(sequence_lengths),
        "median_length": sorted(sequence_lengths)[len(sequence_lengths) // 2],
        "p90": np.percentile(sequence_lengths, 90),
        "p95": np.percentile(sequence_lengths, 95),
        "p99": np.percentile(sequence_lengths, 99),
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to a YAML config.")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    config = load_config(args.config)
    model_name = config["model"]["name"]
    dataset_name = config["dataset"]["name"]

    train_token_distribution = find_token_length_distribution(
        model_name=model_name, dataset_name=dataset_name, split=args.split)
    print(f"Train token length distribution: {train_token_distribution}")




if __name__ == "__main__":
    main()
    # python scripts/analyze_data.py --config configs/qwen.yaml
