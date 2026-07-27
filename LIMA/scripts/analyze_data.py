"""Inspect the LIMA training dataset described by a config file."""

import argparse

from datasets import load_dataset

from lima.config import load_config


def load_lima_dataset(dataset_name: str):
    """Load a dataset from the Hugging Face datasets library.

    Args:
        dataset_name: The name of the dataset to load.

    Returns:
        Dataset: The loaded dataset.
    """
    return load_dataset(dataset_name)

def create_lima_splits(dataset: "DatasetDict", test_size: float = 0.1, seed: int = 42) -> "DatasetDict":
    """
    Create train, validation, and test splits from the LIMA dataset.

    Args:
        dataset: The LIMA dataset.
    Returns:
        DatasetDict: A dictionary containing the train, validation, and test splits.
    """
    # Split the dataset into train, validation, and test sets
    split_ds = dataset["train"].train_test_split(test_size=test_size, seed=seed)
    train_dataset = split_ds["train"]
    val_dataset = split_ds["test"]
    test_dataset = dataset["test"]

    return {"train": train_dataset, "validation": val_dataset, "test": test_dataset}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to a YAML config.")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    config = load_config(args.config)
    print(config)

    dataset = load_lima_dataset(config["dataset"]["name"], split=args.split)
    print(dataset)


if __name__ == "__main__":
    main()
