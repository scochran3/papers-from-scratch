from datasets import load_dataset
from dotenv import load_dotenv
from torch.utils.data import Dataset, DataLoader

from lima.config import load_config
from lima.paths import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")

def create_dataset_splits(config: dict, val_size: float = 0.1):
    dataset = load_dataset(config["dataset"]["name"], trust_remote_code=True)
    split = dataset["train"].train_test_split(test_size=val_size, seed=42)
    train_dataset = split["train"]
    val_dataset = split["test"]
    test_dataset = dataset["test"]

    return {
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    }


class LimaDataset(Dataset):

    def __init__(self, dataset: "DatasetDict"):

        # Explicitly limit to single turn conversations for now, as we don't have a good way to handle multi-turn conversations yet.
        self.dataset = dataset.filter(lambda ex: len(ex["conversations"]) == 2)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self._format_sample(self.dataset[idx])
    
    def _format_sample(self, sample: dict) -> dict:
        """
        Takes a single sample from the dataset and formats it into the standard
        input format for LIMA.

        Args:
            sample (dict): A single sample from the dataset.
        
        Returns:
            list[dict] -> list of dictionaries in messages form
        """
        conversation = sample["conversations"]
        return {
            "messages": [
                {"role": "user", "content": conversation[0]},
                {"role": "assistant", "content": conversation[1]}
            ]
        }


def create_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True):
    """
    Creates a DataLoader for the given dataset.

    Args:
        dataset (Dataset): The dataset to create the DataLoader for.
        batch_size (int): The batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the dataset. Default is True.
    
    Returns:
        DataLoader: The created DataLoader.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
   
if __name__ == "__main__":
    config = load_config()
    full_dataset = create_dataset_splits(config=config, val_size=0.1)
    lima_dataset = LimaDataset(dataset=full_dataset["train"])