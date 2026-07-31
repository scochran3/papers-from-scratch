from functools import partial
import torch

from tqdm import tqdm

from lima.config import load_config

from lima.dataset import create_dataloader, create_dataset_splits, LimaDataset
from lima.collator import collate_fn
from lima.model import load_model, load_tokenizer

def evaluate(model, dataloader):
    """
    Runs evaluation on the validation split
    """

    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            output = model(**batch)
            running_loss += output.loss.item()
        
    return running_loss / len(dataloader)

def train(config: dict, split: str):
    """
    Training loop; reads in config to define training parameters, fetches dataset
    and data loader and runs training.

    Args:
        config (dict): read in YAML defining experiment
    """

    # Define config definitions
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = config["model"]["name"]
    epochs = config["training"]["epochs"]
    learning_rate = config["training"]["learning_rate"]
    batch_size = config["training"]["batch_size"]
    gradient_accumulation_steps = learning_rate = config["training"]["gradient_accumulation_steps"]

    # Load model
    model = load_model(model_name=model_name).to(device)
    tokenizer = load_tokenizer(model_name=model_name)

    # Define Dataset
    data = create_dataset_splits(config=config)
    train_dataset = LimaDataset(data["train"])
    eval_dataset = LimaDataset(data["validation"])

    # Define dataloaders
    train_collate_fn = partial(collate_fn, tokenizer=tokenizer)
    train_dataloader = create_dataloader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        collate_fn=train_collate_fn
    )
    eval_dataloader = create_dataloader(
        dataset=eval_dataset, 
        batch_size=batch_size, 
        collate_fn=train_collate_fn
    )

    # Run training loop
    loss_fn = torch.nn.CrossEntropyLoss()
    criterion = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    for epoch in tqdm(range(epochs), desc="Epochs", position=0):

        # Setup for training loop
        model.train()
        batch_bar = tqdm(
            train_dataloader,
            total=len(train_dataloader),
            desc = f"Epoch {epoch+1} / {epochs}",
            leave=False
        )
        running_train_loss = 0.0

        # Loop over dataloader
        for step, batch in enumerate(batch_bar, start=1):

            print (batch["input_ids"].shape)
            print (batch["input_ids"].shape)
            print (batch["input_ids"].shape)
            print (batch["input_ids"].shape)

            # Move batch to devicew
            batch = {k: v.to(device) for k,v in batch.items()}

            # Training loop
            criterion.zero_grad()
            output = model(**batch)
            loss = output.loss
            loss.backward()
            criterion.step()
            running_loss += loss.item()
        
        # Update TQDM
        batch_bar.set_postfix(lost=f"{loss.item():.4f}")

        # Run evaluation
        epoch_avg_train_loss = running_train_loss / len(train_dataloader)
        epoch_avg_eval_loss = evaluate(model=model, dataloader=eval_dataloader)

        print(f"Train Loss: {epoch_avg_train_loss:.4f} | Eval Loss: {epoch_avg_eval_loss:.4f}")



if __name__ == "__main__":
    config = load_config()
    train(
        config=config,
        split="train"
    )