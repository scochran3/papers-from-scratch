from transformers import AutoTokenizer

def load_tokenizer(model_name: str):
    """
    Loads the tokenizer for the given model name.

    Args:
        model_name (str): The name of the model to load the tokenizer for.
    Returns:
        tokenizer: The loaded tokenizer.
    """
    return AutoTokenizer.from_pretrained(model_name)


def collate_fn(batch):

    # Encode the messages in the batch using the tokenizer
    tokenizer.encode()
