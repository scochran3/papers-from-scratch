from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(model_name: str):
    """
    Loads the model for the given model name.

    Args:
        model_name (str): The name of the model to load.
    Returns:
        model: The loaded model.
    """
    return AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)


def load_tokenizer(model_name: str):
    """
    Loads the tokenizer for the given model name.

    Args:
        model_name (str): The name of the model to load the tokenizer for.
    Returns:
        tokenizer: The loaded tokenizer.
    """
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)