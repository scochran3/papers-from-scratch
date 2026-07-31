import torch


def collate_fn(batch, tokenizer):

    # Determine batch length
    max_sequence_length = max([len(tokenizer.apply_chat_template(sample["messages"])) for sample in batch])

    # Loop through tokens and create input_ids, attention_mask and labels
    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []
    for sample in batch:

        # Padding for equal length samples
        original_input_ids = tokenizer.apply_chat_template(sample["messages"])
        num_pads_needed = max_sequence_length - len(original_input_ids)
        padded_input_ids = original_input_ids + [tokenizer.pad_token_id] * num_pads_needed
        batch_input_ids.append(torch.tensor(padded_input_ids))

        # Attention Mask
        attention_mask = [1]*len(original_input_ids) + [0] * num_pads_needed
        batch_attention_mask.append(torch.tensor(attention_mask))

        # Label Mask
        prompt_tokens = tokenizer.apply_chat_template(sample["messages"][:-1], add_generation_prompt=True)
        assert original_input_ids[:len(prompt_tokens)] == prompt_tokens
        labels = [-100] * len(prompt_tokens) + original_input_ids[len(prompt_tokens):] + [-100]*num_pads_needed
        batch_labels.append(torch.tensor(labels))

    return {
        "input_ids": torch.stack(batch_input_ids),
        "attention_mask": torch.stack(batch_attention_mask),
        "labels": torch.stack(batch_labels)
    }