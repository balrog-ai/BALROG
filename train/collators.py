import torch
import numpy as np

from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from collections.abc import Mapping

def pad_without_fast_tokenizer_warning(tokenizer, *pad_args, **pad_kwargs):
    if not hasattr(tokenizer, "deprecation_warnings"):
        return tokenizer.pad(*pad_args, **pad_kwargs)
    warning_state = tokenizer.deprecation_warnings.get("Asking-to-pad-a-fast-tokenizer", False)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    try:
        padded = tokenizer.pad(*pad_args, **pad_kwargs)
    finally:
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = warning_state
    return padded

def _torch_collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]
    length_of_first = examples[0].size(0)
    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return torch.stack(examples, dim=0)
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )
    max_length = max(x.size(0) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result


class DataCollatorMixin:
    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        if return_tensors == "tf":
            return self.tf_call(features)
        elif return_tensors == "pt":
            return self.torch_call(features)
        elif return_tensors == "np":
            return self.numpy_call(features)
        else:
            raise ValueError(f"Framework '{return_tensors}' not recognized!")
    
class DataCollatorForLanguageModeling(DataCollatorMixin):
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = 'left'

    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )
    
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        if isinstance(examples[0], Mapping):
            batch = pad_without_fast_tokenizer_warning(
                self.tokenizer, examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of
            )
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        labels = batch["input_ids"].clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch


############################################
############ VLMDataCollator ###############
############################################

def pad_sequence_left(sequences, batch_first=False, padding_value=-1):
    reversed_sequences = [seq.flip(0) for seq in sequences]
    padded_reversed = torch.nn.utils.rnn.pad_sequence(reversed_sequences, batch_first=batch_first, padding_value=padding_value)
    if batch_first:
        padded = padded_reversed.flip(1)
    else:
        padded = padded_reversed.flip(0)
    return padded

IGNORE_INDEX = -100

class VLMDataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.tokenizer.padding_side = 'left'

    def __call__(self, examples):
        samples = []
        for example in examples:
            prompt = example["prompt"]
            action = example["action"]
            image = example["image"]
            messages = [
                {
                    "role": "user",
                    "content": prompt
                },
                {
                    "role": "assistant",
                    "content": action
                }
            ]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            text += self.tokenizer.eos_token
            sample = self.processor(text, [image], return_tensors="pt")
            labels = sample["input_ids"].clone()
            labels[labels < 0] = -100 
            sample["labels"] = labels
            samples.append(sample)
            
        input_ids, labels = tuple([instance[key][0] for instance in samples] 
                                   for key in ("input_ids", "labels"))

        input_ids = pad_sequence_left(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence_left(labels, batch_first=True, padding_value=IGNORE_INDEX)

        pixel_values = torch.stack([sample["pixel_values"][0] for sample in samples], dim=0)
        image_sizes = torch.stack([sample["image_sizes"][0] for sample in samples], dim=0)
        batch = dict( 
             input_ids=input_ids, 
             labels=labels, 
             pixel_values=pixel_values, 
             image_sizes=image_sizes, 
             attention_mask=input_ids.ne(self.tokenizer.pad_token_id), 
         ) 
        return batch