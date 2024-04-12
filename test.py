# from datasets import Dataset, DatasetDict, load_dataset
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from torch.utils.data import DataLoader
# from transformers import DataCollatorForSeq2Seq

# def create_dataset(data_files, tokenizer, split_ratios={'train': 0.8, 'test': 0.2}):
#     # Load a dataset from a CSV file
#     dataset = load_dataset('csv', data_files={'data': data_files})

#     # Tokenization function
#     def tokenize_function(examples):
#         # Tokenize the prompts with truncation
#         model_inputs = tokenizer(examples['prompt'], truncation=True)
        
#         # Prepare labels which are the tokenized completions without padding
#         with tokenizer.as_target_tokenizer():
#             labels = tokenizer(examples['completion'], truncation=True)
        
#         # We do not pad here; padding will be handled dynamically during batch preparation in training
#         model_inputs["labels"] = labels["input_ids"]
#         return model_inputs
    
#     # Apply tokenization without padding
#     return dataset.map(tokenize_function, batched=True, remove_columns=['prompt', 'completion'])
    
#     return tokenized_datasets

# tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
# model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", device_map="auto")
# dataset = create_dataset("./data/10/data.csv", tokenizer)

# train_dataloader = DataLoader(
#     dataset["data"],
#     shuffle=True,
#     collate_fn=DataCollatorForSeq2Seq(
#         tokenizer=tokenizer, model=model, padding="longest"
#     ),
#     batch_size=8,
# )

# for step, batch in enumerate(train_dataloader):
#     print(batch)
#     if step > 3:
#         break

# # breakpoint()

# # # dataset = load_dataset('csv', data_files={'data': data_files})
# # dataset = load_dataset('csv', data_files=data_files)
# # breakpoint()

from datasets import load_dataset
from trl import SFTTrainer

dataset = load_dataset("imdb", split="train")
breakpoint()