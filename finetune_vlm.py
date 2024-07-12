import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, TrainingArguments, Trainer
from datasets import load_dataset, load_from_disk
from accelerate import PartialState
import logging
import wandb
import torch
from omegaconf import OmegaConf
from tqdm.rich import tqdm
from trl import (
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    get_peft_config,
)
from trl.commands.cli_utils import SFTScriptArguments, TrlParser

tqdm.pandas()

def load(config):
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, use_fast=True, trust_remote_code=True, use_cache = False)
    tokenizer.padding_side = 'left'
    processor = AutoProcessor.from_pretrained(config.model_id, rust_remote_code=True, use_cache = False)
    processor.tokenizer = tokenizer
    
    if config.DDP:
        device_string = PartialState().process_index
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            device_map={'': device_string},
            trust_remote_code=True,
            _attn_implementation='flash_attention_2',
            use_cache = False,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            device_map="auto",
            trust_remote_code=True,
            _attn_implementation='flash_attention_2',
            use_cache = False,
        )
    return model, tokenizer, processor


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
            labels[labels <0] = -100 
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

def main(config):

    config = OmegaConf.load(config_file)    
    
    print("FINETUNING!")
    
    ################
    # Wandb and logging
    ################
    
    wandb.init(
        project="nle_finetune_test", config=OmegaConf.to_container(config, resolve=True)
    )

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.getLogger().addHandler(logging.StreamHandler())

    if config.report_to == "wandb":

        class WandbLoggingHandler(logging.Handler):
            def emit(self, record):
                log_entry = self.format(record)
                wandb.log({"log": log_entry})

        wandb_handler = WandbLoggingHandler()
        wandb_handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(wandb_handler)
        
    ################
    # Model, Tokenizer & Processor
    ################
    logging.info("Loading model and tokenizer")

    model, tokenizer, processor = load(config)

    ################
    # Create a data collator to encode text and image pairs
    ################

    data_collator = VLMDataCollator(processor)

    ################
    # Dataset
    ################
    logging.info("Loading dataset")
    dataset = load_from_disk(config.dataset_path)

    ################
    # Training
    ################

    training_args = SFTConfig(
        output_dir=os.path.join(config.output_dir),
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        learning_rate=config.learning_rate,
        fp16=config.fp16,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        num_train_epochs=config.num_train_epochs,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        lr_scheduler_type=config.lr_scheduler_type,
        report_to=config.report_to,
        remove_unused_columns=False
    )

    logging.info("Setting up trainer")
    tokenizer.padding_side = 'left'

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        callbacks=None,
        data_collator=data_collator,
    )

    # trainer = SFTTrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=dataset,
    #     dataset_text_field="text",  # need a dummy field
    #     tokenizer=tokenizer,
    #     peft_config=get_peft_config(model_config),
    #     callbacks=None,
    #     data_collator=data_collator,
    #     dataset_kwargs={"skip_prepare_dataset": True},
    # )

    trainer.train()


if __name__ == "__main__":
    config_file = "config/finetune_vlm.yaml"

    config = OmegaConf.load(config_file)
    main(config)