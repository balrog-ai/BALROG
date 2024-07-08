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
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, use_fast=True, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(config.model_id, rust_remote_code=True)
    processor.tokenizer = tokenizer
    
    if config.DDP:
        device_string = PartialState().process_index
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            device_map={'': device_string},
            trust_remote_code=True,
            _attn_implementation='flash_attention_2',
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            device_map="auto",
            trust_remote_code=True,
            _attn_implementation='flash_attention_2',
        )
    return model, tokenizer, processor

class VLMDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        texts = []
        images = []
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
            text = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
            images.append(image)

        batch = self.processor(texts, images, return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone()
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        
        return batch

# def main(config):
    
if __name__ == "__main__":
    config_file = "config/finetune_vlm.yaml"

    config = OmegaConf.load(config_file)
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    sft_script_args, training_args, model_config = parser.parse_args_and_config()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    
    
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
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True, rust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_config.model_name_or_path, rust_remote_code=True)
    processor.tokenizer = tokenizer
    model = AutoModelForCausalLM.from_pretrained("/root/Phi-3-vision-128k-instruct", device_map="auto", trust_remote_code=True, _attn_implementation='flash_attention_2')
    

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

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     data_collator=collator,
    #     train_dataset=dataset
    # )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",  # need a dummy field
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_config),
        callbacks=None,
        data_collator=data_collator,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    trainer.train()


# if __name__ == "__main__":
#     config_file = "config/finetune.yaml"

#     config = OmegaConf.load(config_file)
#     main(config)