import os

from train.collators import VLMDataCollator
from train.trainers import CustomSFTTrainer
from train.utils import *

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, Trainer
from datasets import load_from_disk
from accelerate import PartialState
import logging
import wandb
from omegaconf import OmegaConf
from tqdm.rich import tqdm
from trl import (
    SFTConfig,
)
from peft import LoraConfig, get_peft_model


tqdm.pandas()

def load(config):
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, use_fast=True, trust_remote_code=True, use_cache=False)
    tokenizer.padding_side = 'left'
    processor = AutoProcessor.from_pretrained(config.model_id, trust_remote_code=True, use_cache=False)
    processor.tokenizer = tokenizer
    
    if config.DDP:
        device_string = PartialState().process_index
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            device_map={'': device_string},
            trust_remote_code=True,
            _attn_implementation='flash_attention_2',
            use_cache=False,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            device_map="auto",
            trust_remote_code=True,
            _attn_implementation='flash_attention_2',
            use_cache=False,
        )
    return model, tokenizer, processor

local_rank = None

def rank0_print(*args):
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)

def main(config):    
    wandb.init(
        project="nle_finetune_test", config=OmegaConf.to_container(config, resolve=True)
    )

    model, tokenizer, processor = load(config)
    data_collator = VLMDataCollator(processor)
    dataset = load_from_disk(config.dataset_path)
    
    if config.use_lora:
        peft_config = LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.alpha,
            target_modules=find_target_linear_names(model),
            lora_dropout=config.lora.dropout,
            task_type="CAUSAL_LM",
        )
        rank0_print("Adding LoRA to the model...")
        model = get_peft_model(model, peft_config)

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
        report_to="wandb",  # Ensure this is set to wandb
        remove_unused_columns=False,
    )
    
    training_args.vision_lr = config.train.vision_lr
    training_args.projector_lr = config.train.projector_lr

    vlm_img_pipeline_gradient_config(model, training_args, config)

    logging.info("Setting up trainer")
    tokenizer.padding_side = 'left'
    
    trainer = CustomSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",  # need a dummy field
        tokenizer=tokenizer,
        callbacks=None,
        data_collator=data_collator,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    trainer.train()


if __name__ == "__main__":
    config_file = "config/finetune_vlm.yaml"

    config = OmegaConf.load(config_file)
    main(config)