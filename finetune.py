import os
from train.collators import DataCollatorForLanguageModeling
from train.utils import *

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, AutoProcessor
from datasets import load_dataset

from trl import SFTTrainer
from accelerate import PartialState
from peft import LoraConfig, get_peft_model
import wandb
from omegaconf import OmegaConf


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

def main(config):
    print("FINETUNING!")
    wandb.init(
        project="nle_finetune_test", config=OmegaConf.to_container(config, resolve=True)
    )

    model, tokenizer, _ = load(config)

    dataset = load_dataset("csv", data_files=config.dataset_path, split="train")
    
    data_collator = DataCollatorForLanguageModeling(tokenizer)
    
    if config.use_lora:
        peft_config = LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.alpha,
            target_modules=find_target_linear_names(model),
            lora_dropout=config.lora.dropout,
            task_type="CAUSAL_LM",
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)
        
        # Ensure all LoRA parameters require gradients
        for name, param in model.named_parameters():
            if 'lora' in name:
                param.requires_grad = True
                print(f"Set requires_grad=True for {name}")
    else:
        peft_config = None

    training_args = TrainingArguments(
        output_dir=os.path.join(config.output_dir, config.model_id + "_lora"),
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
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=8192,
        peft_config=peft_config,
        data_collator=data_collator,
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
    )

    trainer.train()

if __name__ == "__main__":
    config_file = "config/finetune.yaml"
    config = OmegaConf.load(config_file)
    main(config)