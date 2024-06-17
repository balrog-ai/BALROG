import os
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig
import logging
import wandb
from omegaconf import OmegaConf
import argparse


def load(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    return tokenizer, model


def main(config):
    print("FINETUNING!")
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

    logging.info("Loading model and tokenizer")
    tokenizer, model = load(config.model_id)

    logging.info("Loading dataset")
    dataset = load_dataset("csv", data_files=config.dataset_path, split="train")

    logging.info("Setting up data collator")
    collator = DataCollatorForCompletionOnlyLM(
        response_template="### Response:", tokenizer=tokenizer
    )

    if config.use_lora:
        logging.info("Using LoRA for finetuning")
        peft_config = LoraConfig(
            lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout, r=config.r
        )
    else:
        peft_config = None

    logging.info("Setting training arguments")
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

    logging.info("Setting up trainer")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=8192,
        peft_config=peft_config,
        data_collator=collator,
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
    )

    logging.info("Starting training")
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetuning script with config file")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config file",
        default="configs/finetune.yaml",
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    main(config)
