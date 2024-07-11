import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim
import logging
import wandb
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
from tqdm.rich import tqdm

tqdm.pandas()

class CustomDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenizer.padding_side = 'left'  # Set padding side to left

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        text = f"<|user|>\n{sample['prompt']}<|end|><|assistant>|\n{sample['action']}"
        
        encodings = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length)
        
        try:
            image = np.array(sample["image"])
        except Exception as e:
            return None
        
        encodings['pixel_values'] = image
        return {key: torch.tensor(val) for key, val in encodings.items()}
    
    
def main(config):
    print("FINETUNING!")
    
    # Initialize Wandb and logging
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

    # Load model, tokenizer, and processor
    logging.info("Loading model and tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, use_fast=True, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(config.model_id, trust_remote_code=True)
    processor.tokenizer = tokenizer
    model = AutoModelForCausalLM.from_pretrained(config.model_id, device_map="auto", torch_dtype="auto", trust_remote_code=True, _attn_implementation='flash_attention_2')

    # Load dataset
    logging.info("Loading dataset")
    raw_dataset = load_from_disk(config.dataset_path)
    train_size = int(0.9 * len(raw_dataset))
    val_size = len(raw_dataset) - train_size
    train_dataset, val_dataset = random_split(raw_dataset, [train_size, val_size])

    train_dataset = CustomDataset(train_dataset, tokenizer, max_length=8192)
    val_dataset = CustomDataset(val_dataset, tokenizer, max_length=8192)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    optimizer = optim.AdamW(model.parameters(), lr=5e-5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training parameters
    num_epochs = 5
    accumulation_steps = 64
    eval_interval = 10000
    checkpoint_interval = 100000

    step = 0
    
    # Training loop
    print("START TRAINING")
    for epoch in range(num_epochs):
        total_train_loss = 0
        batch_count = 0

        model.train()
        for batch in train_loader:
            if batch is None:
                continue

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = input_ids.clone().detach()

            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                pixel_values=pixel_values, 
                labels=labels
            )
            loss = outputs.loss
            loss.backward()

            if (step % accumulation_steps) == 0:
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad /= accumulation_steps
                optimizer.step()
                optimizer.zero_grad()
                wandb.log({"Batch Loss": loss.item(), "Step": step})
                print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item()}")
                batch_count += 1

            total_train_loss += loss.item()
            step += 1

            if step % eval_interval == 0:
                avg_train_loss = total_train_loss / batch_count
                val_loss = evaluate(model, val_loader, device, step)
                wandb.log({
                    "Epoch": epoch,
                    "Average Training Loss": avg_train_loss,
                    "Validation Loss": val_loss
                })

            if step % checkpoint_interval == 0:
                model.save_pretrained(f"checkpoints/checkpoint_{epoch}_{step}", safe_serialization=False)
                tokenizer.save_pretrained(f"checkpoints/checkpoint_{epoch}_{step}", safe_serialization=False)

        avg_train_loss = total_train_loss / batch_count
        print(f"Epoch: {epoch}, Average Training Loss: {avg_train_loss}")
        model.save_pretrained(f"checkpoints/checkpoint_{epoch}_{step}", safe_serialization=False)
        tokenizer.save_pretrained(f"checkpoints/checkpoint_{epoch}_{step}", safe_serialization=False)

def evaluate(model, val_loader, device, step):
    model.eval()
    total_loss = 0
    batch_count = 0

    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = input_ids.clone().detach()

            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                pixel_values=pixel_values, 
                labels=labels
            )
            loss = outputs.loss
            total_loss += loss.item()
            batch_count += 1

    avg_loss = total_loss / batch_count
    wandb.log({"Validation Loss": avg_loss, "Step": step})
    model.train()
    return avg_loss

if __name__ == "__main__":
    config_file = "config/finetune_vlm.yaml"
    config = OmegaConf.load(config_file)
    main(config)