import os
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, pipeline
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig
import accelerate

def load(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(model_name, device_map=f'cuda:{accelerate.Accelerator().process_index}')
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    
    # # If these tokens aren't already in the tokenizer, add them
    # num_new_tokens = tokenizer.add_special_tokens(
    #     {
    #         "bos_token": "<s>",
    #         "eos_token": "</s>",
    #         "unk_token": "<unk>",
    #         "pad_token": "<pad>",
    #     }
    # )
    
    # # Add special action and observation delimiter tokens
    # action_token = "<|action|>"
    # observation_token = "<|observation|>"
    # num_new_tokens += tokenizer.add_tokens([action_token, observation_token])
    
    # # resize the model embedding to match the tokenizer
    # embedding_size = model.get_input_embeddings().weight.shape[0]
    # if len(tokenizer) > embedding_size:
    #     model.resize_token_embeddings(len(tokenizer))
    # assert len(tokenizer) == model.get_input_embeddings().weight.shape[0]
    
    return tokenizer, model

@errors.record
def main():
    MODEL_NAME = "google/gemma-2b"
    # MODEL_NAME = "google/gemma-7b"
    # MODEL_NAME = "meta-llama/Llama-2-7b-hf"
    
    tokenizer, model = load(MODEL_NAME)
    dataset = load_dataset("csv", data_files="data/10/data.csv", split="train")
    
    collator = DataCollatorForCompletionOnlyLM(response_template="### Response:", tokenizer=tokenizer)
    
    peft_config = LoraConfig(
        lora_alpha=64,
        lora_dropout=0.1,
        r=32,
    )
    
    training_args = TrainingArguments(
        output_dir=os.path.join("models", MODEL_NAME),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4, 
        save_steps=100,
        logging_steps=10,
        # eval_steps=50,
        learning_rate=3e-5,
        fp16=True, 
        gradient_checkpointing=True,
        # evaluation_strategy="steps", 
        # max_grad_norm=0.2,
        num_train_epochs=5.0,
        weight_decay=0.001,
        warmup_steps=50,
        lr_scheduler_type="linear",
        report_to='wandb',
    )
    
    # generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        peft_config=peft_config,
        data_collator=collator,
        # max_seq_length=4096, 
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
        # callbacks=[LogGenerationsCallback(generator)]
    )
    
    trainer.train()
    
if __name__ == "__main__":    
    main()