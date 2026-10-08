"""
QLoRA SFT Training Script for Qwen-2.5-1.5B on UltraChat.
"""

import os
import yaml
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    TrainingArguments
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
def load_config(config_path: str="training/config.yaml")->dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    # Load Configuration
    cfg=load_config()

    print("=" * 65)
    print("🚀 Initializing QLoRA Fine-Tuning Pipeline for Qwen-2.5-1.5B")
    print("=" * 65)

    # Determine Compute Dtype and Device
    if torch.cuda.is_available():
        device_map="auto"
        compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    else:
        device_map=None 
        compute_dtype=torch.float32

    # Setup 4-bit BitsAndBytes Quantization
    bnb_config=BitsAndBytesConfig(
        load_in_4bit=cfg["quantization"]["load_in_4bit"],
        bnb_4bit_quant_type=cfg["quantization"]["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=cfg["quantization"]["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=compute_dtype,
    )

    # Load Tokenizer and Base Model
    model_id=cfg["model"]["base_model_name"]

    print(f"\n📦 Loading Base Tokenizer: {model_id}...")

    tokenizer=AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side='right'
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token
        
    print(f"📦 Loading 4-bit Quantized Base Model: {model_id}...")
    model=AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config if torch.cuda.is_available() else None,
        torch_dtype=compute_dtype,
        device_map=device_map,
        trust_remote_code=True
    )

    # Prepare Model for LoRA k-bit Training
    model=prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=cfg["training"]["gradient_checkpointing"]
    )

    # Configure PEFT/LoRA
    lora_cfg=cfg["lora"]
    peft_config=LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
        target_modules=lora_cfg["target_modules"]
    )

    model=get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load Processed Datasets
    print("\n📂 Loading Processed Datasets...")
    dataset = load_dataset(
        "json",
        data_files={
            "train": cfg["data"]["train_file"],
            "validation": cfg["data"]["validation_file"]
        }
    )

    print(f"   • Train samples: {len(dataset['train']):,}")
    print(f"   • Val samples:   {len(dataset['validation']):,}")

    # Training Arguments
    t_cfg=cfg["training"]
    training_args=SFTConfig(
        output_dir=t_cfg["output_dir"],
        num_train_epochs=t_cfg["num_train_epochs"],
        per_device_train_batch_size=t_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=t_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=t_cfg["gradient_accumulation_steps"],
        learning_rate=float(t_cfg["learning_rate"]),
        lr_scheduler_type=t_cfg["lr_scheduler_type"],
        warmup_ratio=t_cfg["warmup_ratio"],
        weight_decay=t_cfg["weight_decay"],
        logging_steps=t_cfg["logging_steps"],
        eval_strategy=t_cfg["eval_strategy"],
        eval_steps=t_cfg["eval_steps"],
        save_strategy=t_cfg["save_strategy"],
        save_steps=t_cfg["save_steps"],
        save_total_limit=t_cfg["save_total_limit"],
        fp16=(compute_dtype == torch.float16),
        bf16=(compute_dtype == torch.bfloat16),
        gradient_checkpointing=t_cfg["gradient_checkpointing"],
        optim=t_cfg["optim"],
        seed=t_cfg["seed"],
        max_seq_length=cfg["data"]["max_seq_length"],
        packing=cfg["data"]["packing"],
        report_to="none" # Can set to "tensorboard" or "wandb"
    )

    # Initialize SFT Trainer
    trainer=SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=training_args
    )

    # Start Training
    print("\n🔥 Starting Training Loop...")
    train_result = trainer.train()

    # Save Final LoRA Adapter and Tokenizer
    final_output_dir = Path(t_cfg["output_dir"]) / "final_adapter"

    print(f"\n💾 Saving LoRA Adapter weights to {final_output_dir}...")
    trainer.model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)

    print("=" * 65)
    print("🎉 Fine-tuning finished successfully!")
    print(f"📊 Training Loss: {train_result.training_loss:.4f}")
    print("=" * 65)

if __name__=="__main__":
    main()