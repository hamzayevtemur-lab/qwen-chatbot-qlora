#!/usr/bin/env python3
"""
Merge LoRA Adapter with Base Model Weights.

Fuses low-rank adapter weights into the base Qwen2.5-1.5B model to create
a standalone, production-ready Hugging Face model directory.
"""

import torch
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument(
        "--base_model_id",
        type=str,
        default="Qwen/Qwen2.5-1.5B",
        help="Base model repository ID on Hugging Face"
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="models/checkpoints/qwen2.5-1.5b-ultrachat-qlora/final_adapter",
        help="Path to trained LoRA adapter directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/merged_qwen_chatbot",
        help="Target directory to save merged standalone model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to load model on ('auto', 'cpu', 'cuda')"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("🔄 Fusing LoRA Adapter into Base Model Weights")
    print(f"📦 Base Model:   {args.base_model_id}")
    print(f"🎯 Adapter Path: {args.adapter_path}")
    print(f"💾 Output Path:  {output_path.resolve()}")
    print("=" * 65)

    # 1. Determine precision
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    # 2. Load Base Model in full/half precision (not 4-bit) for clean mathematical merge
    print("\n📦 Step 1/4: Loading Base Model in 16-bit precision...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        torch_dtype=torch_dtype,
        device_map=args.device,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    # 3. Load LoRA Adapter onto Base Model
    adapter_path = Path(args.adapter_path).resolve()
    if not adapter_path.exists():
        raise FileNotFoundError(
            f"❌ Adapter directory not found at: {adapter_path}\n"
            "Please run 'python training/train_qlora.py' first to train and save the adapter!"
        )

    print(f"🎯 Step 2/4: Attaching LoRA Adapter from {adapter_path}...")
    peft_model = PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
        torch_dtype=torch_dtype
    )

    # 4. Mathematically merge weights (W = W_0 + BA)
    print("⚡ Step 3/4: Merging LoRA layers into base weights (merge_and_unload)...")
    merged_model = peft_model.merge_and_unload()

    # 5. Save Merged Model & Tokenizer
    print(f"💾 Step 4/4: Saving standalone model and tokenizer to {output_path}...")
    merged_model.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size="4GB"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    print("=" * 65)
    print(f"🎉 Merged model successfully saved to: {output_path.resolve()}")
    print("✅ Ready for Hugging Face Hub upload or GGUF conversion!")
    print("=" * 65)


if __name__ == "__main__":
    main()
