#!/usr/bin/env python3
"""
Before vs. After Model Comparison & Benchmark Script.

Compares raw Base Qwen2.5-1.5B completions against the SFT LoRA Fine-Tuned Chatbot.
"""

import json
import torch
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(description="Compare Base vs Fine-Tuned Qwen model")
    
    parser.add_argument(
        "--base_model_id",
        type=str,
        default="Qwen/Qwen2.5-1.5B",
        help="Base model identifier on Hugging Face"
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="models/checkpoints/qwen2.5-1.5b-ultrachat-qlora/final_adapter",
        help="Path to trained LoRA adapter directory"
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default="evaluation/test_prompts.json",
        help="Path to benchmark test prompts JSON"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="evaluation/comparison_report.md",
        help="File to save the comparison report"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum generated tokens per response"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    return parser.parse_args()


def generate_response(model, tokenizer, prompt_or_messages, max_new_tokens, temperature):
    """Formats prompt, tokenizes, and generates autoregressive text."""
    if isinstance(prompt_or_messages, list):
        # Multi-turn conversation format
        formatted_prompt = tokenizer.apply_chat_template(
            prompt_or_messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        # Single prompt wrapped in standard ChatML
        messages = [
            {"role": "system", "content": "You are a helpful, respectful, and intelligent AI assistant."},
            {"role": "user", "content": prompt_or_messages}
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0.0),
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    # Slice output to decode only newly generated tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16 if torch.cuda.is_available() else torch.float32

    print("=" * 70)
    print("🔬 Evaluation: Base Model vs. Fine-Tuned LoRA Chatbot Benchmark")
    print(f"📦 Base Model:    {args.base_model_id}")
    print(f"🎯 Adapter Path:  {args.adapter_path}")
    print(f"💻 Device:        {device} ({dtype})")
    print("=" * 70)

    # 1. Load Tokenizer
    print("\n📦 Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, trust_remote_code=True)

    # 2. Load Base Model
    print(f"📦 Loading Base Model: {args.base_model_id}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    base_model.eval()

    # 3. Load LoRA Fine-Tuned Model
    adapter_dir = Path(args.adapter_path)
    has_adapter = adapter_dir.exists()

    if has_adapter:
        print(f"🎯 Loading LoRA Adapter from {adapter_dir}...")
        ft_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_id,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        ft_model = PeftModel.from_pretrained(ft_model, str(adapter_dir))
        ft_model.eval()
    else:
        print(f"⚠️  Adapter directory '{adapter_dir}' not found. Skipping fine-tuned inference for now.")
        ft_model = None

    # 4. Load Test Prompts
    with open(args.prompts_file, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    report_lines = [
        "# 🔬 Qwen-2.5-1.5B Base vs. Fine-Tuned (QLoRA) Benchmark Report\n",
        f"**Base Model:** `{args.base_model_id}`  \n",
        f"**Adapter:** `{args.adapter_path}`  \n\n",
        "---\n"
    ]

    print("\n🚀 Running Comparative Benchmark...\n")

    for idx, test_case in enumerate(prompts, 1):
        category = test_case.get("category", "General")
        prompt_data = test_case.get("prompt") or test_case.get("messages")

        print(f"[{idx}/{len(prompts)}] Testing Category: {category}")

        # Base Model Output
        base_output = generate_response(
            base_model, tokenizer, prompt_data, args.max_new_tokens, args.temperature
        )

        # Fine-Tuned Model Output
        if ft_model is not None:
            ft_output = generate_response(
                ft_model, tokenizer, prompt_data, args.max_new_tokens, args.temperature
            )
        else:
            ft_output = "*(Adapter not trained/loaded yet)*"

        # Display in Console
        display_prompt = prompt_data if isinstance(prompt_data, str) else prompt_data[-1]["content"]
        print(f"💬 Prompt: {display_prompt}")
        print(f"🔴 Base Model:       {base_output[:120]}...")
        print(f"🟢 Fine-Tuned Model: {ft_output[:120]}...\n" + "-" * 50)

        # Append to Markdown Report
        report_lines.append(f"## Test Case {idx}: {category}\n")
        report_lines.append(f"**User Prompt:**\n> {display_prompt}\n\n")
        report_lines.append(f"### 🔴 Base Model (Before SFT):\n```text\n{base_output}\n```\n\n")
        report_lines.append(f"### 🟢 Fine-Tuned Qwen2.5 Chatbot (After QLoRA):\n```text\n{ft_output}\n```\n\n")
        report_lines.append("---\n")

    # 5. Save Benchmark Markdown Report
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(report_lines)

    print(f"✅ Benchmark complete! Report saved to {output_path.resolve()}\n")


if __name__ == "__main__":
    main()
