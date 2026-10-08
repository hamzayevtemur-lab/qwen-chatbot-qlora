#!/usr/bin/env python3
"""
UltraChat Dataset Preparation Pipeline for Qwen-2.5-1.5B QLoRA Fine-Tuning.

This script:
1. Streams/loads the HuggingFaceH4/ultrachat_200k dataset.
2. Filters out malformed, single-turn, or excessively long dialogues.
3. Standardizes conversations into the ChatML / Hugging Face messages format:
   {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
4. Splits into Train and Validation sets (default 95% / 5%).
5. Saves clean .jsonl files in data/processed/.
"""

import os
import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any
from datasets import load_dataset
from tqdm import tqdm

DEFAULT_SYSTEM_PROMPT = "You are a helpful, respectful, and intelligent AI assistant."

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare UltraChat dataset for Qwen SFT")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="HuggingFaceH4/ultrachat_200k",
        help="Hugging Face dataset identifier"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train_sft",
        help="Dataset split to pull from"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=15000,
        help="Target number of high-quality conversational samples to extract"
    )
    parser.add_argument(
        "--max_turns",
        type=int,
        default=6,
        help="Maximum turns per conversation (user+assistant pair = 2 turns)"
    )
    parser.add_argument(
        "--min_turns",
        type=int,
        default=2,
        help="Minimum turns per conversation"
    )
    parser.add_argument(
        "--max_char_length",
        type=int,
        default=8000,
        help="Maximum total characters per dialogue to prevent context overflow"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.05,
        help="Fraction of data reserved for validation (e.g. 0.05 = 5%)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).parent / "processed"),
        help="Directory to save output jsonl files"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    return parser.parse_args()


def clean_and_format_dialogue(
    raw_messages: List[Dict[str, str]],
    system_prompt: str,
    min_turns: int,
    max_turns: int,
    max_char_length: int
) -> Dict[str, Any] | None:
    """
    Validates, filters, and standardizes a single conversation.
    Returns standard {'messages': [...]} dictionary or None if invalid.
    """
    if not raw_messages or len(raw_messages) < min_turns:
        return None

    cleaned_messages = []
    has_system = False

    # Check if first turn is already a system prompt
    first_role = raw_messages[0].get("role", "").lower().strip()
    if first_role == "system":
        cleaned_messages.append({
            "role": "system",
            "content": raw_messages[0].get("content", "").strip()
        })
        has_system = True
        dialogue_turns = raw_messages[1:]
    else:
        cleaned_messages.append({
            "role": "system",
            "content": system_prompt
        })
        dialogue_turns = raw_messages

    # Limit maximum turns
    dialogue_turns = dialogue_turns[:max_turns]

    # Validate turn sequence (user -> assistant -> user -> assistant...)
    expected_role = "user"
    valid_turns_count = 0
    total_chars = sum(len(m.get("content", "")) for m in cleaned_messages)

    for turn in dialogue_turns:
        role = turn.get("role", "").lower().strip()
        content = turn.get("content", "").strip()

        if not content:
            return None

        # Map role aliases
        if role in ["human", "prompter"]:
            role = "user"
        elif role in ["gpt", "bot", "model"]:
            role = "assistant"

        if role != expected_role:
            return None

        cleaned_messages.append({
            "role": role,
            "content": content
        })
        total_chars += len(content)
        valid_turns_count += 1
        expected_role = "assistant" if expected_role == "user" else "user"

    # Ensure dialogue ends with assistant turn and meets minimum turns requirement
    if cleaned_messages[-1]["role"] != "assistant":
        cleaned_messages.pop()
        valid_turns_count -= 1

    if valid_turns_count < min_turns or total_chars > max_char_length:
        return None

    return {"messages": cleaned_messages}


def main():
    args = parse_args()
    random.seed(args.seed)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    train_file = output_path / "train.jsonl"
    val_file = output_path / "validation.jsonl"

    print("=" * 65)
    print("🚀 UltraChat Dataset Preparation for Qwen-2.5-1.5B Fine-Tuning")
    print(f"📦 Source: {args.dataset_name} ({args.split})")
    print(f"🎯 Target Samples: {args.num_samples:,}")
    print(f"🔄 Turns Range: [{args.min_turns}, {args.max_turns}]")
    print(f"💾 Output Directory: {output_path.resolve()}")
    print("=" * 65)

    print("\n⏳ Streaming dataset from Hugging Face...")
    dataset = load_dataset(args.dataset_name, split=args.split, streaming=True)

    processed_records = []
    total_checked = 0

    pbar = tqdm(total=args.num_samples, desc="Processing & Filtering")
    for item in dataset:
        total_checked += 1
        raw_msgs = item.get("messages", [])

        formatted = clean_and_format_dialogue(
            raw_messages=raw_msgs,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            min_turns=args.min_turns,
            max_turns=args.max_turns,
            max_char_length=args.max_char_length
        )

        if formatted is not None:
            processed_records.append(formatted)
            pbar.update(1)

        if len(processed_records) >= args.num_samples:
            break

    pbar.close()

    print(f"\n📊 Quality Filter Summary:")
    print(f"   • Total Inspected: {total_checked:,}")
    print(f"   • High-Quality Selected: {len(processed_records):,} (Yield: {len(processed_records)/total_checked*100:.1f}%)")

    # Shuffle before splitting
    random.shuffle(processed_records)

    val_count = max(1, int(len(processed_records) * args.val_ratio))
    train_count = len(processed_records) - val_count

    train_data = processed_records[:train_count]
    val_data = processed_records[train_count:]

    print(f"   • Train Split: {len(train_data):,} samples -> {train_file.name}")
    print(f"   • Val Split:   {len(val_data):,} samples -> {val_file.name}")

    # Write Train JSONL
    print(f"\n💾 Saving {train_file}...")
    with open(train_file, "w", encoding="utf-8") as f:
        for record in train_data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Write Validation JSONL
    print(f"💾 Saving {val_file}...")
    with open(val_file, "w", encoding="utf-8") as f:
        for record in val_data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Preview 1 sample
    print("\n" + "=" * 65)
    print("📝 Sample Processed Record Preview:")
    print("=" * 65)
    sample = val_data[0]
    for msg in sample["messages"]:
        prefix = f"[{msg['role'].upper()}]:"
        content = msg['content'][:140] + ("..." if len(msg['content']) > 140 else "")
        print(f"{prefix:<14} {content}")
    print("=" * 65)
    print(f"✅ Data preparation complete! Output ready in {output_path.resolve()}\n")


if __name__ == "__main__":
    main()
