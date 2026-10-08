#!/usr/bin/env python3
"""
GGUF Export & Quantization Script using llama.cpp.

Converts a merged Hugging Face model directory into a 4-bit/8-bit GGUF binary
optimized for low-latency CPU, Mac Metal, or Hugging Face Space deployments.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Export Hugging Face model to GGUF format")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models/merged_qwen_chatbot",
        help="Path to merged Hugging Face model directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/gguf",
        help="Directory to save exported GGUF files"
    )
    parser.add_argument(
        "--quant_type",
        type=str,
        default="Q4_K_M",
        choices=["Q4_K_M", "Q5_K_M", "Q8_0", "F16"],
        help="Quantization level (Q4_K_M is recommended for optimal speed/size)"
    )
    parser.add_argument(
        "--llama_cpp_dir",
        type=str,
        default="../../DL/Homework/llama.cpp",
        help="Path to local llama.cpp clone"
    )
    return parser.parse_args()


def run_command(cmd: str):
    """Executes a shell command and streams output."""
    print(f"\n💻 Running: {cmd}\n")
    result = subprocess.run(cmd, shell=True, text=True)
    if result.returncode != 0:
        print(f"❌ Command failed with return code {result.returncode}")
        sys.exit(result.returncode)


def main():
    args = parse_args()
    model_path = Path(args.model_dir).resolve()
    output_path = Path(args.output_dir).resolve()
    llama_cpp_path = Path(args.llama_cpp_dir).resolve()

    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("📦 GGUF Export & Quantization Pipeline")
    print(f"📂 Source Model:      {model_path}")
    print(f"💾 Target Directory:  {output_path}")
    print(f"⚙️  Quantization Type: {args.quant_type}")
    print("=" * 65)

    if not model_path.exists():
        print(f"❌ Error: Model directory '{model_path}' not found. Run merge_lora.py first.")
        sys.exit(1)

    convert_script = llama_cpp_path / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        # Fallback to current working directory llama.cpp if not found
        convert_script = Path("llama.cpp/convert_hf_to_gguf.py").resolve()

    f16_gguf_file = output_path / "qwen2.5-1.5b-chatbot-f16.gguf"
    quant_gguf_file = output_path / f"qwen2.5-1.5b-chatbot-{args.quant_type.lower()}.gguf"

    # Step 1: Convert Hugging Face weights to F16 GGUF
    print("\n🔄 Step 1/2: Converting Hugging Face weights to F16 GGUF...")
    convert_cmd = f"python3 {convert_script} {model_path} --outfile {f16_gguf_file} --outtype f16"
    run_command(convert_cmd)

    # Step 2: Quantize to target type (if not F16)
    if args.quant_type != "F16":
        print(f"\n⚡ Step 2/2: Quantizing to {args.quant_type}...")
        quantize_bin = llama_cpp_path / "build" / "bin" / "llama-quantize"
        if not quantize_bin.exists():
            quantize_bin = llama_cpp_path / "llama-quantize"

        quant_cmd = f"{quantize_bin} {f16_gguf_file} {quant_gguf_file} {args.quant_type}"
        run_command(quant_cmd)

        print(f"\n🗑️  Cleaning up temporary F16 file ({f16_gguf_file.name})...")
        if f16_gguf_file.exists():
            f16_gguf_file.unlink()

    print("=" * 65)
    print(f"🎉 GGUF Export Complete! Saved to: {quant_gguf_file if args.quant_type != 'F16' else f16_gguf_file}")
    print("=" * 65)


if __name__ == "__main__":
    main()
