---
title: Qwen2.5-1.5B Chatbot (QLoRA SFT)
emoji: 💬
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---

# 💬 Qwen2.5-1.5B Conversational Chatbot — Fine-Tuned via QLoRA 🚀

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers%20%7C%20PEFT-yellow.svg)](https://huggingface.co/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production%20API-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

An end-to-end Large Language Model project fine-tuning a raw base foundation model (**`Qwen/Qwen2.5-1.5B`**) into an instruction-following conversational assistant using **QLoRA (4-bit NormalFloat quantization)** on multi-turn dialogue data (**UltraChat 200k**).

The system includes an asynchronous **FastAPI backend with real-time Server-Sent Events (SSE) token streaming**, a dark-mode glassmorphic chat interface, GGUF quantization export, and Docker containerization for zero-cost cloud deployment.

---

## 🌟 Key Technical Highlights

1. **Raw Base Foundation Model SFT:** Trained starting from the base completion model (`Qwen2.5-1.5B`), demonstrating the transformation from raw next-token prediction into a multi-turn assistant.
2. **QLoRA Parameter-Efficient Fine-Tuning:**
   - 4-bit NormalFloat (`NF4`) base model weight quantization with double quantization.
   - Low-Rank Adapters ($r=16, \alpha=32$) targeted across all 7 linear projection layers (`q, k, v, o, gate, up, down`).
   - Paged AdamW 8-bit optimizer and gradient checkpointing fitting within free Google Colab T4 VRAM (< 7 GB).
3. **Multi-Turn ChatML Data Pipeline:** Automated cleaning and structuring of UltraChat conversations into standardized ChatML format.
4. **Real-Time Token Streaming:** Fast multi-threaded asynchronous generator utilizing `TextIteratorStreamer` and Server-Sent Events (`/api/chat/stream`).
5. **Multi-Tier Deployment:**
   - Standalone Hugging Face merged weights (`merge_and_unload`).
   - 4-bit GGUF binary export for local edge inference via `llama.cpp` and Ollama.
   - Production Docker container ready for Hugging Face Spaces.

---

## 📊 Evaluation: Base Model vs. Fine-Tuned Chatbot

| Benchmark Capability | 🔴 Base Model (`Qwen2.5-1.5B`) | 🟢 Fine-Tuned Model (After QLoRA) |
| :--- | :--- | :--- |
| **Conversational Greeting** | Continues generating random repetitive text / website headers | Introduces itself politely and offers assistance |
| **Multi-Turn Dialogue** | Loses contextual thread, forgets persona | Maintains multi-turn context and conversational memory |
| **Code Generation** | Incomplete code snippets without markdown | Complete, formatted Python functions with docstrings |
| **Explanation / Reasoning** | Dense unformatted text | Structured explanations with bullet points and clear analogies |

---

## 🏗️ System Architecture

```mermaid
flowchart LR
    A[Client Web UI] -->|HTTP POST /api/chat/stream| B[FastAPI Backend]
    B --> C[ChatModelService]
    C --> D[ChatML Template Formatter]
    D --> E[TextIteratorStreamer Thread]
    E -->|SSE Token Stream| A
