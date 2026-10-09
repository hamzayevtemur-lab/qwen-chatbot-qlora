"""
Model Service Manager for Streaming LLM Inference.
Handles tokenizer initialization, ChatML message formatting,
and asynchronous token streaming using Hugging Face TextIteratorStreamer.
"""

import os
import torch
from threading import Thread
from typing import List, Dict, Generator, AsyncGenerator
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer


class ChatModelService:
    def __init__(self, model_path: str="models/merged_qwen_chatbot"):
        self.model_path=model_path
        self.device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        self.dtype=(
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
            if torch.cuda.is_available() or torch.backends.mps.is_available()
            else torch.float32
        )
        self.tokenizer=None
        self.model=None

    def load_model(self):
        """Loads tokenizer and weights into memory."""
        # Check environment variable, local directory, or fallback to Hugging Face Hub
        model_env = os.environ.get("MODEL_PATH") or os.environ.get("MODEL_ID")
        if model_env:
            target_path = model_env
        elif os.path.exists(self.model_path):
            target_path = self.model_path
        else:
            target_path = "TemurbekHamzaev/qwen2.5-1.5b-chatbot"

        print(f"📦 [ModelService] Loading model from: {target_path} on {self.device} ({self.dtype})...")
        self.tokenizer = AutoTokenizer.from_pretrained(target_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model=AutoModelForCausalLM.from_pretrained(
            target_path,
            torch_dtype=self.dtype,
            device_map="auto" if self.device=="cuda" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        if self.device !="cuda":
            self.model.to(self.device)

        self.model.eval()
        print("✅ [ModelService] Model successfully loaded and ready for inference!")

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> Generator[str, None, None]:
        """
        Generates token stream using TextIteratorStreamer in a background thread.
        Yields individual text tokens in real-time.
        """
        if self.model is None or self.tokenizer is None:
            self.load_model()

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)

        # Setup streamer to yield tokens as they are decoded
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=(temperature > 0.0),
            pad_token_id=self.tokenizer.eos_token_id
        )

        # Launch generation in separate thread so generator can yield tokens asynchronously
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text

        thread.join()







    