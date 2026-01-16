"""
Local Model Management
======================

Manages local reasoning models for in-depth financial analysis.

Two-tier architecture:
- Tier 1 (Screening): GPT-5-nano for quick filtering
- Tier 2 (Deep Analysis): Local reasoning model for final investment decisions

Supported Models (12GB VRAM):
- DeepSeek-R1-Distill-Qwen-14B (recommended - best balance of speed and quality)
- Qwen2.5-14B-Instruct (faster, good quality)
"""

import os
import torch
from typing import Optional, Literal
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Model selection (QwQ-32B removed - requires 19GB+ VRAM with 4-bit, 33GB+ with 8-bit)
ModelChoice = Literal["deepseek-r1-14b", "qwen2.5-14b"]

# Global model cache
_reasoning_model = None
_reasoning_tokenizer = None
_current_model_name = None

def get_reasoning_model(
    model_choice: ModelChoice = "deepseek-r1-14b",
    force_reload: bool = False
):
    """
    Lazy-loads local reasoning model for financial analysis.

    Args:
        model_choice: Which model to load
        force_reload: Force reload even if already loaded

    Returns:
        (model, tokenizer) tuple
    """
    global _reasoning_model, _reasoning_tokenizer, _current_model_name

    # Return cached model if already loaded and same choice
    if not force_reload and _reasoning_model is not None and _current_model_name == model_choice:
        return _reasoning_model, _reasoning_tokenizer

    print(f"[Local Model] Loading {model_choice} for deep financial analysis...")

    # Model mappings
    model_paths = {
        "deepseek-r1-14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct"
    }

    model_path = model_paths[model_choice]

    # 4-bit quantization config for 12GB VRAM
    # DeepSeek-R1-14B: ~26GB FP16 → ~9GB with 4-bit (7GB weights + 2GB overhead)
    # Qwen2.5-14B: ~28GB FP16 → ~8.4GB with 4-bit (7GB weights + 1.4GB overhead)
    # Both fit comfortably in 12GB VRAM with headroom
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    print(f"[Local Model] Using 4-bit NF4 quantization (~9GB VRAM expected)")

    # Load tokenizer
    _reasoning_tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    # Load model with quantization
    _reasoning_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",  # Automatic GPU placement
        trust_remote_code=True,
        torch_dtype=torch.float16
    )

    _current_model_name = model_choice

    print(f"[Local Model] {model_choice} loaded successfully")
    print(f"[Local Model] Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    return _reasoning_model, _reasoning_tokenizer


def generate_reasoning_response(
    prompt: str,
    system_prompt: str = "You are a financial analysis expert.",
    model_choice: ModelChoice = "deepseek-r1-14b",
    max_new_tokens: int = 2048,
    temperature: float = 0.1
) -> str:
    """
    Generate response using local reasoning model.

    Args:
        prompt: User query
        system_prompt: System instructions
        model_choice: Which model to use
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (lower = more deterministic)

    Returns:
        Generated response
    """
    model, tokenizer = get_reasoning_model(model_choice)

    # Format messages (Qwen/DeepSeek format)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode (skip prompt)
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


def unload_reasoning_model():
    """Free GPU memory by unloading the reasoning model."""
    global _reasoning_model, _reasoning_tokenizer, _current_model_name

    if _reasoning_model is not None:
        del _reasoning_model
        del _reasoning_tokenizer
        _reasoning_model = None
        _reasoning_tokenizer = None
        _current_model_name = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[Local Model] Reasoning model unloaded, GPU memory freed")


def get_available_vram() -> float:
    """Check available VRAM in GB."""
    if not torch.cuda.is_available():
        return 0.0

    total_memory = torch.cuda.get_device_properties(0).total_memory
    allocated = torch.cuda.memory_allocated(0)
    return (total_memory - allocated) / 1024**3


def check_model_compatibility(model_choice: ModelChoice) -> dict:
    """
    Check if model can fit in available VRAM.

    Returns:
        Dict with compatibility info
    """
    available_vram = get_available_vram()

    # Estimated VRAM requirements (with quantization)
    vram_requirements = {
        "deepseek-r1-14b": 4.5,  # 4-bit quantized
        "qwq-32b": 10.0,         # 8-bit quantized
        "qwen2.5-14b": 4.0       # 4-bit quantized
    }

    required = vram_requirements.get(model_choice, 8.0)
    compatible = available_vram >= required

    return {
        "model": model_choice,
        "required_vram_gb": required,
        "available_vram_gb": available_vram,
        "compatible": compatible,
        "message": f"{'✓' if compatible else '✗'} {model_choice} requires {required}GB, {available_vram:.1f}GB available"
    }


if __name__ == "__main__":
    """Test model loading"""
    print("="*60)
    print("Local Model Compatibility Check")
    print("="*60)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available - GPU required for local models")
        exit(1)

    print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"✓ Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Check each model
    for model_name in ["deepseek-r1-14b", "qwq-32b", "qwen2.5-14b"]:
        result = check_model_compatibility(model_name)
        print(result["message"])

    print("\n" + "="*60)
    print("Testing model loading (this will download ~7GB on first run)...")
    print("="*60)

    # Test load (default: DeepSeek-R1-14B)
    response = generate_reasoning_response(
        prompt="Calculate the intrinsic value of a company with: FCF=$500M, growth rate=5%, discount rate=10%. Show your work.",
        model_choice="deepseek-r1-14b",
        max_new_tokens=512
    )

    print("\n[Response]")
    print(response)

    # Check memory usage
    print(f"\n[Memory] GPU allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
