# model.py (corrected + ready to drop in)
# Fixes:
# - adds missing imports
# - removes duplicate/unused quant_config (keep ONE config)
# - makes max_new_tokens consistent (uses the function arg, not hardcoded 384)
# - avoids relying on implicit globals where possible (still supports your current style)
# - safer device selection

from __future__ import annotations

from typing import Optional, Tuple, Any, Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def load_txgemma_submit_safe(model_id: str, token: Optional[str] = None) -> Tuple[Any, Any, str]:
    """
    Loads TxGemma in 4-bit (bnb nf4). Requires CUDA.
    Returns (tokenizer, model, mode_string).
    """
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required (torch.cuda.is_available() is False).")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token, use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=token,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model.eval()

    # Cache off can reduce memory spikes in some setups; keep consistent.
    model.config.use_cache = False
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = False

    return tokenizer, model, "4bit_auto"


def generate_report(
    disease: str,
    drug: str,
    snippets: List[Dict[str, Any]],
    *,
    build_prompt_fn,
    tokenizer,
    model,
    mol_pack: Optional[Dict[str, Any]] = None,
    max_new_tokens: int = 384,
    max_length: int = 2048,
) -> str:
    """
    Runs TxGemma inference:
    - builds prompt via build_prompt_fn(disease, drug, snippets, mol_pack=...)
    - wraps prompt into a chat-style message
    - returns only generated continuation
    """
    prompt = build_prompt_fn(disease, drug, snippets, mol_pack=mol_pack)
    messages = [{"role": "user", "content": prompt}]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,  # ✅ use the arg (was hardcoded 384)
            do_sample=False,
            repetition_penalty=1.05,
        )

    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1] :],
        skip_special_tokens=True,
    )


def generate_report_from_prompt(
    prompt: str,
    *,
    tokenizer,
    model,
    max_new_tokens: int = 384,
    max_length: Optional[int] = None,
) -> str:
    """
    Same as generate_report but takes a raw prompt string.
    """
    messages = [{"role": "user", "content": prompt}]

    kwargs = dict(
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    if max_length is not None:
        kwargs.update(truncation=True, max_length=max_length)

    inputs = tokenizer.apply_chat_template(messages, **kwargs)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
        )

    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1] :],
        skip_special_tokens=True,
    )
