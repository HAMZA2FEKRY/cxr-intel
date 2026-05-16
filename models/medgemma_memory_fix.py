"""
DROP-IN PATCH for models/medgemma_model.py
Fixes: CUDA out of memory during Streamlit inference
Fix:   Load MedGemma in 4-bit quantization → uses ~4GB instead of ~12GB
       Add torch.cuda.empty_cache() before every inference call
"""

import os
import torch
import gc
from pathlib import Path

# ── Try to import BitsAndBytes for 4-bit quantization ──
try:
    from transformers import BitsAndBytesConfig
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False


def get_quantization_config():
    """Return 4-bit quantization config if bitsandbytes is available."""
    if BNB_AVAILABLE and torch.cuda.is_available():
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    return None


def free_gpu_memory():
    """Aggressively free GPU memory before inference."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_medgemma(model_id: str = "google/medgemma-1.5-4b-it",
                  hf_token: str = None):
    """
    Load MedGemma with 4-bit quantization to fit in 16GB T4 GPU
    alongside CLIP and ColPali indexes.

    Memory usage:
      - Without quantization: ~12 GB
      - With 4-bit:           ~4  GB  ← fits comfortably
    """
    from transformers import AutoProcessor, AutoModelForImageTextToText

    free_gpu_memory()

    token = hf_token or os.environ.get("HF_TOKEN", "")
    quant_config = get_quantization_config()

    print(f"Loading {model_id} ...")
    if quant_config:
        print("  Using 4-bit quantization (saves ~8GB VRAM)")
    else:
        print("  WARNING: 4-bit not available, loading in float16")

    processor = AutoProcessor.from_pretrained(
        model_id,
        token=token,
    )

    model_kwargs = dict(
        token=token,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    if quant_config:
        model_kwargs["quantization_config"] = quant_config

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        **model_kwargs,
    )
    model.eval()

    vram_used = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"MedGemma loaded. VRAM used: {vram_used:.1f} GB")

    return model, processor


def generate_report(model, processor, image, max_new_tokens: int = 300):
    """
    Generate a radiology report from a chest X-ray image.
    Always frees GPU cache before inference.
    """
    free_gpu_memory()  # ← KEY FIX: clear fragmented memory first

    prompt = (
        "You are an expert radiologist. Analyze this chest X-ray and generate "
        "a structured radiology report with the following sections:\n\n"
        "FINDINGS:\n"
        "IMPRESSION:\n\n"
        "Be specific and clinical."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    try:
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=torch.float16)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        # Decode only the new tokens
        input_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0][input_len:]
        report = processor.decode(new_tokens, skip_special_tokens=True)

        free_gpu_memory()  # clean up after inference
        return report.strip()

    except torch.cuda.OutOfMemoryError:
        free_gpu_memory()
        # Retry with shorter output
        try:
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=100,  # shorter on OOM
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )
            input_len = inputs["input_ids"].shape[1]
            report = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
            free_gpu_memory()
            return report.strip() + "\n[Note: truncated due to GPU memory]"
        except Exception as e:
            free_gpu_memory()
            return f"[ERROR] GPU OOM even with reduced tokens: {e}"

    except Exception as e:
        free_gpu_memory()
        return f"[ERROR] Generation failed: {e}"


def answer_question(model, processor, image, question: str,
                    context: str = "", max_new_tokens: int = 200):
    """
    Answer a clinical question about a chest X-ray using RAG context.
    """
    free_gpu_memory()

    context_block = f"\n\nRelevant context from similar cases:\n{context}" if context else ""

    prompt = (
        f"You are an expert radiologist.{context_block}\n\n"
        f"Looking at this chest X-ray, please answer: {question}"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    try:
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=torch.float16)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        input_len = inputs["input_ids"].shape[1]
        answer = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
        free_gpu_memory()
        return answer.strip()

    except torch.cuda.OutOfMemoryError:
        free_gpu_memory()
        return "[ERROR] GPU out of memory. Try restarting the app to reload the model."

    except Exception as e:
        free_gpu_memory()
        return f"[ERROR] {e}"
