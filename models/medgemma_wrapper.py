import os
import yaml
try:
    from transformers import AutoProcessor, AutoModelForImageTextToText
    import torch
    from PIL import Image
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

class MedGemmaGenerator:
    def __init__(self, use_mock_mode=None):
        with open("configs/config.yaml", "r") as f:
            config = yaml.safe_load(f)

        self.model_id = config["models"]["medgemma_model_id"]

        if use_mock_mode is None:
            env_mock = os.environ.get("USE_MOCK_MODE", "").lower()
            cfg_mock = str(config["models"]["use_mock_mode"]).lower()
            self.use_mock_mode = (env_mock in ("true", "1")) or (env_mock == "" and cfg_mock in ("true", "1"))
        else:
            self.use_mock_mode = use_mock_mode

        self.processor = None
        self.model = None

        # CHANGE the __init__ loading section to:
        if not self.use_mock_mode and HAS_TRANSFORMERS:
            try:
                hf_token = os.environ.get("HF_TOKEN", None)
                if not hf_token:
                    token_path = os.path.expanduser("~/.cache/huggingface/token")
                    if os.path.exists(token_path):
                        with open(token_path, "r") as f:
                            hf_token = f.read().strip()
        
                print(f"Loading {self.model_id}...")
                from transformers import pipeline
                import torch
                self.pipe = pipeline(
                    "image-text-to-text",
                    model=self.model_id,
                    token=hf_token,
                    torch_dtype=torch.bfloat16,
                    device_map="auto"
                )
                self.processor = None
                self.model = None
                 print("MedGemma loaded successfully.")
            except Exception as e:
                print(f"Failed to load MedGemma: {e}")
                self.use_mock_mode = True
        else:
            if not HAS_TRANSFORMERS:
                print("transformers not installed.")
            print("Using MedGemma in MOCK mode.")
            self.use_mock_mode = True

    def generate_report(self, image_path: str) -> str:
        if self.use_mock_mode:
            return (
                f"[MOCK REPORT for {os.path.basename(image_path)}]\n"
                "Findings: The lungs are clear.\n"
                "Impression: Normal chest radiograph."
            )
        try:
            from PIL import Image
            img = Image.open(image_path).convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "You are a radiologist. Generate a structured chest X-ray report with Findings and Impression sections."}
            ]}]
            out = self.pipe(messages, max_new_tokens=200)
            return out[0]["generated_text"][-1]["content"]
        except Exception as e:
            return f"[ERROR] {e}"

    def answer_question(self, image_path: str, question: str, context: str) -> str:
        if self.use_mock_mode:
            return f"[MOCK] Based on context: {context[:50]}..."
        try:
            from PIL import Image
            img = Image.open(image_path).convert("RGB")
            prompt = (
                f"Context: {context[:300]}\n"
                f"Question: {question}\n"
                "Answer concisely based on the image and context:"
            )
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt}
            ]}]
            out = self.pipe(messages, max_new_tokens=80)
            return out[0]["generated_text"][-1]["content"]
        except Exception as e:
            return f"[ERROR] {e}"
