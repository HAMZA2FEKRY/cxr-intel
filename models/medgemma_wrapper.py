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

        if not self.use_mock_mode and HAS_TRANSFORMERS:
            try:
                # Get HF token from environment or cached file
                hf_token = os.environ.get("HF_TOKEN", None)
                if not hf_token:
                    token_path = os.path.expanduser("~/.cache/huggingface/token")
                    if os.path.exists(token_path):
                        with open(token_path, "r") as f:
                            hf_token = f.read().strip()

                print(f"Loading {self.model_id}...")

                self.processor = AutoProcessor.from_pretrained(
                    self.model_id,
                    token=hf_token
                )

                if torch.cuda.is_available():
                    self.model = AutoModelForImageTextToText.from_pretrained(
                        self.model_id,
                        token=hf_token,
                        device_map="auto",
                        torch_dtype=torch.float16
                    )
                else:
                    # CPU mode — float32, no device_map, low memory usage
                    self.model = AutoModelForImageTextToText.from_pretrained(
                        self.model_id,
                        token=hf_token,
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True
                    )

                print("MedGemma loaded successfully.")

            except Exception as e:
                print(f"Failed to load MedGemma: {e}")
                print("Falling back to mock mode.")
                self.use_mock_mode = True
        else:
            if not HAS_TRANSFORMERS:
                print("transformers not installed.")
            print("Using MedGemma in MOCK mode.")
            self.use_mock_mode = True

    def generate_report(self, image_path: str) -> str:
        prompt = "You are a medical imaging assistant for an academic project. Generate a concise structured chest X-ray report with Findings and Impression. Do not invent unsupported findings."

        if self.use_mock_mode:
            return (
                f"[MOCK REPORT for {os.path.basename(image_path)}]\n"
                "Findings: The lungs are clear. No pleural effusion or pneumothorax.\n"
                "Impression: Normal chest radiograph."
            )

        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = inputs.to(self.model.device)
            outputs = self.model.generate(**inputs, max_new_tokens=150)
            generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            return generated_text.replace(prompt, "").strip()
        except Exception as e:
            print(f"Generation failed: {e}")
            return "[ERROR] Failed to generate report."

    def answer_question(self, image_path: str, question: str, context: str) -> str:
        prompt = (
            "You are a medical imaging assistant for an academic project. "
            "Answer the question using only the provided image and retrieved context. "
            "If evidence is insufficient, say so. Do not provide a final clinical diagnosis. "
            f"Return a concise grounded answer.\nContext: {context}\nQuestion: {question}"
        )

        if self.use_mock_mode:
            return (
                f"[MOCK ANSWER]\n"
                f"Based on context: {context[:50]}...\n"
                f"Answer to '{question}' is: It appears normal based on the mock context."
            )

        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = inputs.to(self.model.device)
            outputs = self.model.generate(**inputs, max_new_tokens=100)
            generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            return generated_text.replace(prompt, "").strip()
        except Exception as e:
            print(f"QA failed: {e}")
            return "[ERROR] Failed to generate answer."