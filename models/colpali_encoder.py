import os
import yaml
import numpy as np

class ColPaliEncoder:
    def __init__(self, use_mock_mode=None):
        with open("configs/config.yaml", "r") as f:
            config = yaml.safe_load(f)
            
        if use_mock_mode is None:
            env_mock = os.environ.get("USE_MOCK_MODE", "").lower()
            cfg_mock = str(config["models"]["use_mock_mode"]).lower()
            self.use_mock_mode = (env_mock in ("true", "1")) or (env_mock == "" and cfg_mock in ("true", "1"))
        else:
            self.use_mock_mode = use_mock_mode
            
        self.model = None
        self.processor = None
        
        if not self.use_mock_mode:
            try:
                import torch
                from colpali_engine.models import ColPali, ColPaliProcessor
                print("Loading ColPali...")
                model_name = "vidore/colpali-v1.2"
                self.processor = ColPaliProcessor.from_pretrained(model_name)
                
                # Determine device and dtype
                if torch.cuda.is_available():
                    self.model = ColPali.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,
                        device_map="auto"
                    ).eval()
                    self.device = self.model.device
                else:
                    # CPU: load in float32, no device_map to avoid disk offload error
                    self.model = ColPali.from_pretrained(
                        model_name,
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True
                    ).eval()
                    self.device = torch.device("cpu")
                print("ColPali loaded successfully.")
            except ImportError:
                print("colpali-engine not installed. Falling back to mock mode.")
                self.use_mock_mode = True
            except Exception as e:
                print(f"Failed to load ColPali: {e}")
                print("Falling back to mock mode.")
                self.use_mock_mode = True
        else:
            print("Using ColPali in MOCK mode.")
            
    def encode_images(self, image_paths: list[str]) -> np.ndarray:
        if self.use_mock_mode:
            return np.random.rand(len(image_paths), 128).astype('float32')
            
        import torch
        from PIL import Image
        try:
            images = [Image.open(p).convert("RGB") for p in image_paths]
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                embeddings = self.model(**inputs).embeddings
            return embeddings.mean(dim=1).cpu().numpy().astype('float32')
        except Exception as e:
            print(f"ColPali encoding failed: {e}")
            return np.random.rand(len(image_paths), 128).astype('float32')

    def encode_query(self, image_path: str, question: str | None = None) -> np.ndarray:
        if self.use_mock_mode:
            return np.random.rand(1, 128).astype('float32')
            
        import torch
        from PIL import Image
        try:
            query_text = question if question else "What is in this image?"
            inputs = self.processor(text=query_text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                embeddings = self.model(**inputs).embeddings
            return embeddings.mean(dim=1).cpu().numpy().astype('float32')
        except Exception as e:
            print(f"ColPali query encoding failed: {e}")
            return np.random.rand(1, 128).astype('float32')
