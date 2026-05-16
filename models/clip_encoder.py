import os
import yaml
import numpy as np

class CLIPEncoder:
    def __init__(self, use_mock_mode=None):
        if use_mock_mode is None:
            # Read from config like the other models
            try:
                with open("configs/config.yaml", "r") as f:
                    config = yaml.safe_load(f)
                env_mock = os.environ.get("USE_MOCK_MODE", "").lower()
                cfg_mock = str(config["models"]["use_mock_mode"]).lower()
                self.use_mock_mode = (env_mock in ("true", "1")) or (env_mock == "" and cfg_mock in ("true", "1"))
            except Exception:
                self.use_mock_mode = False
        else:
            self.use_mock_mode = use_mock_mode
            
        self.model = None
        self.processor = None
        
        if not self.use_mock_mode:
            try:
                import torch
                from PIL import Image
                from transformers import CLIPProcessor, CLIPModel
                print("Loading CLIP baseline...")
                self.model_id = "openai/clip-vit-base-patch32"
                self.processor = CLIPProcessor.from_pretrained(self.model_id)
                self.model = CLIPModel.from_pretrained(self.model_id)
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model.to(self.device)
                print("CLIP loaded successfully.")
            except Exception as e:
                print(f"Failed to load CLIP: {e}")
                self.use_mock_mode = True
        else:
            print("Using CLIP in MOCK mode.")

    def encode_images(self, image_paths: list[str]) -> np.ndarray:
        if self.use_mock_mode:
            return np.random.rand(len(image_paths), 512).astype('float32')
            
        import torch
        from PIL import Image
        try:
            # Process images in batches to avoid memory issues
            all_embeddings = []
            batch_size = 16
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i+batch_size]
                images = [Image.open(p).convert("RGB") for p in batch_paths]
                inputs = self.processor(images=images, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    embeddings = self.model.get_image_features(**inputs)
                all_embeddings.append(embeddings.cpu().numpy())
            return np.concatenate(all_embeddings, axis=0).astype('float32')
        except Exception as e:
            print(f"CLIP encoding failed: {e}")
            return np.random.rand(len(image_paths), 512).astype('float32')

    def encode_query(self, image_path: str, question: str | None = None) -> np.ndarray:
        if self.use_mock_mode:
            return np.random.rand(1, 512).astype('float32')
            
        import torch
        from PIL import Image
        try:
            query_text = question if question else "A chest x-ray image."
            inputs = self.processor(text=[query_text], return_tensors="pt").to(self.device)
            with torch.no_grad():
                embeddings = self.model.get_text_features(**inputs)
            return embeddings.cpu().numpy().astype('float32')
        except Exception as e:
            print(f"CLIP query encoding failed: {e}")
            return np.random.rand(1, 512).astype('float32')
