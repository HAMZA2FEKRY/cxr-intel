import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.clip_encoder import CLIPEncoder
from models.colpali_encoder import ColPaliEncoder
from rag.faiss_store import FAISSStore

class CXRRetriever:
    def __init__(self, retriever_type: str):
        self.retriever_type = retriever_type.lower()
        self.store = FAISSStore()
        
        index_dir = "rag/indexes"
        
        if self.retriever_type == "clip":
            self.encoder = CLIPEncoder()
            index_path = os.path.join(index_dir, "clip.index")
            meta_path = os.path.join(index_dir, "clip_metadata.json")
        elif self.retriever_type == "colpali":
            self.encoder = ColPaliEncoder()
            prefix = "colpali_mock" if getattr(self.encoder, 'use_mock_mode', False) else "colpali"
            index_path = os.path.join(index_dir, f"{prefix}.index")
            meta_path = os.path.join(index_dir, f"{prefix}_metadata.json")
            
            if not os.path.exists(index_path) and prefix == "colpali":
                fallback_path = os.path.join(index_dir, "colpali_mock.index")
                if os.path.exists(fallback_path):
                    index_path = fallback_path
                    meta_path = os.path.join(index_dir, "colpali_mock_metadata.json")
        else:
            raise ValueError("Unknown retriever type.")
            
        try:
            self.store.load(index_path, meta_path)
            print(f"Loaded {self.retriever_type} index from {index_path}")
        except FileNotFoundError:
            print(f"Warning: Index not found at {index_path}. Retrieval will return empty results.")
            
    def retrieve(self, image_path: str, question: str, top_k: int = 3) -> list[dict]:
        if self.store.index is None:
            return []
            
        query_embedding = self.encoder.encode_query(image_path, question)
        results = self.store.search(query_embedding, top_k)
        return results
