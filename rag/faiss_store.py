import os
import json
import faiss
import numpy as np

class FAISSStore:
    def __init__(self):
        self.index = None
        self.metadata = []

    def build(self, embeddings: np.ndarray, metadata: list[dict]):
        if len(embeddings) == 0:
            print("No embeddings to build index.")
            return
            
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embeddings)
        self.metadata = metadata

    def save(self, index_path: str, metadata_path: str):
        if self.index is None:
            return
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def load(self, index_path: str, metadata_path: str):
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Missing {index_path} or {metadata_path}")
            
        self.index = faiss.read_index(index_path)
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[dict]:
        if self.index is None or len(self.metadata) == 0:
            return []
            
        D, I = self.index.search(query_embedding, top_k)
        results = []
        for i, (score, idx) in enumerate(zip(D[0], I[0])):
            if idx < len(self.metadata):
                item = self.metadata[idx].copy()
                item['rank'] = i + 1
                item['score'] = float(score)
                results.append(item)
        return results
