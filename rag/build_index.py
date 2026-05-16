import os
import argparse
import pandas as pd
import yaml
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.clip_encoder import CLIPEncoder
from models.colpali_encoder import ColPaliEncoder
from rag.faiss_store import FAISSStore

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["clip", "colpali"], required=True)
    parser.add_argument("--input", type=str, default=None)
    args = parser.parse_args()
    
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    input_csv = args.input or config.get("dataset", {}).get("sample_metadata_path", "data/samples/sample_metadata.csv")
    index_dir = config["rag"]["index_dir"]
    os.makedirs(index_dir, exist_ok=True)
    
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found.")
        return
        
    df = pd.read_csv(input_csv)
    df = df.dropna(subset=['sample_image_path'])
    
    image_paths = df['sample_image_path'].tolist()
    metadata = df.to_dict('records')
    
    print(f"Encoding {len(image_paths)} images using {args.model}...")
    
    if args.model == "clip":
        encoder = CLIPEncoder()
        embeddings = encoder.encode_images(image_paths)
        prefix = "clip"
    else:
        encoder = ColPaliEncoder()
        embeddings = encoder.encode_images(image_paths)
        prefix = "colpali_mock" if getattr(encoder, 'use_mock_mode', False) else "colpali"
        
    store = FAISSStore()
    store.build(embeddings, metadata)
    
    index_path = os.path.join(index_dir, f"{prefix}.index")
    meta_path = os.path.join(index_dir, f"{prefix}_metadata.json")
    
    store.save(index_path, meta_path)
    print(f"Saved index to {index_path} and metadata to {meta_path}")

if __name__ == "__main__":
    main()
