import os
import json
import pandas as pd
import yaml
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.retriever import CXRRetriever

def evaluate_retriever(retriever, df, top_k):
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    total = len(df)
    
    print(f"Evaluating {retriever.retriever_type} retrieval on {total} samples...")
    for _, row in df.iterrows():
        image_id = row['image_id']
        image_path = row['sample_image_path']
        
        results = retriever.retrieve(image_path, question=None, top_k=5)
        retrieved_ids = [r['image_id'] for r in results]
        
        if len(retrieved_ids) > 0 and retrieved_ids[0] == image_id:
            hits_at_1 += 1
        if image_id in retrieved_ids[:3]:
            hits_at_3 += 1
        if image_id in retrieved_ids[:5]:
            hits_at_5 += 1
            
    return {
        "Retrieval@1": hits_at_1 / total if total > 0 else 0,
        "Retrieval@3": hits_at_3 / total if total > 0 else 0,
        "Retrieval@5": hits_at_5 / total if total > 0 else 0
    }

def main():
    input_csv = "data/samples/sample_metadata.csv"
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found.")
        return
        
    df = pd.read_csv(input_csv)
    
    clip_retriever = CXRRetriever("clip")
    colpali_retriever = CXRRetriever("colpali")
    
    clip_metrics = evaluate_retriever(clip_retriever, df, 5)
    colpali_metrics = evaluate_retriever(colpali_retriever, df, 5)
    
    metrics = {
        "clip": clip_metrics,
        "colpali": colpali_metrics
    }
    
    out_json = "results/retrieval_metrics.json"
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"Saved retrieval metrics to {out_json}")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
