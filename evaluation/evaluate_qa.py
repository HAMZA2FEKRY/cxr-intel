import os
import json
import pandas as pd

def compute_similarity(ref, gen):
    ref_tokens = set(str(ref).lower().split())
    gen_tokens = set(str(gen).lower().split())
    
    if not ref_tokens:
        return 0.0
        
    overlap = len(ref_tokens.intersection(gen_tokens))
    return overlap / len(ref_tokens)

def main():
    input_csv = "results/qa_results.csv"
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found.")
        return
        
    df = pd.read_csv(input_csv)
    
    metrics = {
        "clip_exact_match": 0.0,
        "clip_similarity": 0.0,
        "colpali_exact_match": 0.0,
        "colpali_similarity": 0.0,
        "count": 0
    }
    
    print(f"Evaluating QA for {len(df)} questions...")
    for _, row in df.iterrows():
        ref = str(row['reference_answer']).strip().lower()
        clip_ans = str(row['clip_answer']).strip().lower()
        colpali_ans = str(row['colpali_answer']).strip().lower()
        
        ref_start = ref.split(',')[0] if ',' in ref else ref.split()[0]
        clip_start = clip_ans.split(',')[0] if ',' in clip_ans else clip_ans.split()[0]
        colpali_start = colpali_ans.split(',')[0] if ',' in colpali_ans else colpali_ans.split()[0]
        
        if ref_start == clip_start:
            metrics["clip_exact_match"] += 1
        if ref_start == colpali_start:
            metrics["colpali_exact_match"] += 1
            
        metrics["clip_similarity"] += compute_similarity(ref, clip_ans)
        metrics["colpali_similarity"] += compute_similarity(ref, colpali_ans)
        
        metrics["count"] += 1
        
    if metrics["count"] > 0:
        for k in metrics:
            if k != "count":
                metrics[k] /= metrics["count"]
                
    out_json = "results/qa_metrics.json"
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"Saved QA metrics to {out_json}")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
