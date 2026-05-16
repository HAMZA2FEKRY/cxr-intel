import os
import json
import pandas as pd

def main():
    retrieval_file = "results/retrieval_metrics.json"
    qa_file = "results/qa_metrics.json"
    report_file = "results/report_generation_metrics.json"
    
    data = []
    
    if os.path.exists(report_file):
        with open(report_file, "r") as f:
            rep = json.load(f)
            data.append({
                "Model": "MedGemma (Report Generation)",
                "Metric": "ROUGE-L",
                "Score": rep.get("rougeL", 0)
            })
            data.append({
                "Model": "MedGemma (Report Generation)",
                "Metric": "BLEU",
                "Score": rep.get("bleu", 0)
            })
            
    if os.path.exists(retrieval_file):
        with open(retrieval_file, "r") as f:
            ret = json.load(f)
            for m in ["clip", "colpali"]:
                for k, v in ret.get(m, {}).items():
                    data.append({
                        "Model": f"{m.upper()} Retriever",
                        "Metric": k,
                        "Score": v
                    })
                    
    if os.path.exists(qa_file):
        with open(qa_file, "r") as f:
            qa = json.load(f)
            data.append({
                "Model": "CLIP + MedGemma QA",
                "Metric": "Exact Match",
                "Score": qa.get("clip_exact_match", 0)
            })
            data.append({
                "Model": "CLIP + MedGemma QA",
                "Metric": "Token Similarity",
                "Score": qa.get("clip_similarity", 0)
            })
            data.append({
                "Model": "ColPali + MedGemma QA",
                "Metric": "Exact Match",
                "Score": qa.get("colpali_exact_match", 0)
            })
            data.append({
                "Model": "ColPali + MedGemma QA",
                "Metric": "Token Similarity",
                "Score": qa.get("colpali_similarity", 0)
            })
            
    if not data:
        print("No metrics found to compare. Run evaluations first.")
        return
        
    df = pd.DataFrame(data)
    out_csv = "results/model_comparison.csv"
    df.to_csv(out_csv, index=False)
    
    print("\n=== Model Comparison ===")
    print(df.to_string(index=False))
    print(f"\nSaved to {out_csv}")

if __name__ == "__main__":
    main()
