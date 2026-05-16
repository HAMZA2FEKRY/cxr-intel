import os
import json
import pandas as pd
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu

def main():
    input_csv = "results/report_generation_results.csv"
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found.")
        return
        
    df = pd.read_csv(input_csv)
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    metrics = {
        "rougeL": 0.0,
        "bleu": 0.0,
        "count": 0
    }
    
    print("Evaluating reports...")
    for _, row in df.iterrows():
        ref = str(row['reference_report'])
        gen = str(row['generated_report'])
        
        scores = scorer.score(ref, gen)
        metrics["rougeL"] += scores['rougeL'].fmeasure
        
        ref_tokens = ref.split()
        gen_tokens = gen.split()
        bleu = sentence_bleu([ref_tokens], gen_tokens, weights=(0.5, 0.5))
        metrics["bleu"] += bleu
        
        metrics["count"] += 1
        
    if metrics["count"] > 0:
        metrics["rougeL"] /= metrics["count"]
        metrics["bleu"] /= metrics["count"]
        
    out_json = "results/report_generation_metrics.json"
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"Evaluation complete. Metrics saved to {out_json}")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
