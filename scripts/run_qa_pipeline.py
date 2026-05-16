import os
import pandas as pd
import json
import sys
from tqdm import tqdm
import yaml
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.medgemma_wrapper import MedGemmaGenerator
from rag.retriever import CXRRetriever

def main():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    top_k = config["rag"]["top_k"]
    input_json = "data/qa_dataset/qa_pairs.json"

    if not os.path.exists(input_json):
        print(f"Error: {input_json} not found.")
        return

    with open(input_json, "r") as f:
        qa_pairs = json.load(f)

    if len(qa_pairs) > 50:
        qa_pairs = qa_pairs[:50]

    generator = MedGemmaGenerator()
    clip_retriever = CXRRetriever("clip")
    colpali_retriever = CXRRetriever("colpali")

    results = []

    print(f"Running QA pipeline for {len(qa_pairs)} questions...")
    for item in tqdm(qa_pairs):
        image_id = item['image_id']
        image_path = item['image_path']
        question = item['question']
        ref_ans = item['answer']

        if pd.isna(image_path) or not os.path.exists(str(image_path)):
            continue

        # CLIP
        clip_res = clip_retriever.retrieve(image_path, question, top_k)
        clip_context = "\n".join([r['report'] for r in clip_res])
        clip_answer = generator.answer_question(image_path, question, clip_context)

        # ColPali
        colpali_res = colpali_retriever.retrieve(image_path, question, top_k)
        colpali_context = "\n".join([r['report'] for r in colpali_res])
        colpali_answer = generator.answer_question(image_path, question, colpali_context)

        results.append({
            "image_id": image_id,
            "question": question,
            "reference_answer": ref_ans,
            "colpali_context": colpali_context,
            "colpali_answer": colpali_answer,
            "clip_context": clip_context,
            "clip_answer": clip_answer,
            "retriever_top_k": top_k
        })

    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "qa_results.csv")

    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"Saved results to {out_csv}")

if __name__ == "__main__":
    main()