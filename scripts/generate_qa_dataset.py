import os
import json
import pandas as pd
import yaml

def main():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    input_csv = config.get("dataset", {}).get("sample_metadata_path", "data/samples/sample_metadata.csv")
    
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found.")
        return
        
    df = pd.read_csv(input_csv)
    
    findings_to_check = [
        "atelectasis", "cardiomegaly", "consolidation", "edema",
        "enlarged cardiomediastinum", "fracture", "lung lesion",
        "lung opacity", "pleural effusion", "pleural abnormality",
        "pneumonia", "pneumothorax", "support devices", "heart size",
        "no acute finding"
    ]
    
    negation_words = ["no ", "no evidence of ", "without ", "negative for ", "absence of "]
    
    qa_pairs = []
    
    print(f"Generating QA dataset for {len(df)} samples...")
    for _, row in df.iterrows():
        report = str(row['report']).lower()
        image_id = row['image_id']
        image_path = row['sample_image_path']
        
        for finding in findings_to_check:
            question = f"Is {finding} observed?"
            if finding == "cardiomegaly":
                question = "Does the radiograph show cardiomegaly?"
            elif finding == "support devices":
                question = "Are support devices present?"
                
            if finding in report:
                is_negated = False
                for neg in negation_words:
                    if f"{neg}{finding}" in report:
                        is_negated = True
                        break
                
                if is_negated:
                    answer = f"No, {finding} is not observed according to the report."
                else:
                    answer = f"Yes, {finding} is observed or suggested in the report."
            else:
                answer = f"The report does not mention {finding}."
                
            qa_pairs.append({
                "image_id": image_id,
                "image_path": image_path,
                "question": question,
                "answer": answer,
                "context": str(row['report']),
                "finding_type": finding,
                "source": "rule_based_from_report"
            })
            
    out_json = "data/qa_dataset/qa_pairs.json"
    out_csv = "data/qa_dataset/qa_pairs.csv"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    
    with open(out_json, "w") as f:
        json.dump(qa_pairs, f, indent=2)
        
    pd.DataFrame(qa_pairs).to_csv(out_csv, index=False)
    
    print(f"Generated {len(qa_pairs)} QA pairs.")
    print(f"Saved to {out_json} and {out_csv}")

if __name__ == "__main__":
    main()
