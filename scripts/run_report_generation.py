import os
import pandas as pd
import sys
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.medgemma_wrapper import MedGemmaGenerator

def main():
    input_csv = "data/samples/sample_metadata.csv"
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found.")
        return

    df = pd.read_csv(input_csv)

    generator = MedGemmaGenerator()
    mode_used = "mock" if getattr(generator, 'use_mock_mode', False) else "real"

    results = []

    print(f"Generating reports for {len(df)} samples...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        image_path = row['sample_image_path']
        if pd.isna(image_path) or not os.path.exists(image_path):
            continue

        gen_report = generator.generate_report(image_path)

        results.append({
            "image_id": row['image_id'],
            "image_path": image_path,
            "reference_report": row['report'],
            "generated_report": gen_report,
            "mode_used": mode_used
        })

    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "report_generation_results.csv")

    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"Saved results to {out_csv}")

if __name__ == "__main__":
    main()