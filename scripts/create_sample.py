import os
import argparse
import pandas as pd
import shutil
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_size", type=int, default=None, help="Sample size")
    args = parser.parse_args()
    
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    sample_size = args.sample_size or config["dataset"]["sample_size"]
    
    processed_dir = config["dataset"]["processed_dir"]
    input_csv = os.path.join(processed_dir, "image_report_pairs.csv")
    
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found. Run prepare_dataset.py first.")
        return
        
    df = pd.read_csv(input_csv)
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        
    samples_dir = "data/samples/images"
    os.makedirs(samples_dir, exist_ok=True)
    
    sample_image_paths = []
    missing = 0
    
    print(f"Copying {len(df)} images to {samples_dir}...")
    for idx, row in df.iterrows():
        src = row['image_path']
        dest = os.path.join(samples_dir, os.path.basename(src))
        try:
            if not os.path.exists(dest):
                shutil.copy2(src, dest)
            sample_image_paths.append(dest)
        except Exception as e:
            print(f"Failed to copy {src}: {e}")
            sample_image_paths.append(None)
            missing += 1
            
    df['sample_image_path'] = sample_image_paths
    df = df.dropna(subset=['sample_image_path'])
    
    out_csv = "data/samples/sample_metadata.csv"
    df.to_csv(out_csv, index=False)
    
    print(f"\nCreated sample with {len(df)} records.")
    print(f"Missing/failed copies: {missing}")
    print(f"Saved metadata to {out_csv}")

if __name__ == "__main__":
    main()
