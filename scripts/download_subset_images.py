import os
import argparse
import pandas as pd
import subprocess
import yaml
import ast
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_size", type=int, default=1000)
    args = parser.parse_args()
    
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    csv_path = config["dataset"].get("raw_train_csv", "data/raw/mimic_cxr/mimic_cxr_aug_train.csv")
    sample_size = args.sample_size
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    df = df.dropna(subset=['image', 'text'])
    
    df['report'] = df['text'].astype(str).str.strip()
    df = df[df['report'].str.len() > 30]
    
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        
    out_dir = "data/samples/images"
    os.makedirs(out_dir, exist_ok=True)
    
    failed_file = "results/failed_image_downloads.txt"
    os.makedirs(os.path.dirname(failed_file), exist_ok=True)
    
    kaggle_slug = config["dataset"]["kaggle_slug"]
    
    sample_metadata = []
    failed_downloads = []
    
    print(f"Downloading {len(df)} images from Kaggle...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # The CSV stores 'image' as a string representation of a list
            paths = ast.literal_eval(row['image'])
            if not paths: 
                continue
            orig_image_path = str(paths[0]).replace('\\', '/')
        except Exception:
            continue
            
        kaggle_path = orig_image_path
        if kaggle_path.startswith("files/"):
            kaggle_path = "official_data_iccv_final/" + kaggle_path
            
        file_name = os.path.basename(kaggle_path)
        dest_path = os.path.join(out_dir, file_name)
        
        image_id = row.get('subject_id', f"img_{idx}")
        image_id = f"{image_id}_{idx}"
        
        if not os.path.exists(dest_path):
            import sys
            cmd = [
                sys.executable, "-m", "kaggle", "datasets", "download", "-d", kaggle_slug, 
                "-f", kaggle_path, "-p", out_dir, "--unzip"
            ]
            try:
                res = subprocess.run(cmd, capture_output=True, text=True)
                if res.returncode != 0 or not os.path.exists(dest_path):
                    failed_downloads.append(f"{kaggle_path} | Error: {res.stderr.strip()}")
                    continue
            except Exception as e:
                failed_downloads.append(f"{kaggle_path} | Exception: {e}")
                continue
                
        sample_metadata.append({
            "image_id": image_id,
            "image_path": orig_image_path,
            "report": row['report'],
            "sample_image_path": dest_path,
            "subject_id": row.get('subject_id', ''),
            "view": row.get('view', '')
        })
        
    meta_df = pd.DataFrame(sample_metadata)
    meta_path = config["dataset"].get("sample_metadata_path", "data/samples/sample_metadata.csv")
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    meta_df.to_csv(meta_path, index=False)
    
    with open(failed_file, "w") as f:
        f.write("\n".join(failed_downloads))
        
    print(f"\nSuccessfully created sample with {len(meta_df)} records.")
    print(f"Failed downloads: {len(failed_downloads)}")
    print(f"Metadata saved to {meta_path}")

if __name__ == "__main__":
    main()