import os
import pandas as pd
import yaml
import glob

def find_image_files(raw_dir):
    image_files = {}
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.dcm']:
        for path in glob.glob(os.path.join(raw_dir, '**', ext), recursive=True):
            basename = os.path.basename(path)
            name_without_ext = os.path.splitext(basename)[0]
            image_files[name_without_ext] = path
    return image_files

def main():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    raw_dir = config["dataset"]["raw_dir"]
    processed_dir = config["dataset"]["processed_dir"]
    os.makedirs(processed_dir, exist_ok=True)
    
    report_candidates = config["columns"]["report_candidates"]
    image_candidates = config["columns"]["image_candidates"]
    id_candidates = config["columns"]["id_candidates"]
    
    csv_files = []
    for root, _, files in os.walk(raw_dir):
        for file in files:
            if file.lower().endswith('.csv'):
                csv_files.append(os.path.join(root, file))
                
    if not csv_files:
        print("No CSV files found in the raw dataset.")
        return
        
    # Assume the largest CSV has the reports
    main_csv = max(csv_files, key=os.path.getsize)
    print(f"Loading {main_csv}...")
    
    df = pd.read_csv(main_csv)
    total_initial = len(df)
    
    report_col = next((c for c in df.columns if c.lower() in report_candidates), None)
    image_col = next((c for c in df.columns if c.lower() in image_candidates), None)
    id_col = next((c for c in df.columns if c.lower() in id_candidates), None)
    
    if not report_col:
        print("Could not find a report column. Exiting.")
        return
        
    if not id_col:
        print("Could not find an ID column. Generating sequential IDs.")
        df['image_id'] = [f"img_{i}" for i in range(len(df))]
        id_col = 'image_id'
        
    print("Cleaning reports...")
    df = df.dropna(subset=[report_col])
    df['report'] = df[report_col].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()
    df = df[df['report'].str.len() > 30]
    
    valid_reports = len(df)
    
    print("Matching image paths...")
    if image_col and df[image_col].notna().any():
        def fix_path(p):
            if pd.isna(p): return None
            full = os.path.join(raw_dir, str(p))
            if os.path.exists(full): return full
            return None
        df['image_path'] = df[image_col].apply(fix_path)
    else:
        print("Searching for images by ID...")
        images_on_disk = find_image_files(raw_dir)
        df['image_path'] = df[id_col].astype(str).map(images_on_disk)
        
    missing_images = df['image_path'].isna().sum()
    df = df.dropna(subset=['image_path'])
    valid_paths = len(df)
    
    df['image_id'] = df[id_col]
    
    out_df = df[['image_id', 'image_path', 'report']]
    out_path = os.path.join(processed_dir, "image_report_pairs.csv")
    out_df.to_csv(out_path, index=False)
    
    print("\n--- Summary ---")
    print(f"Total initial rows: {total_initial}")
    print(f"Valid reports (>30 chars): {valid_reports}")
    print(f"Valid image paths found: {valid_paths}")
    print(f"Missing image paths: {missing_images}")
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
