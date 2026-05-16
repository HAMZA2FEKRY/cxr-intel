import os
import pandas as pd
import yaml

def main():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    raw_dir = config["dataset"]["raw_dir"]
    output_file = "results/dataset_inspection.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    report_candidates = config["columns"]["report_candidates"]
    image_candidates = config["columns"]["image_candidates"]
    
    lines = []
    lines.append(f"Inspecting directory: {raw_dir}\n")
    
    csv_files = []
    image_files = []
    
    for root, _, files in os.walk(raw_dir):
        for file in files:
            ext = file.lower()
            if ext.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
            elif ext.endswith(('.png', '.jpg', '.jpeg', '.dcm')):
                image_files.append(os.path.join(root, file))
                
    lines.append(f"Found {len(csv_files)} CSV files.")
    lines.append(f"Found {len(image_files)} image files.\n")
    
    for csv_file in csv_files:
        lines.append(f"--- CSV: {csv_file} ---")
        try:
            df = pd.read_csv(csv_file, nrows=5)
            lines.append(f"Columns: {list(df.columns)}")
            
            # Find possible report column
            report_cols = [c for c in df.columns if c.lower() in report_candidates]
            image_cols = [c for c in df.columns if c.lower() in image_candidates]
            
            lines.append(f"Possible report columns: {report_cols}")
            lines.append(f"Possible image columns: {image_cols}")
            
            lines.append("\nFirst 5 rows:")
            lines.append(df.to_string())
            
            # Count total rows properly
            with open(csv_file, 'r', encoding='utf-8') as f:
                row_count = sum(1 for _ in f) - 1 # subtract header
            lines.append(f"\nTotal rows: {row_count}\n")
            
        except Exception as e:
            lines.append(f"Error reading {csv_file}: {e}\n")
            
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
        
    print(f"Inspection complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()
