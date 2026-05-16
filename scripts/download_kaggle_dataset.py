import os
import argparse
import subprocess
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force redownload")
    args = parser.parse_args()
    
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    kaggle_slug = config["dataset"]["kaggle_slug"]
    raw_dir = config["dataset"]["raw_dir"]
    
    kaggle_json_path = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_json_path):
        print(f"Error: kaggle.json not found at {kaggle_json_path}")
        print("Please follow these instructions:")
        print("1. Go to Kaggle Account Settings (https://www.kaggle.com/settings)")
        print("2. Create New API token")
        print(f"3. Place kaggle.json in {kaggle_json_path}")
        print("4. Never commit it to git")
        return
        
    os.makedirs(raw_dir, exist_ok=True)
    
    # Check if dataset already downloaded
    if os.listdir(raw_dir) and not args.force:
        print(f"Dataset already exists in {raw_dir}. Use --force to redownload.")
        return
        
    print(f"Downloading {kaggle_slug} to {raw_dir}...")
    cmd = ["kaggle", "datasets", "download", "-d", kaggle_slug, "-p", raw_dir, "--unzip"]
    if args.force:
        cmd.append("--force")
        
    subprocess.run(cmd, check=True)
    print("Download completed successfully!")

if __name__ == "__main__":
    main()
