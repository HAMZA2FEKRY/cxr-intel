import os
import json

def main():
    username = os.environ.get("KAGGLE_USERNAME", "YOUR_USERNAME")
    key = os.environ.get("KAGGLE_KEY", "YOUR_KAGGLE_KEY")
    
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    
    kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")
    
    if not os.path.exists(kaggle_json_path):
        with open(kaggle_json_path, "w") as f:
            json.dump({"username": username, "key": key}, f)
        print(f"Created {kaggle_json_path}")
        if os.name == 'posix':
            os.chmod(kaggle_json_path, 0o600)
    else:
        print(f"{kaggle_json_path} already exists.")

if __name__ == "__main__":
    main()
