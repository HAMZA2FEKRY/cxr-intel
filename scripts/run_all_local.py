import os
import subprocess
import yaml
import sys

def run_step(name, cmd):
    print(f"\n{'='*50}\nRunning {name}...\n{'='*50}")
    subprocess.run(cmd, check=False)

def main():
    run_step("Inspect Dataset", [sys.executable, "scripts/inspect_dataset.py"])
    run_step("Prepare Dataset", [sys.executable, "scripts/prepare_dataset.py"])
    run_step("Create Sample", [sys.executable, "scripts/download_subset_images.py", "--sample_size", "1000"])
    run_step("Generate QA Dataset", [sys.executable, "scripts/generate_qa_dataset.py"])
    
    run_step("Build CLIP Index", [sys.executable, "rag/build_index.py", "--model", "clip"])
    run_step("Build ColPali Index", [sys.executable, "rag/build_index.py", "--model", "colpali"])
    
    run_step("Report Generation Pipeline", [sys.executable, "scripts/run_report_generation.py"])
    run_step("QA Pipeline", [sys.executable, "scripts/run_qa_pipeline.py"])
    
    run_step("Evaluate Reports", [sys.executable, "evaluation/evaluate_reports.py"])
    run_step("Evaluate Retrieval", [sys.executable, "evaluation/evaluate_retrieval.py"])
    run_step("Evaluate QA", [sys.executable, "evaluation/evaluate_qa.py"])
    run_step("Compare Models", [sys.executable, "evaluation/compare_models.py"])
    
    print("\nAll steps completed! You can now run the Streamlit app:")
    print("streamlit run app/streamlit_app.py")

if __name__ == "__main__":
    main()
