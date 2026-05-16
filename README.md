# CXR-Intel: Multi-Modal Chest X-Ray Intelligence System

## 1. Project Overview
CXR-Intel is a dual-mode AI system designed for analyzing chest X-ray images. It provides structured report generation and grounded Visual Question Answering (QA-RAG) using advanced multi-modal models.

## 2. Assignment Objective
Build a dual-mode chest X-ray AI system using MedGemma for text generation and ColPali/CLIP for retrieval augmented generation (RAG).

## 3. Architecture Diagram

```
[ Chest X-Ray Image ]
       |
       v
+------------------+
|  MedGemma (4B)   | -----> [ Structured Radiology Report ]
+------------------+

[ Chest X-Ray Image ] + [ User Question ]
       |
       v
+------------------+
| ColPali or CLIP  | -----> [ Retrieved Context ]
+------------------+             |
                                 v
                        +------------------+
                        |  MedGemma (4B)   | -----> [ Grounded Answer ]
                        +------------------+
```

## 4. Setup Steps for Windows VS Code
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## 5. Kaggle Dataset Download
Create `~/.kaggle/kaggle.json` using your Kaggle API key, or use `scripts/setup_kaggle.py`.
Then run:
```powershell
python scripts/download_kaggle_dataset.py
```

## 6. Hugging Face MedGemma Access
1. Go to `google/medgemma-1.5-4b-it` on HuggingFace.
2. Accept the model conditions.
3. Get your HF_TOKEN and add to `.env`.

## 7. How to Prepare Data
```powershell
python scripts/inspect_dataset.py
python scripts/prepare_dataset.py
python scripts/create_sample.py --sample_size 300
```

## 8. How to Generate QA Dataset
```powershell
python scripts/generate_qa_dataset.py
```

## 9. How to Build Indexes
```powershell
python rag/build_index.py --model clip
python rag/build_index.py --model colpali
```

## 10. How to Run Streamlit App
```powershell
streamlit run app/streamlit_app.py
```

## 11. How to Run Evaluations
```powershell
python scripts/run_report_generation.py
python scripts/run_qa_pipeline.py
python evaluation/evaluate_reports.py
python evaluation/evaluate_retrieval.py
python evaluation/evaluate_qa.py
python evaluation/compare_models.py
```

## 12. GitHub Upload Notes
Do not commit raw datasets, model weights, or API keys (`.env`, `kaggle.json`). Check the `.gitignore`.

## 13. Limitations
This is an academic prototype and MUST NOT be used for clinical diagnosis. High hardware requirements (16GB+ VRAM) are needed for full models; mock mode is available for local testing.
