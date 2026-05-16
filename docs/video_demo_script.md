# Video Demo Script

1. **Show GitHub repo structure:** Open VS Code, show folders and clean setup.
2. **Show dataset preparation:** Run `python scripts/prepare_dataset.py` and show the output CSV.
3. **Show generated QA dataset:** Open `data/qa_dataset/qa_pairs.json`.
4. **Show report generation tab:** Open the Streamlit app, navigate to Tab 1, upload an image, and click Generate.
5. **Show QA-RAG with ColPali:** Navigate to Tab 2, select ColPali, ask "Is there evidence of pneumonia?", and view the answer and retrieved context.
6. **Show QA-RAG with CLIP:** Repeat the above with CLIP to compare.
7. **Show comparison results:** Show the output of `results/model_comparison.csv`.
8. **Explain limitations:** Clearly state this is an academic prototype not for clinical use.
