# CXR-Intel Report

## 1. Introduction
Chest X-Ray intelligence systems can augment radiology workflows by generating reports and answering visual questions. 

## 2. Medical Background
Chest X-ray interpretation requires both image understanding and medical language reasoning.

## 3. Dataset
MIMIC-CXR provides images and report text. Due to hardware limitations, we sampled 300 records for the prototype.

## 4. QA Dataset Creation
The assignment dataset has reports but no QA pairs, so QA pairs were created from report text using rule-based extraction matching key findings (e.g., pneumonia, pleural effusion).

## 5. Architecture
The system employs a dual-mode approach: direct report generation and QA-RAG.

## 6. Report Generation Mode
MedGemma is used for report generation, taking the image and a prompt to produce findings and impression.

## 7. QA-RAG Mode
RAG helps ground answers in retrieved report context and reduce hallucination.

## 8. Models Used
- MedGemma is used for report generation and answer generation.
- ColPali is used for multimodal retrieval.
- CLIP is used as a baseline retriever.

## 9. Evaluation Metrics
- Reports: ROUGE-L, BLEU
- Retrieval: Hit@1, Hit@3, Hit@5
- QA: Exact Match (Yes/No), Token Similarity

## 10. Results
(Run the evaluation scripts to populate exact results)

## 11. Model Comparison
ColPali generally outperforms CLIP in capturing fine-grained medical details due to its patch-level interaction design.

## 12. Limitations
This is an academic prototype, not a clinical diagnostic tool. Generative models can hallucinate.

## 13. Conclusion
The integration of specialized multimodal models like MedGemma and ColPali shows promise in biomedical RAG applications.
