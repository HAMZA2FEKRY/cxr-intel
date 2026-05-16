import os
import sys
import streamlit as st
import pandas as pd
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.medgemma_wrapper import MedGemmaGenerator
from rag.retriever import CXRRetriever

st.set_page_config(
    page_title="CXR-Intel",
    page_icon="🫁",
    layout="wide"
)

# ── Cache expensive model loading ──
@st.cache_resource
def get_generator():
    return MedGemmaGenerator()

@st.cache_resource
def get_retriever(name):
    return CXRRetriever(name)

def check_indexes():
    clip_ok = os.path.exists("rag/indexes/clip.index")
    colpali_ok = (
        os.path.exists("rag/indexes/colpali.index") or
        os.path.exists("rag/indexes/colpali_mock.index")
    )
    reports_ok = os.path.exists("results/report_generation_results.csv")
    metrics_ok = os.path.exists("results/report_generation_metrics.json")
    return clip_ok, colpali_ok, reports_ok, metrics_ok

def main():
    st.title("🫁 CXR-Intel: Multi-Modal Chest X-Ray Intelligence System")
    st.caption("⚠️ Academic prototype only — not for clinical use.")

    # ── Sidebar ──
    with st.sidebar:
        st.header("System Status")

        clip_ok, colpali_ok, reports_ok, metrics_ok = check_indexes()

        generator = get_generator()
        mode_str = "🔴 Mock/Fallback" if getattr(generator, 'use_mock_mode', False) else "🟢 Real MedGemma"
        st.write(f"**Model:** {mode_str}")
        st.write(f"**CLIP Index:** {'✅' if clip_ok else '❌ Not found'}")
        st.write(f"**ColPali Index:** {'✅' if colpali_ok else '❌ Not found'}")
        st.write(f"**Reports:** {'✅' if reports_ok else '❌ Not found'}")

        st.divider()

        if metrics_ok:
            import json
            with open("results/report_generation_metrics.json") as f:
                m = json.load(f)
            st.subheader("Evaluation Metrics")
            st.metric("ROUGE-L", f"{m.get('rougeL', 0):.3f}")
            st.metric("BLEU", f"{m.get('bleu', 0):.4f}")
            st.metric("Reports Generated", m.get('count', 0))

        if os.path.exists("results/retrieval_metrics.json"):
            import json
            with open("results/retrieval_metrics.json") as f:
                rm = json.load(f)
            st.subheader("Retrieval Metrics")
            st.write("**CLIP:**")
            for k, v in rm.get('clip', {}).items():
                st.write(f"  {k}: `{v:.3f}`")
            st.write("**ColPali:**")
            for k, v in rm.get('colpali', {}).items():
                st.write(f"  {k}: `{v:.3f}`")

        if os.path.exists("results/comparison.png"):
            st.subheader("Model Comparison")
            st.image("results/comparison.png", use_column_width=True)

    # ── Main tabs ──
    tab1, tab2, tab3 = st.tabs([
        "📋 Report Generation",
        "❓ QA-RAG Mode",
        "📊 Results & Comparison"
    ])

    # ── TAB 1: Report Generation ──
    with tab1:
        st.header("Report Generation Mode")
        st.markdown("Upload a chest X-ray image and MedGemma will generate a structured radiology report.")

        col1, col2 = st.columns([1, 1])
        with col1:
            uploaded = st.file_uploader(
                "Upload Chest X-Ray",
                type=["png", "jpg", "jpeg"],
                key="rg"
            )
            if uploaded:
                image = Image.open(uploaded)
                st.image(image, caption="Uploaded X-Ray", use_column_width=True)

                if st.button("🔬 Generate Report", type="primary", use_container_width=True):
                    with st.spinner("MedGemma is analyzing the image..."):
                        temp_path = "/tmp/cxr_upload.png"
                        image.save(temp_path)
                        report = generator.generate_report(temp_path)
                        if os.path.exists(temp_path):
                            os.remove(temp_path)

                    with col2:
                        st.subheader("Generated Report")
                        st.info(f"Model: {mode_str}")

                        # Parse and display report sections nicely
                        if "Findings" in report or "FINDINGS" in report:
                            parts = report.replace("**", "").split("\n")
                            for part in parts:
                                if part.strip():
                                    st.write(part)
                        else:
                            st.write(report)

                        # Download button
                        st.download_button(
                            "⬇️ Download Report",
                            data=report,
                            file_name="cxr_report.txt",
                            mime="text/plain"
                        )

    # ── TAB 2: QA-RAG ──
    with tab2:
        st.header("QA-RAG Mode")
        st.markdown("Upload an X-ray, ask a clinical question. The system retrieves similar cases and uses MedGemma to answer.")

        col1, col2 = st.columns([1, 1])

        with col1:
            uploaded_qa = st.file_uploader(
                "Upload Chest X-Ray",
                type=["png", "jpg", "jpeg"],
                key="qa"
            )
            if uploaded_qa:
                image_qa = Image.open(uploaded_qa)
                st.image(image_qa, caption="Uploaded X-Ray", use_column_width=True)

            st.subheader("Question Settings")
            question = st.text_input(
                "Clinical Question:",
                "Is there evidence of pneumonia?"
            )

            retriever_choice = st.selectbox(
                "Retrieval Model:",
                ["CLIP", "ColPali"],
                help="CLIP: general visual retrieval. ColPali: document-aware retrieval."
            )

            top_k = st.slider("Retrieved Contexts (Top-K)", 1, 5, 3)

            if uploaded_qa and st.button("🔍 Retrieve & Answer", type="primary", use_container_width=True):
                retriever_name = retriever_choice.lower()

                if retriever_name == "clip" and not clip_ok:
                    st.error("CLIP index not found.")
                elif retriever_name == "colpali" and not colpali_ok:
                    st.error("ColPali index not found.")
                else:
                    with st.spinner(f"Retrieving with {retriever_choice} and generating answer..."):
                        temp_path = "/tmp/cxr_qa_upload.png"
                        image_qa.save(temp_path)

                        try:
                            retriever = get_retriever(retriever_name)
                            results = retriever.retrieve(temp_path, question, top_k=top_k)
                            context = "\n---\n".join([r['report'] for r in results])
                            answer = generator.answer_question(temp_path, question, context)
                        except Exception as e:
                            results = []
                            context = ""
                            answer = f"Error: {e}"

                        if os.path.exists(temp_path):
                            os.remove(temp_path)

                    with col2:
                        st.subheader("Answer")
                        st.success(answer)
                        st.info(f"Retrieved {len(results)} similar cases using {retriever_choice}")

                        with st.expander(f"View {len(results)} Retrieved Reports"):
                            for i, r in enumerate(results):
                                st.markdown(f"**Case {i+1}** — Similarity Score: `{r.get('score', 0):.4f}`")
                                st.write(r.get('report', '')[:300] + "...")
                                st.divider()

    # ── TAB 3: Results ──
    with tab3:
        st.header("Evaluation Results & Model Comparison")

        if os.path.exists("results/comparison.png"):
            st.image("results/comparison.png", caption="Model Comparison Chart", use_column_width=True)

        col1, col2 = st.columns(2)

        with col1:
            if reports_ok:
                st.subheader("Sample Generated Reports")
                df = pd.read_csv("results/report_generation_results.csv")
                st.write(f"Total reports: **{len(df)}**")
                mode = df['mode_used'].iloc[0] if 'mode_used' in df.columns else 'unknown'
                st.write(f"Generation mode: **{mode}**")

                idx = st.slider("View Report #", 0, min(len(df)-1, 10), 0)
                row = df.iloc[idx]
                st.markdown("**Generated:**")
                st.write(str(row['generated_report'])[:600])
                st.markdown("**Reference:**")
                st.write(str(row['reference_report'])[:400])

        with col2:
            if os.path.exists("results/qa_results.csv"):
                st.subheader("Sample QA Results")
                df_qa = pd.read_csv("results/qa_results.csv")
                st.write(f"Total QA pairs: **{len(df_qa)}**")

                idx_qa = st.slider("View QA #", 0, min(len(df_qa)-1, 10), 0)
                row_qa = df_qa.iloc[idx_qa]
                st.markdown(f"**Q:** {row_qa['question']}")
                st.markdown("**CLIP+MedGemma Answer:**")
                st.write(str(row_qa.get('clip_answer', ''))[:400])

        if os.path.exists("results/model_comparison.csv"):
            st.subheader("Full Model Comparison Table")
            df_cmp = pd.read_csv("results/model_comparison.csv")
            st.dataframe(df_cmp, use_container_width=True)

            st.markdown("""
            **Metric explanations:**
            - **ROUGE-L**: Longest common subsequence overlap between generated and reference reports. 
              Low scores are expected due to different clinical vocabulary.
            - **BLEU**: N-gram overlap. Near-zero is normal for free-text generation vs fixed references.
            - **Retrieval@K**: How often the correct image appears in the top-K retrieved results.
            - **Token Similarity**: Jaccard similarity between generated and reference answer tokens.
            """)

if __name__ == "__main__":
    main()
