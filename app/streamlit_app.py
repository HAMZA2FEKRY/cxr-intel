import os
import streamlit as st
import sys
import pandas as pd
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.medgemma_wrapper import MedGemmaGenerator
from rag.retriever import CXRRetriever

st.set_page_config(page_title="CXR-Intel", layout="wide")

@st.cache_resource
def get_generator():
    return MedGemmaGenerator()

@st.cache_resource
def get_retriever(name):
    return CXRRetriever(name)

def main():
    st.title("CXR-Intel: Multi-Modal Chest X-Ray Intelligence System")
    st.markdown("⚠️ **Academic prototype only. Not for clinical diagnosis.**")
    
    st.sidebar.header("System Status")
    
    config_file = "configs/config.yaml"
    sample_size = 300
    if os.path.exists(config_file):
        import yaml
        with open(config_file, "r") as f:
            c = yaml.safe_load(f)
            sample_size = c.get("dataset", {}).get("sample_size", 300)
            
    st.sidebar.write(f"**Dataset Sample Size:** {sample_size}")
    
    generator = get_generator()
    mode_str = "Mock/Fallback" if getattr(generator, 'use_mock_mode', False) else "Real MedGemma"
    st.sidebar.write(f"**Model Status:** {mode_str}")
    
    clip_exists = os.path.exists("rag/indexes/clip.index")
    colpali_exists = os.path.exists("rag/indexes/colpali.index") or os.path.exists("rag/indexes/colpali_mock.index")
    
    st.sidebar.write("**Index Status:**")
    st.sidebar.write(f"- CLIP: {'✅' if clip_exists else '❌'}")
    st.sidebar.write(f"- ColPali: {'✅' if colpali_exists else '❌'}")
    
    tab1, tab2 = st.tabs(["Report Generation", "QA-RAG"])
    
    with tab1:
        st.header("Report Generation Mode")
        uploaded_file = st.file_uploader("Upload CXR Image", type=["png", "jpg", "jpeg"], key="rg")
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=400)
            
            if st.button("Generate Report"):
                with st.spinner("Generating report..."):
                    temp_path = "temp_upload.png"
                    image.save(temp_path)
                    
                    report = generator.generate_report(temp_path)
                    
                    st.subheader("Generated Structured Report")
                    st.write(report)
                    st.info(f"Mode Used: {mode_str}")
                    
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        
    with tab2:
        st.header("QA-RAG Mode")
        uploaded_file_qa = st.file_uploader("Upload CXR Image", type=["png", "jpg", "jpeg"], key="qa")
        
        col1, col2 = st.columns(2)
        with col1:
            retriever_choice = st.selectbox("Select Retriever", ["ColPali", "CLIP"])
            question = st.text_input("Enter your question:", "Is there evidence of pneumonia?")
        with col2:
            top_k = st.slider("Top K Contexts", min_value=1, max_value=5, value=3)
            
        if uploaded_file_qa is not None:
            image_qa = Image.open(uploaded_file_qa)
            st.image(image_qa, caption="Uploaded Image", width=400)
            
            if st.button("Answer"):
                if (retriever_choice == "CLIP" and not clip_exists) or (retriever_choice == "ColPali" and not colpali_exists):
                    st.error(f"{retriever_choice} index not found. Please build it first.")
                else:
                    with st.spinner(f"Retrieving and answering with {retriever_choice}..."):
                        temp_path = "temp_qa_upload.png"
                        image_qa.save(temp_path)
                        
                        retriever = get_retriever(retriever_choice.lower())
                        results = retriever.retrieve(temp_path, question, top_k=top_k)
                        
                        context = "\n---\n".join([r['report'] for r in results])
                        
                        answer = generator.answer_question(temp_path, question, context)
                        
                        st.subheader("Grounded Answer")
                        st.write(answer)
                        
                        with st.expander("View Retrieved Context"):
                            for i, r in enumerate(results):
                                st.markdown(f"**Document {i+1}** (Score: {r.get('score', 0):.4f})")
                                st.write(r['report'])
                                
                        if os.path.exists(temp_path):
                            os.remove(temp_path)

if __name__ == "__main__":
    main()
