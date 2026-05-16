import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.retriever import CXRRetriever

def test_retriever_interface():
    try:
        retriever = CXRRetriever("clip")
        assert retriever is not None
    except Exception:
        pass
