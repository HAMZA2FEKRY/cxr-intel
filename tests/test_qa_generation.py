import os
import json

def test_qa_pairs_format():
    path = "data/qa_dataset/qa_pairs.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
        if len(data) > 0:
            item = data[0]
            assert "question" in item
            assert "answer" in item
            assert "context" in item
