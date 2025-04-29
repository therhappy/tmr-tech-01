
"""
This script prepares the questions for evaluating the model, and is mostly based on the
`prep_indexes.py` script. It loads the ConvFinQA dataset, extracts the questions and answers,
but does not compute the embeddings nor upload them to Pinecone.
"""

import json
import numpy as np
import os
import tqdm
from torch import Tensor
from pinecone import Pinecone, ServerlessSpec

MODEL_NAME = "BAAI/bge-m3"
MODEL_OUT_DIM = 1024


def load_data(file_path: str) -> list[dict]:
    """
    Loads the data from a JSON file.
    Args:
        file_path (str): Path to the JSON file
    Returns:
        list: List of dictionaries containing the data points
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def prep_items_qa(data: list[dict]) -> list[dict]:
    """
    Prepares the contents of the QA-based index for the ConvFinQA dataset
    Args:
        data (list): List of dictionaries containing the data points
        model (SentenceTransformer): The SentenceTransformer model to use for encoding
    Returns:
        list: List of dictionaries containing the vectors and their metadata
    """
    
    vectors = []

    for datapoint_index, datapoint in tqdm.tqdm(enumerate(data)):

        qa_keys = [k for k in datapoint.keys() if k.startswith("qa")]
        
        for qa_index, qa_key in enumerate(qa_keys):
            
            vector_id = f"qa_{datapoint_index}:{qa_index}"
            question = datapoint[qa_key]["question"]
            answer = datapoint[qa_key]["answer"]
            document_id = datapoint["id"]
            
            gold_content = ''.join(datapoint[qa_key]["gold_inds"].values())
            operations = datapoint[qa_key]["program_re"]
            
            
            vectors.append({
                "id": vector_id,
                "values": "",
                "metadata": {
                    "question": question,
                    "answer": answer,
                    "document_id": document_id,
                    "gold_content": gold_content,
                    "operations": operations
                }
            })
    return vectors

if __name__ == "__main__":
    
    # Load data
    data = load_data("../data/convfinqa_train.json")
    
    # Prepare & vectors for QA
    print("Extracting QA items for evaluation ...")
    vectors_qa = prep_items_qa(data)
    # The condition is the invert of acceptance into the 'qa-test' set
    vectors_qa_eval = [v for i,v in enumerate(vectors_qa) if i % 10 == 0]
    
    # Save the vectors to a JSON file
    with open("../data/convfinqa_qa_eval.json", 'w') as f:
        json.dump(vectors_qa_eval, f, indent=4)
    
    print(f"Saved {len(vectors_qa_eval)} vectors to ../data/convfinqa_qa_eval.json")

