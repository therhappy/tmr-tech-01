
"""
This script prepares the indexes for the ConvFinQA dataset in Pinecone.
Given the constrainsts of the task at hand - reasonable dataset size, no need for further additions,
and fully static settings - this is kept as a single synchronous script, and no time was spent on
optimizing the code for speed or memory usage.

Summary of the script:
1. Load the ConvFinQA dataset from a JSON file.
2. Load the BAAI/bge-m3 SentenceTransformer model.
3. Connect to Pinecone and ensure the index exists.
4. Prepare the vectors for the QA, documents, and tables.
5. Upsert the vectors into the Pinecone index in batches.
6. Print the index stats to verify the number of vectors in each namespace.
"""

import json
import numpy as np
import os
import tqdm
from torch import Tensor
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

MODEL_NAME = "BAAI/bge-m3"
MODEL_OUT_DIM = 1024

NAMESPACES = {
    "qa": {"expected_count": 3965},
    "qa-test": {"expected_count": 3568},
    "docs": {"expected_count": 3037},
    "tables": {"expected_count": 3037},
}

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


def ensure_index(pc: Pinecone, index_name: str, dim=int) -> None:
    """"
    Ensures that the index exists in Pinecone. If it doesn't, it creates a new one.
    """

    if not pc.has_index(name=index_name):
        # Create a new index
        pc.create_index(
            name=index_name,
            dimension=dim,
            metric='dotproduct',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )


def prep_vectors_qa(data: list[dict], model:SentenceTransformer) -> list[dict]:
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
            
            question_embedding = model.encode(question)
            
            vectors.append({
                "id": vector_id,
                "values": question_embedding,
                "metadata": {
                    "question": question,
                    "answer": answer,
                    "document_id": document_id,
                    "gold_content": gold_content,
                    "operations": operations
                }
            })
    return vectors


def compile_table_headers(table: dict) -> str:
    """
    Compiles the headers of a table into a single string to encode as vector for the table index
    Args:
        table (dict): The table to compile headers from, as in the original dataset's `table_ori` field
    Returns:
        str: The compiled headers as a string
    """
    header_cols = ", ".join(table[0])  # take the first full row
    header_rows = ", ".join([r[0] for r in table[1:]])  # and the leftmost column
    return header_cols + ", " + header_rows


def format_table(table: list[list[str]]) -> str:
    """
    Formats a table data in a string format to insert it as metadata in the Pinecone index
    Args:
        table (list): List of lists containing the table data, as in the original dataset's `table_ori` field
    Returns:
        str: Formatted string
    """
    row_str = lambda row : ' | '.join([e for e in row])
    return [row_str(row) for row in table]


def aggregate_embeddings(embeddings: list[np.ndarray | Tensor]) -> np.ndarray:
    """
    Aggregates a list of embeddings into a single embedding by taking the mean.
    Args:
        embeddings (list): List of numpy arrays containing the embeddings
    Returns:
        np.ndarray: The aggregated embedding
    """
    embeddings = [e.detach().cpu().numpy() if isinstance(e, Tensor) else e for e in embeddings]
    return np.mean(np.vstack(embeddings), axis=0)


def prep_vectors_docs_tables(data: list[dict], model:SentenceTransformer) -> tuple[list[dict], list[dict]]:
    """
    Prepares the contents of the documents and table indexes for the ConvFinQA dataset
    Args:
        data (list): List of dictionaries containing the data points
        model (SentenceTransformer): The SentenceTransformer model to use for encoding
    Returns:
        list: List of dictionaries containing the vectors and their metadata for the documents
        list: List of dictionaries containing the vectors and their metadata for the tables
    """
    
    doc_vectors = []
    table_vectors = []

    for datapoint_index, datapoint in tqdm.tqdm(enumerate(data)):
        
        # Extract document contents
        pre_text = datapoint["pre_text"]
        post_text = datapoint["post_text"]
        table_ori = datapoint["table_ori"]
        table_headers = compile_table_headers(table_ori)
        table_contents = format_table(table_ori)

        # Define vectors IDs
        doc_vector_id = f"doc_{datapoint_index}"
        table_vector_id = f"table_{datapoint_index}"
        document_id = datapoint["id"]
        
        # Compute embeddings
        pre_text_embedding = model.encode(pre_text)
        table_headers_embedding = model.encode(table_headers)
        post_text_embedding = model.encode(post_text)
        stacked_embedding = np.vstack([pre_text_embedding, table_headers_embedding, post_text_embedding])
        doc_embedding = np.mean(stacked_embedding, axis=0)
        
        doc_vectors.append({
            "id": doc_vector_id,
            "values": doc_embedding,
            "metadata": {
                "pre_text": pre_text,
                "post_text": post_text,
                "table": table_contents,
                "document_id": document_id
            }
            })
        
        table_vectors.append({
            "id": table_vector_id,
            "values": table_headers_embedding,
            "metadata": {
                "table": table_contents,
                "document_id": document_id
            }
        })

    return doc_vectors, table_vectors


def upsert_in_batches(index: Pinecone.Index, vectors: list[dict], namespace: str,
                      batch_size: int = 100) -> None:
    """
    Upserts vectors into the Pinecone index in batches.
    Args:
        index (Pinecone.Index): The Pinecone index to upsert into
        vectors (list): List of vectors to upsert
        batch_size (int): Size of each batch
    """
    for i in tqdm.tqdm(range(0, len(vectors), batch_size)):
        index.upsert(vectors=vectors[i:i + batch_size], namespace=namespace)


if __name__ == "__main__":
    
    # Load encoder model
    model = SentenceTransformer(MODEL_NAME)
    
    # Load data
    data = load_data("../data/convfinqa_train.json")
    
    # Prepare Pinecone client and index
    pc = Pinecone(os.environ["PINECONE_API_KEY"])    
    index_name = "tmr-convfinqa-qa"
    ensure_index(pc, index_name, dim=MODEL_OUT_DIM)
    index = pc.Index(index_name)
    index_stats = index.describe_index_stats()
    
    # Prepare & vectors for QA
    print("Extracting QA vectors ...")
    vectors_qa = prep_vectors_qa(data, model)
    vectors_qa_test = [v for i,v in enumerate(vectors_qa) if i % 10 > 0]

    print("Upserting QA vectors ...")
    #upsert_in_batches(index, vectors_qa, namespace="qa", batch_size=100)
    upsert_in_batches(index, vectors_qa_test, namespace="qa-test", batch_size=100)

    # Use another loop to prepare and upsert docs and tables vectors
    print("Extracting document and table vectors ...")
    vectors_docs, vectors_tables = prep_vectors_docs_tables(data, model)
    print("Upserting docs & table vectors ...")
    upsert_in_batches(index, vectors_docs, namespace="docs", batch_size=100)
    upsert_in_batches(index, vectors_tables, namespace="tables", batch_size=100)
    
    # Show index stats to demonstrate changes
    print("Index stats after upsert:")
    index_stats = index.describe_index_stats()
    print(index_stats)
    for namespace, stats in index_stats.items():
        if namespace in NAMESPACES:
            if stats["total_vector_count"] == NAMESPACES[namespace]["expected_count"]:
                print(f"Namespace {namespace} has the expected number of vectors: \
                        {NAMESPACES[namespace]['expected_count']}")
                continue    
            print(f"Expected {NAMESPACES[namespace]['expected_count']} vectors in {namespace}, \
                but found {stats['total_vector_count']}")
        else:
            print(f"Unexpected namespace: {namespace}")
