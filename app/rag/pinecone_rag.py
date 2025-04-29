"""
Submodule to handle documents retrieval
either in the langchain retrieval format or directly.
"""
import os
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from app.rag.embeddings import get_embedding_model


def assert_index_status(pc: Pinecone, index_name: str) -> None:
    """
    Assert the status of the Pinecone index.
    
    Args:
        index_name (str): The name of the Pinecone index.
        
    Raises:
        Exception: If the index is not found or not ready.
    """
    try:
        index = pc.Index(name=index_name)
        if not index.describe_index_stats():
            raise Exception(f"Index {index_name} is not ready.")
    except ValueError as e:
        raise Exception(f"Index {index_name} not found.") from e


EMBEDDING_MODEL = get_embedding_model()
INDEX_NAME = "tmr-convfinqa-qa"

# Instanciate client and check index status
pc = Pinecone(os.environ["PINECONE_API_KEY"])   
assert_index_status(pc, INDEX_NAME)


def get_pinecone_vectorstore(namespace: str) -> PineconeVectorStore:
    """
    Get the Pinecone vectorstore for the specified namespace.
    
    Args:
        namespace (str): The namespace to use for the vectorstore.
        
    Returns:
        PineconeVectorStore: The Pinecone vectorstore instance.
    """

    # Instanciate langchain-compatible vectorstore
    pinecone_vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME, 
        embedding=EMBEDDING_MODEL, 
        text_key="document_id",
        namespace=namespace,
    )
    return pinecone_vectorstore
