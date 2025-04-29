
import os
import logging
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings

from app.config import get_config

logger = logging.getLogger(__name__)


class LocalEmbeddingModel(Embeddings):
    """Wrap around the local embedding model to match langchain's embedding interface."""
    def __init__(self, model:SentenceTransformer):
        self.model = model
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.model.encode(t).tolist() for t in texts]
            
    def embed_query(self, query: str) -> list[float]:
        return self.model.encode([query]).tolist()


def get_embedding_model() -> LocalEmbeddingModel:
    """
    Get the local embedding model.
    
    Returns:
        LocalEmbeddingModel: The local embedding model instance.
    """
    model_name = get_config("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
    model_path = get_config("EMBEDDING_MODEL_PATH", "./models")
    
    # Prefer local model if it exists
    if os.path.exists(model_path):
        model_arg = model_path
    else:
        logger.warning(f"Model path {model_path} does not exist. \
        Using model name {model_name} from remote or cache instead.")
        model_arg = model_name

    # Load the model and return the wrapper around it
    model = SentenceTransformer(model_arg)
    return LocalEmbeddingModel(model)
