"""
Tests for the embeddings module.
"""

import pytest
import os
from unittest.mock import patch, MagicMock
from sentence_transformers import SentenceTransformer
from app.rag.embeddings import LocalEmbeddingModel, get_embedding_model
import numpy as np

@pytest.fixture
def mock_sentence_transformer():
    """Create a mock SentenceTransformer."""
    with patch('app.rag.embeddings.SentenceTransformer') as mock_st:
        # Configure mock to return predictable embedding vectors as numpy arrays
        mock_instance = MagicMock()
        mock_instance.encode.side_effect = lambda x: np.array([[0.1, 0.2, 0.3]]) if isinstance(x, list) else np.array([0.1, 0.2, 0.3])
        mock_st.return_value = mock_instance
        yield mock_st

@pytest.fixture
def mock_os_path_exists():
    """Create a mock for os.path.exists."""
    with patch('os.path.exists') as mock_exists:
        yield mock_exists

@pytest.fixture
def mock_get_config():
    """Create a mock for get_config."""
    with patch('app.rag.embeddings.get_config') as mock_config:
        yield mock_config

def test_get_embedding_model(mock_sentence_transformer, mock_os_path_exists, mock_get_config):
    """Test the get_embedding_model function."""
    # Configure mocks
    mock_get_config.side_effect = lambda key, default: "BAAI/bge-m3" if key == "EMBEDDING_MODEL_NAME" else "./models"
    
    # Test case: local model exists
    mock_os_path_exists.return_value = True
    embedding_model = get_embedding_model()
    
    # Verify that SentenceTransformer was initialized with the local model path
    mock_sentence_transformer.assert_called_with("./models")
    
    # Verify that the returned object is a LocalEmbeddingModel
    assert isinstance(embedding_model, LocalEmbeddingModel)
    
    # Test case: local model doesn't exist
    mock_os_path_exists.return_value = False
    embedding_model = get_embedding_model()
    
    # Verify that SentenceTransformer was initialized with the model name
    mock_sentence_transformer.assert_called_with("BAAI/bge-m3")

def test_local_embedding_model():
    """Test the LocalEmbeddingModel wrapper."""
    # Create a mock SentenceTransformer instance
    mock_model = MagicMock()
    # Return NumPy arrays instead of lists to support tolist() method
    mock_model.encode.side_effect = lambda x: np.array([[0.1, 0.2, 0.3]]) if isinstance(x, list) else np.array([0.1, 0.2, 0.3])
    
    # Initialize the wrapper with the mock
    embedding_model = LocalEmbeddingModel(mock_model)
    
    # Test embed_documents method
    docs = ["test document 1", "test document 2"]
    embeddings = embedding_model.embed_documents(docs)
    
    assert len(embeddings) == 2
    assert all(isinstance(embed, list) for embed in embeddings)
    assert mock_model.encode.call_count == 2
    
    # Test embed_query method
    query_embedding = embedding_model.embed_query("test query")
    assert isinstance(query_embedding, list)
    assert mock_model.encode.call_count == 3
    assert len(query_embedding[0]) == 3

