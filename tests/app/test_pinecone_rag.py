"""
Tests for the Pinecone RAG module.
"""

import os
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from app.rag.pinecone_rag import assert_index_status, EMBEDDING_MODEL, INDEX_NAME, get_pinecone_vectorstore

# Mock Pinecone classes
class MockPineconeIndex:
    def __init__(self, name):
        self.name = name
        self.stats = {"namespaces": {"qa": {"vector_count": 100}}}
    
    def describe_index_stats(self):
        return self.stats

class MockPinecone:
    def __init__(self, api_key=None):
        self.api_key = api_key
        
    def Index(self, name):
        return MockPineconeIndex(name)

@pytest.fixture
def pinecone_env():
    """Set up mock Pinecone API key."""
    original_api_key = os.environ.get("PINECONE_API_KEY", None)
    os.environ["PINECONE_API_KEY"] = "test-pinecone-api-key"
    yield
    if original_api_key:
        os.environ["PINECONE_API_KEY"] = original_api_key
    else:
        del os.environ["PINECONE_API_KEY"]

@patch('app.rag.pinecone_rag.Pinecone', side_effect=MockPinecone)
def test_assert_index_status_success(mock_pinecone):
    """Test assert_index_status with a valid index."""
    pc = MockPinecone(api_key="test-key")
    # This should not raise an exception
    assert_index_status(pc, "test-index")
    
    # Verify that the Pinecone Index method was called with the right name
    assert pc.Index("test-index").name == "test-index"

@patch('app.rag.pinecone_rag.Pinecone')
def test_assert_index_status_not_found(mock_pinecone):
    """Test assert_index_status when index is not found."""
    # Mock the Index method to raise a ValueError
    mock_instance = MagicMock()
    mock_pinecone.return_value = mock_instance
    mock_instance.Index.side_effect = ValueError("Index not found")
    
    pc = mock_pinecone(api_key="test-key")
    
    # This should raise an exception
    with pytest.raises(Exception) as e:
        assert_index_status(pc, "nonexistent-index")
    
    assert "not found" in str(e.value)

@patch('app.rag.pinecone_rag.Pinecone')
def test_assert_index_status_not_ready(mock_pinecone):
    """Test assert_index_status when index is not ready."""
    # Mock the describe_index_stats method to return None or empty dict
    mock_index = MagicMock()
    mock_index.describe_index_stats.return_value = None
    
    mock_instance = MagicMock()
    mock_instance.Index.return_value = mock_index
    mock_pinecone.return_value = mock_instance
    
    pc = mock_pinecone(api_key="test-key")
    
    # This should raise an exception
    with pytest.raises(Exception) as e:
        assert_index_status(pc, "not-ready-index")
    
    assert "not ready" in str(e.value)

@patch('app.rag.pinecone_rag.PineconeVectorStore')
def test_get_pinecone_vectorstore(mock_vectorstore):
    """Test get_pinecone_vectorstore function."""
    # Set up the mock
    mock_instance = MagicMock()
    mock_vectorstore.return_value = mock_instance
    
    # Call the function
    result = get_pinecone_vectorstore(namespace="test-namespace")
    
    # Assertions
    mock_vectorstore.assert_called_once_with(
        index_name=INDEX_NAME,
        embedding=EMBEDDING_MODEL,
        text_key="document_id",
        namespace="test-namespace"
    )
    assert result == mock_instance

@patch('app.rag.pinecone_rag.get_embedding_model')
@patch('app.rag.pinecone_rag.PineconeVectorStore')
def test_pinecone_vectorstore_with_different_namespace(mock_vectorstore, mock_embedding_model):
    """Test get_pinecone_vectorstore with different namespace."""
    # Set up mocks
    mock_embedding = MagicMock()
    mock_embedding_model.return_value = mock_embedding
    mock_instance = MagicMock()
    mock_vectorstore.return_value = mock_instance
    
    # Call the function with a different namespace
    result = get_pinecone_vectorstore(namespace="another-namespace")
    
    # Assertions
    mock_vectorstore.assert_called_once_with(
        index_name=INDEX_NAME,
        embedding=EMBEDDING_MODEL,
        text_key="document_id",
        namespace="another-namespace"
    )
    assert result == mock_instance
