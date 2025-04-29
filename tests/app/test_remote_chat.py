"""
Tests for the remote chat module.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from app.llm.remote_chat import get_chat_agent
from langchain_openai import ChatOpenAI

@pytest.fixture
def mock_openai_env():
    """Set up mock OpenAI environment variables."""
    os.environ["OPENAI_API_KEY"] = "test-api-key"
    yield
    # Clean up
    if "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]


@patch('app.llm.remote_chat.ChatOpenAI')
def test_get_chat_agent_no_tools_support(mock_chat_openai, mock_openai_env):
    """Test the get_chat_agent function."""
    # Configure the mock to return a specific value
    mock_instance = MagicMock()
    mock_chat_openai.return_value = mock_instance
    
    # Call the function
    agent = get_chat_agent()
    
    # Verify that ChatOpenAI was initialized with the expected parameters
    mock_chat_openai.assert_called_once()
    call_kwargs = mock_chat_openai.call_args[1]
    assert call_kwargs["openai_api_key"] == "test-api-key"
    assert call_kwargs["model_name"] == "gpt-4o-mini"
    
    # Verify that the function returns the expected instance
    assert agent == mock_instance


@patch('app.llm.remote_chat.ChatOpenAI')
def test_get_chat_agent_with_tools_support(mock_chat_openai, mock_openai_env):
    """Test the get_chat_agent function."""
    # Configure the mock to return a specific value
    mock_instance = MagicMock()
    mock_chat_openai.return_value = mock_instance
    
    # Call the function
    agent = get_chat_agent(needs_tools=True)
    
    # Verify that ChatOpenAI was initialized with the expected parameters
    mock_chat_openai.assert_called_once()
    call_kwargs = mock_chat_openai.call_args[1]
    assert call_kwargs["openai_api_key"] == "test-api-key"
    assert call_kwargs["model_name"] == "o4-mini"
    
    # Verify that the function returns the expected instance
    assert agent == mock_instance
