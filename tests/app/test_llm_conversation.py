"""
Tests for the LLM conversation handler.
"""

import pytest
from unittest.mock import patch, MagicMock
from app.llm.conversation import LLMConversation

def test_conversation_creation():
    """Test that a new conversation is created with a unique ID."""
    with patch('app.llm.conversation.get_agent') as mock_get_agent:
        # Mock the agent
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"answer": "This is a test response"}
        mock_get_agent.return_value = mock_agent
        
        handler = LLMConversation()
        message = "Hello, how are you?"
        
        response, conversation_id = handler.process_message(message)
        
        assert conversation_id is not None
        assert isinstance(conversation_id, str)
        assert conversation_id in handler.conversations
        assert len(handler.conversations[conversation_id]) == 2  # User message and assistant response
        assert handler.conversations[conversation_id][0]["content"] == message
        assert handler.conversations[conversation_id][0]["role"] == "user"
        assert handler.conversations[conversation_id][1]["role"] == "qa_assistant"
        assert handler.conversations[conversation_id][1]["content"] == "This is a test response"
        assert "agent_state" in handler.conversations[conversation_id][1]
        assert handler.conversations[conversation_id][1]["agent_state"]["answer"] == "This is a test response"

def test_conversation_continuation():
    """Test that an existing conversation can be continued."""
    with patch('app.llm.conversation.get_agent') as mock_get_agent:
        # Mock the agent
        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = [
            {"answer": "First response"}, 
            {"answer": "Second response"}
        ]
        mock_get_agent.return_value = mock_agent
        
        handler = LLMConversation()
        first_message = "Hello, who are you?"
        second_message = "What can you do?"
        
        _, conversation_id = handler.process_message(first_message)
        response, same_conversation_id = handler.process_message(second_message, conversation_id)
        
        assert same_conversation_id == conversation_id
        assert len(handler.conversations[conversation_id]) == 4  # Two user messages and two assistant responses
        assert handler.conversations[conversation_id][2]["content"] == second_message
        assert handler.conversations[conversation_id][3]["content"] == "Second response"
        
        # Verify that the agent was invoked with both messages
        assert mock_agent.invoke.call_count == 2
        mock_agent.invoke.assert_any_call(first_message)
        mock_agent.invoke.assert_any_call(second_message)

@patch('app.llm.conversation.get_agent')
def test_conversation_with_mocked_agent(mock_get_agent):
    """Test the conversation handler with a mocked agent."""
    # Create a mock agent that returns a predictable response
    mock_agent_instance = MagicMock()
    mock_agent_instance.invoke.return_value = {"answer": "This is a test response"}
    mock_get_agent.return_value = mock_agent_instance
    
    # Initialize the handler
    handler = LLMConversation()
    
    # Process a message
    response, conversation_id = handler.process_message("Test message")
    
    # Verify that the agent was called with the message
    mock_agent_instance.invoke.assert_called_once_with("Test message")
    
    # Verify that the response is what we expect
    assert response == "This is a test response"
    
    # Verify that the conversation was properly stored
    assert len(handler.conversations[conversation_id]) == 2
    assert handler.conversations[conversation_id][0]["content"] == "Test message"
    assert handler.conversations[conversation_id][1]["content"] == "This is a test response"

def test_invalid_conversation_id():
    """Test behavior when an invalid conversation ID is provided."""
    with patch('app.llm.conversation.get_agent') as mock_get_agent:
        # Mock the agent
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"answer": "This is a test response"}
        mock_get_agent.return_value = mock_agent
        
        handler = LLMConversation()
        
        # Process a message with a non-existent conversation ID
        response, conversation_id = handler.process_message("Test message", "non-existent-id")
        
        # A new conversation should be created
        assert conversation_id != "non-existent-id"
        assert conversation_id in handler.conversations
        assert len(handler.conversations[conversation_id]) == 2