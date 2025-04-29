"""
Tests for the answer generation functions in the evaluation.generate_answers module.
"""
import pytest
import json
from unittest.mock import patch, MagicMock
from evaluation.generate_answers import load_questions, get_answer_from_api


@patch('builtins.open', new_callable=MagicMock)
@patch('json.load')
def test_load_questions_success(mock_json_load, mock_open):
    """Test successful loading of questions."""
    # Mock data
    mock_data = [{"metadata": {"question": "Test question"}}]
    mock_json_load.return_value = mock_data
    
    # Call the function
    result = load_questions("test_file.json")
    
    # Assertions
    mock_open.assert_called_once_with("test_file.json", "r")
    assert result == mock_data


@patch('requests.post')
def test_get_answer_from_api_successful(mock_post):
    """Test a successful API response."""
    # Mock response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": "Test answer"}
    mock_post.return_value = mock_response
    
    # Call the function
    result = get_answer_from_api("Test question")
    
    # Assertions
    mock_post.assert_called_once_with(
        "http://localhost:8000/conversation",
        json={"message": "Test question", "conversation_id": "new"}
    )
    assert result == "Test answer"


@patch('requests.post')
def test_get_answer_from_api_empty_response(mock_post):
    """Test an API response with no 'response' field."""
    # Mock response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {}
    mock_post.return_value = mock_response
    
    # Call the function
    result = get_answer_from_api("Test question")
    
    # Assertions
    assert result == "No answer provided"


@patch('requests.post')
def test_get_answer_from_api_error_fail_fast(mock_post):
    """Test when the API returns an error and fail_fast is True."""
    # Mock response
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Server error"
    mock_post.return_value = mock_response
    
    # Call the function with fail_fast=True (default)
    with pytest.raises(Exception) as excinfo:
        get_answer_from_api("Test question")
    
    # Check the error message
    assert "Error: 500" in str(excinfo.value)
    assert "Server error" in str(excinfo.value)


@patch('requests.post')
def test_get_answer_from_api_error_no_fail_fast(mock_post):
    """Test when the API returns an error and fail_fast is False."""
    # Mock response
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Server error"
    mock_post.return_value = mock_response
    
    # Call the function with fail_fast=False
    result = get_answer_from_api("Test question", fail_fast=False)
    
    # Check the result is the error message
    assert "Error: 500" in result
    assert "Server error" in result


@patch('requests.post')
def test_get_answer_from_api_custom_url(mock_post):
    """Test using a custom API URL."""
    # Mock response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": "Test answer"}
    mock_post.return_value = mock_response
    
    # Call the function with a custom URL
    result = get_answer_from_api("Test question", api_url="http://custom-api/chat")
    
    # Assertions
    mock_post.assert_called_once_with(
        "http://custom-api/chat",
        json={"message": "Test question", "conversation_id": "new"}
    )
    assert result == "Test answer"