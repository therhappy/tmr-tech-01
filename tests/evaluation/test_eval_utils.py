"""
Tests for the evaluation utilities in the evaluation.eval_utils module.
"""
import pytest
from unittest.mock import MagicMock, patch
from evaluation.eval_utils import (extract_components,
                                  ajust_number_magnitude,
                                  contains_uncertainty_marker,
                                  get_arbiter_prompt,
                                  get_arbiter_answer)


def test_extract_components_integer():
    """Test extracting an integer from a string."""
    number, other = extract_components("123 dollars")
    assert number == "123"
    assert other == "dollars"


def test_extract_components_float():
    """Test extracting a float from a string."""
    number, other = extract_components("123.45 dollars")
    assert number == "123.45"
    assert other == "dollars"


def test_extract_components_negative_number():
    """Test extracting a negative number from a string."""
    number, other = extract_components("-123.45 dollars")
    assert number == "-123.45"
    assert other == "dollars"


def test_extract_components_no_number():
    """Test extracting when there's no number."""
    number, other = extract_components("dollars")
    assert number is None
    assert other == "dollars"


def test_extract_components_only_number():
    """Test extracting when there's only a number."""
    number, other = extract_components("123")
    assert number == "123"
    assert other == ""


def test_adjust_number_magnitude_hundred():
    """Test adjusting by hundred."""
    result = ajust_number_magnitude("1", "hundred dollars")
    assert result == 100


def test_adjust_number_magnitude_thousand():
    """Test adjusting by thousand."""
    result = ajust_number_magnitude("1", "thousand dollars")
    assert result == 1000


def test_adjust_number_magnitude_million():
    """Test adjusting by million."""
    result = ajust_number_magnitude("1", "million dollars")
    assert result == 1000000


def test_adjust_number_magnitude_billion():
    """Test adjusting by billion."""
    result = ajust_number_magnitude("1", "billion dollars")
    assert result == 1000000000


def test_adjust_number_magnitude_multiple():
    """Test with multiple magnitude words (should apply all)."""
    result = ajust_number_magnitude("1", "hundred million dollars")
    assert result == 100000000


def test_adjust_number_magnitude_none():
    """Test with no magnitude words."""
    result = ajust_number_magnitude("42", "dollars")
    assert result == 42


def test_contains_uncertainty_marker_unknown():
    """Test with 'unknown' in the text."""
    result = contains_uncertainty_marker("The value is unknown")
    assert result is True


def test_contains_uncertainty_marker_unsure():
    """Test with 'unsure' in the text."""
    result = contains_uncertainty_marker("I am unsure about this answer")
    assert result is True


def test_contains_uncertainty_marker_uncertain():
    """Test with 'uncertain' in the text."""
    result = contains_uncertainty_marker("The result is uncertain")
    assert result is True


def test_contains_uncertainty_marker_cannot():
    """Test with 'cannot' in the text."""
    result = contains_uncertainty_marker("I cannot calculate this")
    assert result is True


def test_contains_uncertainty_marker_insufficient():
    """Test with 'insufficient' in the text."""
    result = contains_uncertainty_marker("There is insufficent data")
    assert result is True


def test_contains_uncertainty_marker_none():
    """Test with no uncertainty markers."""
    result = contains_uncertainty_marker("The answer is 42 dollars")
    assert result is False


def test_get_arbiter_prompt_format():
    """Test that the prompt is formatted correctly."""
    context = "Sample context"
    question = "Sample question"
    expected_answer = "Sample expected answer"
    given_answer = "Sample given answer"
    
    prompt = get_arbiter_prompt(context, question, expected_answer, given_answer)
    prompt_text = str(prompt)
    
    # Check that all input variables are included in the prompt
    assert context in prompt_text
    assert question in prompt_text
    assert expected_answer in prompt_text
    assert given_answer in prompt_text


@patch('evaluation.eval_utils.get_chat_agent')
def test_get_arbiter_answer_yes(mock_get_chat_agent):
    """Test when the arbiter responds with 'yes'."""
    # Set up the mock
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "yes"
    mock_llm.bind_tools.return_value = mock_llm
    mock_get_chat_agent.return_value = mock_llm
    
    # Call the function
    result = get_arbiter_answer("context", "question", "expected", "given")
    
    # Check the result
    assert result is True
    # Verify the mock was called correctly
    mock_llm.invoke.assert_called_once()


@patch('evaluation.eval_utils.get_chat_agent')
def test_get_arbiter_answer_no(mock_get_chat_agent):
    """Test when the arbiter responds with 'no'."""
    # Set up the mock
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "no"
    mock_llm.bind_tools.return_value = mock_llm
    mock_get_chat_agent.return_value = mock_llm
    
    # Call the function
    result = get_arbiter_answer("context", "question", "expected", "given")
    
    # Check the result
    assert result is False
    # Verify the mock was called correctly
    mock_llm.invoke.assert_called_once()


@patch('evaluation.eval_utils.get_chat_agent')
def test_get_arbiter_answer_invalid(mock_get_chat_agent):
    """Test when the arbiter gives an invalid response."""
    # Set up the mock
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "maybe"
    mock_llm.bind_tools.return_value = mock_llm
    mock_get_chat_agent.return_value = mock_llm
    
    # Call the function and expect an assertion error
    with pytest.raises(AssertionError):
        get_arbiter_answer("context", "question", "expected", "given")