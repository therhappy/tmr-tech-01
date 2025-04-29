"""
Tests for the evaluation answer functions in the evaluation.eval_answers module.
"""
import pytest
from unittest.mock import patch, mock_open
from evaluation.eval_answers import evaluate_answer


def test_evaluate_answer_empty():
    """Test with an empty answer."""
    result = evaluate_answer("", "expected answer")
    assert result["is_empty"] is True
    assert result["is_exact_match"] is False
    assert result["is_number_match"] is False
    assert result["is_number_close"] is False
    assert result["needs_arbiter"] is False


def test_evaluate_answer_exact_match():
    """Test with an exact match."""
    result = evaluate_answer("hello world", "hello world")
    assert result["is_empty"] is False
    assert result["is_exact_match"] is True
    assert result["is_number_match"] is True
    assert result["needs_arbiter"] is False


def test_evaluate_answer_case_insensitive_match():
    """Test with a case-insensitive match."""
    result = evaluate_answer("Hello World", "hello world")
    assert result["is_empty"] is False
    assert result["is_exact_match"] is True
    assert result["is_number_match"] is True
    assert result["needs_arbiter"] is False


def test_evaluate_answer_number_exact_match():
    """Test with numbers that match exactly."""
    result = evaluate_answer("42 dollars", "42 euros")
    assert result["is_empty"] is False
    assert result["is_exact_match"] is False
    assert result["is_number_match"] is True
    assert result["needs_arbiter"] is False


def test_evaluate_answer_number_close_match_within_1_percent():
    """Test with numbers that are within 1% of each other."""
    # 99.5 is within 1% of 100
    result = evaluate_answer("99.5 dollars", "100 dollars")
    assert result["is_empty"] is False
    assert result["is_exact_match"] is False
    assert result["is_number_match"] is True
    assert result["is_number_close"] is False
    assert result["needs_arbiter"] is False


def test_evaluate_answer_number_close_match_within_10_percent():
    """Test with numbers that are within 10% of each other."""
    # 92 is within 10% of 100 but not within 1%
    result = evaluate_answer("92 dollars", "100 dollars")
    assert result["is_empty"] is False
    assert result["is_exact_match"] is False
    assert result["is_number_match"] is False
    assert result["is_number_close"] is True
    assert result["needs_arbiter"] is False


def test_evaluate_answer_magnitude_adjustment():
    """Test with numbers that have different magnitudes."""
    result = evaluate_answer("1 million dollars", "1000000 dollars")
    assert result["is_empty"] is False
    assert result["is_exact_match"] is False
    assert result["is_number_match"] is True
    assert result["needs_arbiter"] is False


def test_evaluate_answer_numbers_too_different():
    """Test with numbers that are too different."""
    result = evaluate_answer("75 dollars", "100 dollars")
    assert result["is_empty"] is False
    assert result["is_exact_match"] is False
    assert result["is_number_match"] is False
    assert result["is_number_close"] is False
    assert result["needs_arbiter"] is True


def test_evaluate_answer_no_numbers():
    """Test with answers that don't contain numbers."""
    result = evaluate_answer("The answer is yes", "No, that's incorrect")
    assert result["is_empty"] is False
    assert result["is_exact_match"] is False
    assert result["is_number_match"] is False
    assert result["is_number_close"] is False
    assert result["needs_arbiter"] is True