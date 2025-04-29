'''
Gallery of utility functions to evaluate the awnsers of the app.
The LLM arbiter reuses as much as possible the methods from the app.
'''

import re
from langchain.prompts import PromptTemplate

from app.llm.remote_chat import get_chat_agent
from app.llm.agent_tools import add, substract, multiply, divide


def extract_components(text:str) -> tuple[str, str]:
    """
    Extracts the leading number from a string. Expects the input to be in the form:
    NUMBER [units] [additional info]
    where NUMBER can be an integer or a float, and [units] can be any string.
    Does not support numbers in full text (e.g., "one hundred").
    Args:
        text (str): The input string to extract from.
    Returns:
        tuple (str, str): A tuple containing the leading number and the rest of the string.
    """
    # Extract the leading number (integer or float)
    number_match = re.search(r'^-?\d+(\.\d+)?', text)
    number = number_match.group(0) if number_match else None
    
    # Extract non-empty characters (units, additional info)
    other_parts = text.strip()
    if number_match:
        other_parts = text[len(number_match.group(0)):].strip()
    
    return number, other_parts


def ajust_number_magnitude(num: str, other:str) -> float:
    """
    Multiply a number by the orders of magnitude in plain text contained in a string.
    Args:
        num (str): The number to adjust, as a string.
        other (str): The context string that could contain the magnitude.
    Returns:
        float: The adjusted number value
    """
    num = float(num)
    if "hundred" in other:
        num *= 100
    if "thousand" in other:
        num *= 1000
    if "million" in other:
        num *= 1000000
    if "billion" in other:
        num *= 1000000000
    return num


def contains_uncertainty_marker(text: str) -> bool:
    """
    Check if a string contains admission that a value is unknown or
    cannot be computed
    
    Args:
        text (str): The input string to check.
    
    Returns:
        bool: True if the string contains uncertainty markers, False otherwise.
    """
    # List of common words/phrases that indicate uncertainty
    uncertainty_markers = [
        "unknown",
        "unsure",
        "uncertain",
        "cannot",
        "insufficent"
    ]
    
    # Check if any uncertainty marker is in the text
    for marker in uncertainty_markers:
        if marker in text.lower():
            return True

    return False


prompt = """You are an arbiter. You have to decide whether the given answer is acceptable or not,
as sometimes correct answers can reference the same value expressed differently. An example of this is
the expression of a value in percentage or in absolute value.
Given the following contextual data excerpt: `{context}`,
a correct answer to the question `{question}` is `{expected_answer}`.
Is the given answer `{given_answer}` acceptable?
Answer only with "yes" or "no".
"""

prompt_eval = PromptTemplate(
    input_variables=["context", "question", "expected_answer", "given_answer"],
    template=prompt,
)


def get_arbiter_prompt(context: str, question: str, expected_answer: str, given_answer: str) -> str:
    """
    Generate the arbiter prompt for the evaluation of a single question-answer pair.
    Args:
        context (str): The context string, from the QA original metadata, which provides data relevant
            to assert if numbers are equal.
        question (str): The question string.
        expected_answer (str): The expected answer string.
        given_answer (str): The given answer string.
    Returns:
        str: The generated prompt.
    """
    return prompt_eval.invoke({
        "context":context,
        "question":question,
        "expected_answer":expected_answer,
        "given_answer":given_answer
    })


def get_arbiter_answer(context: str, question: str, expected_answer: str, given_answer: str) -> bool:
    """
    Get the arbiter's answer for a given question-answer pair.
    Args:
        context (str): The context string, from the QA original metadata, which provides data relevant
            to assert if numbers are equal.
        question (str): The question string.
        expected_answer (str): The expected answer string.
        given_answer (str): The given answer string.
    Returns:
        bool: The arbiter's assertion of the given answer being correct.
    """
    llm = get_chat_agent(needs_tools=True)
    llm = llm.bind_tools([add, substract, multiply, divide])
    response = llm.invoke(get_arbiter_prompt(context, question, expected_answer, given_answer))
    decision = response.content.strip().lower()
    assert decision in ["yes", "no"], f"Unexpected arbiter response: {decision}"
    # Return True if the arbiter says "yes", False otherwise
    return decision == "yes"
