"""
Submodule for elementary and self-supported tools to provide the agent with
"""

from langchain_core.tools import tool

# Define elementary tools
@tool
def multiply(a: float, b: float) -> float:
    """Multiply a and b."""
    return a * b

@tool
def add(a: float, b: float) -> float:
    """Sum a and b."""
    return a + b

@tool
def substract(a: float, b: float) -> float:
    """Substract b from a."""
    return a - b

@tool
def divide(a: float, b: float) -> float:
    """Divide a and b."""
    if b == 0:
        return float("inf")  # Handle division by zero
    return a / b