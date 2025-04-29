
from langchain_tests.unit_tests import ToolsUnitTests
from langchain_core.tools import BaseTool
from app.llm.agent import add, substract, multiply, divide

# def test_elementary_tools():
#     """Test the elementary math tools in the agent."""
#     assert add.invoke(5, 3) == 8
#     assert substract.invoke(10.8, 4) == 6.2
#     assert multiply.invoke(7, 6.0) == 42.0
#     assert divide.invoke(10, 2.5) == 4
#     assert divide.invoke(10, 0) == float('inf')  # Test division by zero

class TestAddToolUnit(ToolsUnitTests):
    """Test the add tool."""
    @property
    def tool_constructor(self) -> BaseTool:
        return lambda : add

    @property
    def tool_constructor_params(self) -> dict:
        # if your tool constructor instead required initialization arguments like
        # `def __init__(self, some_arg: int):`, you would return those here
        # as a dictionary, e.g.: `return {'some_arg': 42}`
        return {}

    @property
    def tool_invoke_params_example(self) -> dict:
        """
        Returns a dictionary representing the "args" of an example tool call.

        This should NOT be a ToolCall dict - i.e. it should not
        have {"name", "id", "args"} keys.
        """
        return {"a": 2., "b": 3.}
    
    def test_tool_invoke(self):
        """
        Test the tool invocation with example parameters.
        """
        # Create an instance of the tool
        tool_instance = self.tool_constructor(**self.tool_constructor_params)
        
        # Invoke the tool with example parameters
        result = tool_instance.invoke(self.tool_invoke_params_example)
        
        # Check if the result is as expected
        expected_result = self.tool_invoke_params_example["a"] + self.tool_invoke_params_example["b"]
        assert result == expected_result, f"Expected {expected_result}, but got {result}"


class TestSubstractToolUnit(ToolsUnitTests):
    """Test the add tool."""
    @property
    def tool_constructor(self) -> BaseTool:
        return lambda : substract

    @property
    def tool_constructor_params(self) -> dict:
        # if your tool constructor instead required initialization arguments like
        # `def __init__(self, some_arg: int):`, you would return those here
        # as a dictionary, e.g.: `return {'some_arg': 42}`
        return {}

    @property
    def tool_invoke_params_example(self) -> dict:
        """
        Returns a dictionary representing the "args" of an example tool call.

        This should NOT be a ToolCall dict - i.e. it should not
        have {"name", "id", "args"} keys.
        """
        return {"a": 2., "b": 3.}

    def test_tool_invoke(self):
        """
        Test the tool invocation with example parameters.
        """
        # Create an instance of the tool
        tool_instance = self.tool_constructor(**self.tool_constructor_params)
        
        # Invoke the tool with example parameters
        result = tool_instance.invoke(self.tool_invoke_params_example)
        
        # Check if the result is as expected
        expected_result = self.tool_invoke_params_example["a"] - self.tool_invoke_params_example["b"]
        assert result == expected_result, f"Expected {expected_result}, but got {result}"


class TestMultiplyToolUnit(ToolsUnitTests):
    """Test the add tool."""
    @property
    def tool_constructor(self) -> BaseTool:
        return lambda : multiply

    @property
    def tool_constructor_params(self) -> dict:
        # if your tool constructor instead required initialization arguments like
        # `def __init__(self, some_arg: int):`, you would return those here
        # as a dictionary, e.g.: `return {'some_arg': 42}`
        return {}

    @property
    def tool_invoke_params_example(self) -> dict:
        """
        Returns a dictionary representing the "args" of an example tool call.

        This should NOT be a ToolCall dict - i.e. it should not
        have {"name", "id", "args"} keys.
        """
        return {"a": 2., "b": 3.}

    def test_tool_invoke(self):
        """
        Test the tool invocation with example parameters.
        """
        # Create an instance of the tool
        tool_instance = self.tool_constructor(**self.tool_constructor_params)
        
        # Invoke the tool with example parameters
        result = tool_instance.invoke(self.tool_invoke_params_example)
        
        # Check if the result is as expected
        expected_result = self.tool_invoke_params_example["a"] * self.tool_invoke_params_example["b"]
        assert result == expected_result, f"Expected {expected_result}, but got {result}"


class TestDivideToolUnit(ToolsUnitTests):
    """Test the add tool."""
    @property
    def tool_constructor(self) -> BaseTool:
        return lambda:divide

    @property
    def tool_constructor_params(self) -> dict:
        # if your tool constructor instead required initialization arguments like
        # `def __init__(self, some_arg: int):`, you would return those here
        # as a dictionary, e.g.: `return {'some_arg': 42}`
        return {}

    @property
    def tool_invoke_params_example(self) -> dict:
        """
        Returns a dictionary representing the "args" of an example tool call.

        This should NOT be a ToolCall dict - i.e. it should not
        have {"name", "id", "args"} keys.
        """
        return {"a": 2., "b": 3.}
    
    @property
    def tool_invoke_params_example_divbyzero(self) -> dict:
        """
        Returns a dictionary representing the "args" of an example tool call.

        This should NOT be a ToolCall dict - i.e. it should not
        have {"name", "id", "args"} keys.
        """
        return {"a": 42., "b": 0.}
    
    def test_tool_invoke(self):
        """
        Test the tool invocation with example parameters.
        """
        # Create an instance of the tool
        tool_instance = self.tool_constructor(**self.tool_constructor_params)
        
        # Invoke the tool with example parameters
        result = tool_instance.invoke(self.tool_invoke_params_example)
        
        # Check if the result is as expected
        expected_result = self.tool_invoke_params_example["a"] / self.tool_invoke_params_example["b"]
        assert result == expected_result, f"Expected {expected_result}, but got {result}"
        
        divbyzero_result = tool_instance.invoke(self.tool_invoke_params_example_divbyzero)
        expected_result = float("inf")
        assert divbyzero_result == expected_result, f"Expected {expected_result} for a division by 0, but got {divbyzero_result}"