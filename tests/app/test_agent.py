"""
Tests for the agent module.
"""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from langgraph.graph import START
from app.llm.agent import Agent, State, get_agent, retrieve, generate


@patch('app.llm.agent.pinecone_vectorstore_table')
@patch('app.llm.agent.pinecone_vectorstore_docs')
@patch('app.llm.agent.pinecone_vectorstore_qa')
def test_retrieve(mock_vectorstore_qa, mock_vectorstore_docs, mock_vectorstore_table):
    """Test the retrieve function."""
    # Mock the vector store responses
    mock_qa_docs = [
        Document(
            page_content="python-doc-001",
            metadata={
                "question": "What is Python?",
                "answer": "Python is a programming language",
                "gold_content": "Python is a high-level language",
                "operations": "retrieve, read"
            }
        )
    ]
    mock_docs = [
        Document(
            page_content="python-documentation-table",
            metadata={
                "pre_text": "something before the table",
                "post_text": "Python is a versatile programming language and the table is above",
                "table": ["a row", "another row"],
            }
        )
    ]
    mock_table_docs = [
        Document(
            page_content="python-table",
            metadata={
                "table": ["a row", "another row"],
            }
        )
    ]
    
    mock_vectorstore_qa.similarity_search.return_value = mock_qa_docs
    mock_vectorstore_docs.similarity_search.return_value = mock_docs
    mock_vectorstore_table.similarity_search.return_value = mock_table_docs
    
    # Create a test state
    state = State(question="What is Python?", qa_context="", doc_context="", table_context="", answer="")
    
    # Call the function
    result = retrieve(state)
    
    # Verify the vectorstores were queried
    mock_vectorstore_qa.similarity_search.assert_called_once_with("What is Python?", k=2)
    mock_vectorstore_docs.similarity_search.assert_called_once_with("What is Python?", k=2)
    mock_vectorstore_table.similarity_search.assert_called_once_with("What is Python?", k=2)
    
    # Verify the result contains all formatted contexts
    assert "qa_context" in result
    assert "doc_context" in result
    assert "table_context" in result
    assert "What is Python?" in result["qa_context"]
    assert "Python is a programming language" in result["qa_context"]


@patch('app.llm.agent.get_chat_agent')
@patch('app.llm.agent.prompt_qa')
def test_generate(mock_prompt_qa, mock_get_chat_agent):
    """Test the generate function."""
    # Mock the prompt template and chat agent
    mock_prompt_qa.invoke.return_value = ["Test prompt"]
    
    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm
    mock_response = MagicMock()
    mock_response.content = "This is a test answer"
    mock_llm.invoke.return_value = mock_response
    mock_get_chat_agent.return_value = mock_llm
    
    # Create a test state
    state = State(
        question="What is Python?",
        qa_context="Sample context about Python",
        doc_context="Sample documentation",
        table_context="Sample table data",
        answer=""
    )
    
    # Call the function
    result = generate(state)
    
    # Verify that prompt was invoked with the right data
    mock_prompt_qa.invoke.assert_called_once_with({
        "question": "What is Python?",
        "qa_context": "Sample context about Python",
        "doc_context": "Sample documentation",
        "table_context": "Sample table data"
    })
    
    # Verify that the llm was bound with tools and invoked
    mock_llm.bind_tools.assert_called_once()
    tools_arg = mock_llm.bind_tools.call_args[0][0]
    assert len(tools_arg) == 4  # add, substract, multiply, divide
    
    mock_llm.invoke.assert_called_once_with(["Test prompt"])
    
    # Verify the result
    assert "answer" in result
    assert result["answer"] == "This is a test answer"


@patch('app.llm.agent.StateGraph')
def test_get_agent(mock_stategraph):
    """Test the get_agent function that creates the agent instance."""
    # Mock the StateGraph and its methods
    mock_graph_builder = MagicMock()
    mock_graph_builder.add_sequence.return_value = mock_graph_builder  # Return self for method chaining
    mock_graph_builder.add_edge.return_value = mock_graph_builder  # Return self for method chaining
    mock_compiled_graph = MagicMock()
    mock_graph_builder.compile.return_value = mock_compiled_graph
    mock_stategraph.return_value = mock_graph_builder
    
    # Call get_agent
    agent = get_agent()
    
    # Verify StateGraph was initialized with the correct state type
    mock_stategraph.assert_called_once_with(State)
    
    # Verify the sequence was added correctly
    mock_graph_builder.add_sequence.assert_called_once()
    sequence_arg = mock_graph_builder.add_sequence.call_args[0][0]
    assert len(sequence_arg) == 2
    assert sequence_arg[0] == retrieve
    assert sequence_arg[1] == generate
    
    # Verify edge was added from START to "retrieve"
    mock_graph_builder.add_edge.assert_called_once()
    edge_args = mock_graph_builder.add_edge.call_args[0]
    assert edge_args[0] == "__start__" or edge_args[0] == START
    assert edge_args[1] == "retrieve"
    
    # Verify the graph was compiled
    mock_graph_builder.compile.assert_called_once()
    
    # Verify the agent was created with the compiled graph
    assert isinstance(agent, Agent)
    assert agent.graph == mock_compiled_graph


def test_agent_invoke():
    """Test the agent invoke method."""
    # Create a mock compiled graph
    mock_compiled_graph = MagicMock()
    mock_compiled_graph.invoke.return_value = {
        "question": "Test question",
        "qa_context": "Test context",
        "doc_context": "Test doc",
        "table_context": "Test table",
        "answer": "Test answer"
    }
    
    # Create an agent with the mock graph
    agent = Agent(mock_compiled_graph)
    
    # Call the invoke method
    result = agent.invoke("Test question")
    
    # Verify the graph was invoked with the right state
    mock_compiled_graph.invoke.assert_called_once()
    invoked_state = mock_compiled_graph.invoke.call_args[0][0]
    assert invoked_state["question"] == "Test question"
    assert invoked_state["qa_context"] == ""
    assert invoked_state["doc_context"] == ""
    assert invoked_state["table_context"] == ""
    assert invoked_state["answer"] == ""
    
    # Verify the result
    assert result["question"] == "Test question"
    assert result["qa_context"] == "Test context"
    assert result["doc_context"] == "Test doc"
    assert result["table_context"] == "Test table"
    assert result["answer"] == "Test answer"
