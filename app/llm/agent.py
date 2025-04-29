"""
Submodule defining the answering agent as a LangGraph
"""
from logging import getLogger

from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict

from app.llm.prompt import prompt_qa, format_docs_qa_context, format_docs_context, format_tables_context
from app.llm.agent_tools import add, substract, multiply, divide
from app.llm.remote_chat import get_chat_agent
from app.rag.pinecone_rag import get_pinecone_vectorstore
from app.config import is_eval_mode

logger = getLogger(__name__)

# Define the vectorstores for different contexts
qa_namespace = "qa-test" if is_eval_mode() else "qa"
pinecone_vectorstore_qa = get_pinecone_vectorstore(qa_namespace)
pinecone_vectorstore_docs = get_pinecone_vectorstore("docs")
pinecone_vectorstore_table = get_pinecone_vectorstore("tables")


# Define state for application
class State(TypedDict):
    question: str
    qa_context: str
    doc_context: str
    table_context: str
    answer: str


# Define application steps
def retrieve(state: State):
    """
    State for retrieval of all contexts relevant to the question.
    Update states values: `qa_context`, `doc_context`
    
    Args:
        state (State): The current state of the application.
    """
    
    # Query knowledge base
    query = state["question"]
    retrieved_docs = pinecone_vectorstore_docs.similarity_search(query, k=2)
    retrieved_docs_qa = pinecone_vectorstore_qa.similarity_search(query, k=2)
    retrieved_docs_table = pinecone_vectorstore_table.similarity_search(query, k=2)

    # Format retrieved items
    qa_context = format_docs_qa_context(retrieved_docs_qa)
    doc_context = format_docs_context(retrieved_docs)
    table_context = format_tables_context(retrieved_docs_table)

    return {"qa_context": qa_context, "doc_context": doc_context, "table_context": table_context}


def generate(state: State):
    """
    State for generating the answer to the question.
    Update state value: `answer`

    Args:
        state (State): The current state of the application.
    """
    messages = prompt_qa.invoke({
        "question": state["question"],
        "qa_context": state["qa_context"],
        "doc_context": state["doc_context"],
        "table_context": state["table_context"]
        })
    llm = get_chat_agent(needs_tools=True)
    llm = llm.bind_tools([add, substract, multiply, divide])
    response = llm.invoke(messages)
    logger.debug(f"LLM response: {response.content}")
    return {"answer": response.content}


class Agent:
    """
    Wrapper around a compiled graph handling the agent state
    to allow direct invocation on a message
    """
    def __init__(self, graph: "CompiledStateGraph"):  # type: ignore # sadly CompiledStateGraph is not an importable type
        """
        Initialize the agent with a compiled state graph.
        Args:
            graph (CompiledStateGraph): The compiled state graph for the agent.
        """
        self.graph = graph

    def invoke(self, question: str) -> State:
        """
        Invoke the agent with a question.
        
        Args:
            question (str): The question to ask the agent.
        
        Returns:
            State: The state after processing the question.
        """
        state = State(question=question, qa_context="", doc_context="", table_context="", answer="")
        return self.graph.invoke(state)


# Expose the agent to use
def get_agent() -> Agent:
    """
    Get the agent graph for execution.
    
    Returns:
        Agent: The agent to invoke on questions
    """    
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    compiled_graph = graph_builder.compile()
    return Agent(compiled_graph)
