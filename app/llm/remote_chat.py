import os
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI


def get_chat_agent(needs_tools: bool=False) -> BaseChatModel:
    """
    Get the chat agent for the LLM.
    The choice for the model to use is based on the need for tools,
    as `gpt-4o-mini` is very inexpensive for tests but does not offer tool call.
    The `o4-mini` model is used for the agent with tools and intended for deployment.
    See model cards:
    https://platform.openai.com/docs/models/o4-mini
    https://platform.openai.com/docs/models/gpt-4o-mini
    
    Args:
        needs_tools (bool): Whether the agent needs tools or not, defaults to False.
    Returns:
        ChatOpenAI: The chat agent instance.
    """
    
    openai_model_name = 'gpt-4o-mini'
    if needs_tools:
        openai_model_name = 'o4-mini'
    
    llm = ChatOpenAI(
        openai_api_key=os.environ["OPENAI_API_KEY"],
        model_name=openai_model_name
    )
    # Temperature cannot be set for the o4-mini model
    return llm
