"""
LLM conversation handler module.
"""

import logging
import uuid
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.llm.agent import get_agent

logger = logging.getLogger(__name__)


class LLMConversation:
    """
    Handler for LLM-based conversations
    """
    
    def __init__(self):
        """Initialize the LLM conversation handler."""
        self.conversations: dict[str, list[dict]] = {}
        logger.info("Initialized LLM conversation handler")
        self.agent = get_agent()

    def process_message(
        self, 
        message: str, 
        conversation_id: str | None = None,
    ) -> tuple[str, str]:
        """
        Process a message and return the LLM response.
        
        Args:
            message: The user's message
            conversation_id: Optional ID for an existing conversation
            user_id: Optional ID of the user
            
        Returns:
            Tuple containing (response_text, conversation_id)
        """
        # Create a new conversation if necessary
        if not conversation_id or conversation_id not in self.conversations:
            conversation_id = str(uuid.uuid4())
            self.conversations[conversation_id] = []
            logger.info(f"Created new conversation with ID: {conversation_id}")
        
        # Add the user message to the conversation history
        self.conversations[conversation_id].append({
            "role": "user",
            "content": message
        })

        # Since the agent inits the state for the graph, we can just pass the message
        response_state = self.agent.invoke(message)
        response = response_state["answer"]

        logger.info(f"Response state for conversation {conversation_id}: {response_state}")
        
        # Add the assistant's response to the conversation history
        self.conversations[conversation_id].append({
            "role": "qa_assistant",
            "content": response,
            "agent_state": response_state
        })
        
        return response, conversation_id
