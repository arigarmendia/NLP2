import logging
from typing import Dict
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def wrap_with_memory(
    chain,
    history_store: Dict,
    session_id: str,
    input_key: str = "question"
):
    """
    Wrap a chain with conversational memory.
    
    Args:
        chain: The base chain to wrap
        history_store: Dictionary to store chat histories by session
        session_id: Session identifier
        input_key: Key for the input message in the chain
    
    Returns:
        Chain with message history support
    """
    try:
        # Ensure session exists in history store
        if session_id not in history_store:
            history_store[session_id] = InMemoryChatMessageHistory()
            logger.info(f"Created new chat history for session: {session_id}")
        
        # Create chain with message history
        chain_with_memory = RunnableWithMessageHistory(
            chain,
            lambda sid: history_store.setdefault(sid, InMemoryChatMessageHistory()),
            input_messages_key=input_key,
            history_messages_key="chat_history",
        )
        
        logger.info(f"Chain wrapped with memory for session: {session_id}")
        return chain_with_memory
        
    except Exception as e:
        logger.error(f"Failed to wrap chain with memory: {e}")
        raise


def clear_session_history(history_store: Dict, session_id: str):
    """Clear chat history for a specific session."""
    if session_id in history_store:
        history_store[session_id].clear()
        logger.info(f"Cleared history for session: {session_id}")
    else:
        logger.warning(f"No history found for session: {session_id}")


def get_session_message_count(history_store: Dict, session_id: str) -> int:
    """Get the number of messages in a session's history."""
    if session_id in history_store:
        return len(history_store[session_id].messages)
    return 0