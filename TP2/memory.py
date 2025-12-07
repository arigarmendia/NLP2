from typing import Dict

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory


def wrap_with_memory(chain, history_store: Dict, session_id: str, input_key: str = "question"):
    if session_id not in history_store:
        history_store[session_id] = InMemoryChatMessageHistory()

    return RunnableWithMessageHistory(
        chain,
        lambda sid: history_store.setdefault(sid, InMemoryChatMessageHistory()),
        input_messages_key=input_key,
        history_messages_key="chat_history",
    )
