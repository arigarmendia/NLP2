from langchain_groq import ChatGroq
from config import GROQ_API_KEY


def get_chat_model(model_name: str = "llama-3.1-8b-instant", temperature: float = 0.7):
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set in the environment.")
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=model_name,
        temperature=temperature,
        max_tokens=1000,
    )
