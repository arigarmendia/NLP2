import logging
from langchain_groq import ChatGroq
from config import GROQ_API_KEY, DEFAULT_TEMPERATURE, MAX_TOKENS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_chat_model(
    model_name: str = "llama-3.1-8b-instant",
    temperature: float = None,
    max_tokens: int = None
):
    """
    Create a Groq chat model.
    
    Args:
        model_name: Name of the Groq model
        temperature: Sampling temperature (defaults to config value)
        max_tokens: Maximum tokens to generate (defaults to config value)
    """
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set in the environment.")
    
    if temperature is None:
        temperature = DEFAULT_TEMPERATURE
    
    if max_tokens is None:
        max_tokens = MAX_TOKENS
    
    try:
        logger.info(f"Initializing ChatGroq with model: {model_name}")
        
        return ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        logger.error(f"Failed to initialize ChatGroq: {e}")
        raise