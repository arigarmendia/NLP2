import logging
import time
from pinecone import Pinecone, ServerlessSpec
from config import (
    PINECONE_API_KEY,
    PINECONE_CLOUD,
    PINECONE_REGION,
    PINECONE_INDEX_NAME,
    PINECONE_EMBEDDING_DIMENSION,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache for Pinecone client and index
_pinecone_client = None
_index_cache = None


def get_pinecone_client() -> Pinecone:
    """Get or create Pinecone client (cached)."""
    global _pinecone_client
    
    if _pinecone_client is None:
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY is not set in the environment.")
        
        logger.info("Initializing Pinecone client...")
        _pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
    
    return _pinecone_client


def ensure_index(pc: Pinecone):
    """Ensure the Pinecone index exists, create if necessary. Returns cached index."""
    global _index_cache
    
    if _index_cache is not None:
        return _index_cache
    
    try:
        if not pc.has_index(PINECONE_INDEX_NAME):
            logger.info(f"Index '{PINECONE_INDEX_NAME}' not found. Creating...")
            
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=PINECONE_EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=PINECONE_CLOUD,
                    region=PINECONE_REGION
                )
            )
            
            # Wait for index to be ready
            logger.info("Waiting for index to be ready...")
            while not pc.describe_index(PINECONE_INDEX_NAME).status['ready']:
                time.sleep(1)
            
            logger.info(f"Index '{PINECONE_INDEX_NAME}' created successfully.")
        else:
            logger.info(f"Index '{PINECONE_INDEX_NAME}' already exists.")
        
        _index_cache = pc.Index(PINECONE_INDEX_NAME)
        return _index_cache
        
    except Exception as e:
        logger.error(f"Failed to create/access Pinecone index: {e}")
        raise RuntimeError(f"Failed to create/access Pinecone index: {e}")


def get_index():
    """Get the cached Pinecone index (preferred method for retrieval)."""
    global _index_cache
    
    if _index_cache is None:
        pc = get_pinecone_client()
        _index_cache = ensure_index(pc)
    
    return _index_cache


def reset_cache():
    """Reset the cached client and index (useful for testing)."""
    global _pinecone_client, _index_cache
    _pinecone_client = None
    _index_cache = None
    logger.info("Pinecone cache reset.")