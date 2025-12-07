from pinecone import Pinecone, ServerlessSpec
from config import (
    PINECONE_API_KEY,
    PINECONE_CLOUD,
    PINECONE_REGION,
    PINECONE_INDEX_NAME,
    PINECONE_EMBEDDING_MODEL,
)


def get_pinecone_client() -> Pinecone:
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is not set in the environment.")
    return Pinecone(api_key=PINECONE_API_KEY)


def ensure_index(pc: Pinecone):
    if not pc.has_index(PINECONE_INDEX_NAME):
        pc.create_index_for_model(
            name=PINECONE_INDEX_NAME,
            cloud=PINECONE_CLOUD,
            region=PINECONE_REGION,
            embed={"model": PINECONE_EMBEDDING_MODEL},
        )
    return pc.Index(PINECONE_INDEX_NAME)
