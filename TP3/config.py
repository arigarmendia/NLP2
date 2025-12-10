import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Pinecone Configuration
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "cv-rag-index")
PINECONE_EMBEDDING_MODEL = os.getenv("PINECONE_EMBEDDING_MODEL", "multilingual-e5-large")
PINECONE_EMBEDDING_DIMENSION = 1024

# Document Processing Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# Retrieval Configuration
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "3"))

# LLM Configuration
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama-3.1-8b-instant")
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "300"))

# Ingestion Configuration
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))


# Multi-Person Configuration
PERSONS = {
    "martin_brocca": {
        "name": "Martin Brocca",
        "file": "data/resumes/Resume-Martin-Brocca-2025.pdf",
        "person_id": "martin_brocca"
    },
    "ariadna_garmendia": {
        "name": "Ariadna Garmendia",
        "file": "data/resumes/Ariadna_Garmendia_resume.pdf",
        "person_id": "ariadna_garmendia"
    },
}
