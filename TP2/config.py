import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Pinecone serverless settings for Starter plan
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
PINECONE_INDEX_NAME = "cv-rag-index"

# Pinecone built-in embedding model (serverless)
PINECONE_EMBEDDING_MODEL = "multilingual-e5-large"

RESUME_PATH = "data/Ariadna_Garmendia_resume.pdf"
