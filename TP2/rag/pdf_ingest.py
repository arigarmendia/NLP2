import logging
import sys
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_core.embeddings import Embeddings
from typing import List
from config import (
    RESUME_PATH,
    PINECONE_EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    BATCH_SIZE,
    PINECONE_API_KEY,
)
from rag.pinecone_client import get_pinecone_client, get_index

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PineconeInferenceEmbeddings(Embeddings):
    """Pinecone Inference API embeddings - compatible with Pinecone SDK 6.x+"""
    
    def __init__(self, model: str, pinecone_api_key: str):
        self.model = model
        self.pc = Pinecone(api_key=pinecone_api_key)
        logger.info(f"Initialized Pinecone Inference embeddings with model: {model}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents (passages)."""
        if not texts:
            return []
        try:
            response = self.pc.inference.embed(
                model=self.model,
                inputs=texts,
                parameters={"input_type": "passage"}
            )
            return [item["values"] for item in response.data]
        except Exception as e:
            logger.error(f"Failed to embed documents: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        try:
            response = self.pc.inference.embed(
                model=self.model,
                inputs=[text],
                parameters={"input_type": "query"}
            )
            return response.data[0]["values"]
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            raise


def load_and_split_resume(path: str) -> list[Document]:
    """Load PDF and split into chunks."""
    logger.info(f"Loading resume from {path}...")
    
    try:
        # Try with extract_images=False to avoid character spacing issues
        loader = PyPDFLoader(path, extract_images=False)
        docs = loader.load()
        
        # If text has excessive spacing, try to clean it
        for doc in docs:
            if '  ' in doc.page_content[:100]:  # Check for double spaces
                logger.warning("Detected spacing issues in PDF extraction, attempting cleanup...")
                # Remove extra spaces between characters
                import re
                doc.page_content = re.sub(r'(?<=\w)\s(?=\w)', '', doc.page_content)
        
        logger.info(f"Loaded {len(docs)} pages from resume.")
    except Exception as e:
        logger.error(f"Failed to load PDF: {e}")
        raise
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )
    
    split_docs = splitter.split_documents(docs)
    logger.info(f"Split into {len(split_docs)} chunks.")
    
    return split_docs


def check_existing_vectors(index) -> int:
    """Check how many vectors are already in the index."""
    try:
        stats = index.describe_index_stats()
        return stats.total_vector_count
    except Exception as e:
        logger.warning(f"Could not get index stats: {e}")
        return 0


def ingest_resume(force_reingest: bool = False):
    """
    Ingest resume into Pinecone index with duplicate checking.
    
    Args:
        force_reingest: If True, clear existing vectors and re-ingest.
    """
    try:
        logger.info("Starting resume ingestion process...")
        
        # Initialize Pinecone
        pc = get_pinecone_client()
        index = get_index()
        
        # Log index info
        stats = index.describe_index_stats()
        logger.info(f"Using embedding model: {PINECONE_EMBEDDING_MODEL}")
        logger.info(f"Index dimension: {stats.dimension}")
        
        # Check for existing vectors
        existing_count = check_existing_vectors(index)
        
        if existing_count > 0:
            logger.info(f"Index already contains {existing_count} vectors.")
            
            if not force_reingest:
                response = input(
                    "Index already contains data. Clear and re-ingest? (y/n): "
                ).lower()
                force_reingest = response == 'y'
            
            if force_reingest:
                logger.info("Clearing existing vectors...")
                index.delete(delete_all=True)
                logger.info("Index cleared.")
            else:
                logger.info("Skipping ingestion. Use force_reingest=True to override.")
                return
        
        # Load and split documents
        docs = load_and_split_resume(RESUME_PATH)
        
        # Add metadata
        for d in docs:
            if d.metadata is None:
                d.metadata = {}
            d.metadata.setdefault("source", "resume")
        
        # Initialize embeddings using Pinecone Inference API
        embeddings = PineconeInferenceEmbeddings(
            model=PINECONE_EMBEDDING_MODEL,
            pinecone_api_key=PINECONE_API_KEY
        )
        
        vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
        
        # Batch processing
        total_batches = (len(docs) + BATCH_SIZE - 1) // BATCH_SIZE
        logger.info(f"Ingesting {len(docs)} chunks in {total_batches} batch(es)...")
        
        for i in range(0, len(docs), BATCH_SIZE):
            batch = docs[i:i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            
            try:
                vectorstore.add_documents(batch)
                logger.info(
                    f"Batch {batch_num}/{total_batches}: "
                    f"Ingested {len(batch)} chunks."
                )
            except Exception as e:
                logger.error(f"Error ingesting batch {batch_num}: {e}")
                raise
        
        # Verify ingestion
        final_count = check_existing_vectors(index)
        logger.info(
            f"✓ Ingestion complete! Index now contains {final_count} vectors."
        )
        
    except Exception as e:
        logger.error(f"Failed to ingest resume: {e}")
        raise


if __name__ == "__main__":
    print("=" * 60)
    print("Starting PDF Ingestion Script")
    print("=" * 60)
    
    force = "--force" in sys.argv or "-f" in sys.argv
    print(f"Force reingest mode: {force}")
    print()
    
    try:
        ingest_resume(force_reingest=force)
        print()
        print("=" * 60)
        print("Ingestion completed successfully!")
        print("=" * 60)
    except Exception as e:
        print()
        print("=" * 60)
        print(f"ERROR: Ingestion failed - {e}")
        print("=" * 60)
        sys.exit(1)