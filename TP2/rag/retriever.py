import logging
from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_core.embeddings import Embeddings
from config import PINECONE_EMBEDDING_MODEL, RETRIEVAL_K, PINECONE_API_KEY
from rag.pinecone_client import get_index
from llm_client import get_chat_model

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
            # Debug logging
            logger.info(f"embed_query called with: {repr(text)[:200]}")
            logger.info(f"Type: {type(text)}")
            
            # Ensure text is a string
            if isinstance(text, dict):
                logger.warning(f"Received dict, extracting text: {text}")
                text = text.get("text", text.get("question", str(text)))
            elif not isinstance(text, str):
                logger.warning(f"Converting non-string to string: {type(text)}")
                text = str(text)
            
            response = self.pc.inference.embed(
                model=self.model,
                inputs=[text],
                parameters={"input_type": "query"}
            )
            return response.data[0]["values"]
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            logger.error(f"Input was: {repr(text)[:200]} (type: {type(text)})")
            raise


# Cache the embeddings instance
_embeddings_cache = None


def get_embeddings():
    """Get or create cached embeddings instance."""
    global _embeddings_cache
    if _embeddings_cache is None:
        _embeddings_cache = PineconeInferenceEmbeddings(
            model=PINECONE_EMBEDDING_MODEL,
            pinecone_api_key=PINECONE_API_KEY
        )
    return _embeddings_cache


def get_resume_retriever(k: int = None):
    """
    Create a retriever for the resume vector store.
    
    Args:
        k: Number of documents to retrieve. Defaults to RETRIEVAL_K from config.
    """
    if k is None:
        k = RETRIEVAL_K
    
    try:
        # Use cached index
        index = get_index()
        
        # Initialize embeddings
        embeddings = get_embeddings()
        
        # Create vectorstore
        vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
        
        logger.info(f"Retriever initialized with k={k}")
        return vectorstore.as_retriever(search_kwargs={"k": k})
        
    except Exception as e:
        logger.error(f"Failed to create retriever: {e}")
        raise


def format_docs(docs: List[Document]) -> str:
    """Format retrieved documents for the prompt."""
    logger.info(f"Retrieved {len(docs)} documents")
    
    if not docs:
        logger.warning("No documents retrieved!")
        return "No relevant context found."
    
    parts = []
    for i, d in enumerate(docs):
        # Log first 100 chars of each chunk
        preview = d.page_content[:100].replace('\n', ' ')
        logger.info(f"  Doc {i+1}: {preview}...")
        parts.append(f"[Chunk {i+1}] {d.page_content}")
    
    return "\n\n".join(parts)


def get_rag_chain(system_prompt: str, model_name: str):
    """
    Create a RAG chain with conversational memory support.
    
    Args:
        system_prompt: System message for the LLM
        model_name: Name of the Groq model to use
    """
    try:
        llm = get_chat_model(model_name=model_name)
        retriever = get_resume_retriever()
        
        # Create prompt template with memory support
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "system",
                "Use only the following resume context to answer. "
                "If the context is not enough, say you are not sure.\n\n{context}"
            ),
            ("human", "{question}"),
        ])
        
        # Build RAG pipeline with proper chat_history passing
        rag_pipeline = (
            RunnableParallel(
                question=RunnablePassthrough(),
                docs=retriever,
                # Pass through chat_history if it exists
                chat_history=lambda x: x.get("chat_history", []),
            )
            | (lambda inputs: {
                "question": inputs["question"],
                "context": format_docs(inputs["docs"]),
                "chat_history": inputs["chat_history"],
            })
            | prompt
            | llm
        )
        
        logger.info(f"RAG chain created with model: {model_name}")
        return rag_pipeline
        
    except Exception as e:
        logger.error(f"Failed to create RAG chain: {e}")
        raise