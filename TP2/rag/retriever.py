import logging
from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from config import PINECONE_EMBEDDING_MODEL, RETRIEVAL_K
from rag.pinecone_client import get_index
from llm_client import get_chat_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        embeddings = PineconeEmbeddings(model=PINECONE_EMBEDDING_MODEL)
        
        # Create vectorstore
        vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
        
        logger.info(f"Retriever initialized with k={k}")
        return vectorstore.as_retriever(search_kwargs={"k": k})
        
    except Exception as e:
        logger.error(f"Failed to create retriever: {e}")
        raise


def format_docs(docs: List[Document]) -> str:
    """Format retrieved documents for the prompt."""
    if not docs:
        return "No relevant context found."
    
    parts = []
    for i, d in enumerate(docs):
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