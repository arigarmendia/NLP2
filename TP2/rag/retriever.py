from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings

from config import PINECONE_EMBEDDING_MODEL
from .pinecone_client import get_pinecone_client, ensure_index
from llm_client import get_chat_model


def get_resume_retriever(k: int = 4):
    pc = get_pinecone_client()
    index = ensure_index(pc)

    embeddings = PineconeEmbeddings(model=PINECONE_EMBEDDING_MODEL)
    vectorstore = PineconeVectorStore(index=index, embedding=embeddings)

    return vectorstore.as_retriever(search_kwargs={"k": k})


def get_rag_chain(system_prompt: str, model_name: str):
    llm = get_chat_model(model_name=model_name)

    retriever = get_resume_retriever()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "system",
                "Use only the following resume context to answer. "
                "If the context is not enough, say you are not sure.\n\n{context}",
            ),
            ("human", "{question}"),
        ]
    )

    def format_docs(docs: List[Document]) -> str:
        parts = []
        for i, d in enumerate(docs):
            parts.append(f"[Chunk {i+1}] {d.page_content}")
        return "\n\n".join(parts)

    rag_pipeline = (
        RunnableParallel(
            question=RunnablePassthrough(),
            docs=retriever,
        )
        | (lambda inputs: {"question": inputs["question"], "context": format_docs(inputs["docs"])})
        | prompt
        | llm
    )

    return rag_pipeline
