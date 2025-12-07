import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings

from config import RESUME_PATH, PINECONE_EMBEDDING_MODEL
from .pinecone_client import get_pinecone_client, ensure_index


def load_and_split_resume(path: str) -> list[Document]:
    loader = PyPDFLoader(path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )
    return splitter.split_documents(docs)


def ingest_resume():
    pc = get_pinecone_client()
    index = ensure_index(pc)

    docs = load_and_split_resume(RESUME_PATH)

    embeddings = PineconeEmbeddings(model=PINECONE_EMBEDDING_MODEL)

    for d in docs:
        if d.metadata is None:
            d.metadata = {}
        d.metadata.setdefault("source", "resume")

    vectorstore = PineconeVectorStore(index=index, embedding=embeddings)

    vectorstore.add_documents(docs)

    print(f"Ingested {len(docs)} chunks from resume into index.")


if __name__ == "__main__":
    ingest_resume()
