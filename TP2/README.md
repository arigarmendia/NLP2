# CV RAG Chatbot (Groq + Pinecone + LangChain)

This project is a Streamlit application that lets you chat with an AI assistant about the contents of my resume. It uses:
- Groq LLMs for fast, low-latency generation.  
- Pinecone as a vector database and embedding provider.  
- LangChain for RAG orchestration and conversational memory.  

The project ingests the uploaded resume into Pinecone, and the chatbot answers questions strictly based on that document.

## Features

- Retrieval-Augmented Generation (RAG) over a particular resume.
- Persistent conversation memory within a Streamlit session.
- Configurable assistant persona and Groq model.
- Automatic Pinecone index creation if it does not exist.
- Duplicate detection to prevent re-ingesting data unnecessarily.

## Project structure
```text
chatbot_cv/
├── app.py               # Streamlit UI entry point
├── config.py            # Environment variables and constants
├── llm_client.py        # Groq chat model factory
├── memory.py            # Conversation memory utilities
├── rag/
│   ├── __init__.py
│   ├── pdf_ingest.py    # One-off: ingest data/resume.pdf into Pinecone
│   ├── pinecone_client.py
│   └── retriever.py     # RAG chain definition (retriever + Groq model)
└── data/                # Resume
    └── Ariadna_Garmendia_resume.pdf       
```

## Prerequisites

- Python 3.10+ recommended.
- A Groq API key.
- A Pinecone API key (Starter/free plan is sufficient for a single resume and light chat usage).
- Your resume saved as `data/resume.pdf`.

Create a `.env` file in the project root:
```bash
GROQ_API_KEY=your_groq_key_here
PINECONE_API_KEY=your_pinecone_key_here
RESUME_PATH=data/Ariadna_Garmendia_resume.pdf
```

## Installation

1. Clone or copy this repository:
```bash
git clone <your-repo-url> chatbot_cv
cd chatbot_cv
```

2. (Optional) Create and activate a virtual environment.

3. Install dependencies (adjust versions as needed):
```bash
pip install streamlit python-dotenv groq langchain langchain-groq \
           langchain-community langchain-text-splitters langchain-pinecone \
           pinecone-client pypdf
```

4. Place your resume at:
```text
data/Ariadna_Garmendia_resume.pdf
```

## Configuration

All configuration is handled in `config.py`:
- `RESUME_PATH` points to your resume PDF.
- `PINECONE_INDEX_NAME` is set to `cv-rag-index`.
- Pinecone serverless settings are configured for the Starter plan (cloud `aws`, region `us-east-1`), which aligns with Pinecone's free tier defaults.
- Chunk size, overlap, and retrieval parameters can be customized via environment variables.

Groq model defaults or embedding model can be changed if needed.

## Step 1: Ingest your resume into Pinecone

Before using the chatbot, run the ingestion script once (or whenever the resume file is updated):
```bash
python -m rag.pdf_ingest
```

This will:
- Load your resume PDF.
- Split it into text chunks.
- Use Pinecone's embedding model to generate vector embeddings.
- Upsert the chunks into the `cv-rag-index` Pinecone index.

The index is created automatically if it does not exist. If data already exists in the index, you'll be prompted to confirm before re-ingesting. To force re-ingestion without prompts:
```bash
python -m rag.pdf_ingest --force
```

## Step 2: Run the Streamlit app

Start the chatbot UI:
```bash
streamlit run app.py
```

Then open the URL shown in your terminal `http://localhost:8501`.



