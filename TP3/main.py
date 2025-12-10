#!/usr/bin/env python3
"""CLI for ingesting multiple resumes into Pinecone."""

import sys
import logging
from rag.pdf_ingest import ingest_resume
from config import PERSONS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    print("=" * 60)
    print("TP3: Multi-Resume Ingestion")
    print("=" * 60)
    
    force = "--force" in sys.argv or "-f" in sys.argv
    
    print(f"\nResumes to ingest:")
    for person_id, info in PERSONS.items():
        print(f"  - {info['name']}: {info['file']}")
    
    print(f"\nForce reingest: {force}\n")
    
    try:
        ingest_resume(force_reingest=force)
        print("\n" + "=" * 60)
        print("✓ All resumes ingested successfully!")
        print("=" * 60)
        print("\nYou can now run: streamlit run app.py")
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ Error: {e}")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
