"""
Name extraction from resume name (to be used as metadata).
"""
import os
import re
import logging
from typing import Dict, Optional, List
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetadataExtractor:
    """Extract metadata from resumes for production systems."""
    
    @staticmethod
    def extract_from_filename(filepath: str) -> Dict[str, str]:
        """
        Extract person info from filename.
        
        Handles patterns like:
        - "john_doe_resume.pdf" → {id: "john_doe", name: "John Doe"}
        - "Jane_Smith_2024.pdf" → {id: "jane_smith", name: "Jane Smith"}
        - "12345_Alice_Johnson.pdf" → {id: "12345", name: "Alice Johnson"}
        """
        filename = os.path.basename(filepath).replace('.pdf', '')
        
        # Remove common suffixes
        filename = re.sub(
            r'_(resume|Resume|CV|cv|curriculum|vitae|2024|2025)$',
            '',
            filename,
            flags=re.IGNORECASE
        )
        
        # Check for ID prefix (e.g., "12345_John_Doe")
        match = re.match(r'^(\d+)_(.+)$', filename)
        if match:
            person_id = match.group(1)
            name = match.group(2).replace('_', ' ').title()
        else:
            # Use filename as base
            person_id = filename.lower().replace(' ', '_').replace('-', '_')
            name = filename.replace('_', ' ').replace('-', ' ').title()
        
        return {
            "person_id": person_id,
            "person_name": name,
            "original_filename": os.path.basename(filepath)
        }
    
    @staticmethod
    def extract_name_from_text(text: str, fallback: str = "Unknown Candidate") -> str:
        """
        Extract candidate name from resume text.
        Assumes name appears in first few lines (common resume format).
        """
        # Get first few lines (name usually at top)
        lines = [line.strip() for line in text.split('\n')[:10] if line.strip()]
        
        for line in lines:
            # Look for title case pattern (2-4 words, title case)
            # e.g., "John Doe", "Mary Jane Smith", "Dr. Robert Johnson"
            pattern = r'^(?:Dr\.|Mr\.|Ms\.|Mrs\.)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})$'
            match = re.match(pattern, line)
            if match:
                return match.group(1).strip()
        
        # Fallback: look for email and derive name
        email_match = re.search(r'([a-z]+)[._]([a-z]+)@', text.lower())
        if email_match:
            first = email_match.group(1).capitalize()
            last = email_match.group(2).capitalize()
            return f"{first} {last}"
        
        return fallback
    
    @staticmethod
    def extract_email(text: str) -> Optional[str]:
        """Extract email address from resume text."""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        match = re.search(email_pattern, text)
        return match.group(0) if match else None
    
    @staticmethod
    def extract_phone(text: str) -> Optional[str]:
        """Extract phone number from resume text."""
        # US phone patterns
        phone_patterns = [
            r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',  # (555) 123-4567
            r'\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}',  # International
        ]
        
        for pattern in phone_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        
        return None
    
    @staticmethod
    def generate_candidate_id(name: str, email: Optional[str] = None) -> str:
        """
        Generate a unique, stable candidate ID.
        
        In production:
        - Use UUID from database
        - Use employee ID from HR system
        - Use email hash for consistency
        """
        if email:
            # Hash email for consistent ID
            return hashlib.md5(email.lower().encode()).hexdigest()[:12]
        else:
            # Use name-based ID (less reliable)
            return name.lower().replace(' ', '_').replace('-', '_')
    
    @staticmethod
    def extract_all_metadata(filepath: str, text: str) -> Dict:
        """
        Extract comprehensive metadata from resume.
        
        Returns:
            {
                "candidate_id": str,
                "candidate_name": str,
                "email": Optional[str],
                "phone": Optional[str],
                "source_file": str,
                "extraction_method": str
            }
        """
        # Try filename extraction first
        filename_metadata = MetadataExtractor.extract_from_filename(filepath)
        
        # Extract from content
        name_from_text = MetadataExtractor.extract_name_from_text(text)
        email = MetadataExtractor.extract_email(text)
        phone = MetadataExtractor.extract_phone(text)
        
        # Prefer text extraction if it looks valid
        if name_from_text != "Unknown Candidate":
            candidate_name = name_from_text
            extraction_method = "content"
        else:
            candidate_name = filename_metadata["person_name"]
            extraction_method = "filename"
        
        # Generate stable ID
        candidate_id = MetadataExtractor.generate_candidate_id(candidate_name, email)
        
        metadata = {
            "candidate_id": candidate_id,
            "candidate_name": candidate_name,
            "email": email,
            "phone": phone,
            "source_file": os.path.basename(filepath),
            "extraction_method": extraction_method,
            "original_filename_id": filename_metadata["person_id"]
        }
        
        logger.info(
            f"Extracted metadata: {candidate_name} "
            f"(ID: {candidate_id}, Method: {extraction_method})"
        )
        
        return metadata


def discover_resumes_from_directory(directory: str) -> Dict[str, Dict]:
    """
    Auto-discover all resumes in a directory.

    Returns:
        Dictionary mapping candidate_id to metadata
    """
    if not os.path.exists(directory):
        logger.warning(f"Resume directory not found: {directory}")
        return {}
    
    persons = {}
    pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
    
    logger.info(f"Discovered {len(pdf_files)} PDF files in {directory}")
    
    for filename in pdf_files:
        filepath = os.path.join(directory, filename)
        
        # Quick metadata extraction from filename only
        # (full text extraction happens during ingestion)
        metadata = MetadataExtractor.extract_from_filename(filepath)
        
        person_id = metadata["person_id"]
        persons[person_id] = {
            "name": metadata["person_name"],
            "file": filepath,
            "person_id": person_id
        }
        
        logger.info(f"  - {metadata['person_name']} ({person_id})")
    
    return persons


# Example usage in production config:
def get_persons_config(resumes_dir: str = "data/resumes", use_auto_discovery: bool = True):
    """
    Get persons configuration.
    
    Args:
        resumes_dir: Directory containing resumes
        use_auto_discovery: If True, auto-discover from directory.
                           If False, use hardcoded config.
    """
    if use_auto_discovery:
        return discover_resumes_from_directory(resumes_dir)
    else:
        # Fallback to hardcoded (for demo/testing)
        from config import PERSONS_HARDCODED
        return PERSONS_HARDCODED
