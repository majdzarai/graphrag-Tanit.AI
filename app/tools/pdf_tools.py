"""
PDF processing tools.

This module provides functions for loading and processing PDF documents
for ingestion into the knowledge graph.
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def load_pdf(
    path: str,
    chunk_size: int = 200,
    chunk_overlap: int = 40,
    clean_newlines: bool = True,
) -> list[Document]:
    """
    Load and process a PDF file into document chunks.
    
    Uses PyPDFLoader to extract text and RecursiveCharacterTextSplitter
    to split into manageable chunks for processing.
    
    Args:
        path: Path to the PDF file.
        chunk_size: Maximum size of each text chunk.
        chunk_overlap: Overlap between consecutive chunks.
        clean_newlines: Whether to replace newlines with spaces.
    
    Returns:
        List of Document objects with cleaned text.
    
    Example:
        >>> docs = load_pdf("medical_report.pdf")
        >>> print(f"Loaded {len(docs)} chunks")
    """
    logger.info(f"Loading PDF from: {path}")
    
    # Validate path
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")
    
    if not pdf_path.suffix.lower() == ".pdf":
        raise ValueError(f"File is not a PDF: {path}")
    
    # Load PDF
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load_and_split()
    
    logger.info(f"Loaded {len(pages)} pages from PDF")
    
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    docs = splitter.split_documents(pages)
    
    logger.info(f"Split into {len(docs)} chunks")
    
    # Clean documents
    if clean_newlines:
        cleaned_docs = []
        for doc in docs:
            cleaned_content = doc.page_content.replace("\n", " ")
            cleaned_docs.append(
                Document(
                    page_content=cleaned_content,
                    metadata=doc.metadata,
                )
            )
        docs = cleaned_docs
        logger.info("Cleaned newlines from document chunks")
    
    return docs


def process_uploaded_pdf(
    file_bytes: bytes,
    filename: str,
    chunk_size: int = 200,
    chunk_overlap: int = 40,
    clean_newlines: bool = True,
) -> list[Document]:
    """
    Process an uploaded PDF file from bytes.
    
    Saves the bytes to a temporary file, processes it, and returns
    document chunks. Used by web interfaces (Streamlit, FastAPI).
    
    Args:
        file_bytes: Raw bytes of the PDF file.
        filename: Original filename (used for metadata).
        chunk_size: Maximum size of each text chunk.
        chunk_overlap: Overlap between consecutive chunks.
        clean_newlines: Whether to replace newlines with spaces.
    
    Returns:
        List of Document objects with source metadata set to filename.
    
    Example:
        >>> with open("report.pdf", "rb") as f:
        ...     file_bytes = f.read()
        >>> docs = process_uploaded_pdf(file_bytes, "report.pdf")
    """
    logger.info(f"Processing uploaded PDF: {filename} ({len(file_bytes)} bytes)")
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    
    logger.info(f"Saved to temporary file: {tmp_path}")
    
    try:
        # Load and process
        docs = load_pdf(
            path=tmp_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            clean_newlines=clean_newlines,
        )
        
        # Update metadata with original filename
        for doc in docs:
            doc.metadata["source"] = filename
        
        logger.info(f"Processed {len(docs)} document chunks from {filename}")
        return docs
    
    finally:
        # Clean up temp file
        try:
            Path(tmp_path).unlink()
            logger.debug(f"Cleaned up temporary file: {tmp_path}")
        except Exception as e:
            logger.warning(f"Failed to delete temp file {tmp_path}: {e}")


def get_document_stats(docs: list[Document]) -> dict:
    """
    Get statistics about processed documents.
    
    Args:
        docs: List of Document objects.
    
    Returns:
        Dictionary with document statistics.
    """
    if not docs:
        return {
            "total_chunks": 0,
            "total_characters": 0,
            "avg_chunk_size": 0,
            "sources": [],
        }
    
    total_chars = sum(len(doc.page_content) for doc in docs)
    sources = list(set(doc.metadata.get("source", "unknown") for doc in docs))
    
    return {
        "total_chunks": len(docs),
        "total_characters": total_chars,
        "avg_chunk_size": total_chars // len(docs),
        "sources": sources,
    }

