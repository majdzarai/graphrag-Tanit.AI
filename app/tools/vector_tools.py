"""
Vector search tools for the agent.

This module provides LangChain tools for semantic search over the
Neo4j vector index. Used by the agent for similarity-based retrieval.
"""

import logging
from typing import Optional

from langchain.tools import tool
from langchain_community.vectorstores import Neo4jVector

from app.graph.graph_rag import get_vector_store

logger = logging.getLogger(__name__)


@tool
def tool_vector_search(query: str, k: int = 5) -> str:
    """
    Search for semantically similar content in the knowledge graph.
    
    Use this tool when you need to find relevant information based on
    meaning rather than exact keyword matches. Good for:
    - Finding related concepts
    - Searching for specific topics
    - Retrieving context for answering questions
    
    Args:
        query: Natural language query describing what to search for.
        k: Number of results to return (default: 5, max: 20).
    
    Returns:
        Formatted string of relevant document excerpts, or error message.
    
    Examples:
    - "What symptoms does the patient have?"
    - "medications for diabetes"
    - "treatment plans for heart disease"
    """
    logger.info(f"Vector search for: {query[:50]}...")
    
    # Validate k
    k = min(max(1, k), 20)  # Clamp between 1 and 20
    
    try:
        vector_store = get_vector_store()
        
        if vector_store is None:
            return "Vector store not initialized. Please ingest documents first."
        
        # Perform similarity search
        results = vector_store.similarity_search(query, k=k)
        
        if not results:
            return f"No relevant content found for: {query}"
        
        # Format results
        output_parts = [f"Found {len(results)} relevant excerpts:\n"]
        
        for i, doc in enumerate(results, 1):
            content = doc.page_content[:500]  # Limit content length
            source = doc.metadata.get("source", "unknown")
            output_parts.append(f"[{i}] (Source: {source})\n{content}\n")
        
        logger.info(f"Vector search returned {len(results)} results")
        return "\n".join(output_parts)
    
    except Exception as e:
        error_msg = f"Vector search error: {str(e)}"
        logger.error(error_msg)
        return error_msg


def vector_search_with_scores(
    query: str,
    vector_store: Optional[Neo4jVector] = None,
    k: int = 5,
    score_threshold: float = 0.0,
) -> list[tuple[str, float, dict]]:
    """
    Perform vector search and return results with similarity scores.
    
    This is a utility function (not a tool) for more advanced usage.
    
    Args:
        query: Search query.
        vector_store: Neo4jVector instance. Uses cached if not provided.
        k: Number of results.
        score_threshold: Minimum similarity score (0-1).
    
    Returns:
        List of tuples (content, score, metadata).
    """
    _store = vector_store or get_vector_store()
    
    if _store is None:
        return []
    
    try:
        # Use similarity_search_with_score if available
        results_with_scores = _store.similarity_search_with_score(query, k=k)
        
        # Filter by threshold and format
        filtered = []
        for doc, score in results_with_scores:
            if score >= score_threshold:
                filtered.append((doc.page_content, score, doc.metadata))
        
        return filtered
    
    except Exception as e:
        logger.error(f"Vector search with scores failed: {e}")
        return []


def hybrid_search(
    query: str,
    vector_store: Optional[Neo4jVector] = None,
    k: int = 5,
) -> str:
    """
    Perform hybrid search combining vector and keyword search.
    
    This leverages Neo4jVector's hybrid search capability if configured.
    
    Args:
        query: Search query.
        vector_store: Neo4jVector instance.
        k: Number of results.
    
    Returns:
        Formatted string of search results.
    """
    logger.info(f"Hybrid search for: {query[:50]}...")
    
    _store = vector_store or get_vector_store()
    
    if _store is None:
        return "Vector store not initialized."
    
    try:
        # Neo4jVector with search_type="hybrid" will do hybrid search
        results = _store.similarity_search(query, k=k)
        
        if not results:
            return f"No results found for: {query}"
        
        output_parts = [f"Hybrid search found {len(results)} results:\n"]
        
        for i, doc in enumerate(results, 1):
            content = doc.page_content[:400]
            output_parts.append(f"[{i}] {content}\n")
        
        return "\n".join(output_parts)
    
    except Exception as e:
        return f"Hybrid search error: {str(e)}"


# Export tools for the agent
VECTOR_TOOLS = [
    tool_vector_search,
]

