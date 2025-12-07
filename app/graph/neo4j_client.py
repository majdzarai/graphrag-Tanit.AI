"""
Neo4j client module - Wrapper for Neo4jGraph with singleton pattern.

This module provides a clean interface to the Neo4j graph database,
with functions for querying, schema retrieval, and graph management.
"""

import logging
from typing import Any, Optional

from langchain_community.graphs import Neo4jGraph

from app.config import get_settings

logger = logging.getLogger(__name__)

# Singleton instance
_graph_instance: Optional[Neo4jGraph] = None


def get_graph(
    url: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> Neo4jGraph:
    """
    Get Neo4jGraph instance (singleton pattern).
    
    Creates a connection to Neo4j on first call, reuses it on subsequent calls.
    Pass explicit credentials to override environment variables.
    
    Args:
        url: Neo4j database URL. Uses env var if not provided.
        username: Neo4j username. Uses env var if not provided.
        password: Neo4j password. Uses env var if not provided.
    
    Returns:
        Neo4jGraph instance connected to the database.
    
    Raises:
        ValueError: If connection credentials are missing.
        Exception: If connection to Neo4j fails.
    """
    global _graph_instance
    
    settings = get_settings()
    
    # Use provided values or fall back to settings
    _url = url or settings.neo4j_url
    _username = username or settings.neo4j_username
    _password = password or settings.neo4j_password
    
    if _graph_instance is None:
        if not all([_url, _username, _password]):
            raise ValueError(
                "Neo4j credentials are required. "
                "Set NEO4J_URL, NEO4J_USERNAME, NEO4J_PASSWORD env vars or pass parameters."
            )
        
        logger.info(f"Connecting to Neo4j at {_url}")
        _graph_instance = Neo4jGraph(
            url=_url,
            username=_username,
            password=_password,
        )
        logger.info("Successfully connected to Neo4j")
    
    return _graph_instance


def reset_graph_instance() -> None:
    """
    Reset the graph instance singleton.
    Useful when reconnecting with different credentials.
    """
    global _graph_instance
    _graph_instance = None
    logger.info("Graph instance reset")


def clear_graph() -> None:
    """
    Clear all nodes and relationships from the graph.
    
    Executes: MATCH (n) DETACH DELETE n
    
    WARNING: This permanently deletes all data in the graph!
    """
    graph = get_graph()
    logger.warning("Clearing all nodes and relationships from graph")
    graph.query("MATCH (n) DETACH DELETE n;")
    graph.refresh_schema()
    logger.info("Graph cleared successfully")


def get_schema() -> str:
    """
    Get the current graph schema.
    
    Returns:
        String representation of the graph schema including
        node labels, relationship types, and properties.
    """
    graph = get_graph()
    schema = graph.get_schema
    logger.debug(f"Retrieved schema: {schema[:100]}...")
    return schema


def run_cypher(
    query: str,
    params: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """
    Execute a Cypher query against the graph.
    
    Args:
        query: Cypher query string.
        params: Optional dictionary of query parameters.
    
    Returns:
        List of dictionaries containing query results.
    
    Example:
        >>> results = run_cypher("MATCH (p:Patient) RETURN p.name LIMIT 5")
        >>> results = run_cypher(
        ...     "MATCH (p:Patient {name: $name}) RETURN p",
        ...     params={"name": "John"}
        ... )
    """
    graph = get_graph()
    logger.debug(f"Executing Cypher: {query[:100]}...")
    
    if params:
        results = graph.query(query, params=params)
    else:
        results = graph.query(query)
    
    logger.debug(f"Query returned {len(results)} results")
    return results


def refresh_schema() -> None:
    """
    Refresh the cached graph schema.
    
    Call this after adding or modifying nodes/relationships
    to update the schema information.
    """
    graph = get_graph()
    graph.refresh_schema()
    logger.info("Graph schema refreshed")


def get_node_count() -> int:
    """Get total number of nodes in the graph."""
    results = run_cypher("MATCH (n) RETURN count(n) as count")
    return results[0]["count"] if results else 0


def get_relationship_count() -> int:
    """Get total number of relationships in the graph."""
    results = run_cypher("MATCH ()-[r]->() RETURN count(r) as count")
    return results[0]["count"] if results else 0


def get_node_labels() -> list[str]:
    """Get all node labels in the graph."""
    results = run_cypher("CALL db.labels() YIELD label RETURN label")
    return [r["label"] for r in results]


def get_relationship_types() -> list[str]:
    """Get all relationship types in the graph."""
    results = run_cypher("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")
    return [r["relationshipType"] for r in results]

