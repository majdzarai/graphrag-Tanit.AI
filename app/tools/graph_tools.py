"""
Graph query tools for the agent.

This module provides LangChain tools for querying the Neo4j knowledge graph.
These tools are used by the LangGraph agent to retrieve information.
"""

import json
import logging
from typing import Any

from langchain.tools import tool

from app.graph.neo4j_client import get_schema, run_cypher

logger = logging.getLogger(__name__)


@tool
def tool_cypher_query(cypher: str) -> str:
    """
    Execute a Cypher query against the Neo4j knowledge graph.
    
    Use this tool when you need to query the graph database directly.
    Write valid Cypher queries to retrieve nodes, relationships, or patterns.
    
    Args:
        cypher: A valid Cypher query string.
    
    Returns:
        JSON-formatted string of query results, or error message if query fails.
    
    Examples of valid queries:
    - "MATCH (p:Patient) RETURN p.id, p.name LIMIT 10"
    - "MATCH (p:Patient)-[:HAS_DISEASE]->(d:Disease) RETURN p.id, d.id"
    - "MATCH (n) RETURN labels(n)[0] as label, count(*) as count"
    """
    logger.info(f"Executing Cypher query: {cypher[:100]}...")
    
    try:
        # Validate basic query structure
        cypher_upper = cypher.upper().strip()
        if not any(cypher_upper.startswith(kw) for kw in ["MATCH", "RETURN", "CALL", "WITH"]):
            return "Error: Query must start with MATCH, RETURN, CALL, or WITH"
        
        # Block potentially dangerous operations
        dangerous_keywords = ["DELETE", "REMOVE", "SET", "CREATE", "MERGE", "DROP"]
        if any(kw in cypher_upper for kw in dangerous_keywords):
            return "Error: Write operations (DELETE, REMOVE, SET, CREATE, MERGE, DROP) are not allowed"
        
        # Execute query
        results = run_cypher(cypher)
        
        if not results:
            return "Query returned no results."
        
        # Format results as JSON
        formatted = json.dumps(results, indent=2, default=str)
        logger.info(f"Query returned {len(results)} results")
        
        # Truncate if too long
        if len(formatted) > 4000:
            formatted = formatted[:4000] + "\n... (results truncated)"
        
        return formatted
    
    except Exception as e:
        error_msg = f"Cypher query error: {str(e)}"
        logger.error(error_msg)
        return error_msg


@tool
def tool_graph_schema() -> str:
    """
    Get the schema of the Neo4j knowledge graph.
    
    Use this tool to understand what node types, relationship types,
    and properties exist in the graph before writing queries.
    
    Returns:
        String description of the graph schema including:
        - Node labels and their properties
        - Relationship types
        - Sample data patterns
    """
    logger.info("Retrieving graph schema")
    
    try:
        schema = get_schema()
        logger.info("Schema retrieved successfully")
        return schema
    
    except Exception as e:
        error_msg = f"Error retrieving schema: {str(e)}"
        logger.error(error_msg)
        return error_msg


def tool_count_nodes(label: str | None = None) -> str:
    """
    Count nodes in the graph, optionally filtered by label.
    
    Args:
        label: Optional node label to filter by.
    
    Returns:
        Count of nodes as a string.
    """
    try:
        if label:
            cypher = f"MATCH (n:{label}) RETURN count(n) as count"
        else:
            cypher = "MATCH (n) RETURN count(n) as count"
        
        results = run_cypher(cypher)
        count = results[0]["count"] if results else 0
        
        if label:
            return f"There are {count} {label} nodes in the graph."
        return f"There are {count} total nodes in the graph."
    
    except Exception as e:
        return f"Error counting nodes: {str(e)}"


def tool_find_relationships(source_label: str, target_label: str) -> str:
    """
    Find relationships between two node types.
    
    Args:
        source_label: Label of source nodes.
        target_label: Label of target nodes.
    
    Returns:
        Description of relationships found.
    """
    try:
        cypher = f"""
        MATCH (s:{source_label})-[r]->(t:{target_label})
        RETURN DISTINCT type(r) as relationship, count(*) as count
        """
        
        results = run_cypher(cypher)
        
        if not results:
            return f"No relationships found between {source_label} and {target_label}."
        
        lines = [f"Relationships from {source_label} to {target_label}:"]
        for row in results:
            lines.append(f"  - {row['relationship']}: {row['count']} instances")
        
        return "\n".join(lines)
    
    except Exception as e:
        return f"Error finding relationships: {str(e)}"


# Export all tools for the agent
GRAPH_TOOLS = [
    tool_cypher_query,
    tool_graph_schema,
]

