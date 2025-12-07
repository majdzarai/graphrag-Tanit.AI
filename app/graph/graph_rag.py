"""
GraphRAG pipeline module.

This module handles:
- Document ingestion and graph construction using LLMGraphTransformer
- Vector index creation with Neo4jVector
- Hybrid retrieval combining vector similarity and graph traversal
"""

import logging
from typing import Any, Optional

from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain.schema import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config import get_settings
from app.graph.neo4j_client import get_graph, refresh_schema, run_cypher

logger = logging.getLogger(__name__)

# Store vector index instance for reuse
_vector_store: Optional[Neo4jVector] = None


def ingest_documents(
    docs: list[Document],
    llm: ChatOpenAI,
    graph: Optional[Neo4jGraph] = None,
    allowed_nodes: Optional[list[str]] = None,
    allowed_relationships: Optional[list[str]] = None,
) -> tuple[int, int]:
    """
    Ingest documents into the Neo4j knowledge graph.
    
    Uses LLMGraphTransformer to extract entities and relationships
    from documents and adds them to the graph.
    
    Args:
        docs: List of Document objects to process.
        llm: ChatOpenAI instance for entity extraction.
        graph: Optional Neo4jGraph instance. Uses singleton if not provided.
        allowed_nodes: List of allowed node types. Uses defaults if not provided.
        allowed_relationships: List of allowed relationship types. Uses defaults if not provided.
    
    Returns:
        Tuple of (total_nodes, total_relationships) extracted.
    
    Example:
        >>> from app.tools.pdf_tools import load_pdf
        >>> docs = load_pdf("medical_report.pdf")
        >>> nodes, rels = ingest_documents(docs, llm)
        >>> print(f"Extracted {nodes} nodes and {rels} relationships")
    """
    settings = get_settings()
    
    # Use provided or default values
    _graph = graph or get_graph()
    _allowed_nodes = allowed_nodes or list(settings.allowed_nodes)
    _allowed_rels = allowed_relationships or list(settings.allowed_relationships)
    
    logger.info(f"Processing {len(docs)} documents for graph ingestion")
    logger.info(f"Allowed nodes: {_allowed_nodes}")
    logger.info(f"Allowed relationships: {_allowed_rels}")
    
    # Create transformer with LLM
    transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=_allowed_nodes,
        allowed_relationships=_allowed_rels,
        node_properties=True,
        relationship_properties=True,
    )
    
    # Convert documents to graph documents
    logger.info("Extracting entities and relationships with LLM...")
    graph_docs = transformer.convert_to_graph_documents(docs)
    
    # Count extracted entities
    total_nodes = sum(len(doc.nodes) for doc in graph_docs)
    total_rels = sum(len(doc.relationships) for doc in graph_docs)
    
    logger.info(f"Extracted {total_nodes} nodes and {total_rels} relationships")
    
    # Add to graph
    logger.info("Adding graph documents to Neo4j...")
    _graph.add_graph_documents(graph_docs, include_source=True)
    
    # Refresh schema after adding documents
    refresh_schema()
    
    logger.info("Document ingestion complete")
    return total_nodes, total_rels


def build_vector_index(
    embeddings: OpenAIEmbeddings,
    url: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    database: str = "neo4j",
    node_label: str = "Patient",
    text_node_properties: Optional[list[str]] = None,
    embedding_node_property: str = "embedding",
    index_name: str = "vector_index",
    keyword_index_name: str = "entity_index",
    search_type: str = "hybrid",
) -> Neo4jVector:
    """
    Build a vector index on the Neo4j graph for similarity search.
    
    Creates embeddings for specified node properties and stores them
    in Neo4j for vector similarity search.
    
    Args:
        embeddings: OpenAIEmbeddings instance for generating embeddings.
        url: Neo4j URL. Uses env var if not provided.
        username: Neo4j username. Uses env var if not provided.
        password: Neo4j password. Uses env var if not provided.
        database: Neo4j database name.
        node_label: Label of nodes to index.
        text_node_properties: Properties to embed. Defaults to ["id", "text"].
        embedding_node_property: Property name for storing embeddings.
        index_name: Name of the vector index.
        keyword_index_name: Name of the keyword index for hybrid search.
        search_type: Search type - "vector", "keyword", or "hybrid".
    
    Returns:
        Neo4jVector instance for similarity search.
    """
    global _vector_store
    
    settings = get_settings()
    
    # Use provided or default values
    _url = url or settings.neo4j_url
    _username = username or settings.neo4j_username
    _password = password or settings.neo4j_password
    _text_props = text_node_properties or ["id", "text"]
    
    logger.info(f"Building vector index on {node_label} nodes")
    logger.info(f"Text properties: {_text_props}")
    logger.info(f"Index name: {index_name}")
    
    try:
        _vector_store = Neo4jVector.from_existing_graph(
            embedding=embeddings,
            url=_url,
            username=_username,
            password=_password,
            database=database,
            node_label=node_label,
            text_node_properties=_text_props,
            embedding_node_property=embedding_node_property,
            index_name=index_name,
            keyword_index_name=keyword_index_name,
            search_type=search_type,
        )
        logger.info("Vector index created successfully")
        return _vector_store
    
    except Exception as e:
        logger.warning(f"Failed to create vector index: {e}")
        logger.info("Continuing without vector index - will use graph queries only")
        raise


def get_vector_store() -> Optional[Neo4jVector]:
    """Get the current vector store instance."""
    return _vector_store


def set_vector_store(store: Neo4jVector) -> None:
    """Set the vector store instance."""
    global _vector_store
    _vector_store = store


def reset_vector_store() -> None:
    """Reset the vector store instance."""
    global _vector_store
    _vector_store = None


def hybrid_retrieve(
    question: str,
    vector_store: Optional[Neo4jVector] = None,
    k: int = 5,
    expand_neighbors: bool = True,
    neighbor_depth: int = 1,
) -> dict[str, Any]:
    """
    Perform hybrid retrieval combining vector similarity and graph traversal.
    
    This function:
    1. Uses vector similarity search to find relevant nodes
    2. Optionally expands to neighboring nodes via graph traversal
    3. Returns structured context for LLM consumption
    
    Args:
        question: User question to search for.
        vector_store: Neo4jVector instance. Uses cached instance if not provided.
        k: Number of results to return from vector search.
        expand_neighbors: Whether to include neighboring nodes.
        neighbor_depth: How many hops to traverse for neighbors.
    
    Returns:
        Dictionary with structure:
        {
            "texts": [str, ...],        # Retrieved text content
            "graph_facts": [str, ...],  # Graph relationships as text
            "metadata": {...}           # Additional retrieval metadata
        }
    """
    _store = vector_store or _vector_store
    
    result = {
        "texts": [],
        "graph_facts": [],
        "metadata": {
            "question": question,
            "k": k,
            "vector_results": 0,
            "graph_facts_count": 0,
        }
    }
    
    # Vector similarity search
    if _store is not None:
        try:
            logger.info(f"Performing vector search for: {question[:50]}...")
            docs = _store.similarity_search(question, k=k)
            
            result["texts"] = [doc.page_content for doc in docs]
            result["metadata"]["vector_results"] = len(docs)
            
            logger.info(f"Vector search returned {len(docs)} results")
        
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
    else:
        logger.info("No vector store available, skipping vector search")
    
    # Graph traversal for additional context
    if expand_neighbors:
        try:
            # Get nodes and their relationships
            cypher = """
            MATCH (n)-[r]-(m)
            RETURN DISTINCT 
                labels(n)[0] as source_label,
                n.id as source_id,
                type(r) as relationship,
                labels(m)[0] as target_label,
                m.id as target_id
            LIMIT 50
            """
            
            graph_results = run_cypher(cypher)
            
            # Format as readable facts
            facts = []
            for row in graph_results:
                fact = f"{row['source_label']}({row['source_id']}) -{row['relationship']}-> {row['target_label']}({row['target_id']})"
                facts.append(fact)
            
            result["graph_facts"] = facts
            result["metadata"]["graph_facts_count"] = len(facts)
            
            logger.info(f"Retrieved {len(facts)} graph facts")
        
        except Exception as e:
            logger.warning(f"Graph traversal failed: {e}")
    
    return result


def format_context_for_llm(retrieval_result: dict[str, Any]) -> str:
    """
    Format retrieval results into a context string for LLM.
    
    Args:
        retrieval_result: Output from hybrid_retrieve().
    
    Returns:
        Formatted string suitable for LLM context.
    """
    parts = []
    
    # Add vector search results
    if retrieval_result["texts"]:
        parts.append("=== Retrieved Documents ===")
        for i, text in enumerate(retrieval_result["texts"], 1):
            parts.append(f"{i}. {text}")
    
    # Add graph facts
    if retrieval_result["graph_facts"]:
        parts.append("\n=== Knowledge Graph Facts ===")
        for fact in retrieval_result["graph_facts"][:20]:  # Limit to 20 facts
            parts.append(f"- {fact}")
    
    if not parts:
        return "No relevant context found."
    
    return "\n".join(parts)

