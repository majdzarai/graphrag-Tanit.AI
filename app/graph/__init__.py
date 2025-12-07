"""Graph layer - Neo4j client and GraphRAG pipeline."""

from app.graph.neo4j_client import (
    get_graph,
    clear_graph,
    get_schema,
    run_cypher,
    reset_graph_instance,
    refresh_schema,
    get_node_count,
    get_relationship_count,
    get_node_labels,
    get_relationship_types,
)
from app.graph.graph_rag import (
    ingest_documents,
    build_vector_index,
    hybrid_retrieve,
    get_vector_store,
    set_vector_store,
    reset_vector_store,
    format_context_for_llm,
)

__all__ = [
    # Neo4j client
    "get_graph",
    "clear_graph",
    "get_schema",
    "run_cypher",
    "reset_graph_instance",
    "refresh_schema",
    "get_node_count",
    "get_relationship_count",
    "get_node_labels",
    "get_relationship_types",
    # GraphRAG
    "ingest_documents",
    "build_vector_index",
    "hybrid_retrieve",
    "get_vector_store",
    "set_vector_store",
    "reset_vector_store",
    "format_context_for_llm",
]

