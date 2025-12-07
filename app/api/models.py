"""
Pydantic models for the FastAPI endpoints.

This module defines request and response schemas for the API.
"""

from typing import Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Request model for the /ask endpoint."""
    
    question: str = Field(
        ...,
        description="The question to ask the knowledge graph",
        min_length=1,
        max_length=1000,
        examples=["What diseases are mentioned?", "Who are the patients?"]
    )


class AskResponse(BaseModel):
    """Response model for the /ask endpoint."""
    
    answer: str = Field(
        ...,
        description="The generated answer from the agent"
    )
    used_tools: list[str] = Field(
        default_factory=list,
        description="List of tools used to generate the answer"
    )
    context: Optional[list[str]] = Field(
        default=None,
        description="Context snippets used for the answer"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if processing failed"
    )
    workflow_steps: Optional[list[dict]] = Field(
        default=None,
        description="Workflow execution steps for visualization"
    )


class GraphInfoResponse(BaseModel):
    """Response model for the /graph-info endpoint."""
    
    schema: str = Field(
        ...,
        description="The current graph schema"
    )
    node_count: Optional[int] = Field(
        default=None,
        description="Total number of nodes in the graph"
    )
    relationship_count: Optional[int] = Field(
        default=None,
        description="Total number of relationships in the graph"
    )
    node_labels: Optional[list[str]] = Field(
        default=None,
        description="List of node labels in the graph"
    )
    relationship_types: Optional[list[str]] = Field(
        default=None,
        description="List of relationship types in the graph"
    )


class IngestRequest(BaseModel):
    """Request model for PDF ingestion endpoint (optional)."""
    
    filename: str = Field(
        ...,
        description="Name of the uploaded file"
    )
    # Note: File bytes are sent as form data, not in JSON body


class IngestResponse(BaseModel):
    """Response model for PDF ingestion endpoint."""
    
    success: bool = Field(
        ...,
        description="Whether ingestion was successful"
    )
    message: str = Field(
        ...,
        description="Status message"
    )
    nodes_created: int = Field(
        default=0,
        description="Number of nodes created"
    )
    relationships_created: int = Field(
        default=0,
        description="Number of relationships created"
    )
    chunks_processed: int = Field(
        default=0,
        description="Number of document chunks processed"
    )


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    
    status: str = Field(
        ...,
        description="Service status"
    )
    version: str = Field(
        ...,
        description="API version"
    )
    neo4j_connected: bool = Field(
        ...,
        description="Whether Neo4j is connected"
    )
    vector_store_ready: bool = Field(
        ...,
        description="Whether vector store is initialized"
    )


class ErrorResponse(BaseModel):
    """Standard error response model."""
    
    error: str = Field(
        ...,
        description="Error type"
    )
    message: str = Field(
        ...,
        description="Error message"
    )
    detail: Optional[str] = Field(
        default=None,
        description="Additional error details"
    )

