"""
FastAPI backend for GraphRAG München.

This module provides the REST API endpoints for:
- /ask - Query the knowledge graph using the agent
- /graph-info - Get graph schema and statistics
- /ingest - Upload and process PDF documents
- /health - Health check endpoint

Run with: uvicorn app.api.main:app --reload --port 8000
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.agent.graph_agent import run_agent
from app.api.models import (
    AskRequest,
    AskResponse,
    ErrorResponse,
    GraphInfoResponse,
    HealthResponse,
    IngestResponse,
)
from app.config import get_embeddings, get_llm, get_settings
from app.graph.graph_rag import (
    build_vector_index,
    get_vector_store,
    ingest_documents,
    set_vector_store,
)
from app.graph.neo4j_client import (
    clear_graph,
    get_graph,
    get_node_count,
    get_node_labels,
    get_relationship_count,
    get_relationship_types,
    get_schema,
)
from app.tools.pdf_tools import process_uploaded_pdf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup
    logger.info("Starting GraphRAG München API...")
    settings = get_settings()
    logger.info(f"App: {settings.app_name} by {settings.author}")
    
    # Try to connect to Neo4j on startup
    try:
        get_graph()
        logger.info("Neo4j connection established")
    except Exception as e:
        logger.warning(f"Neo4j not available on startup: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down GraphRAG München API...")


# Create FastAPI app
app = FastAPI(
    title="GraphRAG München API",
    description="""
    Agentic AI system for querying medical knowledge graphs.
    
    This API provides:
    - **Agent-based Q&A**: Ask questions and get answers from the knowledge graph
    - **Graph Information**: View schema and statistics
    - **Document Ingestion**: Upload PDFs to build the knowledge graph
    
    Created by Majd Zarai
    """,
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_error",
            message="An unexpected error occurred",
            detail=str(exc)
        ).model_dump()
    )


# ============================================
# Endpoints
# ============================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    settings = get_settings()
    return {
        "name": settings.app_name,
        "author": settings.author,
        "version": __version__,
        "docs": "/docs",
        "endpoints": {
            "ask": "POST /ask",
            "graph_info": "GET /graph-info",
            "ingest": "POST /ingest",
            "health": "GET /health",
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns the status of the API and connected services.
    """
    neo4j_connected = False
    vector_ready = False
    
    try:
        get_graph()
        neo4j_connected = True
    except:
        pass
    
    vector_ready = get_vector_store() is not None
    
    return HealthResponse(
        status="healthy" if neo4j_connected else "degraded",
        version=__version__,
        neo4j_connected=neo4j_connected,
        vector_store_ready=vector_ready,
    )


@app.post("/ask", response_model=AskResponse, tags=["Agent"])
async def ask_question(request: AskRequest):
    """
    Ask a question to the knowledge graph agent.
    
    The agent will:
    1. Analyze your question
    2. Search the vector store and/or query the graph
    3. Generate an answer based on the retrieved context
    
    **Example request:**
    ```json
    {
        "question": "What diseases are mentioned in the documents?"
    }
    ```
    
    **Example response:**
    ```json
    {
        "answer": "Based on the knowledge graph, the following diseases are mentioned...",
        "used_tools": ["vector_search", "graph_traversal"],
        "context": ["Relevant excerpt 1...", "Relevant excerpt 2..."]
    }
    ```
    """
    logger.info(f"Received question: {request.question[:100]}...")
    
    try:
        # Run the agent
        result = run_agent(request.question)
        
        return AskResponse(
            answer=result["answer"],
            used_tools=result["used_tools"],
            context=result.get("context_snippets"),
            error=result.get("error"),
            workflow_steps=result.get("workflow_steps", []),  # Include workflow steps
        )
    
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process question: {str(e)}"
        )


@app.get("/graph-info", response_model=GraphInfoResponse, tags=["Graph"])
async def get_graph_info():
    """
    Get information about the knowledge graph.
    
    Returns the graph schema, node/relationship counts, and labels.
    """
    try:
        schema = get_schema()
        
        # Get statistics (may fail if graph is empty)
        node_count = None
        rel_count = None
        labels = None
        rel_types = None
        
        try:
            node_count = get_node_count()
            rel_count = get_relationship_count()
            labels = get_node_labels()
            rel_types = get_relationship_types()
        except Exception as e:
            logger.warning(f"Could not get graph statistics: {e}")
        
        return GraphInfoResponse(
            schema=schema,
            node_count=node_count,
            relationship_count=rel_count,
            node_labels=labels,
            relationship_types=rel_types,
        )
    
    except Exception as e:
        logger.error(f"Error getting graph info: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get graph info: {str(e)}"
        )


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_pdf(
    file: UploadFile = File(..., description="PDF file to ingest"),
    clear_existing: bool = False,
):
    """
    Upload and ingest a PDF document into the knowledge graph.
    
    The document will be:
    1. Parsed and split into chunks
    2. Processed by LLM to extract entities and relationships
    3. Added to the Neo4j knowledge graph
    4. Indexed for vector similarity search
    
    **Parameters:**
    - `file`: PDF file to upload
    - `clear_existing`: If true, clears existing graph data before ingestion
    
    **Note:** This endpoint may take several minutes for large documents.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    logger.info(f"Ingesting PDF: {file.filename}")
    
    try:
        # Read file bytes
        file_bytes = await file.read()
        
        # Process PDF
        docs = process_uploaded_pdf(file_bytes, file.filename)
        logger.info(f"Processed {len(docs)} document chunks")
        
        # Clear graph if requested
        if clear_existing:
            logger.info("Clearing existing graph data")
            clear_graph()
        
        # Get LLM for extraction
        llm = get_llm()
        
        # Ingest documents
        nodes_created, rels_created = ingest_documents(docs, llm)
        
        # Build vector index
        try:
            settings = get_settings()
            embeddings = get_embeddings()
            vector_store = build_vector_index(
                embeddings=embeddings,
                url=settings.neo4j_url,
                username=settings.neo4j_username,
                password=settings.neo4j_password,
            )
            set_vector_store(vector_store)
            logger.info("Vector index created successfully")
        except Exception as e:
            logger.warning(f"Vector index creation failed: {e}")
        
        return IngestResponse(
            success=True,
            message=f"Successfully ingested {file.filename}",
            nodes_created=nodes_created,
            relationships_created=rels_created,
            chunks_processed=len(docs),
        )
    
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to ingest document: {str(e)}"
        )


@app.delete("/graph", tags=["Graph"])
async def clear_graph_data():
    """
    Clear all data from the knowledge graph.
    
    **WARNING:** This permanently deletes all nodes and relationships!
    """
    try:
        clear_graph()
        return {"message": "Graph cleared successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear graph: {str(e)}"
        )


@app.post("/query/cypher", tags=["Graph"])
async def execute_cypher(query: str):
    """
    Execute a raw Cypher query against the graph.
    
    **Note:** Only read queries are allowed. Write operations are blocked.
    """
    from app.tools.graph_tools import tool_cypher_query
    
    result = tool_cypher_query.invoke({"cypher": query})
    return {"result": result}


# ============================================
# Main entry point
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )

