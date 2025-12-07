# graphrag-Tanit.AI

**Agentic AI System for Knowledge Graph Retrieval-Augmented Generation**

Created by **Majd Zarai**

> A modular Agentic AI system that combines knowledge graphs with Large Language Models for intelligent question answering using Neo4j, LangGraph, and FastAPI.

---

## Overview

this  is a modular Agentic AI system that:

1. **Extracts knowledge** from PDF documents using LLM-powered entity extraction
2. **Builds a knowledge graph** in Neo4j with nodes, relationships, and properties
3. **Enables natural language queries** via a LangGraph-based agent that decides when to use tools
4. **Provides both API and UI** interfaces for interaction

### Key Features

- 🧠 **LangGraph Agent** - Multi-step reasoning with tool selection
- 🕸️ **Neo4j GraphRAG** - Hybrid retrieval (vector + graph traversal)
- ⚡ **FastAPI Backend** - RESTful API with `/ask` and `/graph-info` endpoints
- 🎨 **Streamlit UI** - Professional web interface for document upload and Q&A
- 🔧 **Custom Tools** - Cypher queries, vector search, schema lookup

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         User                                 │
└───────────────────────────┬─────────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
┌───────────────────┐           ┌───────────────────┐
│   Streamlit UI    │◄─────────►│   FastAPI API     │
│  (app/ui/)        │   HTTP    │  (app/api/)       │
└─────────┬─────────┘           └─────────┬─────────┘
          │                               │
          └───────────────┬───────────────┘
                          ▼
               ┌───────────────────┐
               │  LangGraph Agent  │
               │  (app/agent/)     │
               └─────────┬─────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ Graph Tools │ │Vector Tools │ │ PDF Tools   │
│             │ │             │ │             │
└──────┬──────┘ └──────┬──────┘ └─────────────┘
       │               │
       └───────┬───────┘
               ▼
┌─────────────────────────────────────────┐
│              Neo4j Database              │
│   ┌─────────────┐  ┌─────────────────┐  │
│   │ Graph Store │  │  Vector Index   │  │
│   │ (Nodes/Rels)│  │  (Embeddings)   │  │
│   └─────────────┘  └─────────────────┘  │
└─────────────────────────────────────────┘
```

### Project Structure

```
app/
├── __init__.py
├── config.py                 # Settings, env vars, singletons
├── graph/
│   ├── neo4j_client.py       # Neo4jGraph wrapper
│   └── graph_rag.py          # Ingestion, vector index, retrieval
├── tools/
│   ├── pdf_tools.py          # PDF loading and processing
│   ├── graph_tools.py        # Cypher query tool, schema tool
│   └── vector_tools.py       # Vector similarity search tool
├── agent/
│   ├── state.py              # AgentState TypedDict
│   └── graph_agent.py        # LangGraph workflow
├── api/
│   ├── models.py             # Pydantic request/response models
│   └── main.py               # FastAPI endpoints
└── ui/
    └── streamlit_app.py      # Streamlit web interface
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Neo4j Database (local or cloud)
- OpenRouter API key (or OpenAI API key)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/majdzarai/graphrag-Tanit.AI.git
cd GRAPHY-V1-MAIN
```

2. **Create virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure environment variables**

Create a `.env` file in the project root:

```env
# OpenRouter/OpenAI
OPENAI_API_KEY=sk-or-your-key-here
OPENAI_BASE_URL=https://openrouter.ai/api/v1

# Neo4j
NEO4J_URL=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password
```

### Running Neo4j

**Option 1: Docker**

```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest
```

**Option 2: Neo4j Desktop**

Download from [neo4j.com](https://neo4j.com/download/) and create a local database.

**Option 3: Neo4j Aura (Cloud)**

Create a free instance at [neo4j.com/cloud/aura](https://neo4j.com/cloud/aura/).

---

## Usage

### Option 1: Run the Streamlit UI

```bash
streamlit run app/ui/streamlit_app.py
```

Open http://localhost:8501 in your browser.

### Option 2: Run the FastAPI Backend

```bash
uvicorn app.api.main:app --reload --port 8000
```

API documentation available at http://localhost:8000/docs

### Option 3: Run Both (Recommended)

Terminal 1 - Start FastAPI:
```bash
uvicorn app.api.main:app --reload --port 8000
```

Terminal 2 - Start Streamlit with API mode:
```bash
USE_API=true streamlit run app/ui/streamlit_app.py
```

---

## API Reference

### POST /ask

Query the knowledge graph using the agent.

**Request:**
```json
{
  "question": "What diseases are mentioned in the documents?"
}
```

**Response:**
```json
{
  "answer": "Based on the knowledge graph, the following diseases are mentioned...",
  "used_tools": ["vector_search", "graph_traversal"],
  "context": ["Relevant excerpt 1...", "Relevant excerpt 2..."]
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What patients are in the graph?"}'
```

### GET /graph-info

Get information about the knowledge graph.

**Response:**
```json
{
  "schema": "Node properties...",
  "node_count": 42,
  "relationship_count": 38,
  "node_labels": ["Patient", "Disease", "Medication"],
  "relationship_types": ["HAS_DISEASE", "TAKES_MEDICATION"]
}
```

**cURL Example:**
```bash
curl http://localhost:8000/graph-info
```

### POST /ingest

Upload and process a PDF document.

**cURL Example:**
```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@medical_report.pdf" \
  -F "clear_existing=true"
```

### GET /health

Health check endpoint.

```bash
curl http://localhost:8000/health
```

---

## Agent Workflow

The LangGraph agent follows this workflow:

```
[Question] → [Router] → [Tools?]
                │
    ┌───────────┴───────────┐
    ▼                       ▼
[Call Tools]           [Direct Answer]
    │                       │
    ▼                       │
[LLM Answer]                │
    │                       │
    └───────────┬───────────┘
                ▼
            [Response]
```

### Tools Available

| Tool | Description |
|------|-------------|
| `vector_search` | Semantic similarity search over document embeddings |
| `graph_traversal` | Explore relationships between entities |
| `cypher_query` | Execute Cypher queries against Neo4j |
| `schema_lookup` | Get the current graph schema |

---

## Graph Schema

The default schema extracts medical entities:

**Nodes:**
- `Patient` - Medical patients
- `Disease` - Diseases and conditions
- `Medication` - Drugs and treatments
- `Test` - Medical tests and procedures
- `Symptom` - Symptoms and complaints
- `Doctor` - Healthcare providers

**Relationships:**
- `HAS_DISEASE` - Patient has a disease
- `TAKES_MEDICATION` - Patient takes medication
- `UNDERWENT_TEST` - Patient underwent a test
- `HAS_SYMPTOM` - Patient has a symptom
- `TREATED_BY` - Patient treated by doctor

---

## Development

### Running Tests

```bash
pytest tests/
```

### Code Structure

- **Config layer** (`app/config.py`): Environment variables, singletons
- **Graph layer** (`app/graph/`): Neo4j operations, RAG pipeline
- **Tools layer** (`app/tools/`): LangChain tools for the agent
- **Agent layer** (`app/agent/`): LangGraph workflow
- **API layer** (`app/api/`): FastAPI endpoints
- **UI layer** (`app/ui/`): Streamlit interface

### Adding New Tools

1. Create a new function in `app/tools/`
2. Decorate with `@tool` from LangChain
3. Add to the agent's tool list in `app/agent/graph_agent.py`

Example:
```python
from langchain.tools import tool

@tool
def my_custom_tool(input: str) -> str:
    """Description of what this tool does."""
    # Implementation
    return result
```

---

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.

---

## Author

**Majd Zarai**

Built as part of a Generative AI internship assignment demonstrating:
- Agentic AI systems with LangGraph
- GraphRAG pipelines with Neo4j
- Tool-using LLM reasoning
- Clean architecture and modular design

---

## Acknowledgments

- [LangChain](https://python.langchain.com/) - LLM application framework
- [LangGraph](https://langchain-ai.github.io/langgraph/) - Agent workflow orchestration
- [Neo4j](https://neo4j.com/) - Graph database
- [OpenRouter](https://openrouter.ai/) - LLM API gateway
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [Streamlit](https://streamlit.io/) - Data app framework
