"""
Streamlit UI for GraphRAG München.

This is the web frontend that provides:
- PDF upload and ingestion
- Chat interface for querying the knowledge graph
- Graph statistics display

The UI can work in two modes:
1. Direct mode: Uses local modules directly
2. API mode: Calls the FastAPI backend (requires backend to be running)

Run with: streamlit run app/ui/streamlit_app.py
"""

import os
import sys
from pathlib import Path

import requests
import streamlit as st

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.config import get_settings, get_llm, get_embeddings, configure_from_credentials
from app.graph.neo4j_client import get_graph, clear_graph, get_schema, reset_graph_instance
from app.graph.graph_rag import ingest_documents, build_vector_index, set_vector_store
from app.tools.pdf_tools import process_uploaded_pdf

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
USE_API = os.getenv("USE_API", "false").lower() == "true"

settings = get_settings()
APP_NAME = settings.app_name
AUTHOR = settings.author


# ============================================
# Custom CSS
# ============================================

def inject_custom_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

:root {
    --primary: #00d4aa;
    --primary-light: #00f5c4;
    --primary-dark: #00b894;
    --accent: #667eea;
    --accent-light: #764ba2;
    --success: #00b894;
    --warning: #fdcb6e;
    --error: #e17055;
    --bg-dark: #0a0a0f;
    --bg-card: #12121a;
    --text-primary: #ffffff;
    --text-secondary: #8892b0;
    --border: #1e1e2e;
}

.stApp {
    background: radial-gradient(ellipse at top, #0d1117 0%, #0a0a0f 50%, #000000 100%);
}

.main .block-container {
    padding: 2rem 3rem;
    max-width: 1200px;
}

h1, h2, h3, h4, h5, h6, p, span, div, label {
    font-family: 'Space Grotesk', sans-serif !important;
}

.hero-container {
    text-align: center;
    padding: 2rem 0 1rem 0;
}

.hero-title {
    font-size: 4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00d4aa 0%, #667eea 50%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
    letter-spacing: -0.03em;
}

.hero-subtitle {
    font-size: 1.1rem;
    color: #8892b0;
    margin-bottom: 0.5rem;
}

.hero-author {
    font-size: 0.85rem;
    color: #5a6a8a;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.glass-card {
    background: rgba(18, 18, 26, 0.6);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(30, 30, 46, 0.8);
    border-radius: 16px;
    padding: 2rem;
    margin: 1rem 0;
}

.glass-card h3 {
    color: #ffffff;
    margin-bottom: 0.75rem;
}

.glass-card p {
    color: #8892b0;
}

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 0.4rem 0.8rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 500;
}

.status-connected {
    background: rgba(0, 184, 148, 0.15);
    color: #00b894;
    border: 1px solid rgba(0, 184, 148, 0.3);
}

.status-disconnected {
    background: rgba(90, 106, 138, 0.15);
    color: #5a6a8a;
    border: 1px solid rgba(90, 106, 138, 0.3);
}

.metric-card {
    background: linear-gradient(135deg, rgba(0, 212, 170, 0.08) 0%, rgba(102, 126, 234, 0.08) 100%);
    border: 1px solid rgba(0, 212, 170, 0.2);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}

.metric-value {
    font-size: 2.5rem;
    font-weight: 700;
    color: #00d4aa;
    line-height: 1;
}

.metric-label {
    font-size: 0.8rem;
    color: #5a6a8a;
    margin-top: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.chat-message {
    padding: 1.25rem;
    border-radius: 12px;
    margin-bottom: 1rem;
}

.user-message {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.1) 100%);
    border-left: 3px solid #667eea;
}

.assistant-message {
    background: rgba(18, 18, 26, 0.8);
    border-left: 3px solid #00d4aa;
}

.tool-badge {
    display: inline-block;
    padding: 0.2rem 0.5rem;
    background: rgba(102, 126, 234, 0.2);
    border-radius: 4px;
    font-size: 0.75rem;
    color: #667eea;
    margin-right: 0.5rem;
}

/* Workflow Visualization */
.workflow-container {
    background: rgba(18, 18, 26, 0.8);
    border: 1px solid rgba(0, 212, 170, 0.3);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
}

.workflow-step {
    display: flex;
    align-items: flex-start;
    padding: 0.75rem;
    margin: 0.5rem 0;
    border-left: 3px solid #1e1e2e;
    background: rgba(30, 30, 46, 0.5);
    border-radius: 4px;
    transition: all 0.3s ease;
}

.workflow-step.running {
    border-left-color: #fdcb6e;
    background: rgba(253, 203, 110, 0.1);
}

.workflow-step.completed {
    border-left-color: #00b894;
    background: rgba(0, 184, 148, 0.1);
}

.workflow-step.error {
    border-left-color: #e17055;
    background: rgba(225, 112, 85, 0.1);
}

.workflow-step-icon {
    font-size: 1.2rem;
    margin-right: 0.75rem;
    min-width: 24px;
}

.workflow-step-content {
    flex: 1;
}

.workflow-step-title {
    font-weight: 600;
    color: #ffffff;
    margin-bottom: 0.25rem;
    font-size: 0.9rem;
}

.workflow-step-message {
    color: #8892b0;
    font-size: 0.85rem;
}

.workflow-node {
    font-weight: 700;
    color: #00d4aa;
    text-transform: uppercase;
    font-size: 0.75rem;
    letter-spacing: 0.05em;
}

.workflow-arrow {
    text-align: center;
    color: #5a6a8a;
    font-size: 1.5rem;
    margin: 0.25rem 0;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d14 0%, #0a0a0f 100%);
    border-right: 1px solid #1e1e2e;
}

.stButton > button {
    background: linear-gradient(135deg, #00d4aa 0%, #00b894 100%);
    color: #0a0a0f;
    border: none;
    border-radius: 8px;
    padding: 0.75rem 2rem;
    font-weight: 600;
    font-family: 'Space Grotesk', sans-serif;
    box-shadow: 0 4px 20px rgba(0, 212, 170, 0.3);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 30px rgba(0, 212, 170, 0.4);
}

.stFormSubmitButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}

.stTextInput > div > div > input {
    background: rgba(18, 18, 26, 0.8);
    border: 1px solid #1e1e2e;
    border-radius: 8px;
    color: #ffffff;
}

[data-testid="stFileUploader"] {
    background: rgba(18, 18, 26, 0.5);
    border: 2px dashed #1e1e2e;
    border-radius: 12px;
    padding: 2rem;
}

[data-testid="stFileUploader"]:hover {
    border-color: #00d4aa;
}

.schema-box {
    background: rgba(10, 10, 15, 0.9);
    border: 1px solid #1e1e2e;
    border-radius: 8px;
    padding: 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #8892b0;
    max-height: 300px;
    overflow-y: auto;
    white-space: pre-wrap;
}

.footer-credit {
    text-align: center;
    padding: 2rem 0;
    color: #3a4a6a;
    font-size: 0.8rem;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
    """, unsafe_allow_html=True)


def render_status_badge(connected: bool, label: str) -> str:
    status_class = "status-connected" if connected else "status-disconnected"
    icon = "●" if connected else "○"
    return f'<span class="status-badge {status_class}">{icon} {label}</span>'


def render_metric_card(value: str, label: str) -> str:
    return f'''
<div class="metric-card">
    <div class="metric-value">{value}</div>
    <div class="metric-label">{label}</div>
</div>
    '''


def display_workflow(workflow_steps: list[dict]) -> None:
    """Display workflow steps in a visual format."""
    if not workflow_steps:
        st.info("No workflow steps recorded.")
        return
    
    # Group steps by node
    nodes = {}
    tools = []
    
    for step in workflow_steps:
        node_name = step.get("node", "unknown")
        if node_name == "tool":
            tools.append(step)
        else:
            if node_name not in nodes:
                nodes[node_name] = []
            nodes[node_name].append(step)
    
    # Display workflow
    html_parts = ['<div class="workflow-container">']
    
    # Main nodes
    node_order = ["router", "call_tools", "llm_answer", "direct_answer"]
    
    for node_name in node_order:
        if node_name in nodes:
            for step in nodes[node_name]:
                status = step.get("status", "unknown")
                message = step.get("message", "")
                node_display = step.get("node", "").replace("_", " ").title()
                
                status_class = status
                icon = "⏳" if status == "running" else "✅" if status == "completed" else "❌"
                
                html_parts.append(f'''
<div class="workflow-step {status_class}">
    <div class="workflow-step-icon">{icon}</div>
    <div class="workflow-step-content">
        <div class="workflow-step-title">
            <span class="workflow-node">{node_display}</span>
        </div>
        <div class="workflow-step-message">{message}</div>
    </div>
</div>
                ''')
            
            # Show tools under call_tools
            if node_name == "call_tools" and tools:
                html_parts.append('<div style="margin-left: 2rem; margin-top: 0.5rem;">')
                for tool_step in tools:
                    tool_name = tool_step.get("tool_name", "unknown").replace("_", " ").title()
                    tool_status = tool_step.get("status", "unknown")
                    tool_message = tool_step.get("message", "")
                    
                    tool_status_class = tool_status
                    tool_icon = "⏳" if tool_status == "running" else "✅" if tool_status == "completed" else "❌"
                    
                    html_parts.append(f'''
<div class="workflow-step {tool_status_class}" style="margin-left: 1rem;">
    <div class="workflow-step-icon">{tool_icon}</div>
    <div class="workflow-step-content">
        <div class="workflow-step-title" style="font-size: 0.85rem;">
            🔧 {tool_name}
        </div>
        <div class="workflow-step-message" style="font-size: 0.8rem;">{tool_message}</div>
    </div>
</div>
                    ''')
                html_parts.append('</div>')
            
            # Add arrow between nodes
            if node_name != node_order[-1]:
                html_parts.append('<div class="workflow-arrow">↓</div>')
    
    html_parts.append('</div>')
    
    st.markdown("".join(html_parts), unsafe_allow_html=True)
    
    # Summary
    completed_steps = [s for s in workflow_steps if s.get("status") == "completed"]
    error_steps = [s for s in workflow_steps if s.get("status") == "error"]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Steps", len(workflow_steps))
    with col2:
        st.metric("Completed", len(completed_steps), delta=f"{len(completed_steps)/len(workflow_steps)*100:.0f}%")
    with col3:
        if error_steps:
            st.metric("Errors", len(error_steps), delta_color="inverse")
        else:
            st.metric("Errors", 0)


# ============================================
# API Helpers
# ============================================

def api_ask(question: str) -> dict:
    """Call the /ask endpoint."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/ask",
            json={"question": question},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"answer": "API not available. Please start the FastAPI backend.", "error": "connection_error"}
    except Exception as e:
        return {"answer": f"Error: {str(e)}", "error": str(e)}


def api_graph_info() -> dict:
    """Call the /graph-info endpoint."""
    try:
        response = requests.get(f"{API_BASE_URL}/graph-info", timeout=10)
        response.raise_for_status()
        return response.json()
    except:
        return None


def api_health() -> bool:
    """Check API health."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


# ============================================
# Main App
# ============================================

def main():
    st.set_page_config(
        layout="wide",
        page_title=f"{APP_NAME}",
        page_icon="🔮",
        initial_sidebar_state="expanded"
    )
    
    inject_custom_css()
    
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "api_connected" not in st.session_state:
        st.session_state["api_connected"] = False
    
    # ============================================
    # Sidebar
    # ============================================
    with st.sidebar:
        # Logo
        logo_path = project_root / "logo.png"
        if logo_path.exists():
            st.image(str(logo_path), use_container_width=True)
        else:
            st.markdown(f"### 🔮 {APP_NAME}")
        
        st.markdown("---")
        
        # Connection Status
        st.markdown("### 📡 Status")
        
        api_connected = "OPENAI_API_KEY" in st.session_state
        neo4j_connected = st.session_state.get("neo4j_connected", False)
        backend_available = api_health() if USE_API else True
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(render_status_badge(api_connected, "API"), unsafe_allow_html=True)
        with col2:
            st.markdown(render_status_badge(neo4j_connected, "Neo4j"), unsafe_allow_html=True)
        
        if USE_API:
            st.markdown(render_status_badge(backend_available, "Backend"), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # API Key Setup
        if not api_connected:
            st.markdown("### 🔑 API Key")
            api_key = st.text_input(
                "OpenRouter Key",
                type="password",
                value=os.getenv("OPENAI_API_KEY", ""),
                placeholder="sk-or-...",
                label_visibility="collapsed"
            )
            
            if st.button("Connect API", use_container_width=True):
                if api_key:
                    with st.spinner("Connecting..."):
                        st.session_state["OPENAI_API_KEY"] = api_key
                        os.environ["OPENAI_API_KEY"] = api_key
                        
                        try:
                            # Initialize LLM and embeddings
                            llm = get_llm(api_key=api_key)
                            embeddings = get_embeddings(api_key=api_key)
                            st.session_state["llm"] = llm
                            st.session_state["embeddings"] = embeddings
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to connect: {e}")
        else:
            st.markdown("### 🔑 API")
            st.success("Connected", icon="✅")
        
        st.markdown("---")
        
        # Neo4j Connection
        if not neo4j_connected:
            st.markdown("### 🗄️ Neo4j")
            neo4j_url = st.text_input("URL", value=os.getenv("NEO4J_URL", "bolt://localhost:7687"))
            neo4j_user = st.text_input("User", value=os.getenv("NEO4J_USERNAME", "neo4j"))
            neo4j_pass = st.text_input("Pass", type="password", value=os.getenv("NEO4J_PASSWORD", ""))
            
            if st.button("Connect DB", use_container_width=True, disabled=not api_connected):
                try:
                    with st.spinner("Connecting..."):
                        # Reset and reconnect
                        reset_graph_instance()
                        graph = get_graph(url=neo4j_url, username=neo4j_user, password=neo4j_pass)
                        
                        st.session_state["neo4j_connected"] = True
                        st.session_state["neo4j_url"] = neo4j_url
                        st.session_state["neo4j_user"] = neo4j_user
                        st.session_state["neo4j_pass"] = neo4j_pass
                        
                        # Configure for later use
                        configure_from_credentials(
                            api_key=st.session_state["OPENAI_API_KEY"],
                            neo4j_url=neo4j_url,
                            neo4j_username=neo4j_user,
                            neo4j_password=neo4j_pass,
                        )
                        
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")
        else:
            st.markdown("### 🗄️ Neo4j")
            st.success("Connected", icon="✅")
        
        st.markdown("---")
        
        # About
        with st.expander("ℹ️ About"):
            st.markdown(f"""
**{APP_NAME}** - Agentic AI for Knowledge Graphs

**Stack:**
- 🧠 GPT-4o Mini (OpenRouter)
- 🕸️ Neo4j Graph Database
- 🔗 LangGraph Agent
- ⚡ FastAPI Backend

**Created by:** {AUTHOR}
            """)
        
        # Mode indicator
        mode = "API Mode" if USE_API else "Direct Mode"
        st.caption(f"Running in {mode}")
        
        # Reset
        if st.button("🔄 Reset", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # ============================================
    # Main Content
    # ============================================
    
    # Hero
    st.markdown(f'''
<div class="hero-container">
    <h1 class="hero-title">{APP_NAME}</h1>
    <p class="hero-subtitle">Agentic AI for Intelligent Knowledge Graphs</p>
    <p class="hero-author">Created by {AUTHOR}</p>
</div>
    ''', unsafe_allow_html=True)
    
    # Check prerequisites
    if "OPENAI_API_KEY" not in st.session_state:
        st.markdown(f'''
<div class="glass-card">
    <h3>👋 Welcome to {APP_NAME}</h3>
    <p>Enter your <strong>OpenRouter API key</strong> in the sidebar to get started.</p>
    <p style="color: #5a6a8a; font-size: 0.9rem;">
        Get your key at <a href="https://openrouter.ai" target="_blank" style="color: #00d4aa;">openrouter.ai</a>
    </p>
</div>
        ''', unsafe_allow_html=True)
        return
    
    if not st.session_state.get("neo4j_connected"):
        st.markdown('''
<div class="glass-card">
    <h3>🗄️ Connect Database</h3>
    <p>Enter your Neo4j credentials in the sidebar.</p>
    <p style="color: #5a6a8a;">Local default: <code style="color: #00d4aa;">bolt://localhost:7687</code></p>
</div>
        ''', unsafe_allow_html=True)
        return
    
    # ============================================
    # PDF Upload Section
    # ============================================
    if "graph_ready" not in st.session_state:
        st.markdown('''
<div class="glass-card">
    <h3>📄 Upload Document</h3>
    <p>Upload a PDF to extract entities and build your knowledge graph.</p>
</div>
        ''', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Upload PDF",
            type="pdf",
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            st.markdown(f"**Selected:** `{uploaded_file.name}` ({uploaded_file.size / 1024:.1f} KB)")
            
            if st.button("🚀 Process Document", use_container_width=True):
                progress = st.progress(0)
                status = st.empty()
                
                # Step 1: Read file
                status.info("📁 Reading file...")
                progress.progress(10)
                file_bytes = uploaded_file.read()
                
                # Step 2: Process PDF
                status.info("📖 Processing PDF...")
                progress.progress(25)
                docs = process_uploaded_pdf(file_bytes, uploaded_file.name)
                
                # Step 3: Clear graph
                status.info("🗑️ Preparing graph...")
                progress.progress(35)
                clear_graph()
                
                # Step 4: Extract entities
                status.info("🧠 Extracting entities with AI...")
                progress.progress(50)
                llm = st.session_state["llm"]
                nodes_created, rels_created = ingest_documents(docs, llm)
                
                # Step 5: Build vector index
                status.info("🔍 Building vector index...")
                progress.progress(80)
                try:
                    embeddings = st.session_state["embeddings"]
                    vector_store = build_vector_index(
                        embeddings=embeddings,
                        url=st.session_state["neo4j_url"],
                        username=st.session_state["neo4j_user"],
                        password=st.session_state["neo4j_pass"],
                    )
                    set_vector_store(vector_store)
                except Exception as e:
                    st.warning(f"Vector index creation failed: {e}")
                
                # Complete
                progress.progress(100)
                status.success("✅ Graph ready!")
                
                # Store stats
                st.session_state["graph_ready"] = True
                st.session_state["total_nodes"] = nodes_created
                st.session_state["total_rels"] = rels_created
                st.session_state["doc_name"] = uploaded_file.name
                st.session_state["doc_chunks"] = len(docs)
                st.session_state["graph_schema"] = get_schema()
                
                st.rerun()
    
    # ============================================
    # Query Interface
    # ============================================
    else:
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(render_metric_card(
                str(st.session_state.get("total_nodes", 0)),
                "Nodes"
            ), unsafe_allow_html=True)
        
        with col2:
            st.markdown(render_metric_card(
                str(st.session_state.get("total_rels", 0)),
                "Relations"
            ), unsafe_allow_html=True)
        
        with col3:
            st.markdown(render_metric_card(
                str(st.session_state.get("doc_chunks", 0)),
                "Chunks"
            ), unsafe_allow_html=True)
        
        with col4:
            st.markdown(render_metric_card(
                "●",
                "Ready"
            ), unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Document info
        doc_name = st.session_state.get("doc_name", "Unknown")
        st.markdown(f"**📄 Active Document:** `{doc_name}`")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Chat Interface
        st.markdown("### 💬 Query Your Knowledge Graph")
        
        # Display chat history
        for i, msg in enumerate(st.session_state["chat_history"]):
            if msg["role"] == "user":
                st.markdown(f'''
<div class="chat-message user-message">
    <strong style="color: #667eea;">You</strong><br>
    <span style="color: #ffffff;">{msg["content"]}</span>
</div>
                ''', unsafe_allow_html=True)
            else:
                tools_html = ""
                if msg.get("tools"):
                    tools_html = "<br>" + "".join([f'<span class="tool-badge">{t}</span>' for t in msg["tools"]])
                st.markdown(f'''
<div class="chat-message assistant-message">
    <strong style="color: #00d4aa;">{APP_NAME}</strong>{tools_html}<br>
    <span style="color: #ffffff;">{msg["content"]}</span>
</div>
                ''', unsafe_allow_html=True)
                
                # Display workflow for this message
                workflow_steps = msg.get("workflow_steps", [])
                with st.expander(f"🔄 View Backend Workflow", expanded=False):
                    if workflow_steps:
                        display_workflow(workflow_steps)
                    else:
                        st.info("No workflow data recorded for this response. (This message may have been created before workflow tracking was added.)")
        
        # Query Input
        with st.form("query_form", clear_on_submit=True):
            query = st.text_input(
                "Ask",
                placeholder="Ask anything about your document...",
                label_visibility="collapsed"
            )
            submitted = st.form_submit_button("Ask →", use_container_width=True)
        
        if submitted and query:
            st.session_state["chat_history"].append({"role": "user", "content": query})
            
            with st.spinner("🔄 Processing... (Agent is working)"):
                if USE_API:
                    # Use API
                    result = api_ask(query)
                    answer = result.get("answer", "No answer")
                    tools = result.get("used_tools", [])
                    workflow_steps = result.get("workflow_steps", [])
                else:
                    # Use local agent
                    from app.agent.graph_agent import run_agent
                    result = run_agent(query)
                    answer = result.get("answer", "No answer")
                    tools = result.get("used_tools", [])
                    workflow_steps = result.get("workflow_steps", [])
                
                st.session_state["chat_history"].append({
                    "role": "assistant",
                    "content": answer,
                    "tools": tools,
                    "workflow_steps": workflow_steps,
                })
            
            st.rerun()
        
        # Schema expander
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("🔍 Graph Schema"):
            schema = st.session_state.get("graph_schema", "No schema")
            st.markdown(f'<div class="schema-box">{schema}</div>', unsafe_allow_html=True)
        
        # Clear chat
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.session_state["chat_history"]:
                if st.button("🗑️ Clear Chat"):
                    st.session_state["chat_history"] = []
                    st.rerun()
        
        with col1:
            if st.button("📄 Upload New Document"):
                del st.session_state["graph_ready"]
                st.session_state["chat_history"] = []
                st.rerun()
    
    # Footer
    st.markdown(f'''
<div class="footer-credit">
    Built with ❤️ by <strong>{AUTHOR}</strong> • {APP_NAME}
</div>
    ''', unsafe_allow_html=True)


if __name__ == "__main__":
    main()

