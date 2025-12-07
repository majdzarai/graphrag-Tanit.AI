import os
import tempfile

from dotenv import load_dotenv
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Neo4jVector
from langchain_community.graphs import Neo4jGraph
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.chains.graph_qa.cypher import GraphCypherQAChain


# ---------------------------------------
# Load .env
# ---------------------------------------
load_dotenv()

DEFAULT_NEO4J_URL = os.getenv("NEO4J_URL", "bolt://localhost:7687")
DEFAULT_NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
DEFAULT_NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

DEFAULT_OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

# App Info
APP_NAME = "GraphRAG "
AUTHOR = "Majd Zarai"


# ---------------------------------------
# Custom CSS for Professional UI
# ---------------------------------------
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
    --bg-card-hover: #1a1a25;
    --text-primary: #ffffff;
    --text-secondary: #8892b0;
    --border: #1e1e2e;
    --glow: rgba(0, 212, 170, 0.3);
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

/* Hero Section */
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
    font-weight: 400;
}

.hero-author {
    font-size: 0.85rem;
    color: #5a6a8a;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* Progress Steps - Simplified */
.steps-container {
    display: flex;
    justify-content: center;
    gap: 1rem;
    margin: 2rem 0;
    flex-wrap: wrap;
}

.step-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.6rem 1.2rem;
    background: rgba(18, 18, 26, 0.8);
    border: 1px solid #1e1e2e;
    border-radius: 30px;
    font-size: 0.85rem;
    color: #5a6a8a;
    transition: all 0.3s ease;
}

.step-item.active {
    background: linear-gradient(135deg, rgba(0, 212, 170, 0.15) 0%, rgba(102, 126, 234, 0.15) 100%);
    border-color: #00d4aa;
    color: #00d4aa;
    box-shadow: 0 0 20px rgba(0, 212, 170, 0.2);
}

.step-item.completed {
    background: rgba(0, 184, 148, 0.1);
    border-color: #00b894;
    color: #00b894;
}

.step-num {
    width: 24px;
    height: 24px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    font-size: 0.75rem;
    background: #1e1e2e;
}

.step-item.active .step-num {
    background: #00d4aa;
    color: #0a0a0f;
}

.step-item.completed .step-num {
    background: #00b894;
    color: #0a0a0f;
}

/* Glass Cards */
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
    font-weight: 600;
}

.glass-card p {
    color: #8892b0;
    margin-bottom: 0.5rem;
}

/* Status Badges */
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

/* Metric Cards */
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

/* Chat Messages */
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

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d14 0%, #0a0a0f 100%);
    border-right: 1px solid #1e1e2e;
}

[data-testid="stSidebar"] .stMarkdown h3 {
    color: #ffffff !important;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #00d4aa 0%, #00b894 100%);
    color: #0a0a0f;
    border: none;
    border-radius: 8px;
    padding: 0.75rem 2rem;
    font-weight: 600;
    font-family: 'Space Grotesk', sans-serif;
    transition: all 0.3s ease;
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

/* Text Input */
.stTextInput > div > div > input {
    background: rgba(18, 18, 26, 0.8);
    border: 1px solid #1e1e2e;
    border-radius: 8px;
    color: #ffffff;
    font-family: 'Space Grotesk', sans-serif;
}

.stTextInput > div > div > input:focus {
    border-color: #00d4aa;
    box-shadow: 0 0 0 2px rgba(0, 212, 170, 0.2);
}

/* File Uploader */
[data-testid="stFileUploader"] {
    background: rgba(18, 18, 26, 0.5);
    border: 2px dashed #1e1e2e;
    border-radius: 12px;
    padding: 2rem;
}

[data-testid="stFileUploader"]:hover {
    border-color: #00d4aa;
    background: rgba(0, 212, 170, 0.03);
}

/* Expander */
.streamlit-expanderHeader {
    background: rgba(18, 18, 26, 0.6);
    border-radius: 8px;
    color: #8892b0 !important;
}

/* Schema Box */
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

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Info/Success/Warning boxes */
.stAlert {
    border-radius: 8px;
}

/* Document Badge */
.doc-badge {
    background: rgba(18, 18, 26, 0.8);
    border: 1px solid #1e1e2e;
    border-radius: 8px;
    padding: 0.75rem 1.25rem;
    display: inline-flex;
    align-items: center;
    gap: 0.75rem;
}

.doc-badge-icon {
    font-size: 1.2rem;
}

.doc-badge-name {
    color: #00d4aa;
    font-weight: 500;
}

/* Footer Credit */
.footer-credit {
    text-align: center;
    padding: 2rem 0;
    color: #3a4a6a;
    font-size: 0.8rem;
}

.footer-credit a {
    color: #00d4aa;
    text-decoration: none;
}
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


def render_steps(current_step: int):
    steps = [
        ("1", "API Key"),
        ("2", "Database"),
        ("3", "Upload"),
        ("4", "Query")
    ]
    
    html = '<div class="steps-container">'
    for i, (num, label) in enumerate(steps):
        if i < current_step:
            cls = "completed"
            icon = "✓"
        elif i == current_step:
            cls = "active"
            icon = num
        else:
            cls = ""
            icon = num
        
        html += f'''<div class="step-item {cls}">
            <span class="step-num">{icon}</span>
            <span>{label}</span>
        </div>'''
    
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def main():
    st.set_page_config(
        layout="wide",
        page_title=f"{APP_NAME}",
        page_icon="🔮",
        initial_sidebar_state="expanded"
    )
    
    inject_custom_css()
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    
    # -------------------------
    # Sidebar
    # -------------------------
    with st.sidebar:
        # Logo
        try:
            st.image("logo.png", use_container_width=True)
        except:
            st.markdown(f"### 🔮 {APP_NAME}")
        
        st.markdown("---")
        
        # Connection Status
        st.markdown("### 📡 Status")
        
        api_connected = "OPENAI_API_KEY" in st.session_state
        neo4j_connected = st.session_state.get("neo4j_connected", False)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(render_status_badge(api_connected, "API"), unsafe_allow_html=True)
        with col2:
            st.markdown(render_status_badge(neo4j_connected, "Neo4j"), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # API Key Setup
        if not api_connected:
            st.markdown("### 🔑 API Key")
            api_key = st.text_input(
                "OpenRouter Key",
                type="password",
                value=DEFAULT_OPENAI_KEY,
                placeholder="sk-or-...",
                label_visibility="collapsed"
            )
            
            if st.button("Connect API", use_container_width=True):
                if api_key:
                    with st.spinner("Connecting..."):
                        st.session_state["OPENAI_API_KEY"] = api_key
                        os.environ["OPENAI_API_KEY"] = api_key
                        os.environ["OPENAI_BASE_URL"] = DEFAULT_BASE_URL
                        
                        embeddings = OpenAIEmbeddings(
                            model="text-embedding-3-small",
                            api_key=api_key,
                            base_url=DEFAULT_BASE_URL,
                        )
                        
                        llm = ChatOpenAI(
                            model="openai/gpt-4o-mini",
                            api_key=api_key,
                            base_url=DEFAULT_BASE_URL,
                            temperature=0,
                        )
                        
                        st.session_state["embeddings"] = embeddings
                        st.session_state["llm"] = llm
                    st.rerun()
        else:
            st.markdown("### 🔑 API")
            st.success("Connected", icon="✅")
        
        st.markdown("---")
        
        # Neo4j Connection
        if not neo4j_connected:
            st.markdown("### 🗄️ Neo4j")
            neo4j_url = st.text_input("URL", value=DEFAULT_NEO4J_URL)
            neo4j_user = st.text_input("User", value=DEFAULT_NEO4J_USERNAME)
            neo4j_pass = st.text_input("Pass", type="password", value=DEFAULT_NEO4J_PASSWORD)
            
            if st.button("Connect DB", use_container_width=True, disabled=not api_connected):
                try:
                    with st.spinner("Connecting..."):
                        graph = Neo4jGraph(
                            url=neo4j_url,
                            username=neo4j_user,
                            password=neo4j_pass,
                        )
                        
                        st.session_state["neo4j_connected"] = True
                        st.session_state["neo4j_graph"] = graph
                        st.session_state["neo4j_url"] = neo4j_url
                        st.session_state["neo4j_user"] = neo4j_user
                        st.session_state["neo4j_pass"] = neo4j_pass
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
**{APP_NAME}** transforms documents into queryable knowledge graphs.

**Stack:**
- 🧠 GPT-4o Mini
- 🕸️ Neo4j Graph DB
- 🔗 LangChain

**Created by:** {AUTHOR}
            """)
        
        # Reset
        if st.button("🔄 Reset", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # -------------------------
    # Main Content
    # -------------------------
    
    # Hero
    st.markdown(f'''
<div class="hero-container">
    <h1 class="hero-title">{APP_NAME}</h1>
    <p class="hero-subtitle">Transform documents into intelligent knowledge graphs</p>
    <p class="hero-author">Created by {AUTHOR}</p>
</div>
    ''', unsafe_allow_html=True)
    
    # Calculate step
    current_step = 0
    if "OPENAI_API_KEY" in st.session_state:
        current_step = 1
    if st.session_state.get("neo4j_connected"):
        current_step = 2
    if "qa" in st.session_state:
        current_step = 3
    
    render_steps(current_step)
    
    # Step-based content
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
    <p style="color: #5a6a8a; font-size: 0.9rem;">
        Local default: <code style="color: #00d4aa;">bolt://localhost:7687</code>
    </p>
</div>
        ''', unsafe_allow_html=True)
        return
    
    embeddings = st.session_state.get("embeddings")
    llm = st.session_state.get("llm")
    graph: Neo4jGraph = st.session_state["neo4j_graph"]
    
    # -------------------------
    # PDF Upload
    # -------------------------
    if "qa" not in st.session_state:
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
            st.markdown(f'''
<div class="doc-badge">
    <span class="doc-badge-icon">📎</span>
    <span class="doc-badge-name">{uploaded_file.name}</span>
    <span style="color: #5a6a8a;">({uploaded_file.size / 1024:.1f} KB)</span>
</div>
            ''', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("🚀 Process Document", use_container_width=True):
                progress = st.progress(0)
                status = st.empty()
                
                # Step 1
                status.info("📁 Saving file...")
                progress.progress(10)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    pdf_path = tmp.name
                
                # Step 2
                status.info("📖 Reading PDF...")
                progress.progress(25)
                
                loader = PyPDFLoader(pdf_path)
                pages = loader.load_and_split()
                
                splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
                docs = splitter.split_documents(pages)
                
                cleaned_docs = [
                    Document(
                        page_content=doc.page_content.replace("\n", " "),
                        metadata={"source": uploaded_file.name},
                    )
                    for doc in docs
                ]
                
                # Step 3
                status.info("🧠 Extracting entities with AI...")
                progress.progress(45)
                
                graph.query("MATCH (n) DETACH DELETE n;")
                
                allowed_nodes = ["Patient", "Disease", "Medication", "Test", "Symptom", "Doctor"]
                allowed_rels = ["HAS_DISEASE", "TAKES_MEDICATION", "UNDERWENT_TEST", "HAS_SYMPTOM", "TREATED_BY"]
                
                transformer = LLMGraphTransformer(
                    llm=llm,
                    allowed_nodes=allowed_nodes,
                    allowed_relationships=allowed_rels,
                    node_properties=True,
                    relationship_properties=True,
                )
                
                graph_docs = transformer.convert_to_graph_documents(cleaned_docs)
                
                total_nodes = sum(len(doc.nodes) for doc in graph_docs)
                total_rels = sum(len(doc.relationships) for doc in graph_docs)
                
                # Step 4
                status.info("🕸️ Building graph...")
                progress.progress(70)
                
                graph.add_graph_documents(graph_docs, include_source=True)
                graph.refresh_schema()
                
                # Step 5
                status.info("🔍 Creating index...")
                progress.progress(85)
                
                try:
                    Neo4jVector.from_existing_graph(
                        embedding=embeddings,
                        url=st.session_state["neo4j_url"],
                        username=st.session_state["neo4j_user"],
                        password=st.session_state["neo4j_pass"],
                        database="neo4j",
                        node_label="Patient",
                        text_node_properties=["id", "text"],
                        embedding_node_property="embedding",
                        index_name="vector_index",
                        keyword_index_name="entity_index",
                        search_type="hybrid",
                    )
                except:
                    pass
                
                progress.progress(100)
                status.success("✅ Complete!")
                
                schema = graph.get_schema
                
                prompt = PromptTemplate(
                    template="""
Generate ONLY a Cypher query using the schema below.

Schema:
{schema}

User question:
{question}

Return ONLY the Cypher query, no explanations.
""",
                    input_variables=["schema", "question"],
                )
                
                qa = GraphCypherQAChain.from_llm(
                    llm=llm,
                    graph=graph,
                    cypher_prompt=prompt,
                    verbose=True,
                    allow_dangerous_requests=True,
                )
                
                st.session_state["qa"] = qa
                st.session_state["graph_schema"] = schema
                st.session_state["total_nodes"] = total_nodes
                st.session_state["total_rels"] = total_rels
                st.session_state["doc_name"] = uploaded_file.name
                st.session_state["doc_chunks"] = len(cleaned_docs)
                
                st.rerun()
    
    # -------------------------
    # Query Interface
    # -------------------------
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
        
        # Document badge
        doc_name = st.session_state.get("doc_name", "Unknown")
        st.markdown(f'''
<div class="doc-badge">
    <span class="doc-badge-icon">📄</span>
    <span style="color: #8892b0;">Active:</span>
    <span class="doc-badge-name">{doc_name}</span>
</div>
        ''', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Chat
        st.markdown("### 💬 Query Your Graph")
        
        for msg in st.session_state["chat_history"]:
            if msg["role"] == "user":
                st.markdown(f'''
<div class="chat-message user-message">
    <strong style="color: #667eea;">You</strong><br>
    <span style="color: #ffffff;">{msg["content"]}</span>
</div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
<div class="chat-message assistant-message">
    <strong style="color: #00d4aa;">{APP_NAME}</strong><br>
    <span style="color: #ffffff;">{msg["content"]}</span>
</div>
                ''', unsafe_allow_html=True)
        
        # Input
        with st.form("query_form", clear_on_submit=True):
            query = st.text_input(
                "Ask",
                placeholder="Ask anything about your document...",
                label_visibility="collapsed"
            )
            submitted = st.form_submit_button("Ask →", use_container_width=True)
        
        if submitted and query:
            st.session_state["chat_history"].append({"role": "user", "content": query})
            
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state["qa"].invoke({"query": query})
                    answer = result.get("result", "I couldn't find an answer.")
                except Exception as e:
                    answer = f"Error: {str(e)}"
                
                st.session_state["chat_history"].append({"role": "assistant", "content": answer})
            
            st.rerun()
        
        # Schema
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("🔍 Graph Schema"):
            schema = st.session_state.get("graph_schema", "No schema")
            st.markdown(f'<div class="schema-box">{schema}</div>', unsafe_allow_html=True)
        
        # Clear chat
        if st.session_state["chat_history"]:
            if st.button("🗑️ Clear Chat"):
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
