"""
LangGraph agent workflow for GraphRAG.

This module implements the main agent that orchestrates:
1. Routing decisions (should we use tools?)
2. Tool calling (vector search, cypher queries)
3. LLM response generation

The agent uses a state graph pattern where each node
transforms the state and edges determine the flow.
"""

import logging
from typing import Any, Literal, Optional

from langgraph.graph import END, StateGraph
from langchain_openai import ChatOpenAI

from app.agent.state import AgentState, create_initial_state
from app.config import get_llm, get_settings
from app.graph.graph_rag import (
    format_context_for_llm,
    get_vector_store,
    hybrid_retrieve,
)
from app.graph.neo4j_client import get_schema
from app.tools.graph_tools import tool_cypher_query, tool_graph_schema
from app.tools.vector_tools import tool_vector_search

logger = logging.getLogger(__name__)

# System prompt for the agent
SYSTEM_PROMPT = """You are an intelligent assistant that answers questions based on a medical knowledge graph.

You have access to:
1. A Neo4j knowledge graph containing medical entities (Patients, Diseases, Medications, Tests, Symptoms, Doctors)
2. Vector similarity search over document embeddings
3. Cypher queries for precise graph traversal

Guidelines:
- Use the provided context to answer questions accurately
- If the context doesn't contain relevant information, say "I don't have enough information to answer that"
- Be concise but thorough
- Cite specific entities from the graph when possible
- If asked about relationships, describe the connections between entities

Current Graph Schema:
{schema}
"""

ANSWER_PROMPT = """Based on the following context from the knowledge graph, answer the user's question.

Context:
{context}

Question: {question}

Instructions:
- Answer based ONLY on the provided context
- If the context doesn't contain relevant information, say so
- Be specific and cite entities when possible
- Keep your answer concise but complete

Answer:"""


def router_node(state: AgentState) -> AgentState:
    """
    Router node - decides whether to use tools or answer directly.
    
    Simple heuristic: Always use tools for knowledge graph questions.
    Could be enhanced with LLM-based routing.
    """
    question = state["question"].lower()
    
    # Track workflow step
    workflow_steps = state.get("workflow_steps", [])
    workflow_steps.append({
        "node": "router",
        "status": "running",
        "message": "Analyzing question and deciding routing strategy...",
        "timestamp": None,
    })
    state["workflow_steps"] = workflow_steps
    
    # Keywords that suggest we need graph/vector search
    tool_keywords = [
        "what", "who", "which", "how", "why", "when", "where",
        "patient", "disease", "medication", "symptom", "doctor", "test",
        "find", "list", "show", "tell", "explain", "describe",
        "relationship", "connected", "related",
    ]
    
    # Check if question likely needs tools
    needs_tools = any(kw in question for kw in tool_keywords)
    
    # Default to using tools for most questions
    state["should_use_tools"] = needs_tools or len(question) > 10
    
    # Update workflow step
    workflow_steps[-1]["status"] = "completed"
    workflow_steps[-1]["message"] = f"Decision: {'Use tools' if state['should_use_tools'] else 'Direct answer'}"
    workflow_steps[-1]["details"] = {"should_use_tools": state["should_use_tools"]}
    
    logger.info(f"Router decision: should_use_tools={state['should_use_tools']}")
    
    return state


def call_tools_node(state: AgentState) -> AgentState:
    """
    Call tools node - performs hybrid retrieval using vector search and graph queries.
    
    Aggregates results from multiple sources into the context.
    """
    question = state["question"]
    logger.info(f"Calling tools for question: {question[:50]}...")
    
    # Track workflow step
    workflow_steps = state.get("workflow_steps", [])
    workflow_steps.append({
        "node": "call_tools",
        "status": "running",
        "message": "Calling tools to retrieve context...",
        "tools": [],
    })
    state["workflow_steps"] = workflow_steps
    
    used_tools = []
    context_parts = []
    intermediate_steps = state.get("intermediate_steps", [])
    
    # 1. Vector search
    try:
        workflow_steps.append({
            "node": "tool",
            "tool_name": "vector_search",
            "status": "running",
            "message": "Searching vector embeddings for similar content...",
        })
        state["workflow_steps"] = workflow_steps
        
        vector_store = get_vector_store()
        if vector_store is not None:
            vector_result = tool_vector_search.invoke({"query": question, "k": 5})
            used_tools.append("vector_search")
            context_parts.append("=== Vector Search Results ===\n" + vector_result)
            intermediate_steps.append({
                "tool": "vector_search",
                "input": question,
                "output": vector_result[:500],
            })
            
            # Update workflow step
            workflow_steps[-1]["status"] = "completed"
            workflow_steps[-1]["message"] = f"✅ Vector search: Found {len(vector_result.split('['))-1} relevant results"
            workflow_steps[0]["tools"].append("vector_search")
            
            logger.info("Vector search completed")
    except Exception as e:
        logger.warning(f"Vector search failed: {e}")
        if workflow_steps:
            workflow_steps[-1]["status"] = "error"
            workflow_steps[-1]["message"] = f"❌ Vector search failed: {str(e)[:50]}"
    
    # 2. Hybrid retrieval (includes graph facts)
    try:
        workflow_steps.append({
            "node": "tool",
            "tool_name": "graph_traversal",
            "status": "running",
            "message": "Traversing knowledge graph to find relationships...",
        })
        state["workflow_steps"] = workflow_steps
        
        retrieval_results = hybrid_retrieve(question, k=5, expand_neighbors=True)
        state["retrieval_results"] = retrieval_results
        
        if retrieval_results.get("graph_facts"):
            used_tools.append("graph_traversal")
            facts_text = "\n".join(retrieval_results["graph_facts"][:15])
            context_parts.append("=== Knowledge Graph Facts ===\n" + facts_text)
            intermediate_steps.append({
                "tool": "graph_traversal",
                "input": "expand_neighbors",
                "output": facts_text[:500],
            })
            
            # Update workflow step
            num_facts = len(retrieval_results["graph_facts"])
            workflow_steps[-1]["status"] = "completed"
            workflow_steps[-1]["message"] = f"✅ Graph traversal: Found {num_facts} relationships"
            workflow_steps[0]["tools"].append("graph_traversal")
            
            logger.info("Graph traversal completed")
    except Exception as e:
        logger.warning(f"Hybrid retrieval failed: {e}")
        if workflow_steps:
            workflow_steps[-1]["status"] = "error"
            workflow_steps[-1]["message"] = f"❌ Graph traversal failed: {str(e)[:50]}"
    
    # 3. Get schema for context
    try:
        workflow_steps.append({
            "node": "tool",
            "tool_name": "schema_lookup",
            "status": "running",
            "message": "Retrieving graph schema...",
        })
        state["workflow_steps"] = workflow_steps
        
        schema = get_schema()
        if schema and len(schema) < 2000:
            context_parts.append("=== Graph Schema ===\n" + schema)
            used_tools.append("schema_lookup")
            
            # Update workflow step
            workflow_steps[-1]["status"] = "completed"
            workflow_steps[-1]["message"] = "✅ Schema retrieved"
            workflow_steps[0]["tools"].append("schema_lookup")
    except Exception as e:
        logger.warning(f"Schema lookup failed: {e}")
        if workflow_steps:
            workflow_steps[-1]["status"] = "error"
            workflow_steps[-1]["message"] = "❌ Schema lookup failed"
    
    # Aggregate context
    if context_parts:
        state["context"] = "\n\n".join(context_parts)
    else:
        state["context"] = "No relevant context found in the knowledge graph."
    
    state["used_tools"] = used_tools
    state["intermediate_steps"] = intermediate_steps
    
    # Update main workflow step
    workflow_steps[0]["status"] = "completed"
    workflow_steps[0]["message"] = f"✅ Tools completed: {len(used_tools)} tools used, {len(state['context'])} chars context"
    
    logger.info(f"Tools used: {used_tools}")
    logger.info(f"Context length: {len(state['context'])} chars")
    
    return state


def llm_answer_node(state: AgentState) -> AgentState:
    """
    LLM answer node - generates the final answer using context.
    """
    question = state["question"]
    context = state.get("context", "No context available.")
    
    # Track workflow step
    workflow_steps = state.get("workflow_steps", [])
    workflow_steps.append({
        "node": "llm_answer",
        "status": "running",
        "message": "Generating answer with LLM...",
    })
    state["workflow_steps"] = workflow_steps
    
    logger.info("Generating LLM answer...")
    
    try:
        llm = get_llm()
        
        # Build the prompt
        prompt = ANSWER_PROMPT.format(
            context=context[:6000],  # Limit context length
            question=question,
        )
        
        # Generate answer
        response = llm.invoke(prompt)
        answer = response.content
        
        state["answer"] = answer
        state["error"] = None
        
        # Update workflow step
        workflow_steps[-1]["status"] = "completed"
        workflow_steps[-1]["message"] = f"✅ Answer generated ({len(answer)} characters)"
        
        logger.info(f"Generated answer: {answer[:100]}...")
        
    except Exception as e:
        error_msg = f"Failed to generate answer: {str(e)}"
        logger.error(error_msg)
        state["answer"] = "I encountered an error while processing your question."
        state["error"] = error_msg
        
        # Update workflow step
        workflow_steps[-1]["status"] = "error"
        workflow_steps[-1]["message"] = f"❌ Error: {str(e)[:50]}"
    
    return state


def direct_answer_node(state: AgentState) -> AgentState:
    """
    Direct answer node - for simple questions that don't need tools.
    """
    question = state["question"]
    
    # Track workflow step
    workflow_steps = state.get("workflow_steps", [])
    workflow_steps.append({
        "node": "direct_answer",
        "status": "running",
        "message": "Generating direct answer (no tools needed)...",
    })
    state["workflow_steps"] = workflow_steps
    
    logger.info("Generating direct answer (no tools)...")
    
    try:
        llm = get_llm()
        
        prompt = f"""Answer this question directly and concisely:

Question: {question}

If this is a question about specific medical data or patient information,
say that you need to search the knowledge graph first.

Answer:"""
        
        response = llm.invoke(prompt)
        state["answer"] = response.content
        state["error"] = None
        state["used_tools"] = []
        
        # Update workflow step
        workflow_steps[-1]["status"] = "completed"
        workflow_steps[-1]["message"] = f"✅ Direct answer generated ({len(state['answer'])} characters)"
        
    except Exception as e:
        state["answer"] = "I encountered an error."
        state["error"] = str(e)
        
        # Update workflow step
        workflow_steps[-1]["status"] = "error"
        workflow_steps[-1]["message"] = f"❌ Error: {str(e)[:50]}"
    
    return state


def should_use_tools(state: AgentState) -> Literal["call_tools", "direct_answer"]:
    """Edge function - determines the next node based on router decision."""
    if state.get("should_use_tools", True):
        return "call_tools"
    return "direct_answer"


def create_agent_graph() -> StateGraph:
    """
    Create the LangGraph agent workflow.
    
    Graph structure:
    
        [START]
           │
           ▼
        [router]
           │
           ├─(should_use_tools=True)──► [call_tools] ──► [llm_answer] ──► [END]
           │
           └─(should_use_tools=False)─► [direct_answer] ──► [END]
    
    Returns:
        Compiled StateGraph ready for execution.
    """
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("call_tools", call_tools_node)
    workflow.add_node("llm_answer", llm_answer_node)
    workflow.add_node("direct_answer", direct_answer_node)
    
    # Set entry point
    workflow.set_entry_point("router")
    
    # Add conditional edges from router
    workflow.add_conditional_edges(
        "router",
        should_use_tools,
        {
            "call_tools": "call_tools",
            "direct_answer": "direct_answer",
        }
    )
    
    # Add edges to final answer nodes
    workflow.add_edge("call_tools", "llm_answer")
    workflow.add_edge("llm_answer", END)
    workflow.add_edge("direct_answer", END)
    
    # Compile the graph
    return workflow.compile()


# Cached agent instance
_agent_graph = None


def get_agent() -> StateGraph:
    """Get or create the agent graph instance."""
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = create_agent_graph()
    return _agent_graph


def run_agent(question: str) -> dict[str, Any]:
    """
    Run the agent to answer a question.
    
    This is the main entry point for the agent. It:
    1. Creates initial state from the question
    2. Runs the graph workflow
    3. Returns the final result
    
    Args:
        question: User's question to answer.
    
    Returns:
        Dictionary with:
        - answer: The generated answer
        - used_tools: List of tools that were called
        - context_snippets: Relevant context used (truncated)
        - error: Error message if any
    
    Example:
        >>> result = run_agent("What diseases does patient John have?")
        >>> print(result["answer"])
        >>> print(f"Used tools: {result['used_tools']}")
    """
    logger.info(f"Running agent for question: {question[:100]}...")
    
    # Create initial state
    initial_state = create_initial_state(question)
    
    # Get agent and run
    agent = get_agent()
    final_state = agent.invoke(initial_state)
    
    # Extract results
    answer = final_state.get("answer", "No answer generated.")
    used_tools = final_state.get("used_tools", [])
    context = final_state.get("context", "")
    error = final_state.get("error")
    
    # Create context snippets (truncated for API response)
    context_snippets = []
    if context:
        # Split context into snippets
        lines = context.split("\n")
        current_snippet = []
        for line in lines:
            current_snippet.append(line)
            if len("\n".join(current_snippet)) > 200:
                context_snippets.append("\n".join(current_snippet))
                current_snippet = []
        if current_snippet:
            context_snippets.append("\n".join(current_snippet))
    
    result = {
        "answer": answer,
        "used_tools": used_tools,
        "context_snippets": context_snippets[:5],  # Limit to 5 snippets
        "error": error,
        "workflow_steps": final_state.get("workflow_steps", []),  # Include workflow steps
    }
    
    logger.info(f"Agent completed. Tools used: {used_tools}")
    
    return result


def reset_agent() -> None:
    """Reset the agent instance."""
    global _agent_graph
    _agent_graph = None

