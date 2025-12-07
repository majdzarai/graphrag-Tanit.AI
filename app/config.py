"""
Configuration module for GraphRAG München.
Loads environment variables and provides singleton instances for LLM and embeddings.
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load environment variables from .env file
load_dotenv()


@dataclass
class Settings:
    """Application settings loaded from environment variables."""
    
    # OpenAI / OpenRouter settings
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    
    # Neo4j settings
    neo4j_url: str = os.getenv("NEO4J_URL", "bolt://localhost:7687")
    neo4j_username: str = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "")
    
    # Model settings
    llm_model: str = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0"))
    
    # App info
    app_name: str = "GraphRAG München"
    author: str = "Majd Zarai"
    
    # Graph schema (medical domain)
    allowed_nodes: tuple = (
        "Patient", "Disease", "Medication", "Test", "Symptom", "Doctor"
    )
    allowed_relationships: tuple = (
        "HAS_DISEASE", "TAKES_MEDICATION", "UNDERWENT_TEST", "HAS_SYMPTOM", "TREATED_BY"
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Singleton instances for LLM and embeddings
_llm_instance: Optional[ChatOpenAI] = None
_embeddings_instance: Optional[OpenAIEmbeddings] = None


def get_llm(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> ChatOpenAI:
    """
    Get LLM instance (singleton pattern).
    
    Args:
        api_key: OpenAI/OpenRouter API key. Uses env var if not provided.
        base_url: API base URL. Uses env var if not provided.
        model: Model name. Uses env var if not provided.
        temperature: LLM temperature. Uses env var if not provided.
    
    Returns:
        ChatOpenAI instance configured for the application.
    """
    global _llm_instance
    
    settings = get_settings()
    
    # Use provided values or fall back to settings
    _api_key = api_key or settings.openai_api_key
    _base_url = base_url or settings.openai_base_url
    _model = model or settings.llm_model
    _temperature = temperature if temperature is not None else settings.llm_temperature
    
    if _llm_instance is None:
        if not _api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY env var or pass api_key parameter.")
        
        _llm_instance = ChatOpenAI(
            model=_model,
            api_key=_api_key,
            base_url=_base_url,
            temperature=_temperature,
        )
    
    return _llm_instance


def get_embeddings(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> OpenAIEmbeddings:
    """
    Get embeddings instance (singleton pattern).
    
    Args:
        api_key: OpenAI/OpenRouter API key. Uses env var if not provided.
        base_url: API base URL. Uses env var if not provided.
        model: Embedding model name. Uses env var if not provided.
    
    Returns:
        OpenAIEmbeddings instance configured for the application.
    """
    global _embeddings_instance
    
    settings = get_settings()
    
    # Use provided values or fall back to settings
    _api_key = api_key or settings.openai_api_key
    _base_url = base_url or settings.openai_base_url
    _model = model or settings.embedding_model
    
    if _embeddings_instance is None:
        if not _api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY env var or pass api_key parameter.")
        
        _embeddings_instance = OpenAIEmbeddings(
            model=_model,
            api_key=_api_key,
            base_url=_base_url,
        )
    
    return _embeddings_instance


def reset_instances() -> None:
    """Reset singleton instances. Useful for testing or reconfiguration."""
    global _llm_instance, _embeddings_instance
    _llm_instance = None
    _embeddings_instance = None
    get_settings.cache_clear()


def configure_from_credentials(
    api_key: str,
    neo4j_url: str,
    neo4j_username: str,
    neo4j_password: str,
    base_url: Optional[str] = None,
) -> None:
    """
    Configure the application with provided credentials.
    Used by UI when user enters credentials manually.
    
    Args:
        api_key: OpenAI/OpenRouter API key.
        neo4j_url: Neo4j database URL.
        neo4j_username: Neo4j username.
        neo4j_password: Neo4j password.
        base_url: Optional API base URL.
    """
    # Reset existing instances
    reset_instances()
    
    # Update environment variables
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["NEO4J_URL"] = neo4j_url
    os.environ["NEO4J_USERNAME"] = neo4j_username
    os.environ["NEO4J_PASSWORD"] = neo4j_password
    
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url

