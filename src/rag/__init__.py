"""
RAG (Retrieval-Augmented Generation) Module
============================================

Este módulo implementa un sistema completo de RAG para el análisis de reseñas de Amazon.
Incluye generación de embeddings, almacenamiento vectorial, búsqueda semántica e integración con LLM.

Módulos principales:
- embeddings_generator: Generación de representaciones vectoriales
- vector_store: Almacenamiento y gestión de vectores
- retriever: Búsqueda semántica y recuperación de documentos
- llm_pipeline: Integración con modelos de lenguaje

Autor: Oscar David Hospinal R.
Curso: INF3590 - Big Data
Universidad: Pontificia Universidad Católica de Chile
"""

from .embeddings_generator import EmbeddingsGenerator
from .vector_store import VectorStore
from .retriever import SemanticRetriever
from .llm_pipeline import LLMPipeline

__version__ = "1.0.0"
__author__ = "Oscar David Hospinal R."

__all__ = [
    "EmbeddingsGenerator",
    "VectorStore",
    "SemanticRetriever",
    "LLMPipeline"
]

# Configuración por defecto para RAG
DEFAULT_CONFIG = {
    "embeddings": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384,
        "batch_size": 32
    },
    "vector_store": {
        "type": "chromadb",
        "collection_name": "amazon_reviews",
        "distance_metric": "cosine"
    },
    "retriever": {
        "top_k": 5,
        "similarity_threshold": 0.7
    },
    "llm": {
        "provider": "openai",
        "model": "gpt-3.5-turbo",
        "temperature": 0.3,
        "max_tokens": 500
    }
}


def get_default_config():
    """
    Retorna la configuración por defecto para el sistema RAG.

    Returns:
        dict: Configuración por defecto
    """
    return DEFAULT_CONFIG.copy()


def create_rag_pipeline(config=None):
    """
    Crea un pipeline RAG completo con la configuración especificada.

    Args:
        config (dict, optional): Configuración personalizada. Si no se especifica,
                                usa la configuración por defecto.

    Returns:
        dict: Diccionario con todos los componentes del pipeline RAG
    """
    if config is None:
        config = get_default_config()

    # Inicializar componentes
    embeddings_gen = EmbeddingsGenerator(
        model_name=config["embeddings"]["model_name"],
        batch_size=config["embeddings"]["batch_size"]
    )

    vector_store = VectorStore(
        store_type=config["vector_store"]["type"],
        collection_name=config["vector_store"]["collection_name"]
    )

    retriever = SemanticRetriever(
        vector_store=vector_store,
        top_k=config["retriever"]["top_k"],
        similarity_threshold=config["retriever"]["similarity_threshold"]
    )

    llm_pipeline = LLMPipeline(
        provider=config["llm"]["provider"],
        model=config["llm"]["model"],
        temperature=config["llm"]["temperature"],
        max_tokens=config["llm"]["max_tokens"]
    )

    return {
        "embeddings_generator": embeddings_gen,
        "vector_store": vector_store,
        "retriever": retriever,
        "llm_pipeline": llm_pipeline,
        "config": config
    }


# Información del módulo
def get_module_info():
    """
    Retorna información del módulo RAG.

    Returns:
        dict: Información del módulo
    """
    return {
        "name": "Amazon RAG System",
        "version": __version__,
        "author": __author__,
        "description": "Sistema RAG para análisis de reseñas de Amazon",
        "components": [
            "EmbeddingsGenerator - Generación de vectores",
            "VectorStore - Almacenamiento vectorial",
            "SemanticRetriever - Búsqueda semántica",
            "LLMPipeline - Integración con LLM"
        ]
    }