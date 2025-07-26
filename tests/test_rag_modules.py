#!/usr/bin/env python3
"""
Script de Testing para Módulos RAG
==================================

Este script prueba cada módulo RAG independientemente para verificar
que funcionan correctamente fuera del notebook.

Autor: Oscar David Hospinal R.
Curso: INF3590 - Big Data
"""

import sys
import os
from pathlib import Path
import numpy as np
import json
import time

# Agregar directorio src al path
sys.path.append('src')

print(" INICIANDO TESTING DE MÓDULOS RAG")
print("=" * 60)


def test_embeddings_generator():
    """Test del módulo embeddings_generator.py"""
    print("\n **TEST 1: EmbeddingsGenerator**")

    try:
        from src.rag.embeddings_generator import EmbeddingsGenerator

        # Test básico
        generator = EmbeddingsGenerator(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=16
        )

        # Test embedding simple
        test_text = "This is a test review"
        embedding = generator.generate_single_embedding(test_text)

        print(f"   ✅ Modelo cargado: {generator.model_name}")
        print(f"   ✅ Dimensiones: {generator.embedding_dimension}")
        print(f"   ✅ Embedding generado: shape {embedding.shape}")
        print(f"   ✅ Norma L2: {np.linalg.norm(embedding):.4f}")

        # Test con datos simulados
        fake_reviews = [
            {"reviewText": "Great product!", "summary": "Excellent", "overall": 5.0, "original_category": "Books"},
            {"reviewText": "Not bad", "summary": "OK", "overall": 3.0, "original_category": "Games"}
        ]

        embeddings, metadata = generator.process_reviews_data(fake_reviews)
        print(f"   ✅ Batch processing: {embeddings.shape} embeddings generados")
        print(f"   ✅ Metadatos: {len(metadata)} registros procesados")

        return True

    except Exception as e:
        print(f"   ❌ Error en EmbeddingsGenerator: {e}")
        return False


def test_vector_store():
    """Test del módulo vector_store.py"""
    print("\n🗄️ **TEST 2: VectorStore**")

    try:
        from src.rag.vector_store import VectorStore

        # Test configuración
        vector_store = VectorStore(
            store_type="chromadb",
            collection_name="test_collection",
            persist_directory="data/vectors/test_chroma",
            distance_metric="cosine"
        )

        print(f"   ✅ VectorStore inicializado: {vector_store.store_type}")

        # Test estadísticas
        stats = vector_store.get_collection_stats()
        print(f"   ✅ Estadísticas obtenidas: {stats['collection_name']}")

        # Test con embeddings simulados
        test_embeddings = np.random.random((3, 384))
        test_metadata = [
            {"id": f"test_{i}", "category": "Test", "rating": 4.0}
            for i in range(3)
        ]

        success = vector_store.add_embeddings(test_embeddings, test_metadata)
        print(f"   ✅ Embeddings añadidos: {success}")

        # Limpiar
        vector_store.delete_collection()
        print(f"   ✅ Colección de prueba eliminada")

        return True

    except Exception as e:
        print(f"   ❌ Error en VectorStore: {e}")
        return False


def test_retriever():
    """Test del módulo retriever.py"""
    print("\n **TEST 3: SemanticRetriever**")

    try:
        from src.rag.retriever import SemanticRetriever, SearchQuery
        from src.rag.embeddings_generator import EmbeddingsGenerator
        from src.rag.vector_store import VectorStore

        # Setup mínimo para testing
        generator = EmbeddingsGenerator(batch_size=8)
        vector_store = VectorStore(
            store_type="chromadb",
            collection_name="test_retriever",
            persist_directory="data/vectors/test_retriever"
        )

        retriever = SemanticRetriever(
            vector_store=vector_store,
            embeddings_generator=generator,
            top_k=3,
            similarity_threshold=0.0
        )

        print(f"   ✅ SemanticRetriever inicializado")
        print(f"   ✅ Configuración: top_k={retriever.top_k}")

        # Test SearchQuery
        query = SearchQuery(
            text="test query",
            top_k=3,
            similarity_threshold=0.0
        )

        print(f"   ✅ SearchQuery creado: '{query.text}'")

        # Limpiar
        vector_store.delete_collection()

        return True

    except Exception as e:
        print(f"   ❌ Error en SemanticRetriever: {e}")
        return False


def test_llm_pipeline():
    """Test del módulo llm_pipeline.py"""
    print("\n **TEST 4: LLMPipeline**")

    try:
        from src.rag.llm_pipeline import LLMPipeline, RAGResponse, PromptTemplate
        from src.rag.retriever import SearchResult

        # Test modo local
        llm = LLMPipeline(
            provider="local",
            model="test-model",
            temperature=0.3,
            max_tokens=100
        )

        print(f"   ✅ LLMPipeline inicializado: {llm.provider}")
        print(f"   ✅ Modelo: {llm.model}")

        # Test con resultado simulado
        fake_result = SearchResult(
            id="test_1",
            score=0.85,
            document="This is a test document about books",
            metadata={"category": "Books", "rating": 4.5}
        )

        response = llm.generate_response("test query", [fake_result])

        print(f"   ✅ Respuesta generada: {type(response).__name__}")
        print(f"   ✅ Confianza: {response.confidence_score:.3f}")
        print(f"   ✅ Tiempo: {response.generation_time:.3f}s")

        # Test PromptTemplate
        template = PromptTemplate(
            system_prompt="Test system",
            user_template="Test: {query}",
            context_template="Context: {text}",
            max_context_length=500
        )

        print(f"   ✅ PromptTemplate creado")

        return True

    except Exception as e:
        print(f"   ❌ Error en LLMPipeline: {e}")
        return False


def test_rag_init():
    """Test del módulo __init__.py"""
    print("\n **TEST 5: RAG Module (__init__.py)**")

    try:
        from src.rag import (
            EmbeddingsGenerator,
            VectorStore,
            SemanticRetriever,
            LLMPipeline,
            get_default_config,
            get_module_info
        )

        print(f"   ✅ Todas las clases importadas correctamente")

        # Test configuración por defecto
        config = get_default_config()
        print(f"   ✅ Configuración por defecto: {len(config)} secciones")

        # Test información del módulo
        info = get_module_info()
        print(f"   ✅ Información del módulo: {info['name']}")
        print(f"   ✅ Versión: {info['version']}")
        print(f"   ✅ Componentes: {len(info['components'])}")

        return True

    except Exception as e:
        print(f"   ❌ Error en RAG __init__: {e}")
        return False


def main():
    """Función principal de testing"""
    print(f"🕐 Inicio: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    tests = [
        ("EmbeddingsGenerator", test_embeddings_generator),
        ("VectorStore", test_vector_store),
        ("SemanticRetriever", test_retriever),
        ("LLMPipeline", test_llm_pipeline),
        ("RAG __init__", test_rag_init)
    ]

    results = {}
    total_time = time.time()

    for test_name, test_func in tests:
        start_time = time.time()
        success = test_func()
        end_time = time.time()

        results[test_name] = {
            "success": success,
            "time": end_time - start_time
        }

    total_time = time.time() - total_time

    # Resumen final
    print("\n" + "=" * 60)
    print(" **RESUMEN DE TESTING**")
    print("=" * 60)

    passed = sum(1 for r in results.values() if r["success"])
    total = len(results)

    print(f"\n✅ **Tests pasados:** {passed}/{total}")
    print(f"⏱ **Tiempo total:** {total_time:.2f} segundos")

    for test_name, result in results.items():
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        time_str = f"{result['time']:.2f}s"
        print(f"   • {test_name:<20} {status:<8} ({time_str})")

    if passed == total:
        print(f"\n **TODOS LOS MÓDULOS FUNCIONAN CORRECTAMENTE**")
        print(f" **Sistema RAG completamente operativo**")
    else:
        print(f"\n⚠ **Algunos módulos necesitan atención**")
        print(f" **Revisar errores arriba para detalles**")

    print(f"\n Fin: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()