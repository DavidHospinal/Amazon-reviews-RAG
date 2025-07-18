"""
Semantic Retriever Module
==========================

Módulo responsable de la búsqueda semántica y recuperación de documentos relevantes.
Implementa diferentes estrategias de búsqueda y ranking de resultados.

Este módulo proporciona:
- Búsqueda semántica usando embeddings
- Filtrado por metadatos
- Ranking y reordenamiento de resultados
- Múltiples estrategias de recuperación

Autor: Oscar David Hospinal R.
"""

import numpy as np
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Union, Any
from dataclasses import dataclass
import json

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Imports de módulos locales
from .embeddings_generator import EmbeddingsGenerator
from .vector_store import VectorStore


@dataclass
class SearchResult:
    """Clase para representar un resultado de búsqueda."""
    id: str
    score: float
    document: str
    metadata: Dict
    embedding: Optional[List[float]] = None
    rank: Optional[int] = None


@dataclass
class SearchQuery:
    """Clase para representar una consulta de búsqueda."""
    text: str
    filters: Optional[Dict] = None
    top_k: int = 5
    similarity_threshold: float = 0.0
    rerank: bool = False


class SemanticRetriever:
    """
    Recuperador semántico para el sistema RAG.

    Maneja la búsqueda de documentos relevantes basada en similitud semántica
    y proporciona diferentes estrategias de recuperación y filtrado.
    """

    def __init__(self,
                 vector_store: VectorStore,
                 embeddings_generator: Optional[EmbeddingsGenerator] = None,
                 top_k: int = 5,
                 similarity_threshold: float = 0.7,
                 rerank_results: bool = False):
        """
        Inicializa el recuperador semántico.

        Args:
            vector_store (VectorStore): Almacén vectorial para búsquedas
            embeddings_generator (EmbeddingsGenerator, optional): Generador de embeddings
            top_k (int): Número de resultados por defecto
            similarity_threshold (float): Umbral mínimo de similitud
            rerank_results (bool): Si reordenar resultados
        """
        self.vector_store = vector_store
        self.embeddings_generator = embeddings_generator
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.rerank_results = rerank_results

        # Inicializar generador de embeddings si no se proporciona
        if self.embeddings_generator is None:
            self.embeddings_generator = EmbeddingsGenerator()

        logger.info(f"SemanticRetriever inicializado con top_k={top_k}, threshold={similarity_threshold}")

    def search(self,
               query: Union[str, SearchQuery],
               **kwargs) -> List[SearchResult]:
        """
        Realiza búsqueda semántica.

        Args:
            query (Union[str, SearchQuery]): Consulta de búsqueda
            **kwargs: Parámetros adicionales

        Returns:
            List[SearchResult]: Lista de resultados ordenados por relevancia
        """
        # Convertir string a SearchQuery si es necesario
        if isinstance(query, str):
            search_query = SearchQuery(
                text=query,
                top_k=kwargs.get('top_k', self.top_k),
                similarity_threshold=kwargs.get('similarity_threshold', self.similarity_threshold),
                filters=kwargs.get('filters', None),
                rerank=kwargs.get('rerank', self.rerank_results)
            )
        else:
            search_query = query

        # Generar embedding de la consulta
        query_embedding = self.embeddings_generator.generate_single_embedding(search_query.text)

        # Buscar en el almacén vectorial
        raw_results = self.vector_store.search(
            query_vector=query_embedding,
            k=search_query.top_k * 2,  # Obtener más resultados para filtrado
            filter_criteria=search_query.filters
        )

        # Convertir a SearchResult y filtrar por umbral
        search_results = []
        for i, result in enumerate(raw_results):
            if result['score'] >= search_query.similarity_threshold:
                search_result = SearchResult(
                    id=result['id'],
                    score=result['score'],
                    document=result['document'],
                    metadata=result['metadata'],
                    embedding=result.get('embedding'),
                    rank=i + 1
                )
                search_results.append(search_result)

        # Limitar a top_k después del filtrado
        search_results = search_results[:search_query.top_k]

        # Reordenar si es necesario
        if search_query.rerank:
            search_results = self._rerank_results(search_query.text, search_results)

        logger.info(f"Búsqueda completada: {len(search_results)} resultados para '{search_query.text[:50]}...'")
        return search_results

    def search_by_category(self,
                           query: str,
                           category: str,
                           top_k: Optional[int] = None) -> List[SearchResult]:
        """
        Busca documentos en una categoría específica.

        Args:
            query (str): Texto de consulta
            category (str): Categoría a filtrar
            top_k (int, optional): Número de resultados

        Returns:
            List[SearchResult]: Resultados filtrados por categoría
        """
        filters = {"category": category}
        return self.search(
            query,
            filters=filters,
            top_k=top_k or self.top_k
        )

    def search_by_rating(self,
                         query: str,
                         min_rating: float = 4.0,
                         top_k: Optional[int] = None) -> List[SearchResult]:
        """
        Busca documentos con rating mínimo.

        Args:
            query (str): Texto de consulta
            min_rating (float): Rating mínimo
            top_k (int, optional): Número de resultados

        Returns:
            List[SearchResult]: Resultados filtrados por rating
        """
        filters = {"rating": {"$gte": min_rating}}
        return self.search(
            query,
            filters=filters,
            top_k=top_k or self.top_k
        )

    def find_similar_reviews(self,
                             review_id: str,
                             top_k: Optional[int] = None) -> List[SearchResult]:
        """
        Encuentra reseñas similares a una reseña específica.

        Args:
            review_id (str): ID de la reseña de referencia
            top_k (int, optional): Número de resultados

        Returns:
            List[SearchResult]: Reseñas similares
        """
        # Primero buscar la reseña de referencia
        reference_results = self.vector_store.search(
            query_vector=np.zeros(self.embeddings_generator.embedding_dimension),
            k=1000  # Buscar en muchos para encontrar el ID específico
        )

        reference_doc = None
        for result in reference_results:
            if result['id'] == review_id or result['metadata'].get('original_id') == review_id:
                reference_doc = result['document']
                break

        if reference_doc is None:
            logger.warning(f"No se encontró reseña con ID: {review_id}")
            return []

        # Buscar documentos similares
        return self.search(
            reference_doc,
            top_k=top_k or self.top_k
        )

    def search_with_context(self,
                            query: str,
                            context_filters: Dict,
                            top_k: Optional[int] = None) -> List[SearchResult]:
        """
        Búsqueda contextual con múltiples filtros.

        Args:
            query (str): Texto de consulta
            context_filters (Dict): Filtros contextuales
            top_k (int, optional): Número de resultados

        Returns:
            List[SearchResult]: Resultados contextuales
        """
        return self.search(
            query,
            filters=context_filters,
            top_k=top_k or self.top_k
        )

    def _rerank_results(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """
        Reordena resultados usando criterios adicionales.

        Args:
            query (str): Consulta original
            results (List[SearchResult]): Resultados a reordenar

        Returns:
            List[SearchResult]: Resultados reordenados
        """
        # Implementar estrategias de reordenamiento

        # 1. Boost por rating alto
        for result in results:
            rating = result.metadata.get('rating', 0)
            if rating >= 4.5:
                result.score *= 1.1
            elif rating >= 4.0:
                result.score *= 1.05

        # 2. Boost por longitud de texto adecuada
        query_words = len(query.split())
        for result in results:
            text_length = result.metadata.get('text_length', 0)
            if query_words < 5 and text_length < 500:  # Consultas cortas prefieren textos cortos
                result.score *= 1.02
            elif query_words >= 5 and text_length > 200:  # Consultas largas prefieren textos detallados
                result.score *= 1.02

        # 3. Penalizar textos muy cortos para consultas complejas
        if query_words > 10:
            for result in results:
                text_length = result.metadata.get('text_length', 0)
                if text_length < 100:
                    result.score *= 0.95

        # Reordenar por score ajustado
        results.sort(key=lambda x: x.score, reverse=True)

        # Actualizar rankings
        for i, result in enumerate(results):
            result.rank = i + 1

        return results

    def get_search_statistics(self, results: List[SearchResult]) -> Dict:
        """
        Calcula estadísticas de los resultados de búsqueda.

        Args:
            results (List[SearchResult]): Resultados de búsqueda

        Returns:
            Dict: Estadísticas de los resultados
        """
        if not results:
            return {"total_results": 0}

        scores = [r.score for r in results]
        ratings = [r.metadata.get('rating', 0) for r in results]
        categories = [r.metadata.get('category', 'Unknown') for r in results]
        text_lengths = [r.metadata.get('text_length', 0) for r in results]

        stats = {
            "total_results": len(results),
            "score_stats": {
                "mean": np.mean(scores),
                "max": np.max(scores),
                "min": np.min(scores),
                "std": np.std(scores)
            },
            "rating_stats": {
                "mean": np.mean(ratings),
                "max": np.max(ratings),
                "min": np.min(ratings)
            },
            "category_distribution": {},
            "text_length_stats": {
                "mean": np.mean(text_lengths),
                "max": np.max(text_lengths),
                "min": np.min(text_lengths)
            }
        }

        # Distribución por categorías
        for category in categories:
            stats["category_distribution"][category] = stats["category_distribution"].get(category, 0) + 1

        return stats

    def explain_search(self, query: str, results: List[SearchResult]) -> Dict:
        """
        Proporciona explicación de los resultados de búsqueda.

        Args:
            query (str): Consulta realizada
            results (List[SearchResult]): Resultados obtenidos

        Returns:
            Dict: Explicación de la búsqueda
        """
        explanation = {
            "query": query,
            "search_timestamp": datetime.now().isoformat(),
            "total_results": len(results),
            "search_parameters": {
                "top_k": self.top_k,
                "similarity_threshold": self.similarity_threshold,
                "rerank_enabled": self.rerank_results
            }
        }

        if results:
            explanation["top_result"] = {
                "id": results[0].id,
                "score": results[0].score,
                "category": results[0].metadata.get('category', 'Unknown'),
                "rating": results[0].metadata.get('rating', 0),
                "preview": results[0].document[:100] + "..." if len(results[0].document) > 100 else results[0].document
            }

            explanation["score_range"] = {
                "highest": max(r.score for r in results),
                "lowest": min(r.score for r in results)
            }

            # Distribución de categorías en resultados
            category_counts = {}
            for result in results:
                cat = result.metadata.get('category', 'Unknown')
                category_counts[cat] = category_counts.get(cat, 0) + 1

            explanation["category_breakdown"] = category_counts

        return explanation

    def batch_search(self, queries: List[str], **kwargs) -> Dict[str, List[SearchResult]]:
        """
        Realiza múltiples búsquedas en lote.

        Args:
            queries (List[str]): Lista de consultas
            **kwargs: Parámetros de búsqueda

        Returns:
            Dict[str, List[SearchResult]]: Resultados por consulta
        """
        results = {}

        logger.info(f"Iniciando búsqueda en lote para {len(queries)} consultas")

        for query in queries:
            try:
                search_results = self.search(query, **kwargs)
                results[query] = search_results
            except Exception as e:
                logger.error(f"Error en búsqueda para '{query}': {e}")
                results[query] = []

        logger.info(f"Búsqueda en lote completada")
        return results


# Funciones de utilidad
def create_semantic_retriever(vector_store: VectorStore,
                              embeddings_generator: Optional[EmbeddingsGenerator] = None) -> SemanticRetriever:
    """
    Función de conveniencia para crear un recuperador semántico.

    Args:
        vector_store (VectorStore): Almacén vectorial
        embeddings_generator (EmbeddingsGenerator, optional): Generador de embeddings

    Returns:
        SemanticRetriever: Instancia del recuperador
    """
    return SemanticRetriever(
        vector_store=vector_store,
        embeddings_generator=embeddings_generator
    )


def evaluate_search_quality(retriever: SemanticRetriever,
                            test_queries: List[str],
                            expected_categories: List[str] = None) -> Dict:
    """
    Evalúa la calidad de búsqueda del recuperador.

    Args:
        retriever (SemanticRetriever): Instancia del recuperador
        test_queries (List[str]): Consultas de prueba
        expected_categories (List[str], optional): Categorías esperadas

    Returns:
        Dict: Métricas de evaluación
    """
    evaluation_results = {
        "total_queries": len(test_queries),
        "successful_searches": 0,
        "average_results_per_query": 0,
        "average_score": 0,
        "score_distribution": {"high": 0, "medium": 0, "low": 0}
    }

    all_scores = []
    total_results = 0

    for query in test_queries:
        try:
            results = retriever.search(query)
            if results:
                evaluation_results["successful_searches"] += 1
                total_results += len(results)

                query_scores = [r.score for r in results]
                all_scores.extend(query_scores)

                # Clasificar scores
                for score in query_scores:
                    if score >= 0.8:
                        evaluation_results["score_distribution"]["high"] += 1
                    elif score >= 0.6:
                        evaluation_results["score_distribution"]["medium"] += 1
                    else:
                        evaluation_results["score_distribution"]["low"] += 1

        except Exception as e:
            logger.error(f"Error evaluando consulta '{query}': {e}")

    # Calcular promedios
    if evaluation_results["successful_searches"] > 0:
        evaluation_results["average_results_per_query"] = total_results / evaluation_results["successful_searches"]

    if all_scores:
        evaluation_results["average_score"] = np.mean(all_scores)
        evaluation_results["score_stats"] = {
            "mean": np.mean(all_scores),
            "std": np.std(all_scores),
            "min": np.min(all_scores),
            "max": np.max(all_scores)
        }

    return evaluation_results