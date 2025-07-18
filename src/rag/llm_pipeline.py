"""
LLM Pipeline Module
===================

Módulo para integración con modelos de lenguaje grandes (LLM) en el sistema RAG.
Maneja la generación de respuestas basadas en contexto recuperado.

Este módulo proporciona:
- Integración con múltiples proveedores de LLM (OpenAI, Hugging Face, local)
- Construcción de prompts contextuales
- Generación de respuestas augmentadas
- Detección de alucinaciones

Autor: Oscar David Hospinal R.
"""

import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass
import re
import os

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Imports opcionales para diferentes proveedores
try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI no disponible. Instale con: pip install openai")

try:
    from transformers import pipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers no disponible. Instale con: pip install transformers")

# Imports de módulos locales
from .retriever import SearchResult


@dataclass
class RAGResponse:
    """Clase para representar una respuesta del sistema RAG."""
    query: str
    answer: str
    context_used: List[Dict]
    confidence_score: float
    generation_time: float
    model_used: str
    tokens_used: Optional[int] = None
    has_hallucination_risk: bool = False
    metadata: Dict = None


@dataclass
class PromptTemplate:
    """Plantilla para construcción de prompts."""
    system_prompt: str
    user_template: str
    context_template: str
    max_context_length: int = 2000


class LLMPipeline:
    """
    Pipeline de integración con LLMs para generación de respuestas RAG.

    Maneja la construcción de prompts contextuales y generación de respuestas
    basadas en documentos recuperados.
    """

    def __init__(self,
                 provider: str = "openai",
                 model: str = "gpt-3.5-turbo",
                 api_key: Optional[str] = None,
                 temperature: float = 0.3,
                 max_tokens: int = 500,
                 custom_prompt_template: Optional[PromptTemplate] = None):
        """
        Inicializa el pipeline de LLM.

        Args:
            provider (str): Proveedor del LLM ("openai", "huggingface", "local")
            model (str): Nombre del modelo
            api_key (str, optional): Clave API si es necesaria
            temperature (float): Temperatura para generación
            max_tokens (int): Máximo número de tokens
            custom_prompt_template (PromptTemplate, optional): Plantilla personalizada
        """
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Configurar cliente según proveedor
        self.client = None
        self.pipeline_model = None

        # Plantilla de prompts
        self.prompt_template = custom_prompt_template or self._get_default_prompt_template()

        # Inicializar proveedor
        self._initialize_provider()

        logger.info(f"LLMPipeline inicializado: {provider} - {model}")

    def _initialize_provider(self):
        """Inicializa el proveedor de LLM."""
        if self.provider == "openai":
            self._initialize_openai()
        elif self.provider == "huggingface":
            self._initialize_huggingface()
        elif self.provider == "local":
            self._initialize_local()
        else:
            raise ValueError(f"Proveedor no soportado: {self.provider}")

    def _initialize_openai(self):
        """Inicializa cliente OpenAI."""
        if not OPENAI_AVAILABLE:
            raise RuntimeError("OpenAI no está disponible. Instale con: pip install openai")

        # Usar API key del parámetro o variable de entorno
        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("No se encontró API key de OpenAI. Algunas funciones no estarán disponibles.")
            return

        try:
            openai.api_key = api_key
            self.client = openai
            logger.info("Cliente OpenAI inicializado")
        except Exception as e:
            logger.error(f"Error inicializando OpenAI: {e}")
            raise

    def _initialize_huggingface(self):
        """Inicializa pipeline de Hugging Face."""
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Transformers no está disponible. Instale con: pip install transformers")

        try:
            # Usar modelo por defecto para generación de texto
            model_name = self.model if self.model != "gpt-3.5-turbo" else "microsoft/DialoGPT-medium"

            self.pipeline_model = pipeline(
                "text-generation",
                model=model_name,
                device=-1  # CPU
            )
            logger.info(f"Pipeline Hugging Face inicializado con {model_name}")
        except Exception as e:
            logger.error(f"Error inicializando Hugging Face: {e}")
            raise

    def _initialize_local(self):
        """Inicializa modelo local (simulado para propósitos académicos)."""
        logger.info("Modo local inicializado (simulado)")
        self.client = "local_mock"

    def _get_default_prompt_template(self) -> PromptTemplate:
        """Retorna la plantilla de prompt por defecto."""
        system_prompt = """Eres un asistente experto en análisis de reseñas de productos de Amazon. 
Tu tarea es responder preguntas basándote únicamente en el contexto proporcionado de reseñas reales.

INSTRUCCIONES:
1. Responde basándote SOLO en la información del contexto proporcionado
2. Si no hay información suficiente en el contexto, di que no tienes información suficiente
3. Sé específico y cita aspectos mencionados en las reseñas
4. Mantén un tono profesional y útil
5. NO inventes información que no esté en el contexto"""

        user_template = """PREGUNTA: {query}

CONTEXTO DE RESEÑAS:
{context}

RESPUESTA:"""

        context_template = """Reseña {index}:
Producto: {category}
Calificación: {rating}/5
Resumen: {summary}
Opinión: {review_text}
---"""

        return PromptTemplate(
            system_prompt=system_prompt,
            user_template=user_template,
            context_template=context_template,
            max_context_length=2000
        )

    def _build_context_from_results(self, search_results: List[SearchResult]) -> str:
        """
        Construye el contexto a partir de los resultados de búsqueda.

        Args:
            search_results (List[SearchResult]): Resultados de búsqueda

        Returns:
            str: Contexto formateado
        """
        context_parts = []
        total_length = 0

        for i, result in enumerate(search_results, 1):
            # Extraer información de los metadatos
            metadata = result.metadata
            category = metadata.get('category', 'Desconocido')
            rating = metadata.get('rating', 'N/A')
            summary = metadata.get('summary', 'Sin resumen')

            # Extraer texto de reseña del documento
            review_text = result.document

            # Limpiar el texto del documento que puede incluir metadata
            if '|' in review_text:
                # Si el documento tiene formato "Resumen: X | Reseña: Y | ..."
                parts = review_text.split('|')
                for part in parts:
                    if part.strip().startswith('Reseña:'):
                        review_text = part.replace('Reseña:', '').strip()
                        break

            # Truncar si es muy largo
            if len(review_text) > 300:
                review_text = review_text[:297] + "..."

            # Construir contexto para esta reseña
            context_part = self.prompt_template.context_template.format(
                index=i,
                category=category,
                rating=rating,
                summary=summary,
                review_text=review_text
            )

            # Verificar límite de longitud
            if total_length + len(context_part) > self.prompt_template.max_context_length:
                break

            context_parts.append(context_part)
            total_length += len(context_part)

        return "\n".join(context_parts)

    def _build_prompt(self, query: str, search_results: List[SearchResult]) -> str:
        """
        Construye el prompt completo para el LLM.

        Args:
            query (str): Pregunta del usuario
            search_results (List[SearchResult]): Resultados de búsqueda

        Returns:
            str: Prompt completo
        """
        context = self._build_context_from_results(search_results)

        prompt = self.prompt_template.user_template.format(
            query=query,
            context=context
        )

        return prompt

    def generate_response(self,
                          query: str,
                          search_results: List[SearchResult]) -> RAGResponse:
        """
        Genera una respuesta basada en la consulta y contexto recuperado.

        Args:
            query (str): Pregunta del usuario
            search_results (List[SearchResult]): Resultados de búsqueda

        Returns:
            RAGResponse: Respuesta generada por el LLM
        """
        start_time = datetime.now()

        # Construir prompt
        prompt = self._build_prompt(query, search_results)

        # Generar respuesta según proveedor
        if self.provider == "openai":
            answer, tokens_used = self._generate_openai_response(prompt)
        elif self.provider == "huggingface":
            answer, tokens_used = self._generate_huggingface_response(prompt)
        elif self.provider == "local":
            answer, tokens_used = self._generate_local_response(prompt, search_results)
        else:
            raise ValueError(f"Proveedor no soportado: {self.provider}")

        generation_time = (datetime.now() - start_time).total_seconds()

        # Preparar contexto usado
        context_used = []
        for result in search_results:
            context_info = {
                'id': result.id,
                'score': result.score,
                'category': result.metadata.get('category', ''),
                'rating': result.metadata.get('rating', 0),
                'summary': result.metadata.get('summary', '')
            }
            context_used.append(context_info)

        # Evaluar riesgo de alucinación
        hallucination_risk = self._detect_hallucination_risk(answer, search_results)

        # Calcular confianza
        confidence_score = self._calculate_confidence_score(search_results, answer)

        response = RAGResponse(
            query=query,
            answer=answer,
            context_used=context_used,
            confidence_score=confidence_score,
            generation_time=generation_time,
            model_used=f"{self.provider}-{self.model}",
            tokens_used=tokens_used,
            has_hallucination_risk=hallucination_risk,
            metadata={
                'prompt_length': len(prompt),
                'context_sources': len(search_results),
                'temperature': self.temperature
            }
        )

        logger.info(f"Respuesta generada en {generation_time:.2f}s")
        return response

    def _generate_openai_response(self, prompt: str) -> tuple[str, Optional[int]]:
        """Genera respuesta usando OpenAI."""
        if not self.client:
            return "Error: Cliente OpenAI no inicializado correctamente.", None

        try:
            response = self.client.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.prompt_template.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            answer = response.choices[0].message.content.strip()
            tokens_used = response.usage.total_tokens

            return answer, tokens_used

        except Exception as e:
            logger.error(f"Error generando respuesta OpenAI: {e}")
            return f"Error generando respuesta: {str(e)}", None

    def _generate_huggingface_response(self, prompt: str) -> tuple[str, Optional[int]]:
        """Genera respuesta usando Hugging Face."""
        if not self.pipeline_model:
            return "Error: Pipeline Hugging Face no inicializado.", None

        try:
            # Construir prompt completo
            full_prompt = f"{self.prompt_template.system_prompt}\n\n{prompt}"

            response = self.pipeline_model(
                full_prompt,
                max_length=len(full_prompt.split()) + self.max_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.pipeline_model.tokenizer.eos_token_id
            )

            generated_text = response[0]['generated_text']

            # Extraer solo la respuesta generada
            answer = generated_text[len(full_prompt):].strip()

            return answer, None

        except Exception as e:
            logger.error(f"Error generando respuesta Hugging Face: {e}")
            return f"Error generando respuesta: {str(e)}", None

    def _generate_local_response(self, prompt: str, search_results: List[SearchResult]) -> tuple[str, Optional[int]]:
        """Genera respuesta simulada para modo local (propósitos académicos)."""
        # Respuesta simulada basada en los resultados de búsqueda
        if not search_results:
            return "No se encontraron reseñas relevantes para responder tu pregunta.", 0

        # Analizar resultados para generar respuesta básica
        categories = set(r.metadata.get('category', '') for r in search_results)
        avg_rating = sum(r.metadata.get('rating', 0) for r in search_results) / len(search_results)
        high_rated = sum(1 for r in search_results if r.metadata.get('rating', 0) >= 4.0)

        response_parts = [
            f"Basándome en {len(search_results)} reseñas analizadas"
        ]

        if len(categories) == 1:
            response_parts.append(f"de la categoría {list(categories)[0]}")
        else:
            response_parts.append(f"de {len(categories)} categorías diferentes")

        response_parts.append(f"con una calificación promedio de {avg_rating:.1f}/5")

        if high_rated > len(search_results) * 0.7:
            response_parts.append("La mayoría de usuarios están satisfechos")
        elif high_rated > len(search_results) * 0.5:
            response_parts.append("Las opiniones están divididas")
        else:
            response_parts.append("Hay preocupaciones recurrentes en las reseñas")

        # Agregar ejemplo de reseña
        top_result = search_results[0]
        response_parts.append(f'Un usuario comentó: "{top_result.metadata.get("summary", "Sin resumen disponible")}"')

        answer = ", ".join(response_parts) + "."

        return answer, len(answer.split())

    def _detect_hallucination_risk(self, answer: str, search_results: List[SearchResult]) -> bool:
        """
        Detecta posible riesgo de alucinación en la respuesta.

        Args:
            answer (str): Respuesta generada
            search_results (List[SearchResult]): Contexto original

        Returns:
            bool: True si hay riesgo de alucinación
        """
        # Extraer palabras clave del contexto
        context_words = set()
        for result in search_results:
            text = result.document.lower()
            # Extraer palabras significativas (más de 3 caracteres)
            words = re.findall(r'\b\w{4,}\b', text)
            context_words.update(words)

        # Buscar en la respuesta afirmaciones específicas que no estén en el contexto
        answer_lower = answer.lower()

        # Patrones que pueden indicar alucinación
        hallucination_patterns = [
            r'según estudios',
            r'se ha demostrado que',
            r'estadísticamente',
            r'en general los expertos',
            r'está comprobado que'
        ]

        for pattern in hallucination_patterns:
            if re.search(pattern, answer_lower):
                return True

        # Si la respuesta es muy específica pero el contexto es limitado
        if len(search_results) < 3 and len(answer.split()) > 100:
            return True

        return False

    def _calculate_confidence_score(self, search_results: List[SearchResult], answer: str) -> float:
        """
        Calcula un score de confianza para la respuesta.

        Args:
            search_results (List[SearchResult]): Resultados de búsqueda
            answer (str): Respuesta generada

        Returns:
            float: Score de confianza (0-1)
        """
        if not search_results:
            return 0.0

        # Factores de confianza
        factors = []

        # 1. Calidad de los resultados de búsqueda
        avg_similarity = sum(r.score for r in search_results) / len(search_results)
        factors.append(avg_similarity)

        # 2. Número de fuentes
        source_score = min(len(search_results) / 5.0, 1.0)  # Máximo con 5 fuentes
        factors.append(source_score)

        # 3. Consistencia en las calificaciones
        ratings = [r.metadata.get('rating', 0) for r in search_results]
        rating_variance = sum((r - sum(ratings) / len(ratings)) ** 2 for r in ratings) / len(ratings)
        consistency_score = max(0, 1 - rating_variance)
        factors.append(consistency_score)

        # 4. Longitud apropiada de respuesta
        answer_words = len(answer.split())
        length_score = 1.0 if 20 <= answer_words <= 150 else 0.7
        factors.append(length_score)

        # Promedio ponderado
        weights = [0.4, 0.2, 0.2, 0.2]
        confidence = sum(f * w for f, w in zip(factors, weights))

        return round(confidence, 3)

    def compare_with_without_context(self, query: str, search_results: List[SearchResult]) -> Dict:
        """
        Compara respuestas con y sin contexto RAG.

        Args:
            query (str): Pregunta del usuario
            search_results (List[SearchResult]): Contexto RAG

        Returns:
            Dict: Comparación de respuestas
        """
        # Respuesta con contexto
        rag_response = self.generate_response(query, search_results)

        # Respuesta sin contexto (solo la pregunta)
        no_context_response = self.generate_response(query, [])

        comparison = {
            "query": query,
            "with_rag": {
                "answer": rag_response.answer,
                "confidence": rag_response.confidence_score,
                "sources_used": len(rag_response.context_used),
                "generation_time": rag_response.generation_time
            },
            "without_rag": {
                "answer": no_context_response.answer,
                "confidence": no_context_response.confidence_score,
                "generation_time": no_context_response.generation_time
            },
            "analysis": {
                "rag_improvement": rag_response.confidence_score > no_context_response.confidence_score,
                "length_difference": len(rag_response.answer) - len(no_context_response.answer),
                "time_overhead": rag_response.generation_time - no_context_response.generation_time
            }
        }

        return comparison


# Funciones de utilidad
def create_llm_pipeline(provider: str = "local",
                        model: str = "gpt-3.5-turbo",
                        **kwargs) -> LLMPipeline:
    """
    Función de conveniencia para crear un pipeline LLM.

    Args:
        provider (str): Proveedor del LLM
        model (str): Nombre del modelo
        **kwargs: Parámetros adicionales

    Returns:
        LLMPipeline: Instancia del pipeline
    """
    return LLMPipeline(
        provider=provider,
        model=model,
        **kwargs
    )


def evaluate_rag_responses(llm_pipeline: LLMPipeline,
                           test_queries: List[str],
                           search_results_list: List[List[SearchResult]]) -> Dict:
    """
    Evalúa la calidad de las respuestas RAG.

    Args:
        llm_pipeline (LLMPipeline): Pipeline LLM
        test_queries (List[str]): Consultas de prueba
        search_results_list (List[List[SearchResult]]): Resultados por consulta

    Returns:
        Dict: Métricas de evaluación
    """
    if len(test_queries) != len(search_results_list):
        raise ValueError("Número de consultas y resultados debe coincidir")

    evaluation = {
        "total_queries": len(test_queries),
        "successful_generations": 0,
        "average_confidence": 0,
        "average_generation_time": 0,
        "hallucination_risk_count": 0,
        "responses": []
    }

    confidences = []
    generation_times = []

    for query, search_results in zip(test_queries, search_results_list):
        try:
            response = llm_pipeline.generate_response(query, search_results)

            evaluation["successful_generations"] += 1
            confidences.append(response.confidence_score)
            generation_times.append(response.generation_time)

            if response.has_hallucination_risk:
                evaluation["hallucination_risk_count"] += 1

            evaluation["responses"].append({
                "query": query,
                "confidence": response.confidence_score,
                "generation_time": response.generation_time,
                "hallucination_risk": response.has_hallucination_risk,
                "answer_length": len(response.answer)
            })

        except Exception as e:
            logger.error(f"Error evaluando consulta '{query}': {e}")
            evaluation["responses"].append({
                "query": query,
                "error": str(e)
            })

    # Calcular promedios
    if confidences:
        evaluation["average_confidence"] = sum(confidences) / len(confidences)
        evaluation["confidence_stats"] = {
            "min": min(confidences),
            "max": max(confidences),
            "std": (sum((c - evaluation["average_confidence"]) ** 2 for c in confidences) / len(confidences)) ** 0.5
        }

    if generation_times:
        evaluation["average_generation_time"] = sum(generation_times) / len(generation_times)

    evaluation["hallucination_rate"] = evaluation["hallucination_risk_count"] / evaluation["total_queries"]

    return evaluation