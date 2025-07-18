"""
Embeddings Generator Module
===========================

Módulo responsable de generar representaciones vectoriales de los textos de reseñas
utilizando modelos de sentence-transformers pre-entrenados.

Este módulo maneja:
- Carga y configuración de modelos de embeddings
- Procesamiento de texto en lotes
- Generación eficiente de vectores
- Almacenamiento persistente de embeddings

Autor: Oscar David Hospinal R.
"""

import json
import numpy as np
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    logger.error("sentence-transformers no está instalado. Ejecute: pip install sentence-transformers")
    raise


class EmbeddingsGenerator:
    """
    Generador de embeddings para textos de reseñas de Amazon.

    Esta clase maneja la generación de representaciones vectoriales usando
    modelos pre-entrenados de sentence-transformers.
    """

    def __init__(self,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 batch_size: int = 32,
                 cache_dir: Optional[str] = None):
        """
        Inicializa el generador de embeddings.

        Args:
            model_name (str): Nombre del modelo de sentence-transformers
            batch_size (int): Tamaño del lote para procesamiento
            cache_dir (str, optional): Directorio para cache del modelo
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.model = None
        self.embedding_dimension = None

        # Cargar modelo
        self._load_model()

        logger.info(f"EmbeddingsGenerator inicializado con modelo: {model_name}")
        logger.info(f"Dimensión de embeddings: {self.embedding_dimension}")

    def _load_model(self):
        """Carga el modelo de sentence-transformers."""
        try:
            logger.info(f"Cargando modelo {self.model_name}...")
            self.model = SentenceTransformer(
                self.model_name,
                cache_folder=self.cache_dir
            )
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            logger.info("Modelo cargado exitosamente")
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            raise

    def preprocess_text(self, text: str) -> str:
        """
        Preprocesa el texto antes de generar embeddings.

        Args:
            text (str): Texto a preprocesar

        Returns:
            str: Texto preprocesado
        """
        if not text or not isinstance(text, str):
            return ""

        # Limpieza básica
        text = text.strip()

        # Remover caracteres especiales excesivos
        text = ' '.join(text.split())

        # Truncar si es muy largo (límite del modelo)
        max_length = 512
        if len(text) > max_length:
            text = text[:max_length]

        return text

    def combine_review_fields(self, review: Dict) -> str:
        """
        Combina los campos relevantes de una reseña en un texto único.

        Args:
            review (dict): Diccionario con datos de la reseña

        Returns:
            str: Texto combinado para embeddings
        """
        components = []

        # Título/resumen si existe
        if review.get('summary'):
            components.append(f"Resumen: {review['summary']}")

        # Texto principal de la reseña
        if review.get('reviewText'):
            components.append(f"Reseña: {review['reviewText']}")

        # Categoría para contexto
        if review.get('original_category'):
            components.append(f"Categoría: {review['original_category']}")

        # Rating para contexto
        if review.get('overall'):
            components.append(f"Calificación: {review['overall']}/5")

        return " | ".join(components)

    def generate_single_embedding(self, text: str) -> np.ndarray:
        """
        Genera embedding para un texto individual.

        Args:
            text (str): Texto a vectorizar

        Returns:
            np.ndarray: Vector embedding
        """
        if not text:
            return np.zeros(self.embedding_dimension)

        processed_text = self.preprocess_text(text)

        try:
            embedding = self.model.encode(processed_text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Error generando embedding: {e}")
            return np.zeros(self.embedding_dimension)

    def generate_batch_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Genera embeddings para una lista de textos.

        Args:
            texts (List[str]): Lista de textos

        Returns:
            np.ndarray: Array de embeddings
        """
        if not texts:
            return np.array([])

        # Preprocesar textos
        processed_texts = [self.preprocess_text(text) for text in texts]

        try:
            embeddings = self.model.encode(
                processed_texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=True
            )
            return embeddings
        except Exception as e:
            logger.error(f"Error generando embeddings en lote: {e}")
            return np.zeros((len(texts), self.embedding_dimension))

    def process_reviews_data(self, reviews_data: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
        """
        Procesa una lista de reseñas y genera embeddings.

        Args:
            reviews_data (List[Dict]): Lista de diccionarios con datos de reseñas

        Returns:
            Tuple[np.ndarray, List[Dict]]: Embeddings y metadatos procesados
        """
        logger.info(f"Procesando {len(reviews_data)} reseñas...")

        # Combinar campos de texto para cada reseña
        combined_texts = []
        processed_metadata = []

        for i, review in enumerate(reviews_data):
            # Combinar texto de la reseña
            combined_text = self.combine_review_fields(review)
            combined_texts.append(combined_text)

            # Preparar metadatos
            metadata = {
                'id': f"review_{i}",
                'original_id': review.get('reviewerID', f'unknown_{i}'),
                'asin': review.get('asin', ''),
                'reviewer_name': review.get('reviewerName', ''),
                'summary': review.get('summary', ''),
                'rating': review.get('overall', 0),
                'category': review.get('original_category', ''),
                'category_group': review.get('category_group', ''),
                'review_time': review.get('reviewTime', ''),
                'combined_text': combined_text,
                'text_length': len(combined_text)
            }
            processed_metadata.append(metadata)

        # Generar embeddings
        logger.info("Generando embeddings...")
        embeddings = self.generate_batch_embeddings(combined_texts)

        logger.info(f"Embeddings generados: {embeddings.shape}")
        return embeddings, processed_metadata

    def save_embeddings(self,
                        embeddings: np.ndarray,
                        metadata: List[Dict],
                        output_dir: str = "data/vectors"):
        """
        Guarda embeddings y metadatos en archivos.

        Args:
            embeddings (np.ndarray): Array de embeddings
            metadata (List[Dict]): Lista de metadatos
            output_dir (str): Directorio de salida
        """
        # Crear directorio si no existe
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Guardar embeddings como numpy array
        embeddings_file = os.path.join(output_dir, f"embeddings_{timestamp}.npy")
        np.save(embeddings_file, embeddings)

        # Guardar metadatos como JSON
        metadata_file = os.path.join(output_dir, f"metadata_{timestamp}.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Guardar información del modelo
        model_info = {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dimension,
            'batch_size': self.batch_size,
            'num_embeddings': len(embeddings),
            'generation_timestamp': timestamp,
            'embeddings_file': embeddings_file,
            'metadata_file': metadata_file
        }

        info_file = os.path.join(output_dir, f"model_info_{timestamp}.json")
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2)

        logger.info(f"Embeddings guardados en: {embeddings_file}")
        logger.info(f"Metadatos guardados en: {metadata_file}")
        logger.info(f"Info del modelo guardada en: {info_file}")

        return {
            'embeddings_file': embeddings_file,
            'metadata_file': metadata_file,
            'model_info_file': info_file,
            'model_info': model_info
        }

    def load_embeddings(self, embeddings_file: str, metadata_file: str) -> Tuple[np.ndarray, List[Dict]]:
        """
        Carga embeddings y metadatos desde archivos.

        Args:
            embeddings_file (str): Ruta al archivo de embeddings
            metadata_file (str): Ruta al archivo de metadatos

        Returns:
            Tuple[np.ndarray, List[Dict]]: Embeddings y metadatos cargados
        """
        try:
            embeddings = np.load(embeddings_file)

            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            logger.info(f"Embeddings cargados: {embeddings.shape}")
            logger.info(f"Metadatos cargados: {len(metadata)} registros")

            return embeddings, metadata

        except Exception as e:
            logger.error(f"Error cargando embeddings: {e}")
            raise

    def get_model_info(self) -> Dict:
        """
        Retorna información del modelo actual.

        Returns:
            Dict: Información del modelo
        """
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dimension,
            'batch_size': self.batch_size,
            'cache_dir': self.cache_dir,
            'model_loaded': self.model is not None
        }


# Función de utilidad para uso directo
def create_embeddings_from_file(data_file: str,
                                output_dir: str = "data/vectors",
                                model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Dict:
    """
    Función de conveniencia para generar embeddings desde un archivo de datos.

    Args:
        data_file (str): Ruta al archivo JSON con las reseñas
        output_dir (str): Directorio de salida para embeddings
        model_name (str): Nombre del modelo de embeddings

    Returns:
        Dict: Información de los archivos generados
    """
    # Cargar datos
    with open(data_file, 'r', encoding='utf-8') as f:
        reviews_data = json.load(f)

    # Crear generador
    generator = EmbeddingsGenerator(model_name=model_name)

    # Procesar y generar embeddings
    embeddings, metadata = generator.process_reviews_data(reviews_data)

    # Guardar resultados
    result = generator.save_embeddings(embeddings, metadata, output_dir)

    return result