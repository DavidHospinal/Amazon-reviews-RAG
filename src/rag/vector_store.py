"""
Vector Store Module
===================

Módulo para gestión de almacenamiento vectorial utilizando ChromaDB y Faiss.
Maneja la persistencia, indexación y búsqueda de vectores de embeddings.

Este módulo proporciona:
- Almacenamiento persistente de vectores
- Indexación eficiente para búsquedas
- Soporte para múltiples backends (ChromaDB, Faiss)
- Gestión de metadatos asociados

Autor: Oscar David Hospinal R.
"""

import json
import numpy as np
import os
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Union, Any
from pathlib import Path
import uuid

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Imports opcionales para diferentes backends
try:
    import chromadb
    from chromadb.config import Settings

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("ChromaDB no disponible. Instale con: pip install chromadb")

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("Faiss no disponible. Instale con: pip install faiss-cpu")


class VectorStore:
    """
    Clase principal para gestión de almacenamiento vectorial.

    Soporta múltiples backends y proporciona una interfaz unificada
    para operaciones de almacenamiento y búsqueda vectorial.
    """

    def __init__(self,
                 store_type: str = "chromadb",
                 collection_name: str = "amazon_reviews",
                 persist_directory: str = "data/vectors/chroma_db",
                 distance_metric: str = "cosine"):
        """
        Inicializa el almacén vectorial.

        Args:
            store_type (str): Tipo de almacén ("chromadb" o "faiss")
            collection_name (str): Nombre de la colección
            persist_directory (str): Directorio de persistencia
            distance_metric (str): Métrica de distancia ("cosine", "l2", "ip")
        """
        self.store_type = store_type.lower()
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.distance_metric = distance_metric

        # Variables de estado
        self.client = None
        self.collection = None
        self.index = None
        self.embeddings = None
        self.metadata = None
        self.dimension = None

        # Crear directorio si no existe
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        # Inicializar backend
        self._initialize_backend()

        logger.info(f"VectorStore inicializado: {store_type} en {persist_directory}")

    def _initialize_backend(self):
        """Inicializa el backend de almacenamiento vectorial."""
        if self.store_type == "chromadb":
            self._initialize_chromadb()
        elif self.store_type == "faiss":
            self._initialize_faiss()
        else:
            raise ValueError(f"Tipo de almacén no soportado: {self.store_type}")

    def _initialize_chromadb(self):
        """Inicializa ChromaDB."""
        if not CHROMADB_AVAILABLE:
            raise RuntimeError("ChromaDB no está disponible. Instale con: pip install chromadb")

        try:
            # Configurar cliente persistente
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )

            # Mapear métricas de distancia
            distance_map = {
                "cosine": "cosine",
                "l2": "l2",
                "ip": "ip"
            }

            chroma_distance = distance_map.get(self.distance_metric, "cosine")

            # Crear o obtener colección
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    metadata={"distance": chroma_distance}
                )
                logger.info(f"Colección existente cargada: {self.collection_name}")
            except:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"distance": chroma_distance}
                )
                logger.info(f"Nueva colección creada: {self.collection_name}")

        except Exception as e:
            logger.error(f"Error inicializando ChromaDB: {e}")
            raise

    def _initialize_faiss(self):
        """Inicializa Faiss."""
        if not FAISS_AVAILABLE:
            raise RuntimeError("Faiss no está disponible. Instale con: pip install faiss-cpu")

        # Faiss se inicializará cuando se agreguen vectores
        self.index = None
        logger.info("Backend Faiss preparado")

    def add_embeddings(self,
                       embeddings: np.ndarray,
                       metadata: List[Dict],
                       ids: Optional[List[str]] = None) -> bool:
        """
        Añade embeddings al almacén vectorial.

        Args:
            embeddings (np.ndarray): Array de embeddings
            metadata (List[Dict]): Lista de metadatos
            ids (List[str], optional): IDs únicos para cada embedding

        Returns:
            bool: True si se añadieron correctamente
        """
        if len(embeddings) != len(metadata):
            raise ValueError("Número de embeddings y metadatos debe coincidir")

        # Generar IDs si no se proporcionan
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(embeddings))]

        self.dimension = embeddings.shape[1]

        if self.store_type == "chromadb":
            return self._add_to_chromadb(embeddings, metadata, ids)
        elif self.store_type == "faiss":
            return self._add_to_faiss(embeddings, metadata, ids)

        return False

    def _add_to_chromadb(self, embeddings: np.ndarray, metadata: List[Dict], ids: List[str]) -> bool:
        """Añade datos a ChromaDB."""
        try:
            # Preparar documentos (texto combinado para ChromaDB)
            documents = []
            chroma_metadata = []

            for meta in metadata:
                # Texto para búsqueda textual en ChromaDB
                doc_text = meta.get('combined_text', '')
                documents.append(doc_text)

                # Metadatos (ChromaDB requiere valores simples)
                chroma_meta = {
                    'original_id': str(meta.get('original_id', '')),
                    'asin': str(meta.get('asin', '')),
                    'rating': float(meta.get('rating', 0)),
                    'category': str(meta.get('category', '')),
                    'category_group': str(meta.get('category_group', '')),
                    'text_length': int(meta.get('text_length', 0))
                }
                chroma_metadata.append(chroma_meta)

            # Añadir a la colección
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=documents,
                metadatas=chroma_metadata,
                ids=ids
            )

            logger.info(f"Añadidos {len(embeddings)} embeddings a ChromaDB")
            return True

        except Exception as e:
            logger.error(f"Error añadiendo a ChromaDB: {e}")
            return False

    def _add_to_faiss(self, embeddings: np.ndarray, metadata: List[Dict], ids: List[str]) -> bool:
        """Añade datos a Faiss."""
        try:
            # Inicializar índice Faiss si no existe
            if self.index is None:
                if self.distance_metric == "cosine":
                    # Para cosine similarity, normalizar vectores
                    faiss.normalize_L2(embeddings)
                    self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product
                elif self.distance_metric == "l2":
                    self.index = faiss.IndexFlatL2(self.dimension)
                else:
                    self.index = faiss.IndexFlatIP(self.dimension)  # Por defecto

            # Añadir vectores al índice
            if self.distance_metric == "cosine":
                embeddings_normalized = embeddings.copy()
                faiss.normalize_L2(embeddings_normalized)
                self.index.add(embeddings_normalized)
            else:
                self.index.add(embeddings)

            # Guardar metadatos separadamente
            if self.metadata is None:
                self.metadata = {}

            # Mapear índices a metadatos
            start_idx = len(self.metadata)
            for i, (id_val, meta) in enumerate(zip(ids, metadata)):
                self.metadata[start_idx + i] = {
                    'id': id_val,
                    **meta
                }

            # Guardar embeddings para referencia
            if self.embeddings is None:
                self.embeddings = embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, embeddings])

            logger.info(f"Añadidos {len(embeddings)} embeddings a Faiss")
            return True

        except Exception as e:
            logger.error(f"Error añadiendo a Faiss: {e}")
            return False

    def search(self,
               query_vector: np.ndarray,
               k: int = 5,
               filter_criteria: Optional[Dict] = None) -> List[Dict]:
        """
        Busca vectores similares al vector de consulta.

        Args:
            query_vector (np.ndarray): Vector de consulta
            k (int): Número de resultados a retornar
            filter_criteria (Dict, optional): Criterios de filtrado

        Returns:
            List[Dict]: Lista de resultados con metadata y scores
        """
        if self.store_type == "chromadb":
            return self._search_chromadb(query_vector, k, filter_criteria)
        elif self.store_type == "faiss":
            return self._search_faiss(query_vector, k, filter_criteria)

        return []

    def _search_chromadb(self, query_vector: np.ndarray, k: int, filter_criteria: Optional[Dict]) -> List[Dict]:
        """Busca en ChromaDB."""
        try:
            # Preparar query
            query_embeddings = [query_vector.tolist()]

            # Preparar filtros de ChromaDB
            where_clause = {}
            if filter_criteria:
                for key, value in filter_criteria.items():
                    if key in ['rating']:
                        # Filtros numéricos
                        if isinstance(value, dict):
                            where_clause[key] = value
                        else:
                            where_clause[key] = {"$eq": value}
                    else:
                        # Filtros de texto
                        where_clause[key] = {"$eq": str(value)}

            # Realizar búsqueda
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=k,
                where=where_clause if where_clause else None,
                include=['embeddings', 'documents', 'metadatas', 'distances']
            )

            # Formatear resultados
            formatted_results = []
            if results['ids'][0]:  # Verificar que hay resultados
                for i in range(len(results['ids'][0])):
                    result = {
                        'id': results['ids'][0][i],
                        'score': 1.0 - results['distances'][0][i],  # Convertir distancia a similitud
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'embedding': results['embeddings'][0][i] if results['embeddings'] else None
                    }
                    formatted_results.append(result)

            return formatted_results

        except Exception as e:
            logger.error(f"Error en búsqueda ChromaDB: {e}")
            return []

    def _search_faiss(self, query_vector: np.ndarray, k: int, filter_criteria: Optional[Dict]) -> List[Dict]:
        """Busca en Faiss."""
        try:
            if self.index is None:
                logger.warning("Índice Faiss no inicializado")
                return []

            # Preparar vector de consulta
            query_vec = query_vector.reshape(1, -1)
            if self.distance_metric == "cosine":
                faiss.normalize_L2(query_vec)

            # Buscar
            scores, indices = self.index.search(query_vec, k)

            # Formatear resultados
            formatted_results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1:  # No hay más resultados
                    break

                metadata = self.metadata.get(idx, {})

                # Aplicar filtros si se especifican
                if filter_criteria and not self._matches_filter(metadata, filter_criteria):
                    continue

                result = {
                    'id': metadata.get('id', f'faiss_{idx}'),
                    'score': float(score),
                    'document': metadata.get('combined_text', ''),
                    'metadata': metadata,
                    'embedding': self.embeddings[idx].tolist() if self.embeddings is not None else None
                }
                formatted_results.append(result)

            return formatted_results

        except Exception as e:
            logger.error(f"Error en búsqueda Faiss: {e}")
            return []

    def _matches_filter(self, metadata: Dict, filter_criteria: Dict) -> bool:
        """Verifica si los metadatos coinciden con los criterios de filtro."""
        for key, value in filter_criteria.items():
            if key not in metadata:
                return False

            meta_value = metadata[key]

            if isinstance(value, dict):
                # Filtro complejo (ej: {"$gte": 4.0})
                for op, filter_val in value.items():
                    if op == "$eq" and meta_value != filter_val:
                        return False
                    elif op == "$gte" and meta_value < filter_val:
                        return False
                    elif op == "$lte" and meta_value > filter_val:
                        return False
                    elif op == "$gt" and meta_value <= filter_val:
                        return False
                    elif op == "$lt" and meta_value >= filter_val:
                        return False
            else:
                # Filtro simple
                if meta_value != value:
                    return False

        return True

    def get_collection_stats(self) -> Dict:
        """
        Retorna estadísticas de la colección.

        Returns:
            Dict: Estadísticas de la colección
        """
        stats = {
            'store_type': self.store_type,
            'collection_name': self.collection_name,
            'distance_metric': self.distance_metric,
            'dimension': self.dimension
        }

        if self.store_type == "chromadb" and self.collection:
            try:
                count = self.collection.count()
                stats['total_vectors'] = count
            except:
                stats['total_vectors'] = 0
        elif self.store_type == "faiss" and self.index:
            stats['total_vectors'] = self.index.ntotal
        else:
            stats['total_vectors'] = 0

        return stats

    def delete_collection(self) -> bool:
        """
        Elimina la colección completa.

        Returns:
            bool: True si se eliminó correctamente
        """
        try:
            if self.store_type == "chromadb" and self.client:
                self.client.delete_collection(self.collection_name)
                logger.info(f"Colección ChromaDB eliminada: {self.collection_name}")
                return True
            elif self.store_type == "faiss":
                # Resetear índice Faiss
                self.index = None
                self.metadata = None
                self.embeddings = None
                logger.info("Índice Faiss reseteado")
                return True
        except Exception as e:
            logger.error(f"Error eliminando colección: {e}")
            return False

        return False

    def save_to_disk(self, filepath: Optional[str] = None) -> str:
        """
        Guarda el estado actual en disco.

        Args:
            filepath (str, optional): Ruta del archivo de guardado

        Returns:
            str: Ruta del archivo guardado
        """
        if self.store_type == "chromadb":
            # ChromaDB se persiste automáticamente
            logger.info("ChromaDB se persiste automáticamente")
            return self.persist_directory

        elif self.store_type == "faiss":
            if filepath is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = os.path.join(self.persist_directory, f"faiss_index_{timestamp}")

            try:
                # Guardar índice Faiss
                if self.index:
                    faiss.write_index(self.index, f"{filepath}.index")

                # Guardar metadatos
                if self.metadata:
                    with open(f"{filepath}_metadata.json", 'w', encoding='utf-8') as f:
                        json.dump(self.metadata, f, indent=2, ensure_ascii=False)

                # Guardar embeddings
                if self.embeddings is not None:
                    np.save(f"{filepath}_embeddings.npy", self.embeddings)

                # Guardar configuración
                config = {
                    'store_type': self.store_type,
                    'collection_name': self.collection_name,
                    'distance_metric': self.distance_metric,
                    'dimension': self.dimension,
                    'total_vectors': self.index.ntotal if self.index else 0
                }

                with open(f"{filepath}_config.json", 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2)

                logger.info(f"Índice Faiss guardado en: {filepath}")
                return filepath

            except Exception as e:
                logger.error(f"Error guardando índice Faiss: {e}")
                raise

        return ""

    def load_from_disk(self, filepath: str) -> bool:
        """
        Carga el estado desde disco.

        Args:
            filepath (str): Ruta del archivo a cargar

        Returns:
            bool: True si se cargó correctamente
        """
        if self.store_type == "faiss":
            try:
                # Cargar configuración
                with open(f"{filepath}_config.json", 'r', encoding='utf-8') as f:
                    config = json.load(f)

                self.dimension = config['dimension']

                # Cargar índice
                if os.path.exists(f"{filepath}.index"):
                    self.index = faiss.read_index(f"{filepath}.index")

                # Cargar metadatos
                if os.path.exists(f"{filepath}_metadata.json"):
                    with open(f"{filepath}_metadata.json", 'r', encoding='utf-8') as f:
                        self.metadata = json.load(f)

                # Cargar embeddings
                if os.path.exists(f"{filepath}_embeddings.npy"):
                    self.embeddings = np.load(f"{filepath}_embeddings.npy")

                logger.info(f"Índice Faiss cargado desde: {filepath}")
                return True

            except Exception as e:
                logger.error(f"Error cargando índice Faiss: {e}")
                return False

        return True  # ChromaDB se carga automáticamente


# Funciones de utilidad
def create_vector_store(store_type: str = "chromadb",
                        collection_name: str = "amazon_reviews",
                        persist_directory: str = "data/vectors") -> VectorStore:
    """
    Función de conveniencia para crear un almacén vectorial.

    Args:
        store_type (str): Tipo de almacén ("chromadb" o "faiss")
        collection_name (str): Nombre de la colección
        persist_directory (str): Directorio de persistencia

    Returns:
        VectorStore: Instancia del almacén vectorial
    """
    return VectorStore(
        store_type=store_type,
        collection_name=collection_name,
        persist_directory=persist_directory
    )


def migrate_embeddings_to_vector_store(embeddings_file: str,
                                       metadata_file: str,
                                       vector_store: VectorStore) -> bool:
    """
    Migra embeddings desde archivos a un almacén vectorial.

    Args:
        embeddings_file (str): Archivo de embeddings (.npy)
        metadata_file (str): Archivo de metadatos (.json)
        vector_store (VectorStore): Instancia del almacén vectorial

    Returns:
        bool: True si la migración fue exitosa
    """
    try:
        # Cargar embeddings y metadatos
        embeddings = np.load(embeddings_file)

        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Añadir al almacén vectorial
        success = vector_store.add_embeddings(embeddings, metadata)

        if success:
            logger.info(f"Migración exitosa: {len(embeddings)} vectores añadidos")

        return success

    except Exception as e:
        logger.error(f"Error en migración: {e}")
        return False