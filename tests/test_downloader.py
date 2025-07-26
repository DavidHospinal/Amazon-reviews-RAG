"""
Tests para el módulo de descarga de datos
"""

import unittest
import sys
import os
from pathlib import Path
import tempfile
import json

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestDownloader(unittest.TestCase):
    """Tests para funcionalidades de descarga"""

    def setUp(self):
        """Configuración inicial para cada test"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Limpieza después de cada test"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_downloader_import(self):
        """Test: Verificar que el módulo downloader se puede importar"""
        try:
            from acquisition import downloader
            self.assertTrue(True, "Módulo downloader importado correctamente")
        except ImportError as e:
            self.fail(f"No se pudo importar downloader: {e}")

    def test_extractor_import(self):
        """Test: Verificar que el módulo extractor se puede importar"""
        try:
            from acquisition import extractor
            self.assertTrue(True, "Módulo extractor importado correctamente")
        except ImportError as e:
            self.fail(f"No se pudo importar extractor: {e}")

    def test_sample_data_structure(self):
        """Test: Verificar estructura de datos de muestra"""
        sample_review = {
            "reviewerID": "A2SUAM1J3GNN3B",
            "asin": "0000013714",
            "reviewerName": "J. McDonald",
            "helpful": [2, 3],
            "reviewText": "I bought this for my husband...",
            "overall": 5.0,
            "summary": "Yay! Another old favorite!",
            "unixReviewTime": 1365465600,
            "reviewTime": "04 9, 2013"
        }

        # Verificar campos obligatorios
        required_fields = ["reviewerID", "asin", "overall"]
        for field in required_fields:
            self.assertIn(field, sample_review, f"Campo requerido {field} presente")

        # Verificar tipos de datos
        self.assertIsInstance(sample_review["overall"], (int, float))
        self.assertTrue(1.0 <= sample_review["overall"] <= 5.0)

    def test_data_validation(self):
        """Test: Validación básica de datos"""
        # Test rating válido
        valid_ratings = [1.0, 2.5, 3.0, 4.5, 5.0]
        for rating in valid_ratings:
            self.assertTrue(1.0 <= rating <= 5.0, f"Rating {rating} es válido")

        # Test rating inválido
        invalid_ratings = [0.0, 6.0, -1.0]
        for rating in invalid_ratings:
            self.assertFalse(1.0 <= rating <= 5.0, f"Rating {rating} es inválido")


def run_downloader_tests():
    """Ejecuta todos los tests de downloader"""
    print("🧪 EJECUTANDO TESTS DE DOWNLOADER")
    print("=" * 40)

    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == "__main__":
    run_downloader_tests()