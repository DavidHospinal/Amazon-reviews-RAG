"""
Tests para el módulo de limpieza de datos
"""

import unittest
import sys
import pandas as pd
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

class TestDataCleaner(unittest.TestCase):
    """Tests para la clase DataCleaner"""

    def setUp(self):
        """Configuración inicial"""
        from preprocessing.cleaner import DataCleaner
        self.cleaner = DataCleaner()

        # Datos de prueba
        self.sample_review = {
            "reviewerID": "A123456",
            "asin": "B001234",
            "reviewerName": "Test User",
            "helpful": [2, 3],
            "reviewText": "This is a test review",
            "overall": 4.0,
            "summary": "Good product",
            "unixReviewTime": 1365465600,
            "reviewTime": "04 9, 2013"
        }

        self.invalid_review = {
            "reviewerID": "",           # Vacío - se convertirá a "UNKNOWN"
            "asin": None,              # None - se convertirá a "UNKNOWN"
            "overall": 6.0,            # Rating inválido - se convertirá a 3.0
            "reviewText": "",          # Vacío
            "summary": ""              # Vacío
        }

    def test_clean_valid_review(self):
        """Test: Limpiar review válido"""
        cleaned = self.cleaner.clean_review_data(self.sample_review)

        # Verificar que se mantienen los datos válidos
        self.assertEqual(cleaned["reviewerID"], "A123456")
        self.assertEqual(cleaned["asin"], "B001234")
        self.assertEqual(cleaned["overall"], 4.0)
        self.assertIsInstance(cleaned["helpful"], list)
        self.assertEqual(len(cleaned["helpful"]), 2)

    def test_clean_invalid_review(self):
        """Test: Limpiar review inválido"""
        cleaned = self.cleaner.clean_review_data(self.invalid_review)

        # Verificar valores por defecto (CORREGIDO)
        self.assertEqual(cleaned["reviewerID"], "UNKNOWN")  # ✅ CORREGIDO
        self.assertEqual(cleaned["asin"], "UNKNOWN")
        self.assertEqual(cleaned["overall"], 3.0)           # Valor por defecto para rating inválido
        self.assertEqual(cleaned["helpful"], [0, 0])

    def test_validate_quality_valid(self):
        """Test: Validar calidad de review válido"""
        cleaned = self.cleaner.clean_review_data(self.sample_review)
        is_valid = self.cleaner.validate_review_quality(cleaned)
        self.assertTrue(is_valid, "Review válido debe pasar validación")

    def test_validate_quality_invalid(self):
        """Test: Validar calidad de review inválido"""
        cleaned = self.cleaner.clean_review_data(self.invalid_review)
        is_valid = self.cleaner.validate_review_quality(cleaned)
        self.assertFalse(is_valid, "Review inválido debe fallar validación")

    def test_batch_cleaning(self):
        """Test: Limpieza en lote"""
        reviews = [self.sample_review, self.invalid_review, self.sample_review.copy()]
        cleaned_reviews = self.cleaner.clean_batch(reviews)

        # Solo los reviews válidos deben pasar
        self.assertGreater(len(cleaned_reviews), 0)
        self.assertLess(len(cleaned_reviews), len(reviews))

        # Verificar que todos los reviews limpiados son válidos
        for review in cleaned_reviews:
            self.assertTrue(self.cleaner.validate_review_quality(review))

    def test_text_length_limits(self):
        """Test: Límites de longitud de texto"""
        long_review = self.sample_review.copy()
        long_review["reviewText"] = "x" * 2000  # Muy largo
        long_review["summary"] = "y" * 500     # Muy largo

        cleaned = self.cleaner.clean_review_data(long_review)

        # Verificar límites aplicados
        self.assertLessEqual(len(cleaned["reviewText"]), 1000)
        self.assertLessEqual(len(cleaned["summary"]), 200)

def run_cleaner_tests():
    """Ejecuta todos los tests de cleaner"""
    print("🧪 EJECUTANDO TESTS DE DATA CLEANER")
    print("=" * 40)

    unittest.main(argv=[''], exit=False, verbosity=2)

if __name__ == "__main__":
    run_cleaner_tests()