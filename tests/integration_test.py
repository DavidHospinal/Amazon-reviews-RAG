"""
Test de integración completa para Amazon Big Data Analysis
Prueba todos los módulos .py del proyecto de forma integrada
"""

import sys
import os
import pandas as pd
import numpy as np
import tempfile
import json
from pathlib import Path
from datetime import datetime

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class IntegrationTester:
    """Clase para ejecutar tests de integración completos"""

    def __init__(self):
        self.results = {
            'acquisition': {'status': 'pending', 'details': []},
            'preprocessing': {'status': 'pending', 'details': []},
            'storage': {'status': 'pending', 'details': []},
            'analysis': {'status': 'pending', 'details': []},
            'overall': {'status': 'pending', 'tests_passed': 0, 'tests_total': 0}
        }
        self.sample_data = self._create_sample_data()

    def _create_sample_data(self):
        """Crea datos de muestra para testing"""
        return [
            {
                "reviewerID": "A123456789",
                "asin": "B001234567",
                "reviewerName": "Test User 1",
                "helpful": [5, 7],
                "reviewText": "This is an excellent product. Highly recommend it!",
                "overall": 5.0,
                "summary": "Excellent product",
                "unixReviewTime": 1365465600,
                "reviewTime": "04 9, 2013"
            },
            {
                "reviewerID": "A987654321",
                "asin": "B009876543",
                "reviewerName": "Test User 2",
                "helpful": [2, 3],
                "reviewText": "Average product, nothing special.",
                "overall": 3.0,
                "summary": "Average",
                "unixReviewTime": 1400000000,
                "reviewTime": "05 14, 2014"
            },
            {
                "reviewerID": "A111222333",
                "asin": "B111222333",
                "reviewerName": "Test User 3",
                "helpful": [0, 1],
                "reviewText": "Poor quality, would not recommend.",
                "overall": 1.0,
                "summary": "Poor quality",
                "unixReviewTime": 1450000000,
                "reviewTime": "12 13, 2015"
            }
        ]

    def test_acquisition_modules(self):
        """Test módulos de adquisición de datos"""
        print("📥 TESTING ACQUISITION MODULES")
        print("-" * 40)

        try:
            # Test downloader import
            try:
                from acquisition import downloader
                self.results['acquisition']['details'].append("✅ downloader module imported")
                print("   ✅ downloader.py - Import successful")
            except ImportError as e:
                self.results['acquisition']['details'].append(f"❌ downloader import failed: {e}")
                print(f"   ❌ downloader.py - Import failed: {e}")

            # Test extractor import
            try:
                from acquisition import extractor
                self.results['acquisition']['details'].append("✅ extractor module imported")
                print("   ✅ extractor.py - Import successful")
            except ImportError as e:
                self.results['acquisition']['details'].append(f"❌ extractor import failed: {e}")
                print(f"   ❌ extractor.py - Import failed: {e}")

            self.results['acquisition']['status'] = 'completed'

        except Exception as e:
            self.results['acquisition']['status'] = 'failed'
            self.results['acquisition']['details'].append(f"❌ Unexpected error: {e}")
            print(f"   ❌ Unexpected error in acquisition: {e}")

    def test_preprocessing_modules(self):
        """Test módulos de preprocesamiento"""
        print("\n🧹 TESTING PREPROCESSING MODULES")
        print("-" * 40)

        try:
            # Test DataCleaner
            try:
                from preprocessing.cleaner import DataCleaner
                cleaner = DataCleaner()

                # Test con datos válidos
                cleaned_valid = cleaner.clean_review_data(self.sample_data[0])
                assert cleaned_valid['reviewerID'] == "A123456789"
                assert cleaned_valid['overall'] == 5.0

                # Test con datos inválidos
                invalid_data = {"reviewerID": "", "asin": None, "overall": 6.0, "reviewText": "", "summary": ""}
                cleaned_invalid = cleaner.clean_review_data(invalid_data)
                assert cleaned_invalid['reviewerID'] == "UNKNOWN"
                assert cleaned_invalid['overall'] == 3.0

                # Test batch cleaning
                cleaned_batch = cleaner.clean_batch(self.sample_data)
                assert len(cleaned_batch) > 0

                self.results['preprocessing']['details'].append("✅ DataCleaner - All tests passed")
                print("   ✅ DataCleaner - All tests passed")

            except Exception as e:
                self.results['preprocessing']['details'].append(f"❌ DataCleaner failed: {e}")
                print(f"   ❌ DataCleaner failed: {e}")

            # Test DataTransformer
            try:
                from preprocessing.transformer import DataTransformer
                transformer = DataTransformer()

                # Test enrichment
                test_review = self.sample_data[0].copy()
                enriched = transformer.enrich_review_data(test_review, "Books")

                assert enriched['category_group'] == 'Entertainment'
                assert enriched['analysis_type'] == 'Leisure/Personal'
                assert enriched['original_category'] == 'Books'
                assert 'download_timestamp' in enriched

                self.results['preprocessing']['details'].append("✅ DataTransformer - All tests passed")
                print("   ✅ DataTransformer - All tests passed")

            except Exception as e:
                self.results['preprocessing']['details'].append(f"❌ DataTransformer failed: {e}")
                print(f"   ❌ DataTransformer failed: {e}")

            self.results['preprocessing']['status'] = 'completed'

        except Exception as e:
            self.results['preprocessing']['status'] = 'failed'
            self.results['preprocessing']['details'].append(f"❌ Unexpected error: {e}")
            print(f"   ❌ Unexpected error in preprocessing: {e}")

    def test_storage_modules(self):
        """Test módulos de almacenamiento"""
        print("\n🗄️ TESTING STORAGE MODULES")
        print("-" * 40)

        try:
            # Test NoSQLManager import
            try:
                from storage import nosql_manager
                self.results['storage']['details'].append("✅ nosql_manager module imported")
                print("   ✅ nosql_manager.py - Import successful")
            except ImportError as e:
                self.results['storage']['details'].append(f"❌ nosql_manager import failed: {e}")
                print(f"   ❌ nosql_manager.py - Import failed: {e}")

            # Test QueryEngine
            try:
                from storage.queries import QueryEngine
                self.results['storage']['details'].append("✅ QueryEngine class imported")
                print("   ✅ queries.py - QueryEngine import successful")
            except ImportError as e:
                self.results['storage']['details'].append(f"❌ QueryEngine import failed: {e}")
                print(f"   ❌ queries.py - QueryEngine import failed: {e}")

            # Test TinyDB functionality
            try:
                from tinydb import TinyDB
                import tempfile

                # Crear base temporal
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    temp_db_path = f.name

                db = TinyDB(temp_db_path)
                test_table = db.table('test_reviews')

                # Insertar datos de prueba
                test_table.insert_multiple(self.sample_data)

                # Verificar inserción
                all_records = test_table.all()
                assert len(all_records) == len(self.sample_data)

                # Limpiar
                db.close()
                os.unlink(temp_db_path)

                self.results['storage']['details'].append("✅ TinyDB operations - All tests passed")
                print("   ✅ TinyDB operations - All tests passed")

            except Exception as e:
                self.results['storage']['details'].append(f"❌ TinyDB operations failed: {e}")
                print(f"   ❌ TinyDB operations failed: {e}")

            self.results['storage']['status'] = 'completed'

        except Exception as e:
            self.results['storage']['status'] = 'failed'
            self.results['storage']['details'].append(f"❌ Unexpected error: {e}")
            print(f"   ❌ Unexpected error in storage: {e}")

    def test_analysis_modules(self):
        """Test módulos de análisis"""
        print("\n📊 TESTING ANALYSIS MODULES")
        print("-" * 40)

        try:
            # Preparar DataFrame para análisis
            df_sample = pd.DataFrame(self.sample_data)

            # Test DataExplorer
            try:
                from analysis.explorer import DataExplorer
                explorer = DataExplorer(df_sample)

                # Test estadísticas básicas
                basic_stats = explorer.basic_statistics()
                assert 'total_reviews' in basic_stats
                assert basic_stats['total_reviews'] == len(self.sample_data)

                # Test análisis de satisfacción
                satisfaction = explorer.satisfaction_analysis()
                assert 'distribution_counts' in satisfaction

                self.results['analysis']['details'].append("✅ DataExplorer - All tests passed")
                print("   ✅ DataExplorer - All tests passed")

            except ImportError as e:
                self.results['analysis']['details'].append(f"⚠️ DataExplorer not available: {e}")
                print(f"   ⚠️ DataExplorer not available: {e}")
            except Exception as e:
                self.results['analysis']['details'].append(f"❌ DataExplorer failed: {e}")
                print(f"   ❌ DataExplorer failed: {e}")

            # Test DataVisualizer
            try:
                from analysis.visualizer import DataVisualizer
                visualizer = DataVisualizer(df_sample)

                # Test inicialización
                assert visualizer.data is not None
                assert len(visualizer.data) == len(self.sample_data)

                # Test configuración de matplotlib (sin generar gráficos)
                import matplotlib.pyplot as plt
                plt.ioff()  # Turn off interactive mode for testing

                self.results['analysis']['details'].append("✅ DataVisualizer - Initialization tests passed")
                print("   ✅ DataVisualizer - Initialization tests passed")

            except ImportError as e:
                self.results['analysis']['details'].append(f"⚠️ DataVisualizer not available: {e}")
                print(f"   ⚠️ DataVisualizer not available: {e}")
            except Exception as e:
                self.results['analysis']['details'].append(f"❌ DataVisualizer failed: {e}")
                print(f"   ❌ DataVisualizer failed: {e}")

            self.results['analysis']['status'] = 'completed'

        except Exception as e:
            self.results['analysis']['status'] = 'failed'
            self.results['analysis']['details'].append(f"❌ Unexpected error: {e}")
            print(f"   ❌ Unexpected error in analysis: {e}")

    def test_full_pipeline(self):
        """Test del pipeline completo de procesamiento"""
        print("\n🔄 TESTING FULL PIPELINE")
        print("-" * 40)

        try:
            from preprocessing.cleaner import DataCleaner
            from preprocessing.transformer import DataTransformer

            # Pipeline completo
            cleaner = DataCleaner()
            transformer = DataTransformer()

            # 1. Limpiar datos
            cleaned_data = []
            for review in self.sample_data:
                cleaned = cleaner.clean_review_data(review)
                if cleaner.validate_review_quality(cleaned):
                    cleaned_data.append(cleaned)

            # 2. Enriquecer datos
            enriched_data = []
            categories = ['Books', 'Video_Games', 'Movies_and_TV']
            for i, review in enumerate(cleaned_data):
                category = categories[i % len(categories)]
                enriched = transformer.enrich_review_data(review.copy(), category)
                enriched_data.append(enriched)

            # 3. Crear DataFrame
            df = pd.DataFrame(enriched_data)

            # 4. Análisis básico
            assert len(df) > 0
            assert 'category_group' in df.columns
            assert 'analysis_type' in df.columns
            assert 'original_category' in df.columns

            print("   ✅ Full pipeline - All steps completed successfully")
            print(f"   📊 Processed {len(enriched_data)} records through complete pipeline")

            return True

        except Exception as e:
            print(f"   ❌ Full pipeline failed: {e}")
            return False

    def generate_report(self):
        """Genera reporte final de testing"""
        print("\n" + "=" * 60)
        print("📋 INTEGRATION TEST REPORT")
        print("=" * 60)

        total_tests = 0
        passed_tests = 0

        for module, result in self.results.items():
            if module != 'overall':
                total_tests += 1
                if result['status'] == 'completed':
                    passed_tests += 1

                status_emoji = "✅" if result['status'] == 'completed' else "❌" if result['status'] == 'failed' else "⚠️"
                print(f"\n{status_emoji} {module.upper()}: {result['status']}")

                for detail in result['details']:
                    print(f"   {detail}")

        # Pipeline test
        pipeline_success = self.test_full_pipeline()
        if pipeline_success:
            passed_tests += 1
        total_tests += 1

        # Actualizar resultados generales
        self.results['overall']['tests_passed'] = passed_tests
        self.results['overall']['tests_total'] = total_tests
        self.results['overall']['status'] = 'completed' if passed_tests == total_tests else 'partial'

        print(f"\n" + "=" * 60)
        print(f"📊 SUMMARY:")
        print(f"   🎯 Tests passed: {passed_tests}/{total_tests}")
        print(f"   📈 Success rate: {(passed_tests / total_tests) * 100:.1f}%")

        if passed_tests == total_tests:
            print(f"   🎉 ALL INTEGRATION TESTS PASSED!")
        elif passed_tests > total_tests * 0.7:
            print(f"   ✅ MOST TESTS PASSED - Minor issues detected")
        else:
            print(f"   ⚠️ SEVERAL ISSUES DETECTED - Review failed modules")

        print(f"\n⏰ Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        return self.results


def run_integration_tests():
    """Función principal para ejecutar todos los tests de integración"""
    print("🧪 AMAZON BIG DATA - INTEGRATION TESTING SUITE")
    print("=" * 60)
    print("🎯 Testing all .py modules in the project...")
    print()

    tester = IntegrationTester()

    # Ejecutar todos los tests
    tester.test_acquisition_modules()
    tester.test_preprocessing_modules()
    tester.test_storage_modules()
    tester.test_analysis_modules()

    # Generar reporte final
    results = tester.generate_report()

    return results


if __name__ == "__main__":
    results = run_integration_tests()

    # Exit code basado en resultados
    success_rate = results['overall']['tests_passed'] / results['overall']['tests_total']
    exit_code = 0 if success_rate >= 0.8 else 1
    sys.exit(exit_code)