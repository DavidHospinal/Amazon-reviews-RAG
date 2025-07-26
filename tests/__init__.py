"""
Módulo de tests para Amazon Big Data Analysis Project
Suite de pruebas unitarias para validar funcionalidad de módulos
"""

import sys
import os
from pathlib import Path

# Agregar src al path para importaciones
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

__version__ = "1.0.0"
__test_suite__ = "Amazon Big Data Tests"

# Configuración común para todos los tests
TEST_DATA_SIZE = 100
SAMPLE_CATEGORIES = ['Books', 'Video_Games', 'Movies_and_TV']
SAMPLE_RATINGS = [1.0, 2.0, 3.0, 4.0, 5.0]


def setup_test_environment():
    """
    Configura el entorno de pruebas
    """
    import logging
    logging.basicConfig(level=logging.WARNING)  # Silenciar logs durante tests

    print("🧪 Configurando entorno de tests...")
    print(f"📁 Proyecto: {project_root}")
    print(f"📦 Módulos src: {src_path}")
    return True


if __name__ == "__main__":
    print("🧪 SUITE DE TESTS - AMAZON BIG DATA ANALYSIS")
    print("=" * 50)
    print("📋 Tests disponibles:")
    print("   • test_downloader.py    - Tests de descarga de datos")
    print("   • test_cleaner.py       - Tests de limpieza de datos")
    print("   • test_explorer.py      - Tests de análisis exploratorio")
    print("   • test_visualizer.py    - Tests de visualización")
    print()
    print("🚀 Para ejecutar todos los tests:")
    print("   python -m pytest tests/")
    print()
    print("🎯 Para ejecutar un test específico:")
    print("   python tests/test_cleaner.py")