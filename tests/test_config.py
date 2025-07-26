"""
Test de configuraciones
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_config():
    print("⚙️ TESTING CONFIGURATION MODULES")
    print("=" * 40)

    try:
        from config import PROJECT_CONFIG, DATA_CONFIG, DATABASE_CONFIG
        from config import validate_config, get_data_path, get_db_connection

        print("✅ Configuraciones importadas correctamente")
        print(f"   📚 Proyecto: {PROJECT_CONFIG['name']}")
        print(f"   📁 Categorías: {len(DATA_CONFIG['categories'])}")
        print(f"   🗄️ Tablas BD: {len(DATABASE_CONFIG['tables'])}")

        # Test validación
        if validate_config():
            print("✅ Validación de configuración exitosa")
        else:
            print("⚠️ Problemas en validación de configuración")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    test_config()