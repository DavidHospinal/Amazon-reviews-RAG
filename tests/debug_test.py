"""
Debug del test de cleaner
"""
import sys
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def debug_cleaner():
    print("🔍 DEPURANDO DATA CLEANER")
    print("=" * 40)

    from preprocessing.cleaner import DataCleaner
    cleaner = DataCleaner()

    # Datos de prueba inválidos
    invalid_review = {
        "reviewerID": "",  # Vacío
        "asin": None,  # None
        "overall": 6.0,  # Rating inválido
        "reviewText": "",  # Vacío
        "summary": ""  # Vacío
    }

    print("📋 DATOS DE ENTRADA:")
    for key, value in invalid_review.items():
        print(f"   {key}: {repr(value)} (tipo: {type(value).__name__})")

    print("\n🧹 LIMPIANDO DATOS...")
    cleaned = cleaner.clean_review_data(invalid_review)

    print("\n📋 DATOS LIMPIADOS:")
    for key, value in cleaned.items():
        print(f"   {key}: {repr(value)} (tipo: {type(value).__name__})")

    print(f"\n🔍 VERIFICACIÓN ESPECÍFICA:")
    print(f"   cleaned['reviewerID'] = {repr(cleaned['reviewerID'])}")
    print(f"   ¿Es igual a 'UNKNOWN'? {cleaned['reviewerID'] == 'UNKNOWN'}")
    print(f"   ¿Es igual a ''? {cleaned['reviewerID'] == ''}")

    # Verificar la función clean_review_data directamente
    print(f"\n🔧 VERIFICANDO LÓGICA DE LIMPIEZA:")
    reviewerID_input = invalid_review.get('reviewerID', 'UNKNOWN')
    print(f"   invalid_review.get('reviewerID', 'UNKNOWN') = {repr(reviewerID_input)}")

    # Simular la lógica
    if not reviewerID_input:  # Si es vacío
        result = 'UNKNOWN'
    else:
        result = reviewerID_input
    print(f"   Resultado esperado: {repr(result)}")


if __name__ == "__main__":
    debug_cleaner()