"""Pruebas de la validacion con regex del OCR (no necesitan Tesseract)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ocr"))
pytest.importorskip("pytesseract")
import ocr  # noqa: E402


def test_extrae_y_valida_campos():
    texto = ("Poliza AB-456789 Tomador NIF: 12345678Z Matricula: 1234 BCD\n"
             "Vencimiento: 14/03/2027 Prima: 412,35 EUR Ref CG-AUTO-2026/03 tel 900 123 456")
    c = ocr.extraer_campos(texto)
    assert c["polizas"] == ["AB-456789"]
    assert c["nifs"] == ["12345678Z"]
    assert c["matriculas"] == ["1234BCD"]
    assert c["fechas"] == ["2027-03-14"]
    assert c["importes_eur"] == [412.35]
    assert c["referencias"] == ["CG-AUTO-2026/03"]
    assert c["telefonos"] == ["900123456"]


def test_descarta_lo_que_la_regex_encuentra_pero_no_es_valido():
    c = ocr.extraer_campos("NIF 12345678A, fecha 31/02/2026")
    assert c["nifs"] == [] and c["fechas"] == []
    assert set(c["descartados_por_validacion"]) == {"12345678A", "31/02/2026"}


def test_corrige_confusiones_del_ocr_solo_junto_a_numeros():
    # "2O26" -> "2026", pero "Otros" se queda igual
    assert ocr.normalizar_para_regex("Otros 14/03/2O26") == "Otros 14/03/2026"
