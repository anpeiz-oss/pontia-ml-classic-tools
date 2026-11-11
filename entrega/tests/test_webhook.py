"""
Pruebas del webhook simulando las peticiones que envia Dialogflow ES.
No llaman al LLM: se sustituye por una funcion falsa.

    pytest            (desde la carpeta entrega/)
"""

import os

os.environ["SSML"] = "0"  # en los tests comprobamos el texto plano

import pytest
from fastapi.testclient import TestClient

from webhook import main

SESION = "projects/proyecto-aseguradora-api/agent/sessions/test-123"
CTX = f"{SESION}/contexts/ctx_grua_en_curso"

cliente = TestClient(main.app)


def peticion(intencion, parametros=None, texto="", completos=True, contextos=None):
    return {
        "session": SESION,
        "queryResult": {
            "queryText": texto,
            "parameters": parametros or {},
            "allRequiredParamsPresent": completos,
            "intent": {"displayName": intencion},
            "outputContexts": contextos or [],
        },
    }


def enviar(*args, **kwargs) -> dict:
    r = cliente.post("/webhook", json=peticion(*args, **kwargs))
    assert r.status_code == 200
    return r.json()


def ctx_salida(respuesta):
    return next(c for c in respuesta["outputContexts"] if c["name"] == CTX)


# Flujo con contexto

def test_grua_faltan_parametros_deja_hablar_a_dialogflow():
    assert enviar("solicitar_grua", {"matricula": "1234BCD"}, completos=False) == {}


def test_grua_matricula_con_formato_invalido():
    r = enviar("solicitar_grua", {"matricula": "12AB", "ubicacion": "Madrid"})
    assert "formato válido" in r["fulfillmentText"]
    assert ctx_salida(r)["parameters"]["estado"] == "esperando_matricula"


def test_grua_matricula_hablada_se_normaliza():
    r = enviar("solicitar_grua", {"matricula": "1234 be ce de", "ubicacion": "Madrid"})
    assert "Seat León" in r["fulfillmentText"]


def test_grua_matricula_inexistente():
    r = enviar("solicitar_grua", {"matricula": "9999XYZ", "ubicacion": "Madrid"})
    assert "No encuentro" in r["fulfillmentText"]


def test_grua_sin_cobertura_cierra_contexto():
    r = enviar("solicitar_grua", {"matricula": "5678 FGH", "ubicacion": "Sevilla"})
    assert "no incluye asistencia" in r["fulfillmentText"]
    assert ctx_salida(r)["lifespanCount"] == 0


def test_grua_camino_bueno_pide_confirmacion():
    r = enviar("solicitar_grua", {"matricula": "1234-bcd", "ubicacion": "Madrid"})
    assert "¿Confirmas" in r["fulfillmentText"]
    ctx = ctx_salida(r)
    assert ctx["lifespanCount"] == 5
    assert ctx["parameters"] == {"matricula": "1234BCD", "ubicacion": "Madrid",
                                 "estado": "pendiente_confirmar"}


def test_confirmacion_si_abre_aviso_y_cierra_contexto():
    guardado = [{"name": CTX, "parameters": {"matricula": "1234BCD", "ubicacion": "Madrid",
                                              "estado": "pendiente_confirmar"}}]
    r = enviar("solicitar_grua-yes", contextos=guardado)
    assert "AV-" in r["fulfillmentText"] and "30 minutos" in r["fulfillmentText"]
    assert ctx_salida(r)["lifespanCount"] == 0


def test_confirmacion_no_mantiene_contexto_para_corregir():
    guardado = [{"name": CTX, "parameters": {"matricula": "1234BCD", "ubicacion": "Madrid",
                                              "estado": "pendiente_confirmar"}}]
    r = enviar("solicitar_grua-no", contextos=guardado)
    assert "no pido la grúa" in r["fulfillmentText"]
    assert ctx_salida(r)["parameters"]["estado"] == "corrigiendo"


def test_corregir_matricula_a_mitad_de_flujo():
    guardado = [{"name": CTX, "parameters": {"matricula": "", "ubicacion": "Valencia",
                                              "estado": "esperando_matricula"}}]
    r = enviar("corregir_dato", {"matricula": "9012JKL"}, contextos=guardado)
    assert "Toyota Corolla" in r["fulfillmentText"] and "Valencia" in r["fulfillmentText"]
    assert ctx_salida(r)["parameters"]["estado"] == "pendiente_confirmar"


def test_corregir_ciudad_a_mitad_de_flujo():
    guardado = [{"name": CTX, "parameters": {"matricula": "1234BCD", "ubicacion": "Madrid",
                                              "estado": "corrigiendo"}}]
    r = enviar("corregir_dato", {"ubicacion": "Barcelona"}, contextos=guardado)
    assert "Barcelona" in r["fulfillmentText"]


# Consultas

def test_consultar_coberturas():
    r = enviar("consultar_coberturas", {"matricula": "9012JKL"})
    assert "terceros ampliado" in r["fulfillmentText"]


def test_consultar_poliza_franquicia_y_prima():
    assert "300 euros" in enviar("consultar_poliza", {"matricula": "1234BCD",
                                                      "dato_poliza": "franquicia"})["fulfillmentText"]
    assert "289,90 euros" in enviar("consultar_poliza", {"matricula": "5678FGH",
                                                         "dato_poliza": "prima"})["fulfillmentText"]


def test_hablar_con_agente():
    assert "número de caso" in enviar("hablar_con_agente")["fulfillmentText"]


def test_intencion_desconocida_siempre_responde():
    assert enviar("intencion_rara")["fulfillmentText"]


# Respuesta documental

@pytest.fixture
def llm_falso(monkeypatch):
    llamadas = []

    def falso(prompt):
        llamadas.append(prompt)
        return "NO_CONSTA" if "mascota" in prompt else "El remolque cubre hasta 50 km."

    monkeypatch.setattr(main.documental, "_preguntar", falso)
    return llamadas


def test_pregunta_documental_envia_los_documentos(llm_falso):
    r = enviar("consultar_condiciones", texto="¿Cuántos km cubre el remolque?")
    assert "50 km" in r["fulfillmentText"]
    assert "[Documento: 02_condicionado_asistencia.jpg]" in llm_falso[0]


def test_pregunta_sin_respuesta_no_inventa(llm_falso):
    r = enviar("Default Fallback Intent", texto="¿Cubrís a mi mascota?")
    assert "no aparece en la documentación" in r["fulfillmentText"]


def test_llm_lento_devuelve_respaldo(monkeypatch):
    import time
    monkeypatch.setattr(main.documental, "timeout", 0.2)
    monkeypatch.setattr(main.documental, "_preguntar", lambda p: time.sleep(1) or "tarde")
    r = enviar("consultar_condiciones", texto="¿Plazo para dar un parte?")
    assert "no puedo consultar" in r["fulfillmentText"]
