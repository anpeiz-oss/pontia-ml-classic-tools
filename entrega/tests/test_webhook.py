"""
Pruebas del webhook simulando las peticiones que envia Dialogflow ES.
No llaman al LLM: se sustituye por una funcion falsa.

    pytest            (desde la carpeta entrega/)
"""

import os

os.environ["SSML"] = "0"  # en los tests comprobamos el texto plano
os.environ["LLM_KEEPALIVE"] = "0"  # sin llamadas reales de calentamiento

import pytest
from fastapi.testclient import TestClient

from webhook import main

SESION = "projects/proyecto-aseguradora-api/agent/sessions/test-123"
CTX = f"{SESION}/contexts/solicitar_grua-followup"

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


def ctx_entrada(grua_matricula, grua_ubicacion, grua_estado, **de_dialogflow):
    """Contexto tal como lo reenvia Dialogflow: lo nuestro (grua_*) + los
    parametros de la intencion detectada, que Dialogflow copia encima."""
    return [{"name": CTX, "parameters": {"grua_matricula": grua_matricula,
                                         "grua_ubicacion": grua_ubicacion,
                                         "grua_estado": grua_estado, **de_dialogflow}}]


# Flujo con contexto

def test_grua_faltan_parametros_deja_hablar_a_dialogflow():
    assert enviar("solicitar_grua", {"matricula": "1234BCD"}, completos=False) == {}


def test_grua_matricula_con_formato_invalido():
    r = enviar("solicitar_grua", {"matricula": "12AB", "ubicacion": "Madrid"})
    assert "formato válido" in r["fulfillmentText"]
    assert ctx_salida(r)["parameters"]["grua_estado"] == "esperando_matricula"


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
    assert ctx["parameters"] == {"grua_matricula": "1234BCD", "grua_ubicacion": "Madrid",
                                 "grua_estado": "pendiente_confirmar"}


def test_confirmacion_si_abre_aviso_y_cierra_contexto():
    guardado = ctx_entrada("1234BCD", "Madrid", "pendiente_confirmar")
    r = enviar("solicitar_grua-si", contextos=guardado)
    assert "AV-" in r["fulfillmentText"] and "30 minutos" in r["fulfillmentText"]
    assert ctx_salida(r)["lifespanCount"] == 0


def test_confirmacion_no_mantiene_contexto_para_corregir():
    guardado = ctx_entrada("1234BCD", "Madrid", "pendiente_confirmar")
    r = enviar("solicitar_grua-no", contextos=guardado)
    assert "no pido la grúa" in r["fulfillmentText"]
    assert ctx_salida(r)["parameters"]["grua_estado"] == "corrigiendo"


def test_corregir_matricula_a_mitad_de_flujo():
    guardado = ctx_entrada("", "Valencia", "esperando_matricula")
    r = enviar("corregir_dato", {"matricula": "9012JKL"}, contextos=guardado)
    assert "Toyota Corolla" in r["fulfillmentText"] and "Valencia" in r["fulfillmentText"]
    assert ctx_salida(r)["parameters"]["grua_estado"] == "pendiente_confirmar"


def test_corregir_ciudad_cuando_dialogflow_vacia_la_matricula_del_contexto():
    # Caso real (12/11/2026): tras "no", el usuario dice "estoy en Barcelona".
    # Dialogflow copia en el contexto matricula="" y ubicacion como lista.
    guardado = ctx_entrada("1234BCD", "Madrid", "corrigiendo",
                           matricula="", ubicacion="Barcelona")
    r = enviar("corregir_dato", {"matricula": "", "ubicacion": ["Barcelona"]}, contextos=guardado)
    assert "Seat León" in r["fulfillmentText"] and "Barcelona" in r["fulfillmentText"]
    assert ctx_salida(r)["parameters"]["grua_matricula"] == "1234BCD"


def test_corregir_ciudad_a_mitad_de_flujo():
    guardado = ctx_entrada("1234BCD", "Madrid", "corrigiendo")
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

    def falso(prompt, modelo=None):
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


def test_gana_el_modelo_que_responde_primero(monkeypatch):
    import time

    def por_modelo(prompt, modelo=None):
        if modelo == "principal":
            time.sleep(2)          # el principal se atasca
            return "respuesta lenta"
        return "El plazo para dar el parte es de 7 días."

    monkeypatch.setattr(main.documental, "modelos", ["principal", "respaldo"])
    monkeypatch.setattr(main.documental, "_preguntar", por_modelo)
    inicio = time.perf_counter()
    r = enviar("consultar_condiciones", texto="¿Plazo para dar un parte?")
    assert "7 días" in r["fulfillmentText"]
    assert time.perf_counter() - inicio < 1.5


def test_si_un_modelo_falla_responde_el_otro(monkeypatch):
    def por_modelo(prompt, modelo=None):
        if modelo == "principal":
            raise RuntimeError("503 UNAVAILABLE")
        return "El remolque cubre hasta 50 km."

    monkeypatch.setattr(main.documental, "modelos", ["principal", "respaldo"])
    monkeypatch.setattr(main.documental, "_preguntar", por_modelo)
    r = enviar("consultar_condiciones", texto="¿Cuántos km de remolque?")
    assert "50 km" in r["fulfillmentText"]


def test_prefiere_gemini_si_contesta_a_tiempo(monkeypatch):
    import time

    def por_modelo(prompt, modelo=None):
        if modelo == "ollama:local":
            return "respuesta local"
        time.sleep(0.3)
        return "respuesta de gemini"

    monkeypatch.setattr(main.documental, "modelos", ["gemini", "ollama:local"])
    monkeypatch.setattr(main.documental, "espera_preferido", 1.0)
    monkeypatch.setattr(main.documental, "_preguntar", por_modelo)
    r = enviar("consultar_condiciones", texto="¿Plazo para dar un parte?")
    assert r["fulfillmentText"] == "respuesta de gemini"


def test_usa_el_modelo_local_si_gemini_tarda(monkeypatch):
    import time

    def por_modelo(prompt, modelo=None):
        if modelo == "ollama:local":
            return "El plazo es de 7 días."
        time.sleep(3)
        return "demasiado tarde"

    monkeypatch.setattr(main.documental, "modelos", ["gemini", "ollama:local"])
    monkeypatch.setattr(main.documental, "espera_preferido", 0.5)
    monkeypatch.setattr(main.documental, "timeout", 2.0)
    monkeypatch.setattr(main.documental, "_preguntar", por_modelo)
    inicio = time.perf_counter()
    r = enviar("consultar_condiciones", texto="¿Plazo para dar un parte?")
    assert r["fulfillmentText"] == "El plazo es de 7 días."
    assert time.perf_counter() - inicio < 1.0


def test_usa_el_local_en_cuanto_gemini_falla(monkeypatch):
    def por_modelo(prompt, modelo=None):
        if modelo == "ollama:local":
            return "El remolque cubre hasta 50 km."
        raise RuntimeError("503 UNAVAILABLE")

    monkeypatch.setattr(main.documental, "modelos", ["gemini", "ollama:local"])
    monkeypatch.setattr(main.documental, "espera_preferido", 5.0)
    monkeypatch.setattr(main.documental, "_preguntar", por_modelo)
    r = enviar("consultar_condiciones", texto="¿Km de remolque?")
    assert "50 km" in r["fulfillmentText"]


def test_gemini_colgado_no_bloquea_al_modelo_local(monkeypatch):
    # Regresion: con un pool compartido, varias llamadas a Gemini colgadas
    # dejaban en cola la del modelo local y se perdia el respaldo.
    import threading
    import time
    bloqueo = threading.Event()

    def por_modelo(prompt, modelo=None):
        if modelo == "ollama:local":
            return "El plazo es de 7 días."
        bloqueo.wait(5)            # Gemini colgado
        return "tarde"

    monkeypatch.setattr(main.documental, "modelos", ["gemini", "ollama:local"])
    monkeypatch.setattr(main.documental, "espera_preferido", 0.2)
    monkeypatch.setattr(main.documental, "_preguntar", por_modelo)
    try:
        for _ in range(40):        # mas preguntas que hilos remotos
            r = enviar("consultar_condiciones", texto="¿Plazo para dar un parte?")
            assert r["fulfillmentText"] == "El plazo es de 7 días."
    finally:
        bloqueo.set()


def test_llm_lento_devuelve_respaldo(monkeypatch):
    import time
    monkeypatch.setattr(main.documental, "timeout", 0.2)
    monkeypatch.setattr(main.documental, "espera_preferido", 0.1)
    monkeypatch.setattr(main.documental, "_preguntar", lambda p, m=None: time.sleep(1) or "tarde")
    r = enviar("consultar_condiciones", texto="¿Plazo para dar un parte?")
    assert "no puedo consultar" in r["fulfillmentText"]
