"""
Webhook de fulfillment del agente "Seguros Angel" (Dialogflow ES).

Arrancar (desde la carpeta entrega/):
    python ocr/ocr.py                       # una sola vez: genera ocr/salida/documentos.json
    uvicorn webhook.main:app --port 8000

Exponer a internet (por ejemplo con Cloudflare Tunnel):
    cloudflared tunnel --url http://localhost:8000
y pegar la URL publica + /webhook en Dialogflow -> Fulfillment.

Intenciones que gestiona (ver dialogflow/GUIA_CONSOLA.md):
    solicitar_grua            flujo con contexto solicitar_grua-followup (slot filling)
    solicitar_grua-si         confirmacion: abre el aviso
    solicitar_grua-no         negacion: ofrece corregir
    corregir_dato             correccion de matricula o ciudad a mitad de flujo
    consultar_coberturas      coberturas de la poliza de una matricula
    consultar_poliza          vencimiento, franquicia, prima o modalidad
    consultar_condiciones     preguntas sobre los documentos (OCR + LLM)
    hablar_con_agente         transferencia a agente humano
    despedida
    Default Welcome Intent
    Default Fallback Intent   tambien va a la respuesta documental
"""

import os
import re
import time

from dotenv import load_dotenv
from fastapi import FastAPI, Request

from webhook import negocio
from webhook.rag import Documental

load_dotenv()

# Es el contexto que crea Dialogflow al anadir los follow-up si/no a
# solicitar_grua. Reutilizarlo mantiene la anidacion visual en la consola
# y sigue siendo el unico contexto del agente.
CONTEXTO = "solicitar_grua-followup"
# Con SSML=1 la respuesta va en SSML para el TTS de Dialogflow. Ponlo a 0 si
# pruebas en el simulador o en Dialogflow Messenger y ves las etiquetas.
USAR_SSML = os.environ.get("SSML", "1") == "1"

app = FastAPI(title="Webhook Seguros Angel")

print("Cargando documentos de ocr/salida/documentos.json ...")
documental = Documental()
print(f"Documentos cargados: {len(documental.documentos)}")


# Utilidades de formato Dialogflow ES

def responder(texto: str, ssml: str | None = None, contextos: list | None = None) -> dict:
    cuerpo = {"fulfillmentText": ssml if (USAR_SSML and ssml) else texto}
    if contextos is not None:
        cuerpo["outputContexts"] = contextos
    return cuerpo


def contexto(sesion: str, lifespan: int, **parametros) -> dict:
    return {
        "name": f"{sesion}/contexts/{CONTEXTO}",
        "lifespanCount": lifespan,
        "parameters": parametros,
    }


def parametros_contexto(resultado: dict) -> dict:
    """Parametros guardados en solicitar_grua-followup (si esta activo)."""
    for ctx in resultado.get("outputContexts", []):
        if ctx.get("name", "").endswith(f"/contexts/{CONTEXTO}"):
            return ctx.get("parameters", {})
    return {}


def texto_parametro(valor) -> str:
    """@sys.geo-city puede llegar como texto, lista o diccionario."""
    if isinstance(valor, list):
        valor = valor[0] if valor else ""
    if isinstance(valor, dict):
        valor = valor.get("city") or next(iter(valor.values()), "")
    return str(valor or "").strip()


def a_ssml(texto: str) -> str:
    """Para respuestas del LLM: pausas tras cada frase."""
    texto = re.sub(r"[<>&]", " ", texto)
    return "<speak>" + re.sub(r"([.?!])\s+", r'\1 <break time="250ms"/> ', texto) + "</speak>"


# Flujo con contexto: grua

def paso_grua(sesion: str, matricula_cruda, ubicacion: str) -> dict:
    """Valida y pide confirmacion. Lo usan solicitar_grua y corregir_dato."""
    matricula = negocio.normalizar_matricula(matricula_cruda)
    r = negocio.preparar_grua(matricula, ubicacion)

    if r["estado"] == "pendiente_confirmar":
        ctx = contexto(sesion, 5, matricula=matricula, ubicacion=ubicacion,
                       estado="pendiente_confirmar")
    elif r["estado"] in ("invalida", "sin_poliza"):
        # Mantenemos el contexto para que el usuario solo tenga que repetir la matricula
        ctx = contexto(sesion, 3, matricula="", ubicacion=ubicacion,
                       estado="esperando_matricula")
    else:  # sin_cobertura: el flujo termina
        ctx = contexto(sesion, 0)
    return responder(r["texto"], r["ssml"], [ctx])


def gestionar_solicitar_grua(sesion: str, resultado: dict) -> dict:
    # Con "webhook call for slot filling" nos llaman en cada turno.
    # Si faltan parametros, respuesta vacia: Dialogflow usa sus prompts.
    if not resultado.get("allRequiredParamsPresent", False):
        return {}
    p = resultado.get("parameters", {})
    return paso_grua(sesion, p.get("matricula"), texto_parametro(p.get("ubicacion")))


def gestionar_corregir_dato(sesion: str, resultado: dict) -> dict:
    guardado = parametros_contexto(resultado)
    nuevos = resultado.get("parameters", {})
    matricula = nuevos.get("matricula") or guardado.get("matricula", "")
    ubicacion = texto_parametro(nuevos.get("ubicacion")) or texto_parametro(guardado.get("ubicacion"))
    if not ubicacion:
        return responder("¿En qué ciudad estás?", None,
                         [contexto(sesion, 3, matricula=negocio.normalizar_matricula(matricula),
                                   ubicacion="", estado="esperando_ciudad")])
    return paso_grua(sesion, matricula, ubicacion)


def gestionar_confirmacion(sesion: str, resultado: dict) -> dict:
    guardado = parametros_contexto(resultado)
    if guardado.get("estado") != "pendiente_confirmar":
        return responder("Antes necesito una matrícula válida. ¿Me la dices?", None,
                         [contexto(sesion, 3, **guardado)])
    texto, ssml = negocio.confirmar_grua(guardado["matricula"],
                                         texto_parametro(guardado.get("ubicacion")))
    return responder(texto, ssml, [contexto(sesion, 0)])


def gestionar_negacion(sesion: str, resultado: dict) -> dict:
    guardado = parametros_contexto(resultado)
    texto, ssml = negocio.cancelar_grua()
    return responder(texto, ssml, [contexto(sesion, 3, **{**guardado, "estado": "corrigiendo"})])


# Consultas de poliza (un solo turno)

def gestionar_consultar_coberturas(resultado: dict) -> dict:
    if not resultado.get("allRequiredParamsPresent", False):
        return {}
    matricula = negocio.normalizar_matricula(resultado["parameters"].get("matricula"))
    return responder(*negocio.consultar_coberturas(matricula))


def gestionar_consultar_poliza(resultado: dict) -> dict:
    if not resultado.get("allRequiredParamsPresent", False):
        return {}
    p = resultado["parameters"]
    matricula = negocio.normalizar_matricula(p.get("matricula"))
    return responder(*negocio.consultar_poliza(matricula, texto_parametro(p.get("dato_poliza"))))


def gestionar_documental(pregunta: str) -> dict:
    texto = documental.responder(pregunta)
    return responder(texto, a_ssml(texto))


BIENVENIDA = (
    "Hola, soy el asistente de Seguros Angel. Puedo pedirte una grúa, decirte qué cubre "
    "tu póliza o resolver dudas sobre nuestras condiciones. ¿Qué necesitas?",
    "<speak>Hola, <break time=\"200ms\"/> soy el asistente de Seguros Angel. "
    "<break time=\"300ms\"/> Puedo pedirte una grúa, decirte qué cubre tu póliza "
    "o resolver dudas sobre nuestras condiciones. <break time=\"300ms\"/> "
    "¿Qué necesitas?</speak>",
)


# Endpoint

@app.post("/webhook")
async def webhook(request: Request):
    inicio = time.perf_counter()
    body = await request.json()
    sesion = body.get("session", "")
    resultado = body.get("queryResult", {})
    intencion = resultado.get("intent", {}).get("displayName", "")
    pregunta = resultado.get("queryText", "")

    print("-" * 50)
    print("INTENCION:", intencion)
    print("PARAMETROS:", resultado.get("parameters"))
    print("TEXTO:", pregunta)

    if intencion == "solicitar_grua":
        respuesta = gestionar_solicitar_grua(sesion, resultado)
    elif intencion == "solicitar_grua-si":
        respuesta = gestionar_confirmacion(sesion, resultado)
    elif intencion == "solicitar_grua-no":
        respuesta = gestionar_negacion(sesion, resultado)
    elif intencion == "corregir_dato":
        respuesta = gestionar_corregir_dato(sesion, resultado)
    elif intencion == "consultar_coberturas":
        respuesta = gestionar_consultar_coberturas(resultado)
    elif intencion == "consultar_poliza":
        respuesta = gestionar_consultar_poliza(resultado)
    elif intencion in ("consultar_condiciones", "Default Fallback Intent"):
        respuesta = gestionar_documental(pregunta)
    elif intencion == "hablar_con_agente":
        respuesta = responder(*negocio.hablar_con_agente(), [contexto(sesion, 0)])
    elif intencion == "despedida":
        respuesta = responder(*negocio.despedida(), [contexto(sesion, 0)])
    elif intencion == "Default Welcome Intent":
        respuesta = responder(*BIENVENIDA)
    else:
        # Nunca dejar al agente sin respuesta
        respuesta = responder("Todavía no sé gestionar esa petición. ¿Quieres que te pase con un agente?")

    print(f"RESPUESTA ({time.perf_counter() - inicio:.2f}s):", respuesta.get("fulfillmentText"))
    return respuesta


@app.get("/")
def raiz():
    """Para comprobar de un vistazo que el tunel llega hasta aqui."""
    return {"estado": "ok", "documentos_cargados": len(documental.documentos),
            "modelo": documental.modelo, "ssml": USAR_SSML}
