"""
Webhook de fulfillment para el agente de Dialogflow ES.

Arrancar:
    python -m venv .venv
    source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
    pip install fastapi uvicorn
    uvicorn main:app --reload --port 8000

Luego exponer el puerto 8000 a internet (VS Code PORTS -> Public, ngrok,
Cloudflare Tunnel...) y pegar la URL + /webhook en Dialogflow -> Fulfillment.
Recuerda activar "Enable webhook call for this intent" en cada intencion.
"""

from fastapi import FastAPI, Request
import random

app = FastAPI()

# Base de datos falsa. En un proyecto real, aqui iria una consulta a
# una base de datos o una llamada a otra API.
POLIZAS = {
    "1234BCD": {"titular": "Ana Ruiz", "grua": True},
    "5678FGH": {"titular": "Luis Gómez", "grua": False},
}


def responder(texto: str) -> dict:
    """Formato minimo que espera Dialogflow ES."""
    return {"fulfillmentText": texto}


def gestionar_grua(parametros: dict) -> dict:
    # Normalizamos: el usuario puede decir "1234 bcd" o "1234-BCD"
    matricula = (
        parametros.get("matricula", "")
        .upper()
        .replace(" ", "")
        .replace("-", "")
    )
    ubicacion = parametros.get("ubicacion", "")

    poliza = POLIZAS.get(matricula)

    # Camino de error 1: la matricula no existe
    if poliza is None:
        return responder(
            f"No encuentro ninguna póliza con la matrícula {matricula}. "
            "Puede que la haya entendido mal. ¿Me la repites?"
        )

    # Camino de error 2: existe pero no tiene la cobertura contratada
    if not poliza["grua"]:
        return responder(
            f"{poliza['titular']}, tu póliza no incluye asistencia en carretera. "
            "¿Quieres que te pase con un agente para contratarla?"
        )

    # Camino bueno
    aviso = random.randint(10000, 99999)
    return responder(
        f"Perfecto {poliza['titular']}. He abierto el aviso número {aviso} "
        f"para enviar una grúa a {ubicacion} por la matrícula {matricula}. "
        "¿Confirmas que los datos son correctos?"
    )


@app.post("/webhook")
async def webhook(request: Request):
    body = await request.json()

    intencion = body["queryResult"]["intent"]["displayName"]
    parametros = body["queryResult"]["parameters"]

    # Imprimimos siempre lo que llega que será útil para depurar luego
    print("INTENCION:", intencion)
    print("PARAMETROS:", parametros)

    if intencion == "solicitar_grua":
        # Con "webhook call for slot filling" nos llaman en cada turno.
        # Si aun faltan parametros, respuesta vacia: Dialogflow usa sus prompts.
        if not body["queryResult"].get("allRequiredParamsPresent", False):
            return {}
        return gestionar_grua(parametros)

    if intencion == "Default Fallback Intent":
        return gestionar_fallback(body["queryResult"]["queryText"])

    # Nunca dejar al agente sin respuesta
    return responder("Todavía no sé gestionar esa petición.")

# Fallback con LLM
# Requiere:  pip install google-genai
# y una variable de entorno GOOGLE_API_KEY (ponla en un .env)
# El dia 2 este mismo bloque recibira ademas el texto extraido por OCR

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()  # carga GOOGLE_API_KEY desde el .env

cliente = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

INSTRUCCIONES = """Eres el asistente telefónico de Seguros Alonso.
Respondes en español, en dos o tres frases como mucho, porque tu respuesta
se va a convertir en voz. No inventes datos concretos de pólizas, precios ni
coberturas. Si no sabes algo, dilo claramente y ofrece pasar con un agente."""  

def gestionar_fallback(pregunta: str) -> dict:
    try:
        respuesta = cliente.models.generate_content(
            model="gemini-2.5-flash",
            contents=pregunta,
            config=types.GenerateContentConfig(
                system_instruction=INSTRUCCIONES,
                temperature=0.1,
                max_output_tokens=150,
                # Esto es para no dejar al modelo llamar a funciones de python por su cuenta
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            ),
        )
        return responder(respuesta.text.strip())
    except Exception as e:
        # Dialogflow corta a los 5 segundos: pase lo que pase, devolvemos algo
        print("ERROR LLM:", e)
        return responder(
            "Ahora mismo no puedo responder a eso. ¿Quieres que te pase con un agente?"
        )