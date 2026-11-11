"""
Respuesta documental: preguntas sobre los documentos procesados con OCR.

- Los documentos se leen de ocr/salida/documentos.json, generado offline por
  ocr/ocr.py. Aqui no se hace OCR.
- Son 5 documentos cortos (~3.500 caracteres en total), asi que se envian
  TODOS en el prompt (context stuffing). No hace falta base vectorial.
- Control de alucinaciones: el modelo debe contestar la marca NO_CONSTA si la
  respuesta no esta en los documentos, y el webhook la traduce a una frase
  explicita. Temperatura 0 y respuesta corta.
- Control de latencia: la llamada al LLM tiene un tiempo maximo (LLM_TIMEOUT).
  Si se supera o falla, se devuelve una frase de respaldo: Dialogflow corta
  a los 5 segundos y un webhook que no responde deja al usuario en silencio.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as TimeoutFuturo
from pathlib import Path

FICHERO_DOCUMENTOS = Path(__file__).resolve().parent.parent / "ocr" / "salida" / "documentos.json"

MARCA_NO_CONSTA = "NO_CONSTA"

RESPUESTA_NO_CONSTA = (
    "Esa información no aparece en la documentación que tengo, así que prefiero "
    "no inventármela. ¿Quieres que te pase con un agente?"
)
RESPUESTA_ERROR = (
    "Ahora mismo no puedo consultar la documentación. ¿Quieres que te pase con un agente?"
)

INSTRUCCIONES = """Eres el asistente telefónico de Seguros Angel.
Respondes en español, en una o dos frases como mucho, porque tu respuesta se
convierte en voz. No uses listas, asteriscos ni emojis.
Solo puedes usar la información de los DOCUMENTOS que te da el usuario.
Los documentos vienen de un OCR y pueden tener errores sueltos de letras.
Si la respuesta no está en los documentos, o no estás seguro, responde
exactamente NO_CONSTA y nada más. Nunca completes con conocimiento propio."""

PLANTILLA = """DOCUMENTOS:
{contexto}

PREGUNTA DEL CLIENTE: {pregunta}"""


def cargar_documentos(fichero: Path = FICHERO_DOCUMENTOS) -> list[dict]:
    if not fichero.exists():
        print(f"AVISO: no existe {fichero}. Ejecuta 'python ocr/ocr.py' antes de arrancar.")
        return []
    datos = json.loads(fichero.read_text(encoding="utf-8"))
    return datos.get("documentos", [])


def construir_contexto(documentos: list[dict]) -> str:
    return "\n\n".join(
        f"[Documento: {d['fichero']}]\n{d['texto']}" for d in documentos
    )


class Documental:
    """Envuelve el cliente de Gemini. Se crea una vez al arrancar el webhook."""

    def __init__(self, documentos: list[dict] | None = None, preguntar=None):
        self.documentos = cargar_documentos() if documentos is None else documentos
        self.contexto = construir_contexto(self.documentos)
        self.timeout = float(os.environ.get("LLM_TIMEOUT", "3.5"))
        self.modelo = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
        self._preguntar = preguntar or self._preguntar_gemini
        self._cliente = None
        self._hilos = ThreadPoolExecutor(max_workers=4)

    def _preguntar_gemini(self, prompt: str) -> str:
        # Import aqui para que los tests no necesiten la libreria ni la clave
        from google import genai
        from google.genai import types

        if self._cliente is None:
            # Ojo: la API de Gemini rechaza timeouts HTTP de menos de 10 s, por
            # eso el limite de espera lo ponemos nosotros en responder().
            self._cliente = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        config = types.GenerateContentConfig(
            system_instruction=INSTRUCCIONES,
            temperature=0.0,
            max_output_tokens=200,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        respuesta = self._cliente.models.generate_content(
            model=self.modelo, contents=prompt, config=config
        )
        return (respuesta.text or "").strip()

    def responder(self, pregunta: str) -> str:
        if not self.documentos:
            return RESPUESTA_NO_CONSTA
        prompt = PLANTILLA.format(contexto=self.contexto, pregunta=pregunta)
        inicio = time.perf_counter()
        try:
            texto = self._hilos.submit(self._preguntar, prompt).result(timeout=self.timeout)
        except TimeoutFuturo:
            print(f"LLM: sin respuesta en {self.timeout}s, devuelvo respaldo")
            return RESPUESTA_ERROR
        except Exception as e:
            print("ERROR LLM:", e)
            return RESPUESTA_ERROR
        finally:
            print(f"LLM: {time.perf_counter() - inicio:.2f}s")

        if not texto or MARCA_NO_CONSTA in texto.upper():
            return RESPUESTA_NO_CONSTA
        return texto
