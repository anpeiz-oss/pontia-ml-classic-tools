"""
Respuesta documental: preguntas sobre los documentos procesados con OCR.

- Los documentos se leen de ocr/salida/documentos.json, generado offline por
  ocr/ocr.py. Aqui no se hace OCR.
- Son 5 documentos cortos (~3.500 caracteres en total), asi que se envian
  TODOS en el prompt (context stuffing). No hace falta base vectorial.
- Control de alucinaciones: el modelo debe contestar la marca NO_CONSTA si la
  respuesta no esta en los documentos, y el webhook la traduce a una frase
  explicita. Temperatura 0 y respuesta corta.
- Control de latencia: la latencia de Gemini es muy variable (en pruebas del
  12/11/2026 el mismo modelo tardo entre 1 y 16 s). Por eso:
    * la pregunta se lanza A LA VEZ a varios modelos (GEMINI_MODEL y
      GEMINI_MODELOS_RESPALDO) y se usa la primera respuesta valida;
    * al arrancar se hace una llamada de calentamiento a cada modelo, y
      despues una cada LLM_KEEPALIVE segundos, para que la conexion no se
      enfrie (la primera llamada en frio es la mas lenta);
    * hay un tiempo maximo total (LLM_TIMEOUT). Si se supera o fallan todos,
      se devuelve una frase de respaldo: Dialogflow corta a los 5 segundos y
      un webhook que no responde deja al usuario en silencio.
"""

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        self.timeout = float(os.environ.get("LLM_TIMEOUT", "4"))
        self.modelo = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
        respaldo = os.environ.get("GEMINI_MODELOS_RESPALDO", "gemini-3.1-flash-lite")
        self.modelos = [self.modelo] + [m.strip() for m in respaldo.split(",")
                                        if m.strip() and m.strip() != self.modelo]
        # preguntar(prompt, modelo) -> texto. Se puede inyectar en los tests.
        self._preguntar = preguntar or self._preguntar_gemini
        self._cliente = None
        self._hilos = ThreadPoolExecutor(max_workers=8)
        self._bloqueo = threading.Lock()

        keepalive = float(os.environ.get("LLM_KEEPALIVE", "60"))
        if preguntar is None and os.environ.get("GOOGLE_API_KEY") and keepalive > 0:
            threading.Thread(target=self._mantener_caliente, args=(keepalive,),
                             daemon=True).start()

    def _mantener_caliente(self, cada: float) -> None:
        """Llamada minima a cada modelo al arrancar y cada `cada` segundos."""
        while True:
            for modelo in self.modelos:
                inicio = time.perf_counter()
                try:
                    self._preguntar("Responde solo: ok", modelo)
                    print(f"LLM calentamiento {modelo}: {time.perf_counter() - inicio:.2f}s")
                except Exception as e:
                    print(f"LLM calentamiento {modelo} fallo: {str(e)[:120]}")
            time.sleep(cada)

    def _preguntar_gemini(self, prompt: str, modelo: str | None = None) -> str:
        # Import aqui para que los tests no necesiten la libreria ni la clave
        from google import genai
        from google.genai import types

        with self._bloqueo:
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
            model=modelo or self.modelo, contents=prompt, config=config
        )
        return (respuesta.text or "").strip()

    def responder(self, pregunta: str) -> str:
        if not self.documentos:
            return RESPUESTA_NO_CONSTA
        prompt = PLANTILLA.format(contexto=self.contexto, pregunta=pregunta)
        inicio = time.perf_counter()
        # Misma pregunta a todos los modelos a la vez; gana la primera respuesta
        futuros = {self._hilos.submit(self._preguntar, prompt, m): m for m in self.modelos}
        texto = None
        try:
            for futuro in as_completed(futuros, timeout=self.timeout):
                try:
                    texto = futuro.result()
                except Exception as e:
                    print(f"ERROR LLM ({futuros[futuro]}): {str(e)[:150]}")
                    continue
                print(f"LLM: {futuros[futuro]} respondio en {time.perf_counter() - inicio:.2f}s")
                break
        except TimeoutFuturo:
            print(f"LLM: ningun modelo respondio en {self.timeout}s, devuelvo respaldo")
        # Los que sigan en marcha terminan solos en segundo plano; no se esperan

        if texto is None:
            return RESPUESTA_ERROR

        if not texto or MARCA_NO_CONSTA in texto.upper():
            return RESPUESTA_NO_CONSTA
        return texto
