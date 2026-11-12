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
    * la pregunta se lanza A LA VEZ a los modelos de Gemini (GEMINI_MODEL y
      GEMINI_MODELOS_RESPALDO) y a un modelo local de Ollama (LLM_LOCAL);
    * se prefiere la respuesta de Gemini. Si ninguno de Gemini ha
      contestado en LLM_ESPERA_PREFERIDO segundos, o han fallado todos, se
      usa la respuesta local (qwen3:14b: 11/11 aciertos y 0,3 s de media en
      caliente en la comparativa del README);
    * al arrancar se hace una llamada de calentamiento a cada modelo, y
      despues una cada LLM_KEEPALIVE segundos, para que la conexion no se
      enfrie y Ollama no descargue el modelo de la GPU (la primera carga
      de un 14B tarda ~22 s);
    * hay un tiempo maximo total (LLM_TIMEOUT). Si se supera o fallan todos,
      se devuelve una frase de respaldo: Dialogflow corta a los 5 segundos y
      un webhook que no responde deja al usuario en silencio.
"""

import json
import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

FICHERO_DOCUMENTOS = Path(__file__).resolve().parent.parent / "ocr" / "salida" / "documentos.json"

MARCA_NO_CONSTA = "NO_CONSTA"
PREFIJO_LOCAL = "ollama:"

RESPUESTA_NO_CONSTA = (
    "Esa información no aparece en la documentación que tengo, así que prefiero "
    "no inventármela. ¿Quieres que te pase con un agente?"
)
RESPUESTA_ERROR = (
    "Ahora mismo no puedo consultar la documentación. ¿Quieres que te pase con un agente?"
)

INSTRUCCIONES = """Eres el asistente telefónico de Seguros Angel.
Respondes en español, en una o dos frases completas como mucho, porque tu
respuesta se convierte en voz. No uses listas, asteriscos ni emojis.
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


def es_local(modelo: str) -> bool:
    return modelo.startswith(PREFIJO_LOCAL)


class Documental:
    """Envuelve los modelos (Gemini y Ollama). Se crea una vez al arrancar el webhook."""

    def __init__(self, documentos: list[dict] | None = None, preguntar=None):
        self.documentos = cargar_documentos() if documentos is None else documentos
        self.contexto = construir_contexto(self.documentos)
        self.timeout = float(os.environ.get("LLM_TIMEOUT", "4"))
        self.espera_preferido = float(os.environ.get("LLM_ESPERA_PREFERIDO", "1.5"))
        self.modelo = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
        respaldo = os.environ.get("GEMINI_MODELOS_RESPALDO", "gemini-3.1-flash-lite")
        local = os.environ.get("LLM_LOCAL", "ollama:qwen3:14b").strip()
        self.ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
        self.modelos = [self.modelo] + [m.strip() for m in respaldo.split(",")
                                        if m.strip() and m.strip() != self.modelo]
        if local:
            self.modelos.append(local if es_local(local) else PREFIJO_LOCAL + local)
        # preguntar(prompt, modelo) -> texto. Se puede inyectar en los tests.
        self._preguntar = preguntar or self._preguntar_modelo
        self._cliente = None
        self._http = None
        # Pools separados: una llamada a Gemini puede quedarse colgada mas de un
        # minuto (visto en pruebas). Con un pool compartido, esos hilos ocupados
        # dejaban en cola la llamada al modelo local y se perdia el respaldo.
        self._hilos_remotos = ThreadPoolExecutor(max_workers=32)
        self._hilos_local = ThreadPoolExecutor(max_workers=2)
        self._bloqueo = threading.Lock()

        keepalive = float(os.environ.get("LLM_KEEPALIVE", "60"))
        if preguntar is None and keepalive > 0:
            # Un hilo por modelo: si Gemini tarda en el calentamiento, no
            # retrasa la carga del modelo local en la GPU
            for modelo in self.modelos:
                threading.Thread(target=self._mantener_caliente, args=(modelo, keepalive),
                                 daemon=True).start()

    # Llamadas a los modelos

    def _preguntar_modelo(self, prompt: str, modelo: str | None = None) -> str:
        modelo = modelo or self.modelo
        if es_local(modelo):
            return self._preguntar_ollama(prompt, modelo[len(PREFIJO_LOCAL):])
        return self._preguntar_gemini(prompt, modelo)

    def _preguntar_gemini(self, prompt: str, modelo: str) -> str:
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
            model=modelo, contents=prompt, config=config
        )
        return (respuesta.text or "").strip()

    def _preguntar_ollama(self, prompt: str, modelo: str) -> str:
        import httpx

        with self._bloqueo:
            if self._http is None:
                self._http = httpx.Client(timeout=30)
        r = self._http.post(f"{self.ollama_url}/api/chat", json={
            "model": modelo,
            "messages": [{"role": "system", "content": INSTRUCCIONES},
                         {"role": "user", "content": prompt}],
            "stream": False,
            # Los Qwen 3.x "piensan" antes de responder y eso anade segundos.
            # En modelos sin esa capacidad (qwen2.5) Ollama lo ignora.
            "think": False,
            "keep_alive": "60m",   # que no descargue el modelo de la GPU
            "options": {"temperature": 0, "num_predict": 150, "num_ctx": 4096},
        })
        r.raise_for_status()
        return r.json()["message"]["content"].strip()

    def _mantener_caliente(self, modelo: str, cada: float) -> None:
        """Llamada minima a un modelo al arrancar y cada `cada` segundos."""
        if not es_local(modelo) and not os.environ.get("GOOGLE_API_KEY"):
            return
        while True:
            inicio = time.perf_counter()
            try:
                self._preguntar("Responde solo: ok", modelo)
                print(f"LLM calentamiento {modelo}: {time.perf_counter() - inicio:.2f}s")
            except Exception as e:
                print(f"LLM calentamiento {modelo} fallo: {str(e)[:120]}")
            time.sleep(cada)

    # Eleccion de la respuesta

    def _elegir(self, resultados: dict, pendientes: set, transcurrido: float):
        """
        Devuelve (modelo, texto) si ya se puede responder, o None si hay que
        seguir esperando. Preferencia: cualquier modelo remoto (Gemini) en
        cuanto conteste; el local solo si los remotos tardan mas de
        espera_preferido o han fallado todos.
        """
        remotos = [m for m in self.modelos if not es_local(m)]
        for m in remotos:                      # primer remoto con respuesta
            if resultados.get(m) is not None:
                return m, resultados[m]
        remotos_vivos = [m for m in remotos if m in pendientes]
        if remotos_vivos and transcurrido < self.espera_preferido:
            return None                        # aun merece la pena esperar a Gemini
        for m in self.modelos:                 # local (o lo que haya)
            if resultados.get(m) is not None:
                return m, resultados[m]
        return None

    def responder(self, pregunta: str) -> str:
        if not self.documentos:
            return RESPUESTA_NO_CONSTA
        prompt = PLANTILLA.format(contexto=self.contexto, pregunta=pregunta)
        inicio = time.perf_counter()

        futuros = {
            (self._hilos_local if es_local(m) else self._hilos_remotos).submit(self._preguntar, prompt, m): m
            for m in self.modelos
        }
        pendientes = set(futuros)
        resultados: dict[str, str | None] = {}
        eleccion = None

        while True:
            transcurrido = time.perf_counter() - inicio
            eleccion = self._elegir(resultados, {futuros[f] for f in pendientes}, transcurrido)
            if eleccion or not pendientes or transcurrido >= self.timeout:
                break
            # Esperar hasta la siguiente respuesta o el siguiente "hito" de tiempo
            hito = self.espera_preferido if transcurrido < self.espera_preferido else self.timeout
            hechos, pendientes = wait(pendientes, timeout=max(hito - transcurrido, 0.01),
                                      return_when=FIRST_COMPLETED)
            for f in hechos:
                try:
                    resultados[futuros[f]] = f.result()
                except Exception as e:
                    resultados[futuros[f]] = None
                    print(f"ERROR LLM ({futuros[f]}): {str(e)[:150]}")
        # Los que sigan en marcha terminan solos en segundo plano; no se esperan

        if not eleccion:
            print(f"LLM: ningun modelo respondio en {self.timeout}s, devuelvo respaldo")
            return RESPUESTA_ERROR
        modelo, texto = eleccion
        print(f"LLM: respuesta de {modelo} en {time.perf_counter() - inicio:.2f}s")

        if not texto or MARCA_NO_CONSTA in texto.upper():
            return RESPUESTA_NO_CONSTA
        return texto
