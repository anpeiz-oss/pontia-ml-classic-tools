"""
Cliente de voz: graba por micrófono, lo manda a Dialogflow (STT nativo) y
reproduce la respuesta hablada (TTS nativo). Basado en ejer2/cliente_voz.py.

Cambios respecto al de clase:
  - Voz configurable (DIALOGFLOW_VOZ) y velocidad, para que suene natural.
  - Muestra el texto sin etiquetas SSML y mide la latencia de cada turno.
  - Modo texto (--texto) para probar el flujo sin micrófono.

ANTES DE EJECUTAR:
  1. Cuenta de servicio de Google Cloud con el rol "Dialogflow API Client" y su
     clave JSON descargada. El JSON es una contraseña: fuera del repositorio.
  2. En entrega/.env:
       DIALOGFLOW_PROJECT_ID=proyecto-aseguradora-api
       GOOGLE_APPLICATION_CREDENTIALS=C:\\ruta\\clave.json

Uso (desde la carpeta entrega/):
    python voz/cliente_voz.py               # Enter para hablar, q para salir
    python voz/cliente_voz.py --texto       # escribe en vez de hablar
    python voz/cliente_voz.py --microfonos  # lista los microfonos disponibles

Para usar un microfono distinto del predeterminado de Windows, pon en el .env
MICROFONO=<numero de la lista> o MICROFONO=<parte del nombre>.
"""

import os
import re
import sys
import time
import uuid
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from google.cloud import dialogflow

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

PROJECT_ID = os.environ.get("DIALOGFLOW_PROJECT_ID", "proyecto-aseguradora-api")
IDIOMA = "es"
VOZ = os.environ.get("DIALOGFLOW_VOZ", "es-ES-Neural2-F")
FS = 16000          # Dialogflow espera 16 kHz para LINEAR_16
SEGUNDOS = int(os.environ.get("SEGUNDOS_GRABACION", "5"))
MICROFONO = os.environ.get("MICROFONO", "").strip()
NIVEL_MINIMO = 300  # pico de amplitud (int16) por debajo del cual es silencio
FICHERO_AUDIO = Path(__file__).resolve().parent / "respuesta.wav"

# Un SESSION_ID por conversación, no por frase: de él cuelgan los contextos.
SESSION_ID = str(uuid.uuid4())

clientes = dialogflow.SessionsClient()
sesion = clientes.session_path(PROJECT_ID, SESSION_ID)

SALIDA_AUDIO = dialogflow.OutputAudioConfig(
    audio_encoding=dialogflow.OutputAudioEncoding.OUTPUT_AUDIO_ENCODING_LINEAR_16,
    synthesize_speech_config=dialogflow.SynthesizeSpeechConfig(
        speaking_rate=1.0,
        voice=dialogflow.VoiceSelectionParams(name=VOZ),
    ),
)


# Grabación y reproducción

def microfono():
    """Indice del microfono elegido en MICROFONO, o None para el predeterminado."""
    if not MICROFONO:
        return None
    if MICROFONO.isdigit():
        return int(MICROFONO)
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and MICROFONO.lower() in d["name"].lower():
            return i
    raise SystemExit(f"No encuentro ningun microfono que contenga '{MICROFONO}'. "
                     "Usa --microfonos para ver la lista.")


def listar_microfonos() -> None:
    predeterminado = sd.default.device[0]
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            marca = "  <- predeterminado" if i == predeterminado else ""
            print(f"{i:3}  {d['name']}{marca}")


def escuchar(segundos: int = SEGUNDOS) -> bytes | None:
    print(f"\nHabla ({segundos}s)...")
    audio = sd.rec(int(segundos * FS), samplerate=FS, channels=1, dtype="int16",
                   device=microfono())
    sd.wait()
    pico = int(np.abs(audio).max())
    if pico < NIVEL_MINIMO:
        nombre = sd.query_devices(microfono() if microfono() is not None else sd.default.device[0])["name"]
        print(f"  No llega sonido del microfono '{nombre}' (nivel {pico}). Comprueba que no "
              "este silenciado o elige otro con MICROFONO en el .env (lista: --microfonos).")
        return None
    return audio.tobytes()


def reproducir(wav_bytes: bytes) -> None:
    """El audio LINEAR_16 que devuelve Dialogflow ya es un WAV completo."""
    FICHERO_AUDIO.write_bytes(wav_bytes)
    with wave.open(str(FICHERO_AUDIO), "rb") as w:
        datos = w.readframes(w.getnframes())
        tasa = w.getframerate()
    sd.play(np.frombuffer(datos, dtype=np.int16), tasa)
    sd.wait()


def sin_ssml(texto: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", texto)).strip()


# Llamadas a Dialogflow

def detectar_por_voz(audio_bytes: bytes):
    entrada = dialogflow.QueryInput(
        audio_config=dialogflow.InputAudioConfig(
            audio_encoding=dialogflow.AudioEncoding.AUDIO_ENCODING_LINEAR_16,
            language_code=IDIOMA,
            sample_rate_hertz=FS,
            model="latest_short",          # mejor para frases cortas de diálogo
            phrase_hints=["grúa", "matrícula", "póliza", "franquicia", "siniestro"],
        )
    )
    return clientes.detect_intent(request={
        "session": sesion, "query_input": entrada,
        "input_audio": audio_bytes, "output_audio_config": SALIDA_AUDIO,
    })


def detectar_por_texto(texto: str):
    entrada = dialogflow.QueryInput(text=dialogflow.TextInput(text=texto, language_code=IDIOMA))
    return clientes.detect_intent(request={
        "session": sesion, "query_input": entrada, "output_audio_config": SALIDA_AUDIO,
    })


# Bucle de conversación

def main():
    if "--microfonos" in sys.argv:
        listar_microfonos()
        return
    modo_texto = "--texto" in sys.argv
    print("=" * 60)
    print("Seguros Angel - cliente de voz de Dialogflow")
    print(f"Proyecto: {PROJECT_ID}   Sesión: {SESSION_ID}   Voz: {VOZ}")
    print("Escribe tu frase y Enter." if modo_texto else "Enter para hablar, 'q' + Enter para salir.")
    print("=" * 60)

    while True:
        linea = input("\n> ").strip()
        if linea.lower() == "q":
            break

        if modo_texto:
            if not linea:
                continue
            inicio = time.perf_counter()
            respuesta = detectar_por_texto(linea)
        else:
            audio = escuchar()
            if audio is None:
                continue
            inicio = time.perf_counter()
            respuesta = detectar_por_voz(audio)
        latencia = time.perf_counter() - inicio

        r = respuesta.query_result
        if not r.query_text:
            print("  No se ha reconocido ninguna frase. Habla mas cerca del microfono "
                  "y empieza justo despues de pulsar Enter.")
            continue
        print(f"  Has dicho  : {r.query_text}")
        print(f"  Intención  : {r.intent.display_name} ({r.intent_detection_confidence:.2f})")
        print(f"  Parámetros : {dict(r.parameters) if r.parameters else {}}")
        print(f"  Responde   : {sin_ssml(r.fulfillment_text)}")
        print(f"  Latencia   : {latencia:.2f}s {'(OK)' if latencia < 5 else '(DEMASIADO LENTO)'}")

        if respuesta.output_audio:
            reproducir(respuesta.output_audio)

        if r.intent.end_interaction or r.intent.display_name == "despedida":
            break


if __name__ == "__main__":
    main()
