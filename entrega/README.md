# Seguros Angel: asistente de voz con Dialogflow ES, OCR y LLM

Evaluación final de *Herramientas de IA Clásica*.

## Caso de uso

Seguros Angel atiende por teléfono a sus clientes de seguro de coche. El asistente, al que se le habla y que responde con voz, resuelve tres cosas: pedir una grúa de asistencia en carretera (flujo de varios turnos que recoge matrícula y ciudad, comprueba la póliza y termina con una confirmación sí/no), consultar datos de la póliza de una matrícula (coberturas, vencimiento, franquicia, prima) y responder preguntas sobre las condiciones generales a partir de cinco documentos escaneados procesados con OCR, diciendo explícitamente cuándo algo no aparece en ellos en vez de inventarlo.

## Arquitectura

```mermaid
flowchart LR
    U((Usuario)) -- voz --> CV[voz/cliente_voz.py<br/>micrófono y altavoz]
    CV -- audio 16 kHz --> DF[Dialogflow ES<br/>STT · intenciones · contexto · TTS]
    DF -- audio + SSML --> CV
    DF -- POST /webhook --> WH[Webhook FastAPI<br/>webhook/main.py]
    WH --> NEG[negocio.py<br/>regex + pólizas simuladas]
    WH --> RAG[rag.py<br/>documentos en el prompt]
    RAG -- prompt --> LLM[Gemini flash-lite]
    subgraph Offline una sola vez
      IMG[ocr/documentos/*.jpg] --> OCR[ocr/ocr.py<br/>Tesseract + regex]
      OCR --> JSON[(ocr/salida/documentos.json)]
    end
    JSON -. se carga al arrancar .-> RAG
    CF[Cloudflare Tunnel] -.expone.- WH
```

El diagrama está en Mermaid (GitHub lo dibuja). Para la entrega se puede exportar a PNG en [mermaid.live](https://mermaid.live) y guardarlo como `docs/arquitectura.png`.

## Intenciones

Un único contexto, `solicitar_grua-followup` (el que Dialogflow crea al añadir los follow-up sí/no a `solicitar_grua`), que gestiona el webhook.

| Intención | Propósito | Parámetros | Contexto |
| --- | --- | --- | --- |
| `Default Welcome Intent` | Saludo y lo que sabe hacer | – | – |
| `solicitar_grua` | Inicia el flujo de grúa | `matricula` (@matricula, regex, obligatoria), `ubicacion` (@sys.geo-city, obligatoria) | sale `solicitar_grua-followup` |
| `solicitar_grua-si` | Confirma: abre el aviso con tiempo estimado según ciudad | – | entra y cierra `solicitar_grua-followup` |
| `solicitar_grua-no` | Niega: ofrece corregir o cancelar | – | entra `solicitar_grua-followup` |
| `corregir_dato` | Corrige matrícula o ciudad a mitad de flujo (extra) | `matricula`, `ubicacion` | entra `solicitar_grua-followup` |
| `consultar_coberturas` | Coberturas de la póliza de una matrícula | `matricula` | – |
| `consultar_poliza` | Vencimiento, franquicia, prima o modalidad | `matricula`, `dato_poliza` (entidad propia) | – |
| `consultar_condiciones` | Preguntas sobre los documentos (OCR + LLM) | – | – |
| `hablar_con_agente` | Transferencia a agente humano con número de caso (extra) | – | cierra el contexto |
| `despedida` | Cierra la conversación | – | cierra el contexto |
| `Default Fallback Intent` | Lo no reconocido también va a la respuesta documental | – | – |

El export del agente (intenciones, entidades, frases de entrenamiento y contextos) está en `dialogflow/proyecto-aseguradora-api.zip`.

## Decisiones técnicas

**Voz.** Se usa el STT y el TTS nativos de Dialogflow en una sola llamada `detect_intent` con audio de entrada y de salida, con el modelo `latest_short` para frases cortas y una voz `es-ES-Neural2`. El webhook responde en SSML: pausas con `<break>`, matrículas y números de caso deletreados con `<say-as interpret-as="characters">`, fechas con `say-as date`, y entonación de pregunta con `<prosody>`. La variable `SSML=0` devuelve texto plano para probar en el simulador.

**Validación con regex en el webhook.** Aunque la entidad `@matricula` ya es una regex, el STT devuelve cosas como «1234 be ce de» o «1234-bcd». El webhook normaliza (mayúsculas, quita guiones y espacios, convierte letras deletreadas) y valida contra `^\d{4}[B-DF-HJ-NP-TV-Z]{3}$` (sin vocales, Ñ ni Q, como las matrículas actuales). La respuesta cambia según el resultado: formato inválido, matrícula sin póliza, póliza sin asistencia o petición de confirmación con los datos del vehículo.

**Contexto gestionado por el webhook.** El webhook guarda en `solicitar_grua-followup` la matrícula y la ciudad ya validadas y un `estado`. Así la confirmación usa los datos validados y no los que entendió Dialogflow, y la corrección funciona aunque el usuario solo diga la matrícula nueva. Se guardan con claves propias (`grua_matricula`, `grua_ubicacion`, `grua_estado`) porque Dialogflow copia en el contexto los parámetros de la intención detectada: con los mismos nombres, «estoy en Barcelona» en `corregir_dato` sobrescribía la matrícula validada con un valor vacío (fallo encontrado en las pruebas).

**Follow-ups y un único contexto.** `solicitar_grua-si` y `solicitar_grua-no` se crean como follow-ups de `solicitar_grua`, y el contexto que Dialogflow genera para ellos (`solicitar_grua-followup`) es el único del agente: si se sustituye por uno propio, la consola deja de mostrarlos anidados. `corregir_dato` es una intención normal con ese mismo contexto de entrada, porque un follow-up *custom* añade un segundo contexto (`solicitar_grua-followup-2`).

**Ejemplos negativos.** El `Default Fallback Intent` tiene frases ajenas al negocio («cuéntame un chiste», «¿qué tiempo hace?») para que Dialogflow no las fuerce a una intención. También pasan por el webhook, y el LLM responde que no aparece en la documentación.

**OCR.** Los documentos son cinco imágenes de documentos ficticios de Seguros Angel (póliza, condicionado de asistencia, tabla de modalidades, procedimiento de siniestros y recibo), generadas por `ocr/generar_documentos.py` con giro, ruido y desenfoque de escáner. Son imágenes, no PDF con texto, así que la extracción es OCR real. Preprocesado en `ocr/ocr.py`:

1. Escala de grises, reescalado a 2.000 px de ancho, filtro de mediana contra el ruido y autocontraste.
2. Binarización con umbral de Otsu.
3. Enderezado por perfil de proyección (busca el giro que maximiza la varianza de la suma de tinta por fila).
4. Borrado de líneas de tabla (tramos negros de más de 120 px).
5. Si detecta una rejilla, lee la tabla celda a celda con `--psm 7` y la guarda como filas separadas por `|`. Si no, prueba `--psm 4` y `--psm 6` y se queda con el de mayor confianza media.

Leída de golpe, la tabla de modalidades salía ilegible (sopa de letras con `|` y `[`). Celda a celda sale entera salvo dos celdas («Sí» leído como «St» y «El»). Los errores sueltos de OCR se toleran porque el prompt avisa al LLM de que el texto viene de un OCR.

**Validación con regex sobre el OCR.** Se extraen matrículas, números de póliza, NIF, fechas, importes, teléfonos y referencias. No basta con que la regex encaje: la fecha tiene que existir en el calendario, el NIF tiene que tener la letra de control correcta y el importe se convierte a número. Lo que encaja pero no es válido se guarda en `descartados_por_validacion`. Antes de aplicar las regex se corrigen las confusiones típicas (O/0, l/1, S/5, B/8) solo cuando están pegadas a dígitos, y solo en la copia que va a las regex, nunca en el texto que se da al LLM.

**Respuesta documental sin alucinar.** El JSON se carga al arrancar el webhook (el OCR tarda unos 40 s y Dialogflow corta a los 5). Con cinco documentos (~3.500 caracteres) se envían todos en el prompt; no hace falta base vectorial ni filtrar por palabras clave, que fallaría con sinónimos («coche» frente a «vehículo»). El modelo tiene instrucción de responder la marca `NO_CONSTA` si la respuesta no está en los documentos, con temperatura 0, y el webhook la traduce a «Esa información no aparece en la documentación…». Probado con preguntas reales: remolque, tiempos de llegada, plazos de parte y precios se responden bien; «¿cubrís a mi mascota?», «horario de la oficina de Bilbao» o «el tiempo en Madrid» devuelven el mensaje de que no consta.

**Por qué no la Knowledge Base de Dialogflow.** Es beta, solo admite documentos con texto (no hace OCR), no permite controlar el prompt ni el mensaje cuando no hay respuesta, y no deja ver qué contexto se ha usado. Con el webhook controlamos las tres cosas.

**Modelo y latencia.** La latencia de Gemini es muy variable: el 06/11/2026 `gemini-3.5-flash-lite` respondía en ~1 s, y el 12/11/2026 el mismo modelo tardó entre 1 y 16 s con la misma pregunta; lanzando la pregunta a dos modelos de Gemini a la vez solo 3 de 8 respuestas llegaban a tiempo. Por eso la pregunta se envía en paralelo a `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite` y a un modelo local en Ollama sobre la GPU (`LLM_LOCAL`, por defecto `qwen3:14b`). Se prefiere Gemini; si ninguno de sus modelos contesta en 1,5 s (`LLM_ESPERA_PREFERIDO`) o fallan, se usa la respuesta local. Al arrancar y cada 60 s (`LLM_KEEPALIVE`) se hace una llamada mínima a cada modelo, para que la conexión no se enfríe y Ollama no descargue el modelo (la primera carga de un 14B tarda ~22 s). Las llamadas a Gemini y al modelo local van en grupos de hilos separados, porque en las pruebas una llamada a Gemini llegó a quedarse colgada 93 s y, con un grupo compartido, dejaba en cola la del modelo local. En los Qwen 3.x se desactiva el razonamiento previo (`"think": false`), que añade segundos. Hay un tiempo máximo total (`LLM_TIMEOUT`, 4 s): si nadie responde, el webhook contesta «Ahora mismo no puedo consultar la documentación. ¿Quieres que te pase con un agente?» en lugar de dejar al usuario en silencio. El SDK de Gemini no admite timeouts HTTP de menos de 10 s, por eso el corte es nuestro.

**Elección del modelo local.** Se compararon cinco modelos que caben en 16 GB de VRAM (RX 9070) con 11 preguntas sobre los documentos: 8 con respuesta (incluidas dos trampa: la compensación si la grúa llega tarde y los días de coche de sustitución del todo riesgo sin franquicia, que obligan a leer bien la tabla) y 3 sin respuesta en los documentos. Los Qwen 3.6 y 3.8 no se probaron porque solo existen desde 27B (17–18 GB en Q4).

| Modelo | VRAM | Aciertos | Tiempo medio | Peor | Observaciones |
| --- | --- | --- | --- | --- | --- |
| `qwen2.5:3b` | 1,9 GB | 9/11 | 0,15 s | 0,68 s | Muy escueto; inventa «30 € por cada minuto» y confunde los días de sustitución |
| `qwen3.5:9b` (Q4) | 6,6 GB | 10/11 | 0,50 s | 0,86 s | Dice que no consta el todo riesgo sin franquicia |
| `qwen3.5:9b-q8_0` | 10,4 GB | 11/11 | 0,58 s | 0,89 s | Correcto pero largo para voz («según la tabla de modalidades…») |
| `qwen2.5:14b` | 9,0 GB | 11/11 | 0,48 s | 1,38 s | Correcto, algo más lento |
| **`qwen3:14b`** | 9,3 GB | **11/11** | **0,34 s** | **0,64 s** | **Elegido**: correcto, el más rápido de los 14B y respuestas cortas, adecuadas para voz |

Todos respondieron «NO_CONSTA» a las tres preguntas sin respuesta en los documentos.

**Datos de negocio.** Las pólizas son un diccionario en `webhook/negocio.py` (cuatro matrículas). Coinciden con los documentos: la póliza AB-456789 del Seat León 1234 BCD y el recibo AB-456790 del Golf 5678 FGH.

## Ejecutar el proyecto

Requisitos: Python 3.12, [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) con el idioma español, [Ollama](https://ollama.com) con el modelo `qwen3:14b` (`ollama pull qwen3:14b`), una clave de Gemini y una cuenta de servicio con el rol *Dialogflow API Client*.

```bash
cd entrega
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Rellena `.env` y después:

```bash
python ocr/generar_documentos.py
```

```bash
python ocr/ocr.py
```

```bash
pytest
```

```bash
uvicorn webhook.main:app --port 8000
```

En otra terminal, expón el puerto y pega la URL que muestra + `/webhook` en Dialogflow → Fulfillment:

```bash
cloudflared tunnel --url http://localhost:8000
```

Comprueba que llega abriendo la URL pública: debe responder `{"estado": "ok", "documentos_cargados": 5, ...}`. Por último, la conversación por voz:

```bash
python voz/cliente_voz.py
```

## Pruebas

`pytest` ejecuta 27 pruebas sin red: el flujo de la grúa con todos sus caminos (faltan datos, formato inválido, matrícula hablada, sin póliza, sin cobertura, confirmación sí y no, corrección de matrícula y de ciudad), las consultas, la respuesta documental con un LLM simulado (incluido el caso «no consta», el de un LLM que tarda más del límite y los de varios modelos en paralelo con preferencia por Gemini) y la validación de regex del OCR.

## Estructura

```
entrega/
├── webhook/        main.py (endpoint), negocio.py (regex, pólizas, SSML), rag.py (LLM)
├── ocr/            generar_documentos.py, ocr.py, documentos/, salida/documentos.json
├── voz/            cliente_voz.py
├── dialogflow/     el ZIP exportado del agente
├── docs/           guion_video.md
└── tests/
```
