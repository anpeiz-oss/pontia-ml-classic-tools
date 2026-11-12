# Guion del vídeo (máximo 3 minutos)

Grabación de pantalla con dos ventanas visibles: la terminal de `voz/cliente_voz.py` (muestra lo que se ha entendido, la intención, la respuesta y la latencia) y la terminal del webhook (muestra la intención y los parámetros que llegan). Se debe realizar con `SSML=1` en el `.env`.

| Tiempo | Qué se dice | Qué se enseña |
| --- | --- | --- |
| 0:00–0:15 | Presentación: «Asistente de voz de Seguros Angel con Dialogflow ES, webhook en FastAPI, OCR con Tesseract y Gemini» | Diagrama de arquitectura del README |
| 0:15–0:25 | «Hola» | Bienvenida con voz |
| 0:25–1:05 | «Necesito una grúa» → «La 1234 BCD» → «Estoy en Madrid» | Slot filling, regex de matrícula, contexto `solicitar_grua-followup` en el log del webhook |
| 1:05–1:30 | «No» → «Estoy en Barcelona» → «Sí» | Negación, corrección a mitad de flujo y confirmación con número de aviso |
| 1:30–1:45 | «¿Cuándo vence la póliza del 5678 FGH?» | Consulta de póliza |
| 1:45–2:15 | «¿Cuánto tarda la grúa en llegar en ciudad?» y «¿Qué plazo tengo para dar un parte?» | Respuesta desde los documentos. Enseña `ocr/salida/documentos.json` un segundo |
| 2:15–2:35 | «¿Cubrís los daños a mi mascota?» | El asistente dice que no aparece en la documentación |
| 2:35–2:50 | «Quiero hablar con una persona» | Transferencia con número de caso |
| 2:50–3:00 | «Adiós» | Cierre y latencias por turno (todas < 5 s) |

Consejos: ensayar una vez antes de grabar, lanzar el túnel justo antes y compruebar con `/` en el navegador; hablar cerca del micrófono y deletrear la matrícula despacio.
