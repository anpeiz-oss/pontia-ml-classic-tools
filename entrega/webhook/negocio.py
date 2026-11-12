"""
Logica de negocio de Seguros Angel: polizas simuladas, validacion con regex
y construccion de las respuestas de cada intencion.

No depende de FastAPI ni de Dialogflow: recibe diccionarios y devuelve texto,
asi se puede probar con pytest sin levantar nada.
"""

import random
import re
from datetime import date

# Polizas simuladas
#
# En un sistema real seria una consulta a base de datos. Los datos coinciden
# con los documentos de ocr/documentos/ (poliza AB-456789 y recibo AB-456790).

POLIZAS = {
    "1234BCD": {
        "poliza": "AB-456789",
        "titular": "Alonso de la Fuente",
        "vehiculo": "Seat León",
        "modalidad": "Todo riesgo con franquicia",
        "franquicia": 300,
        "prima": 412.35,
        "vencimiento": date(2027, 3, 14),
        "grua": True,
        "remolque_km": 50,
        "coberturas": ["responsabilidad civil", "daños propios con franquicia",
                       "lunas", "robo e incendio", "asistencia en carretera",
                       "vehículo de sustitución"],
    },
    "5678FGH": {
        "poliza": "AB-456790",
        "titular": "Luis Gómez",
        "vehiculo": "Volkswagen Golf",
        "modalidad": "Terceros básico",
        "franquicia": None,
        "prima": 289.90,
        "vencimiento": date(2026, 11, 20),
        "grua": False,
        "remolque_km": 0,
        "coberturas": ["responsabilidad civil"],
    },
    "9012JKL": {
        "poliza": "AB-456791",
        "titular": "Marta Sanz",
        "vehiculo": "Toyota Corolla",
        "modalidad": "Terceros ampliado",
        "franquicia": None,
        "prima": 355.00,
        "vencimiento": date(2027, 6, 1),
        "grua": True,
        "remolque_km": 25,
        "coberturas": ["responsabilidad civil", "lunas", "robo e incendio",
                       "asistencia en carretera desde el kilómetro 25"],
    },
    "3456KLM": {
        "poliza": "AB-456792",
        "titular": "Javier Ortega",
        "vehiculo": "Renault Clio",
        "modalidad": "Todo riesgo",
        "franquicia": 0,
        "prima": 640.00,
        "vencimiento": date(2027, 1, 10),
        "grua": True,
        "remolque_km": 50,
        "coberturas": ["responsabilidad civil", "daños propios sin franquicia",
                       "lunas", "robo e incendio", "asistencia en carretera",
                       "vehículo de sustitución"],
    },
}

# Tiempo estimado de llegada de la grua por ciudad (minutos). El resto, 45.
TIEMPO_GRUA = {"madrid": 30, "barcelona": 35, "valencia": 35, "sevilla": 40}


# Validacion con expresiones regulares (requisito del webhook)

# Matricula espanola actual: 4 digitos + 3 consonantes sin vocales, N, Q ni Ñ.
RE_MATRICULA = re.compile(r"^\d{4}[B-DF-HJ-NP-TV-Z]{3}$")
# Numero de poliza: dos letras, guion y seis digitos
RE_POLIZA = re.compile(r"^[A-Z]{2}-\d{6}$")

# Lo que el STT de Dialogflow devuelve a veces al deletrear letras
LETRAS_HABLADAS = {"BE": "B", "CE": "C", "DE": "D", "EFE": "F", "GE": "G",
                   "HACHE": "H", "JOTA": "J", "KA": "K", "ELE": "L", "EME": "M",
                   "PE": "P", "ERRE": "R", "ESE": "S", "TE": "T", "UVE": "V",
                   "DOBLEUVE": "W", "EQUIS": "X", "YE": "Y", "ZETA": "Z"}


def normalizar_matricula(valor) -> str:
    """
    Deja la matricula como la escribiria una persona: '1234BCD'.

    El STT de Dialogflow la transcribe de muchas formas segun como se dicte:
    '1234 bcd', '1234-BCD', '1 2 3 4 B C D' (dictada cifra a cifra),
    '1.234 BCD' (lo toma como un numero con separador de miles) o
    '1234 be ce de' (letras deletreadas).
    """
    if isinstance(valor, list):  # Dialogflow manda lista si el parametro es "is list"
        valor = valor[0] if valor else ""
    texto = str(valor or "").upper()
    texto = re.sub(r"[.,\-]", " ", texto)          # 1.234 / 1,234 / 1234-BCD
    palabras = [LETRAS_HABLADAS.get(p, p) for p in texto.split()]
    return "".join(palabras)


def matricula_valida(matricula: str) -> bool:
    return bool(RE_MATRICULA.match(matricula))


# SSML

def ssml_matricula(matricula: str) -> str:
    """Que la voz lea '1 2 3 4 B C D' y no 'mil doscientos treinta y cuatro'."""
    return f'<say-as interpret-as="characters">{matricula}</say-as>'


def euros(valor: float) -> str:
    texto = f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return texto.removesuffix(",00") + " euros"


# Respuestas
#
# Cada funcion devuelve (texto, ssml): el texto se muestra en pantalla y el
# SSML es lo que se convierte en voz.

def respuesta_matricula_invalida(matricula: str) -> tuple[str, str]:
    texto = (f"La matrícula {matricula or 'que me has dicho'} no tiene un formato válido. "
             "Debe tener cuatro números y tres letras, por ejemplo 1234 BCD. ¿Me la repites?")
    ssml = (f"<speak>Vaya, esa matrícula no me cuadra. <break time=\"300ms\"/> "
            "Debe tener cuatro números y tres letras, por ejemplo "
            f"{ssml_matricula('1234BCD')}. <break time=\"200ms\"/> ¿Me la repites?</speak>")
    return texto, ssml


def respuesta_sin_poliza(matricula: str) -> tuple[str, str]:
    texto = (f"No encuentro ninguna póliza con la matrícula {matricula}. "
             "Puede que la haya entendido mal. ¿Me la repites?")
    ssml = (f"<speak>No encuentro ninguna póliza con la matrícula {ssml_matricula(matricula)}. "
            "<break time=\"300ms\"/> Puede que la haya entendido mal. ¿Me la repites?</speak>")
    return texto, ssml


def preparar_grua(matricula: str, ubicacion: str) -> dict:
    """
    Primer paso del flujo con contexto: valida la matricula y prepara el aviso.
    Devuelve un dict con 'texto', 'ssml', 'estado' y los datos del aviso.
      estado = "invalida" | "sin_poliza" | "sin_cobertura" | "pendiente_confirmar"
    """
    if not matricula_valida(matricula):
        texto, ssml = respuesta_matricula_invalida(matricula)
        return {"estado": "invalida", "texto": texto, "ssml": ssml}

    poliza = POLIZAS.get(matricula)
    if poliza is None:
        texto, ssml = respuesta_sin_poliza(matricula)
        return {"estado": "sin_poliza", "texto": texto, "ssml": ssml}

    if not poliza["grua"]:
        texto = (f"{poliza['titular']}, tu póliza de {poliza['modalidad'].lower()} no incluye "
                 "asistencia en carretera. ¿Quieres que te pase con un agente para contratarla?")
        ssml = (f"<speak>{poliza['titular']}, <break time=\"200ms\"/> lo siento, tu póliza de "
                f"{poliza['modalidad'].lower()} no incluye asistencia en carretera. "
                "<break time=\"300ms\"/> Si quieres, te paso con un agente para contratarla.</speak>")
        return {"estado": "sin_cobertura", "texto": texto, "ssml": ssml}

    ciudad = ubicacion or "tu ubicación"
    texto = (f"{poliza['titular']}, voy a pedir una grúa para tu {poliza['vehiculo']} "
             f"con matrícula {matricula} en {ciudad}. Tu póliza cubre el remolque hasta "
             f"{poliza['remolque_km']} km. ¿Confirmas que los datos son correctos?")
    ssml = (f"<speak>{poliza['titular']}, <break time=\"200ms\"/> voy a pedir una grúa para tu "
            f"{poliza['vehiculo']} con matrícula {ssml_matricula(matricula)} en {ciudad}. "
            f"<break time=\"300ms\"/> Tu póliza cubre el remolque hasta {poliza['remolque_km']} "
            "kilómetros. <break time=\"300ms\"/> "
            "<prosody pitch=\"+2st\">¿Confirmas que los datos son correctos?</prosody></speak>")
    return {"estado": "pendiente_confirmar", "texto": texto, "ssml": ssml,
            "titular": poliza["titular"]}


def confirmar_grua(matricula: str, ubicacion: str) -> tuple[str, str]:
    aviso = f"AV-{random.randint(10000, 99999)}"
    minutos = TIEMPO_GRUA.get((ubicacion or "").strip().lower(), 45)
    digitos = " ".join(aviso.split("-")[1])
    texto = (f"Hecho. He abierto el aviso {aviso}. La grúa llegará a {ubicacion} en unos "
             f"{minutos} minutos y te llamará antes de llegar. ¿Puedo ayudarte en algo más?")
    ssml = (f"<speak>Hecho. <break time=\"300ms\"/> He abierto el aviso "
            f"<prosody rate=\"slow\">{digitos}</prosody>. <break time=\"300ms\"/> "
            f"La grúa llegará a {ubicacion} en unos {minutos} minutos y te llamará antes "
            "de llegar. <break time=\"400ms\"/> ¿Puedo ayudarte en algo más?</speak>")
    return texto, ssml


def cancelar_grua() -> tuple[str, str]:
    texto = ("De acuerdo, no pido la grúa. Si algún dato estaba mal, dime la matrícula "
             "o la ciudad correctas y lo corrijo. ¿Qué quieres hacer?")
    ssml = ("<speak>De acuerdo, <break time=\"200ms\"/> no pido la grúa. "
            "Si algún dato estaba mal, dime la matrícula o la ciudad correctas "
            "y lo corrijo. <break time=\"300ms\"/> ¿Qué quieres hacer?</speak>")
    return texto, ssml


def consultar_coberturas(matricula: str) -> tuple[str, str]:
    if not matricula_valida(matricula):
        return respuesta_matricula_invalida(matricula)
    poliza = POLIZAS.get(matricula)
    if poliza is None:
        return respuesta_sin_poliza(matricula)
    lista = ", ".join(poliza["coberturas"][:-1]) + " y " + poliza["coberturas"][-1] \
        if len(poliza["coberturas"]) > 1 else poliza["coberturas"][0]
    texto = (f"Tu {poliza['vehiculo']} tiene la modalidad {poliza['modalidad'].lower()}, "
             f"que incluye {lista}.")
    ssml = (f"<speak>Tu {poliza['vehiculo']} tiene la modalidad "
            f"<emphasis level=\"moderate\">{poliza['modalidad'].lower()}</emphasis>. "
            f"<break time=\"300ms\"/> Incluye {lista}.</speak>")
    return texto, ssml


DATOS_POLIZA = ("vencimiento", "franquicia", "prima", "modalidad")


def consultar_poliza(matricula: str, dato: str, hoy: date | None = None) -> tuple[str, str]:
    """Responde a un dato concreto de la poliza: vencimiento, franquicia, prima o modalidad."""
    if not matricula_valida(matricula):
        return respuesta_matricula_invalida(matricula)
    poliza = POLIZAS.get(matricula)
    if poliza is None:
        return respuesta_sin_poliza(matricula)

    hoy = hoy or date.today()
    dato = (dato or "modalidad").lower()
    if dato not in DATOS_POLIZA:
        dato = "modalidad"

    if dato == "vencimiento":
        dias = (poliza["vencimiento"] - hoy).days
        fecha = poliza["vencimiento"].strftime("%d/%m/%Y")
        texto = f"Tu póliza {poliza['poliza']} vence el {fecha}, dentro de {dias} días."
        if dias <= 30:
            texto += " Se renovará automáticamente salvo que nos avises."
        ssml = (f"<speak>Tu póliza vence el <say-as interpret-as=\"date\" format=\"dmy\">"
                f"{fecha}</say-as>, <break time=\"200ms\"/> dentro de {dias} días.</speak>")
        return texto, ssml

    if dato == "franquicia":
        if poliza["franquicia"] is None:
            texto = (f"Tu modalidad, {poliza['modalidad'].lower()}, no cubre daños propios, "
                     "así que no tiene franquicia.")
        elif poliza["franquicia"] == 0:
            texto = "Tu póliza de todo riesgo no tiene franquicia."
        else:
            texto = f"Tu franquicia es de {euros(poliza['franquicia'])} por siniestro."
        return texto, f"<speak>{texto}</speak>"

    if dato == "prima":
        texto = f"Pagas {euros(poliza['prima'])} al año por el {poliza['vehiculo']}."
        return texto, f"<speak>{texto}</speak>"

    texto = f"Tu {poliza['vehiculo']} está asegurado a {poliza['modalidad'].lower()}."
    return texto, f"<speak>{texto}</speak>"


def hablar_con_agente(motivo: str = "") -> tuple[str, str]:
    caso = f"CS-{random.randint(1000, 9999)}"
    texto = (f"Te paso con un agente. Tu número de caso es {caso}. "
             "El tiempo de espera estimado es de 2 minutos.")
    ssml = (f"<speak>Te paso con un agente. <break time=\"300ms\"/> Tu número de caso es "
            f"<say-as interpret-as=\"characters\">{caso.replace('-', '')}</say-as>. "
            "<break time=\"300ms\"/> El tiempo de espera estimado es de dos minutos.</speak>")
    return texto, ssml


def despedida() -> tuple[str, str]:
    texto = "Gracias por llamar a Seguros Angel. ¡Que tengas un buen día!"
    ssml = ("<speak>Gracias por llamar a Seguros Angel. <break time=\"200ms\"/> "
            "<prosody pitch=\"+1st\">¡Que tengas un buen día!</prosody></speak>")
    return texto, ssml
