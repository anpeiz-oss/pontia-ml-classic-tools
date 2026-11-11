"""
OCR offline con Tesseract + validacion de campos con expresiones regulares.

Se ejecuta UNA SOLA VEZ, antes de arrancar el webhook:

    python ocr/ocr.py

Lee las imagenes de ocr/documentos/, aplica preprocesado, elige el modo de
segmentacion de Tesseract (--psm) con mejor confianza media para cada
documento, extrae y valida campos con regex y guarda todo en
ocr/salida/documentos.json. El webhook solo lee ese JSON: nunca hace OCR
dentro de una peticion, porque Dialogflow corta a los 5 segundos.

Basado en ejer2/ocr.py (clase), ampliado con:
  - preprocesado: escala de grises, reescalado, autocontraste y binarizacion Otsu
  - enderezado, borrado de lineas de tabla y lectura de tablas celda a celda
  - eleccion automatica de --psm por confianza media
  - validacion real de los campos (fecha existente, letra del NIF, importe numerico)
  - salida persistida en JSON
"""

import json
import re
import shutil
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pytesseract
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image, ImageFilter, ImageOps

# Configuracion

if shutil.which("tesseract") is None:
    ejecutable_windows = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if ejecutable_windows.is_file():
        pytesseract.pytesseract.tesseract_cmd = str(ejecutable_windows)

BASE = Path(__file__).parent
CARPETA_DOCUMENTOS = BASE / "documentos"
FICHERO_SALIDA = BASE / "salida" / "documentos.json"
EXTENSIONES = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

IDIOMA = "spa"
# 6 = bloque uniforme de texto, 4 = una columna de lineas de tamano variable.
MODOS_PSM = (6, 4)  # se prueban ambos y se queda el de mayor confianza
ANCHO_MINIMO = 2000  # Tesseract rinde mejor con letras de unos 30 px de alto


# Preprocesado

def umbral_otsu(gris: np.ndarray) -> int:
    """Calcula el umbral de Otsu: el que mejor separa tinta y papel."""
    hist = np.bincount(gris.ravel(), minlength=256).astype(float)
    total = gris.size
    suma_total = np.dot(np.arange(256), hist)
    suma_fondo, peso_fondo, mejor, umbral = 0.0, 0.0, 0.0, 127
    for t in range(256):
        peso_fondo += hist[t]
        if peso_fondo == 0:
            continue
        peso_tinta = total - peso_fondo
        if peso_tinta == 0:
            break
        suma_fondo += t * hist[t]
        media_fondo = suma_fondo / peso_fondo
        media_tinta = (suma_total - suma_fondo) / peso_tinta
        varianza = peso_fondo * peso_tinta * (media_fondo - media_tinta) ** 2
        if varianza > mejor:
            mejor, umbral = varianza, t
    return umbral


def binarizar(imagen: Image.Image) -> Image.Image:
    gris = np.asarray(imagen)
    umbral = umbral_otsu(gris)
    return Image.fromarray(np.where(gris > umbral, 255, 0).astype(np.uint8))


def angulo_inclinacion(binaria: Image.Image) -> float:
    """
    Estima el giro del escaneo por perfil de proyeccion: cuando las lineas de
    texto estan horizontales, la suma de pixeles negros por fila alterna mucho
    entre filas de texto y huecos, asi que su varianza es maxima.
    """
    peq = binaria.resize((binaria.width // 4, binaria.height // 4))
    mejor_var, mejor_ang = -1.0, 0.0
    for angulo in np.arange(-3.0, 3.01, 0.1):
        filas = (np.asarray(peq.rotate(angulo, fillcolor=255)) < 128).sum(axis=1)
        var = float(np.var(filas))
        if var > mejor_var:
            mejor_var, mejor_ang = var, float(angulo)
    return round(mejor_ang, 1)


def mascaras_lineas(binaria: Image.Image, largo: int = 120):
    """
    Detecta las lineas de las tablas: cualquier tramo negro horizontal o
    vertical de mas de `largo` pixeles no es una letra. Devuelve dos mascaras
    booleanas (horizontales, verticales).
    """
    tinta = np.asarray(binaria) < 128
    mascaras = []
    for eje in (1, 0):  # 1 = horizontales, 0 = verticales
        mascara = np.zeros_like(tinta)
        ventanas = sliding_window_view(tinta, largo, axis=eje).all(axis=-1)
        n = ventanas.shape[eje]
        for k in range(largo):
            if eje == 1:
                mascara[:, k:k + n] |= ventanas
            else:
                mascara[k:k + n, :] |= ventanas
        mascaras.append(mascara)
    return mascaras


def quitar_lineas(binaria: Image.Image) -> Image.Image:
    """
    Borra las lineas de tabla. Tesseract confunde los bordes de celda con
    letras (|, l, I, [) y desordena la lectura.
    """
    tinta = np.asarray(binaria) < 128
    horizontales, verticales = mascaras_lineas(binaria)
    limpia = tinta & ~(horizontales | verticales)
    return Image.fromarray(np.where(limpia, 0, 255).astype(np.uint8))


def agrupar(posiciones: np.ndarray, separacion: int = 15) -> list[int]:
    """Convierte filas/columnas de pixeles contiguas en una sola coordenada."""
    grupos, actual = [], []
    for p in posiciones:
        if actual and p - actual[-1] > separacion:
            grupos.append(int(np.mean(actual)))
            actual = []
        actual.append(p)
    if actual:
        grupos.append(int(np.mean(actual)))
    return grupos


def detectar_rejilla(binaria: Image.Image):
    """
    Busca una tabla con bordes. Devuelve (ys, xs) con las coordenadas de las
    lineas horizontales y verticales, o None si no hay tabla.
    """
    horizontales, verticales = mascaras_lineas(binaria)
    alto, ancho = horizontales.shape
    ys = agrupar(np.where(horizontales.sum(axis=1) > ancho * 0.3)[0])
    xs = agrupar(np.where(verticales.sum(axis=0) > alto * 0.15)[0])
    if len(ys) < 3 or len(xs) < 3:
        return None
    return ys, xs


def leer_tabla(binaria: Image.Image, ys: list[int], xs: list[int]) -> str:
    """
    OCR celda a celda (--psm 7 = una sola linea). Leer la tabla entera de
    golpe mezcla columnas; por celdas se conserva la estructura, que despues
    se guarda como filas separadas por " | " para que el LLM la entienda.
    """
    limpia = quitar_lineas(binaria)
    filas = []
    for y0, y1 in zip(ys, ys[1:]):
        celdas = []
        for x0, x1 in zip(xs, xs[1:]):
            celda = limpia.crop((x0 + 6, y0 + 6, x1 - 6, y1 - 6))
            texto = pytesseract.image_to_string(
                ImageOps.expand(celda, border=20, fill=255),
                lang=IDIOMA, config="--psm 7",
            )
            celdas.append(texto.strip())
        filas.append(" | ".join(celdas))
    return "\n".join(filas)


def preprocesar(imagen: Image.Image) -> Image.Image:
    """
    1. Escala de grises: el color no aporta nada y mete ruido.
    2. Reescalado si la imagen es pequena.
    3. Filtro de mediana: quita el ruido "sal y pimienta" del escaner.
    4. Autocontraste: estira el histograma (papel gris -> blanco).
    5. Binarizacion Otsu: tinta negra sobre fondo blanco.
    6. Enderezado: corrige el giro del escaneo.
    (El borrado de lineas de tabla se hace despues, ver leer_documento.)
    """
    imagen = ImageOps.exif_transpose(imagen).convert("L")
    if imagen.width < ANCHO_MINIMO:
        factor = ANCHO_MINIMO / imagen.width
        imagen = imagen.resize(
            (ANCHO_MINIMO, int(imagen.height * factor)), Image.LANCZOS
        )
    imagen = imagen.filter(ImageFilter.MedianFilter(3))
    imagen = ImageOps.autocontrast(imagen, cutoff=1)
    binaria = binarizar(imagen)
    angulo = angulo_inclinacion(binaria)
    if angulo:
        # Se gira la imagen en grises (no la binaria) para no dentar las letras
        binaria = binarizar(imagen.rotate(angulo, fillcolor=255, resample=Image.BICUBIC))
    return binaria


# OCR

def ocr_con_confianza(imagen: Image.Image, psm: int) -> tuple[str, float]:
    """Devuelve (texto, confianza media de las palabras reconocidas)."""
    config = f"--psm {psm}"
    datos = pytesseract.image_to_data(
        imagen, lang=IDIOMA, config=config, output_type=pytesseract.Output.DICT
    )
    confianzas = [float(c) for c, p in zip(datos["conf"], datos["text"])
                  if p.strip() and float(c) >= 0]
    media = sum(confianzas) / len(confianzas) if confianzas else 0.0
    texto = pytesseract.image_to_string(imagen, lang=IDIOMA, config=config)
    return texto, media


def leer_documento(ruta: Path) -> dict:
    original = Image.open(ruta)
    binaria = preprocesar(original)
    rejilla = detectar_rejilla(binaria)

    if rejilla:
        # Documento con tabla: texto de encima + tabla celda a celda + texto de debajo
        ys, xs = rejilla
        arriba = binaria.crop((0, 0, binaria.width, max(ys[0] - 5, 1)))
        abajo = binaria.crop((0, ys[-1] + 5, binaria.width, binaria.height))
        texto_arriba, conf_arriba = ocr_con_confianza(arriba, 6)
        texto_abajo, conf_abajo = ocr_con_confianza(abajo, 6)
        texto = "\n".join([texto_arriba, leer_tabla(binaria, ys, xs), texto_abajo])
        confianza, mejor_psm = (conf_arriba + conf_abajo) / 2, "tabla por celdas (psm 7)"
    else:
        limpia = quitar_lineas(binaria)
        resultados = {psm: ocr_con_confianza(limpia, psm) for psm in MODOS_PSM}
        mejor_psm = max(resultados, key=lambda p: resultados[p][1])
        texto, confianza = resultados[mejor_psm]

    return {
        "fichero": ruta.name,
        "psm": mejor_psm,
        "tabla_detectada": bool(rejilla),
        "confianza_media": round(confianza, 1),
        "texto": limpiar_texto(texto),
    }


def limpiar_texto(texto: str) -> str:
    """Quita lineas vacias y espacios repetidos. NO cambia letras."""
    lineas = [re.sub(r"[ \t]+", " ", l).strip() for l in texto.splitlines()]
    return "\n".join(l for l in lineas if l)


# Extraccion y validacion con regex

PATRONES = {
    # Matricula espanola actual: 4 digitos + 3 consonantes (sin vocales, N ni Q)
    "matricula": r"\b\d{4}\s?[B-DF-HJ-NP-TV-Z]{3}\b",
    # NIF: 8 digitos + letra de control
    "nif": r"\b\d{8}\s?[A-Z]\b",
    # Fechas dd/mm/aaaa o dd-mm-aaaa
    "fecha": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b",
    # Importes: 1.234,56 EUR / 300 euros / 30,00 €
    "importe": r"\b\d{1,3}(?:\.\d{3})*(?:,\d{2})?\s?(?:€|EUR|euros?)(?!\w)",
    # Numero de poliza: AB-123456
    "poliza": r"\b[A-Z]{2}-\d{6}\b",
    # Referencias internas de documento: CG-AUTO-2026/03, PRO-SIN-07, RC-2026-004512
    "referencia": r"\b(?:CG|PRO|RC)-[A-Z0-9]+(?:[-/][A-Z0-9]+)*\b",
    # Telefonos de 9 cifras con o sin espacios: 900 123 456
    "telefono": r"\b[689]\d{2}\s?\d{3}\s?\d{3}\b",
}

LETRAS_NIF = "TRWAGMYFPDXBNJZSQVHLCKE"


def nif_valido(nif: str) -> bool:
    nif = nif.replace(" ", "").upper()
    return LETRAS_NIF[int(nif[:8]) % 23] == nif[8]


def fecha_valida(texto: str) -> str | None:
    """Devuelve la fecha en ISO (aaaa-mm-dd) si existe en el calendario."""
    try:
        d = datetime.strptime(texto.replace("-", "/"), "%d/%m/%Y").date()
    except ValueError:
        return None
    return d.isoformat()


def importe_a_numero(texto: str) -> float:
    numero = re.sub(r"[^\d,\.]", "", texto).replace(".", "").replace(",", ".")
    return float(numero)


def normalizar_para_regex(texto: str) -> str:
    """
    Corrige confusiones tipicas del OCR SOLO en contextos numericos
    (O->0, l/I->1, S->5, B->8 cuando estan rodeadas de digitos).

    Es destructivo: se aplica a la copia que pasa por las regex, nunca al
    texto que se entrega al modelo de lenguaje.
    """
    cambios = {"O": "0", "o": "0", "l": "1", "I": "1", "S": "5", "B": "8"}
    patron = re.compile(r"(?<=\d)[OolISB]|[OolISB](?=\d{2})")
    return patron.sub(lambda m: cambios[m.group()], texto)


def extraer_campos(texto: str) -> dict:
    texto_regex = normalizar_para_regex(texto)
    crudo = {
        campo: sorted(set(re.findall(patron, texto_regex, re.IGNORECASE if campo == "importe" else 0)))
        for campo, patron in PATRONES.items()
    }

    campos = {
        "matriculas": [m.replace(" ", "") for m in crudo["matricula"]],
        "polizas": crudo["poliza"],
        "referencias": crudo["referencia"],
        "telefonos": [t.replace(" ", "") for t in crudo["telefono"]],
        "nifs": [],
        "fechas": [],
        "importes_eur": [],
    }
    descartes = []

    for nif in crudo["nif"]:
        (campos["nifs"] if nif_valido(nif) else descartes).append(nif)
    for f in crudo["fecha"]:
        iso = fecha_valida(f)
        if iso:
            campos["fechas"].append(iso)
        else:
            descartes.append(f)
    for i in crudo["importe"]:
        try:
            campos["importes_eur"].append(importe_a_numero(i))
        except ValueError:
            descartes.append(i)

    campos["fechas"].sort()
    campos["importes_eur"].sort()
    campos["descartados_por_validacion"] = descartes
    return campos


# Programa principal

def procesar_carpeta(carpeta: Path = CARPETA_DOCUMENTOS) -> list[dict]:
    rutas = [r for r in sorted(carpeta.iterdir()) if r.suffix.lower() in EXTENSIONES]
    if not rutas:
        sys.exit(f"No hay imagenes en {carpeta}. Ejecuta antes ocr/generar_documentos.py")

    documentos = []
    for ruta in rutas:
        try:
            doc = leer_documento(ruta)
        except Exception as e:  # un documento roto no tumba el resto
            print(f"  OCR FALLO en {ruta.name}: {e}")
            continue
        doc["campos"] = extraer_campos(doc["texto"])
        documentos.append(doc)
        print(
            f"  OCR OK: {ruta.name}  psm={doc['psm']}  "
            f"confianza={doc['confianza_media']}  "
            f"{len(doc['texto'])} caracteres"
        )
    return documentos


def guardar(documentos: list[dict], destino: Path = FICHERO_SALIDA) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    salida = {
        "generado": date.today().isoformat(),
        "motor": f"Tesseract {pytesseract.get_tesseract_version()} ({IDIOMA})",
        "documentos": documentos,
    }
    destino.write_text(json.dumps(salida, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nGuardado {destino} con {len(documentos)} documentos")


if __name__ == "__main__":
    docs = procesar_carpeta()
    guardar(docs)
    for d in docs:
        print("\n" + "=" * 60)
        print(d["fichero"])
        print("-" * 60)
        for campo, valores in d["campos"].items():
            if valores:
                print(f"  {campo}: {valores}")
