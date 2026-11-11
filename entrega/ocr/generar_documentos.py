"""
Genera los documentos de ejemplo de "Seguros Angel" como IMAGENES escaneadas.

Por que existe este script: el enunciado pide hacer OCR sobre imagenes, no
extraer texto de un PDF que ya lo tiene. Aqui creamos 5 documentos ficticios
(poliza, condicionado de asistencia, tabla de modalidades, procedimiento de
siniestros y recibo) y los "ensuciamos" como si salieran de un escaner:
giro ligero, ruido, desenfoque y fondo grisaceo. Asi el OCR tiene trabajo real
y el preprocesado de ocr.py se nota.

Si prefieres usar documentos reales (fichas IPID de aseguradoras, fotos con el
movil...), basta con dejarlos en ocr/documentos/ y no ejecutar este script.

Uso (una sola vez, desde la carpeta entrega/):
    python ocr/generar_documentos.py
"""

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

CARPETA = Path(__file__).parent / "documentos"
ANCHO = 1700  # ~A4 a 200 ppp
MARGEN = 110

FUENTES = {
    "normal": ["arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"],
    "negrita": ["arialbd.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"],
}


def fuente(estilo: str = "normal", tam: int = 34):
    for nombre in FUENTES[estilo]:
        try:
            return ImageFont.truetype(nombre, tam)
        except OSError:
            continue
    return ImageFont.load_default()


# Contenido de los documentos.
# Cada documento es una lista de bloques: ("titulo", texto), ("texto", texto),
# ("tabla", filas) o ("espacio", px).

DOCUMENTOS = {
    "01_poliza_AB-456789.jpg": [
        ("titulo", "SEGUROS ANGEL - CONDICIONES PARTICULARES"),
        ("texto", "Número de póliza: AB-456789"),
        ("texto", "Tomador: Alonso de la Fuente      NIF: 12345678Z"),
        ("texto", "Vehículo: Seat León 1.5 TSI      Matrícula: 1234 BCD"),
        ("texto", "Modalidad: Todo riesgo con franquicia"),
        ("texto", "Franquicia por siniestro: 300,00 EUR"),
        ("texto", "Fecha de efecto: 14/03/2026      Vencimiento: 14/03/2027"),
        ("texto", "Prima anual total: 412,35 EUR (pago anual domiciliado)"),
        ("espacio", 20),
        ("titulo", "Garantías contratadas"),
        ("texto", "- Responsabilidad civil obligatoria y voluntaria hasta 50.000.000,00 EUR"),
        ("texto", "- Daños propios con franquicia de 300,00 EUR"),
        ("texto", "- Lunas, robo e incendio sin franquicia"),
        ("texto", "- Asistencia en viaje desde el kilómetro 0 con remolque hasta 50 km"),
        ("texto", "- Vehículo de sustitución hasta 7 días por avería o accidente"),
        ("espacio", 20),
        ("texto", "Conductor habitual: el tomador. Se admiten conductores ocasionales"),
        ("texto", "mayores de 25 años con más de 2 años de carné."),
    ],
    "02_condicionado_asistencia.jpg": [
        ("titulo", "CONDICIONADO GENERAL - ASISTENCIA EN CARRETERA"),
        ("texto", "Referencia del documento: CG-AUTO-2026/03"),
        ("texto", "Edición: 01/01/2026"),
        ("espacio", 10),
        ("titulo", "Artículo 1. Ámbito"),
        ("texto", "La asistencia se presta en España, Andorra y el resto de Europa,"),
        ("texto", "las 24 horas del día, los 365 días del año, en el 900 123 456."),
        ("titulo", "Artículo 2. Remolque"),
        ("texto", "Modalidad Terceros básico: no incluye asistencia en carretera."),
        ("texto", "Modalidad Terceros ampliado: remolque hasta 25 km desde el km 25."),
        ("texto", "Modalidades Todo riesgo: remolque hasta 50 km desde el km 0."),
        ("texto", "El exceso de kilómetros se factura a 1,20 EUR por kilómetro."),
        ("titulo", "Artículo 3. Tiempos de llegada"),
        ("texto", "Tiempo máximo de llegada de la grúa: 45 minutos en ciudad y"),
        ("texto", "90 minutos en carretera. Si se supera, se abonan 30,00 EUR."),
        ("titulo", "Artículo 4. Exclusiones"),
        ("texto", "No se cubren vehículos en competiciones, sin ITV en vigor ni los"),
        ("texto", "servicios solicitados directamente a otra empresa sin autorización."),
    ],
    "03_tabla_modalidades.jpg": [
        ("titulo", "SEGUROS ANGEL - MODALIDADES DE SEGURO DE COCHE 2026"),
        ("espacio", 10),
        ("tabla", [
            ["Cobertura", "Terceros básico", "Terceros ampliado", "Todo riesgo franq.", "Todo riesgo"],
            ["Resp. civil", "Sí", "Sí", "Sí", "Sí"],
            ["Lunas", "No", "Sí", "Sí", "Sí"],
            ["Robo e incendio", "No", "Sí", "Sí", "Sí"],
            ["Asistencia", "No", "Desde km 25", "Desde km 0", "Desde km 0"],
            ["Daños propios", "No", "No", "Franquicia 300", "Sin franquicia"],
            ["Coche sustitución", "No", "No", "7 días", "15 días"],
            ["Prima desde", "289,90 EUR", "355,00 EUR", "412,35 EUR", "640,00 EUR"],
        ]),
        ("espacio", 20),
        ("texto", "Precios orientativos para conductor de 35 años en Madrid."),
        ("texto", "Válidos para contrataciones entre el 01/01/2026 y el 31/12/2026."),
    ],
    "04_procedimiento_siniestros.jpg": [
        ("titulo", "CÓMO DECLARAR UN SINIESTRO (PARTE DE ACCIDENTE)"),
        ("texto", "Referencia: PRO-SIN-07      Actualizado: 15/09/2026"),
        ("espacio", 10),
        ("texto", "1. Comunique el siniestro en un plazo máximo de 7 días desde"),
        ("texto", "   que lo conozca, en el 900 123 456 o en la app de Seguros Angel."),
        ("texto", "2. Si hay otro vehículo implicado, rellene la Declaración Amistosa"),
        ("texto", "   de Accidente firmada por ambos conductores."),
        ("texto", "3. Aporte fotos de los daños, el permiso de circulación y el"),
        ("texto", "   número de póliza (formato AB-123456)."),
        ("texto", "4. Un perito contactará con usted en un plazo de 72 horas para"),
        ("texto", "   fijar la cita de peritación en un taller concertado."),
        ("texto", "5. En talleres concertados la reparación no requiere adelantar pago."),
        ("espacio", 10),
        ("texto", "Si hay heridos llame primero al 112."),
    ],
    "05_recibo_prima.jpg": [
        ("titulo", "RECIBO DE PRIMA"),
        ("texto", "Recibo número: RC-2026-004512"),
        ("texto", "Póliza: AB-456790      Tomador: Luis Gómez      NIF: 87654321X"),
        ("texto", "Vehículo: Volkswagen Golf      Matrícula: 5678 FGH"),
        ("texto", "Modalidad: Terceros básico"),
        ("texto", "Periodo: 20/11/2025 a 20/11/2026"),
        ("texto", "Importe total: 289,90 EUR"),
        ("texto", "Fecha de cargo: 20/11/2025"),
        ("espacio", 10),
        ("texto", "Renovación automática salvo aviso con un mes de antelación."),
        ("texto", "Para contratar asistencia en carretera llame al 900 123 456."),
    ],
}


def dibujar(bloques) -> Image.Image:
    alto = 2200
    img = Image.new("L", (ANCHO, alto), 255)
    d = ImageDraw.Draw(img)
    y = MARGEN
    for tipo, contenido in bloques:
        if tipo == "titulo":
            d.text((MARGEN, y), contenido, font=fuente("negrita", 40), fill=0)
            y += 70
        elif tipo == "texto":
            d.text((MARGEN, y), contenido, font=fuente("normal", 34), fill=20)
            y += 52
        elif tipo == "espacio":
            y += contenido
        elif tipo == "tabla":
            cols = len(contenido[0])
            ancho_col = (ANCHO - 2 * MARGEN) // cols
            alto_fila = 64
            for i, fila in enumerate(contenido):
                for j, celda in enumerate(fila):
                    x0 = MARGEN + j * ancho_col
                    d.rectangle([x0, y, x0 + ancho_col, y + alto_fila], outline=0, width=2)
                    f = fuente("negrita" if i == 0 or j == 0 else "normal", 26)
                    d.text((x0 + 10, y + 18), celda, font=f, fill=0)
                y += alto_fila
            y += 20
    return img.crop((0, 0, ANCHO, min(alto, y + MARGEN)))


def ensuciar(img: Image.Image, semilla: int) -> Image.Image:
    """Simula un escaneo: fondo gris, giro leve, desenfoque y ruido."""
    rnd = random.Random(semilla)
    img = img.point(lambda p: 232 if p > 200 else p)          # papel no blanco
    img = img.rotate(rnd.uniform(-1.2, 1.2), expand=True, fillcolor=232,
                     resample=Image.BICUBIC)
    img = img.filter(ImageFilter.GaussianBlur(0.8))
    arr = np.asarray(img).astype(np.int16)
    ruido = np.random.default_rng(semilla).normal(0, 12, arr.shape)
    arr = np.clip(arr + ruido, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def main():
    CARPETA.mkdir(parents=True, exist_ok=True)
    for i, (nombre, bloques) in enumerate(DOCUMENTOS.items()):
        img = ensuciar(dibujar(bloques), semilla=i + 1)
        img.save(CARPETA / nombre, quality=70)  # JPEG, como un escaner real
        print(f"Generado {nombre} ({img.width}x{img.height})")


if __name__ == "__main__":
    main()
