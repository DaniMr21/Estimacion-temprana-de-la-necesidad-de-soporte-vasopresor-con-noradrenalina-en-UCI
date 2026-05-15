"""
Tabla 1 — Características basales MIMIC vs eICU
================================================
Genera una tabla comparativa de case-mix para las tres ventanas temporales.

Variables incluidas (disponibles en ambas cohortes):
  - N pacientes
  - Edad (mediana [IQR])
  - Género masculino n (%)
  - Peso kg (mediana [IQR])
  - SOFA máx (mediana [IQR])
  - Primera estancia UCI n (%)
  - Uso de noradrenalina — positivos n (%)
  - Horas hasta noradrenalina en positivos (mediana [IQR])

Test estadístico:
  - Continuas: Mann-Whitney U → p-value
  - Categóricas: Chi-cuadrado → p-value
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── CONFIGURACIÓN ──────────────────────────────────────────────────────────────
CARPETA_MIMIC  = r'C:\Users\danie\OneDrive\Escritorio\DATA'
CARPETA_EICU   = r'C:\Users\danie\OneDrive\Escritorio\DATA'
CARPETA_SALIDA = os.path.dirname(os.path.abspath(__file__))

VENTANAS = {
    'Corto 3-12h': {
        'mimic':    os.path.join(CARPETA_MIMIC, 'definitivo_v4p.csv'),
        'eicu':     os.path.join(CARPETA_EICU,  'eICU_3_12_definitivo.csv'),
        'etiqueta': 'etiqueta_norad_3_12',
    },
    'Medio 6-24h': {
        'mimic':    os.path.join(CARPETA_MIMIC, 'definitivo_v4.csv'),
        'eicu':     os.path.join(CARPETA_EICU,  'eICU_6_24_definitivo.csv'),
        'etiqueta': 'etiqueta_norad_6_24',
    },
    'Largo 12-48h': {
        'mimic':    os.path.join(CARPETA_MIMIC, 'definitivo_v4l.csv'),
        'eicu':     os.path.join(CARPETA_EICU,  'eICU_12_48_definitivo.csv'),
        'etiqueta': 'etiqueta_norad_12_48',
    },
}


# ── FUNCIONES AUXILIARES ───────────────────────────────────────────────────────

def mediana_iqr(serie):
    """Devuelve string 'mediana [Q1 – Q3]'."""
    s = serie.dropna()
    if len(s) == 0:
        return '—'
    med = np.median(s)
    q1  = np.percentile(s, 25)
    q3  = np.percentile(s, 75)
    return f'{med:.1f} [{q1:.1f}–{q3:.1f}]'


def n_pct(serie, valor=1):
    """Devuelve string 'n (%)' para variable binaria."""
    s = serie.dropna()
    n = int((s == valor).sum())
    pct = 100 * n / len(s) if len(s) > 0 else 0
    return f'{n:,} ({pct:.1f}%)'


def test_continua(s1, s2):
    """Mann-Whitney U, devuelve p formateado."""
    s1 = s1.dropna()
    s2 = s2.dropna()
    if len(s1) < 2 or len(s2) < 2:
        return '—'
    _, p = stats.mannwhitneyu(s1, s2, alternative='two-sided')
    if p < 0.001:
        return '<0.001'
    return f'{p:.3f}'


def test_categorica(s1, s2, valor=1):
    """Chi-cuadrado 2×2; si hay celda cero usa Fisher exacto."""
    s1 = s1.dropna()
    s2 = s2.dropna()
    tabla = [
        [int((s1 == valor).sum()), int((s1 != valor).sum())],
        [int((s2 == valor).sum()), int((s2 != valor).sum())],
    ]
    if any(c == 0 for fila in tabla for c in fila):
        _, p = stats.fisher_exact(tabla)
    else:
        try:
            _, p, _, _ = stats.chi2_contingency(tabla)
        except ValueError:
            _, p = stats.fisher_exact(tabla)
    if p < 0.001:
        return '<0.001'
    return f'{p:.3f}'


def codificar_gender(serie):
    """Normaliza género a 1=masculino, 0=femenino, NaN=desconocido.
    Maneja: 'M', 'F', 'Male', 'Female', 'Unknown', numérico."""
    s = serie.copy().astype(str).str.upper().str.strip()
    resultado = pd.Series(np.nan, index=serie.index)
    resultado[s.isin(['M', 'MALE'])]   = 1.0
    resultado[s.isin(['F', 'FEMALE'])] = 0.0
    # 'UNKNOWN' y cualquier otro valor se quedan como NaN
    return resultado


# ── GENERACIÓN DE TABLA ────────────────────────────────────────────────────────

filas_totales = []

for ventana, cfg in VENTANAS.items():
    dm = pd.read_csv(cfg['mimic'])
    de = pd.read_csv(cfg['eicu'])
    etq = cfg['etiqueta']

    # Normalizar género
    dm['gender_bin'] = codificar_gender(dm['gender'])
    de['gender_bin'] = codificar_gender(de['gender'])

    # Primera estancia UCI
    dm['primera_estancia'] = (dm['contador_estancia_uci'] == 1).astype(float)
    de['primera_estancia'] = (de['contador_estancia_uci'] == 1).astype(float)

    # Horas hasta norad solo en positivos
    horas_m = dm.loc[dm[etq] == 1, 'horas_hasta_norad']
    horas_e = de.loc[de[etq] == 1, 'horas_hasta_norad']

    definiciones = [
        # (etiqueta fila, tipo, col_m, col_e, kwargs)
        ('N',                        'n',    None,             None,             {}),
        ('Edad (años)',               'cont', 'anchor_age',     'anchor_age',     {}),
        ('Género masculino',          'cat',  'gender_bin',     'gender_bin',     {}),
        ('Peso (kg)',                 'cont', 'peso_kg',        'peso_kg',        {}),
        ('SOFA máx',                  'cont', 'sofa_max',       'sofa_max',       {}),
        ('Primera estancia UCI',      'cat',  'primera_estancia','primera_estancia',{}),
        ('Noradrenalina (positivos)', 'cat',  etq,              etq,              {}),
        ('Horas hasta norad *',       'cont_sub', None,         None,             {}),
    ]

    bloque = []
    bloque.append({'Variable': f'── {ventana} ──', 'MIMIC': '', 'eICU': '', 'p': ''})

    for etiqueta_fila, tipo, col_m, col_e, _ in definiciones:
        if tipo == 'n':
            fila = {
                'Variable': 'N',
                'MIMIC':    f'{len(dm):,}',
                'eICU':     f'{len(de):,}',
                'p':        '—',
            }
        elif tipo == 'cont':
            fila = {
                'Variable': etiqueta_fila,
                'MIMIC':    mediana_iqr(dm[col_m]),
                'eICU':     mediana_iqr(de[col_e]),
                'p':        test_continua(dm[col_m], de[col_e]),
            }
        elif tipo == 'cat':
            fila = {
                'Variable': etiqueta_fila,
                'MIMIC':    n_pct(dm[col_m]),
                'eICU':     n_pct(de[col_e]),
                'p':        test_categorica(dm[col_m], de[col_e]),
            }
        elif tipo == 'cont_sub':
            fila = {
                'Variable': etiqueta_fila,
                'MIMIC':    mediana_iqr(horas_m),
                'eICU':     mediana_iqr(horas_e),
                'p':        test_continua(horas_m, horas_e),
            }
        bloque.append(fila)

    filas_totales.extend(bloque)

df_tabla = pd.DataFrame(filas_totales)[['Variable', 'MIMIC', 'eICU', 'p']]

# ── IMPRIMIR ───────────────────────────────────────────────────────────────────
print('\n' + '=' * 80)
print('TABLA 1 — Características basales MIMIC (entrenamiento) vs eICU (validación)')
print('Continuas: mediana [IQR]  |  Categóricas: n (%)  |  p: Mann-Whitney / χ²')
print('=' * 80)
print(df_tabla.to_string(index=False))
print('\n* Solo en pacientes que recibieron noradrenalina (positivos)')
print('  IQR = rango intercuartílico  |  UCI = Unidad de Cuidados Intensivos')

# ── GUARDAR CSV ────────────────────────────────────────────────────────────────
ruta_csv = os.path.join(CARPETA_SALIDA, 'tabla1_mimic_vs_eicu.csv')
df_tabla.to_csv(ruta_csv, index=False, encoding='utf-8-sig')
print(f'\nCSV guardado en: {ruta_csv}')

# ── GUARDAR IMAGEN ─────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(13, len(df_tabla) * 0.42 + 1.5))
ax.axis('off')

cabecera = ['Variable', 'MIMIC\n(entrenamiento)', 'eICU\n(validación externa)', 'p-valor']
filas_img = df_tabla.values.tolist()

tabla_img = ax.table(
    cellText=filas_img,
    colLabels=cabecera,
    loc='center',
    cellLoc='center',
)
tabla_img.auto_set_font_size(False)
tabla_img.set_fontsize(8.5)
tabla_img.scale(1, 1.55)

# Estilo cabecera
for j in range(4):
    tabla_img[0, j].set_facecolor('#2c3e50')
    tabla_img[0, j].set_text_props(color='white', fontweight='bold')

# Estilo filas de ventana (separadores)
for i, fila in enumerate(filas_img):
    if str(fila[0]).startswith('──'):
        for j in range(4):
            tabla_img[i + 1, j].set_facecolor('#d5e8f0')
            tabla_img[i + 1, j].set_text_props(fontweight='bold')
    elif i % 2 == 0:
        for j in range(4):
            tabla_img[i + 1, j].set_facecolor('#f7f9fc')

# Columna Variable alineada a la izquierda
for i in range(len(filas_img) + 1):
    tabla_img[i, 0].set_text_props(ha='left')

fig.suptitle(
    'Tabla 1 — Características basales MIMIC vs eICU\n'
    'Continuas: mediana [IQR]  ·  Categóricas: n (%)  ·  p: Mann-Whitney / χ²',
    fontsize=9, y=0.98
)

plt.tight_layout()
ruta_img = os.path.join(CARPETA_SALIDA, 'tabla1_mimic_vs_eicu.png')
plt.savefig(ruta_img, dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Imagen guardada en: {ruta_img}')
print('\n[Fin]')