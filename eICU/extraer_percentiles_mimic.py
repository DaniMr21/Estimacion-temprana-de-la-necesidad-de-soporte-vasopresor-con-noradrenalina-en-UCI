"""
Paso 1: Extraer percentiles de winsorización desde MIMIC (datos de entrenamiento)
y guardarlos en un CSV para aplicarlos después a eICU.

Ejecutar UNA SOLA VEZ sobre los datos de MIMIC.
"""

import os
import pandas as pd
import numpy as np

CARPETA_MIMIC = r'C:\Users\danie\OneDrive\Escritorio\DATA'
CARPETA_SALIDA = r'C:\Users\danie\OneDrive\Escritorio\DATA'

VENTANAS_MIMIC = {
    'v4p': {
        'entrada':  os.path.join(CARPETA_MIMIC, 'definitivo_v4p.csv'),
        'etiqueta': 'etiqueta_norad_3_12',
        'excluir_extra': [],
    },
    'v4': {
        'entrada':  os.path.join(CARPETA_MIMIC, 'definitivo_v4.csv'),
        'etiqueta': 'etiqueta_norad_6_24',
        'excluir_extra': [],
    },
    'v4l': {
        'entrada':  os.path.join(CARPETA_MIMIC, 'definitivo_v4l.csv'),
        'etiqueta': 'etiqueta_norad_12_48',
        'excluir_extra': ['ventilacion_invasiva_12h'],
    },
}

EXCLUIR_BASE = [
    'subject_id', 'stay_id', 'hadm_id',
    'anchor_age', 'gender', 'peso_kg',
    'contador_estancia_uci',
    'horas_hasta_norad',
    'gcs_min',
    'sofa_max',
]

for nombre, config in VENTANAS_MIMIC.items():
    print(f'\n{"=" * 50}')
    print(f'  {nombre} — extrayendo percentiles de MIMIC')
    print(f'{"=" * 50}')

    df = pd.read_csv(config['entrada'])

    excluir = EXCLUIR_BASE + [config['etiqueta']] + config['excluir_extra']
    cols_clinicas = [c for c in df.columns
                     if c not in excluir and pd.api.types.is_numeric_dtype(df[c])]

    print(f'  Variables: {cols_clinicas}')

    registros = []
    for col in cols_clinicas:
        vals = df[col].dropna().astype(float)
        if len(vals) == 0:
            continue
        p2  = float(np.percentile(vals, 2))
        p98 = float(np.percentile(vals, 98))
        registros.append({'variable': col, 'p2': p2, 'p98': p98})
        print(f'  {col:35s} p2={p2:.4f}  p98={p98:.4f}')

    df_percentiles = pd.DataFrame(registros)
    ruta_salida = os.path.join(CARPETA_SALIDA, f'percentiles_mimic_{nombre}.csv')
    df_percentiles.to_csv(ruta_salida, index=False)
    print(f'\n  Percentiles guardados en: {ruta_salida}')

print('\n[Fin] Percentiles extraídos de MIMIC listos para aplicar a eICU.')
