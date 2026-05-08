import os
import pandas as pd
import numpy as np

CARPETA = r'C:\Users\danie\OneDrive\Escritorio\cosas EICU'

VENTANAS = {
    'v4': {
        'entrada':  os.path.join(CARPETA, 'eICU_6_24.csv'),
        'salida':   os.path.join(CARPETA, 'eICU_6_24_definitivo.csv'),
        'etiqueta': 'etiqueta_norad_6_24',
        'excluir_extra': [],                        # sin binarias ni volúmenes extra en v4
    },
    'v4p': {
        'entrada':  os.path.join(CARPETA, 'eICU_3_12.csv'),
        'salida':   os.path.join(CARPETA, 'eICU_3_12_definitivo.csv'),
        'etiqueta': 'etiqueta_norad_3_12',
        'excluir_extra': [],                        # ídem v4p
    },
    'v4l': {
        'entrada':  os.path.join(CARPETA, 'eICU_12_48.csv'),
        'salida':   os.path.join(CARPETA, 'eICU_12_48_definitivo.csv'),
        'etiqueta': 'etiqueta_norad_12_48',
        'excluir_extra': ['ventilacion_invasiva_12h'],  # binaria 0/1
    },
}

# Columnas excluidas de winsorización en todas las ventanas:
#   - IDs y metadatos
#   - Binarias y escalas acotadas por definición (sofa, gcs)
#   - peso_kg: variable demográfica, no predictora del modelo
EXCLUIR_BASE = [
    'subject_id', 'stay_id',
    'anchor_age', 'gender', 'peso_kg',
    'contador_estancia_uci',
    'horas_hasta_norad',
    'gcs_min',      # solo existe en v4l (escala 3-15, acotada)
    'sofa_max',     # escala 0-24, acotada por definición
]


def compute_outlier_winsorize(arr, left_thresh, right_thresh):
    arr = arr.copy().astype(float)
    if arr.notna().sum() == 0:
        return arr, 0, 0
    perc_bajo = np.nanpercentile(arr, left_thresh)
    perc_alto = np.nanpercentile(arr, right_thresh)
    n_bajo = int((arr < perc_bajo).sum())
    n_alto = int((arr > perc_alto).sum())
    arr[arr < perc_bajo] = perc_bajo
    arr[arr > perc_alto] = perc_alto
    return arr, n_bajo, n_alto


for nombre, config in VENTANAS.items():
    print(f'\n{"=" * 50}')
    print(f'  {nombre}')
    print(f'{"=" * 50}')

    df = pd.read_csv(config['entrada'])

    excluir = EXCLUIR_BASE + [config['etiqueta']] + config['excluir_extra']
    cols_clinicas = [c for c in df.columns
                     if c not in excluir and pd.api.types.is_numeric_dtype(df[c])]

    print(f'  Filas: {len(df)}')
    print(f'  Variables a winsorizar: {len(cols_clinicas)}')
    print(f'  Variables: {cols_clinicas}')

    total_bajos, total_altos = 0, 0
    for col in cols_clinicas:
        df[col], n_bajo, n_alto = compute_outlier_winsorize(df[col], 2, 98)
        total_bajos += n_bajo
        total_altos += n_alto

    print(f'  Valores recortados por abajo: {total_bajos}')
    print(f'  Valores recortados por arriba: {total_altos}')
    print(f'  Positivos: {int(df[config["etiqueta"]].sum())}')
    print(f'  Prevalencia: {100 * df[config["etiqueta"]].mean():.2f}%')

    df.to_csv(config['salida'], index=False)
    print(f'  Guardado en: {config["salida"]}')
