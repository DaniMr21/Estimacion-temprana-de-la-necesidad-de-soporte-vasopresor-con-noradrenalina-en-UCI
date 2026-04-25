import os
import pandas as pd
import numpy as np

CARPETA = r'C:\Users\danie\OneDrive\Escritorio\DATA'
ENTRADA = os.path.join(CARPETA, 'dataset_final_v4p.csv')
SALIDA  = os.path.join(CARPETA, 'definitivo_v4p.csv')

df = pd.read_csv(ENTRADA)

def compute_outlier_winsorize(arr, left_thresh, right_thresh):
    """Winsorización 2-98%.
    Uso nanpercentile porque la tabla ya está agregada por estancia y puede tener NaN."""
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

# Se excluyen de la winsorización:
#  - IDs y etiquetas (no son clínicas).
#  - Binarias (tiene_sepsis, ventilacion_invasiva_3h): 0/1, winsorizar no tiene sentido.
#  - GCS (3-15) y SOFA (0-24): rangos acotados por definición del score.
excluir = [
    'subject_id', 'hadm_id', 'stay_id',
    'anchor_age', 'gender',
    'contador_estancia_uci',
    'horas_hasta_norad', 'etiqueta_norad_3_12',
    'gcs_media', 'gcs_min', 'gcs_max',
    'sofa_media', 'sofa_min', 'sofa_max',
    'tiene_sepsis',
    'ventilacion_invasiva_3h',
]

cols_clinicas = [c for c in df.columns
                 if c not in excluir and pd.api.types.is_numeric_dtype(df[c])]

print(f"Filas antes: {len(df)}")
print(f"Variables a winsorizar: {len(cols_clinicas)}")

total_bajos, total_altos = 0, 0
for col in cols_clinicas:
    df[col], n_bajo, n_alto = compute_outlier_winsorize(df[col], 2, 98)
    total_bajos += n_bajo
    total_altos += n_alto

print(f"Filas después: {len(df)}")
print(f"Valores recortados por abajo: {total_bajos}")
print(f"Valores recortados por arriba: {total_altos}")
print(f"Positivos: {df['etiqueta_norad_3_12'].sum()}")
print(f"Prevalencia: {100*df['etiqueta_norad_3_12'].mean():.2f}%")

df.to_csv(SALIDA, index=False)
print(f"Guardado en {SALIDA}")
