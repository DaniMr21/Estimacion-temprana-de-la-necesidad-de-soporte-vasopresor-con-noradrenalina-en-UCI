import os
import pandas as pd
import numpy as np

CARPETA = r'C:\Users\danie\OneDrive\Escritorio\DATA'
ENTRADA = os.path.join(CARPETA, 'dataset_final_v3_clean.csv')
SALIDA  = os.path.join(CARPETA, 'definitivo_v3.csv')

df = pd.read_csv(ENTRADA)

def compute_outlier_winsorize(arr, left_thresh, right_thresh):
    """Winsorización 2-98% como en Gupta et al. 2022 (MIMIC-IV-Data-Pipeline).
    Uso nanpercentile porque la tabla ya está agregada por estancia y puede tener NaN."""
    arr = arr.copy().astype(float)

    if arr.notna().sum() == 0:
        return arr, 0, 0

    perc_low  = np.nanpercentile(arr, left_thresh)
    perc_high = np.nanpercentile(arr, right_thresh)

    n_low  = int((arr < perc_low).sum())
    n_high = int((arr > perc_high).sum())

    arr[arr < perc_low]  = perc_low
    arr[arr > perc_high] = perc_high
    return arr, n_low, n_high

# GCS excluido por rango acotado (3-15); IDs y etiqueta también
excluir = ['subject_id', 'hadm_id', 'stay_id', 'anchor_age', 'gender',
           'contador_estancia_uci', 'horas_hasta_norad', 'etiqueta_norad_6_24',
           'gcs_media', 'gcs_min', 'gcs_max']

cols_clinicas = [c for c in df.columns
                 if c not in excluir and pd.api.types.is_numeric_dtype(df[c])]

print(f"Filas antes: {len(df)}")
print(f"Variables a winsorizar: {len(cols_clinicas)}")

total_low, total_high = 0, 0
for col in cols_clinicas:
    df[col], n_low, n_high = compute_outlier_winsorize(df[col], 2, 98)
    total_low  += n_low
    total_high += n_high

print(f"Filas después: {len(df)}")
print(f"Valores recortados por abajo: {total_low}")
print(f"Valores recortados por arriba: {total_high}")
print(f"Positivos: {df['etiqueta_norad_6_24'].sum()}")
print(f"Prevalencia: {100*df['etiqueta_norad_6_24'].mean():.2f}%")

df.to_csv(SALIDA, index=False)
print(f"✅ Guardado en {SALIDA}")