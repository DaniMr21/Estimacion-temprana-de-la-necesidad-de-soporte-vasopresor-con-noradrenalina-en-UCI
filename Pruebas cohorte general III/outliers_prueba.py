import os
import pandas as pd
import numpy as np

ruta = os.path.join(os.path.dirname(__file__), 'definitivo.csv')
df = pd.read_csv(ruta)

def compute_outlier_winsorize(arr, left_thresh, right_thresh):
    perc_low = np.percentile(arr, left_thresh)
    perc_high = np.percentile(arr, right_thresh)
    arr = arr.copy().astype(float)  # conversión aquí
    arr[arr < perc_low] = perc_low
    arr[arr > perc_high] = perc_high
    return arr

excluir = ['subject_id', 'hadm_id', 'stay_id', 'anchor_age', 'gender',
           'contador_estancia_uci', 'horas_hasta_norad', 'etiqueta_norad_6_24']

cols_clinicas = [c for c in df.columns if c not in excluir]

print(f"Filas antes: {len(df)}")

for col in cols_clinicas:
    df[col] = compute_outlier_winsorize(df[col], left_thresh=2, right_thresh=98)

print(f"Filas después: {len(df)}")
print(f"Positivos: {df['etiqueta_norad_6_24'].sum()}")
print(f"Prevalencia: {100*df['etiqueta_norad_6_24'].mean():.2f}%")

df.to_csv(os.path.join(os.path.dirname(__file__), 'definitivo_clean.csv'), index=False)
print("Guardado como definitivo_clean.csv")
