"""
Análisis de significancia univariante — ventana LARGA v4l (25 variables).
Observación: 0-12h | Predicción: 12-48h
Etiqueta: etiqueta_norad_12_48

Salidas:
  - tablas/significancia_univariante_v4l.csv
"""

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, chi2_contingency
from statsmodels.stats.multitest import multipletests

# ── CONFIGURACIÓN ──────────────────────────────────────────────────────────────
RUTA_CSV    = r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4l.csv'
ETIQUETA    = 'etiqueta_norad_12_48'

CARPETA_BASE   = os.path.dirname(__file__) if '__file__' in dir() else '.'
CARPETA_TABLAS = os.path.join(CARPETA_BASE, 'tablas')
os.makedirs(CARPETA_TABLAS, exist_ok=True)

variables_predictoras = [
    'anchor_age', 'gender', 'peso_kg', 'contador_estancia_uci',
    'map_min', 'hr_media',
    'pf_min', 'spo2_min', 'rr_max',            # sin fio2_max
    'ventilacion_invasiva_12h', 'gcs_min',      # 12h
    'creatinina_max', 'diuresis_ml_kg_12h',     # 12h
    'lactato_max', 'ph_min', 'bicarbonato_min',
    'bilirrubina_media', 'gpt_max',
    'tp_max', 'plaquetas_min',
    'leucocitos_min', 'hemoglobina_min',
    'glucemia_min', 'temp_min', 'sofa_max',
]
variables_binarias  = ['gender', 'ventilacion_invasiva_12h']
variables_continuas = [v for v in variables_predictoras if v not in variables_binarias]

# ── CARGA Y FILTRADO ───────────────────────────────────────────────────────────
print("─" * 60)
print("SIGNIFICANCIA UNIVARIANTE — v4l LARGA (25 variables)")
print("─" * 60)

df = pd.read_csv(RUTA_CSV)
df = df.dropna(subset=['pf_max'])
print(f"\nEstancias totales      : {len(df)}")
print(f"Pacientes únicos       : {df['subject_id'].nunique()}")

df = (df.sort_values(['subject_id', 'contador_estancia_uci'])
        .drop_duplicates('subject_id', keep='first')
        .reset_index(drop=True))
print(f"Tras filtrar 1ª estancia:")
print(f"  Estancias  : {len(df)}")
print(f"  Positivos  : {df[ETIQUETA].sum()} ({100*df[ETIQUETA].mean():.2f}%)\n")

df_test = df.copy()
df_test['gender'] = (df_test['gender'] == 'M').astype(int)

# ── TESTS ──────────────────────────────────────────────────────────────────────
filas = []

for var in variables_continuas:
    pos = df_test.loc[df_test[ETIQUETA] == 1, var].dropna()
    neg = df_test.loc[df_test[ETIQUETA] == 0, var].dropna()
    if len(pos) < 2 or len(neg) < 2:
        filas.append({'variable': var, 'test': 'Mann-Whitney U',
                      'n_pos': len(pos), 'n_neg': len(neg),
                      'mediana_pos': np.nan, 'mediana_neg': np.nan,
                      'estadistico': np.nan, 'p_valor': np.nan})
        continue
    stat, p = mannwhitneyu(pos, neg, alternative='two-sided')
    filas.append({'variable': var, 'test': 'Mann-Whitney U',
                  'n_pos': len(pos), 'n_neg': len(neg),
                  'mediana_pos': round(pos.median(), 4),
                  'mediana_neg': round(neg.median(), 4),
                  'estadistico': round(stat, 2), 'p_valor': p})

for var in variables_binarias:
    tabla = pd.crosstab(df_test[var], df_test[ETIQUETA])
    chi2, p, _, _ = chi2_contingency(tabla)
    filas.append({'variable': var, 'test': 'Chi-cuadrado',
                  'n_pos': int((df_test[ETIQUETA] == 1).sum()),
                  'n_neg': int((df_test[ETIQUETA] == 0).sum()),
                  'mediana_pos': round(df_test.loc[df_test[ETIQUETA]==1, var].mean(), 4),
                  'mediana_neg': round(df_test.loc[df_test[ETIQUETA]==0, var].mean(), 4),
                  'estadistico': round(chi2, 2), 'p_valor': p})

resultados = pd.DataFrame(filas)

# ── CORRECCIÓN FDR ─────────────────────────────────────────────────────────────
mask = resultados['p_valor'].notna()
rechaza, p_corr, _, _ = multipletests(resultados.loc[mask, 'p_valor'], method='fdr_bh', alpha=0.05)
resultados['p_valor_BH']       = np.nan
resultados['significativa_BH'] = False
resultados.loc[mask, 'p_valor_BH']       = p_corr
resultados.loc[mask, 'significativa_BH'] = rechaza
resultados['significativa_bruta'] = resultados['p_valor'] < 0.05
resultados = resultados.sort_values('p_valor').reset_index(drop=True)

# ── RESULTADOS ─────────────────────────────────────────────────────────────────
def fmt_p(p):
    return 'NaN' if pd.isna(p) else (f'{p:.2e}' if p < 1e-4 else f'{p:.4f}')

rp = resultados.copy()
rp['p_valor']    = rp['p_valor'].apply(fmt_p)
rp['p_valor_BH'] = rp['p_valor_BH'].apply(fmt_p)
print(rp.to_string(index=False))

n_sig = resultados['significativa_BH'].sum()
print(f"\nSignificativas tras BH-FDR: {n_sig} / {len(resultados)}")
print("Significativas:")
for v in resultados[resultados['significativa_BH']]['variable']:
    print(f"  - {v}")
print("NO significativas:")
for v in resultados[~resultados['significativa_BH']]['variable']:
    print(f"  - {v}")

ruta = os.path.join(CARPETA_TABLAS, 'significancia_univariante_v4l.csv')
resultados.to_csv(ruta, index=False)
print(f"\nTabla guardada en: {ruta}")
