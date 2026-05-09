import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, chi2_contingency
from statsmodels.stats.multitest import multipletests


# CONFIGURACIÓN

RUTA_CSV = r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4.csv'

CARPETA_BASE = os.path.dirname(__file__) if '__file__' in dir() else '.'
CARPETA_TABLAS = os.path.join(CARPETA_BASE, 'tablas')
os.makedirs(CARPETA_TABLAS, exist_ok=True)

ETIQUETA = 'etiqueta_norad_6_24'

variables_predictoras = [
    # Demografía y contexto (4)
    'anchor_age',
    'gender',
    'peso_kg',
    'contador_estancia_uci',

    # Hemodinámica (2)
    'map_min',
    'hr_media',

    # Respiratorio (4)
    'pf_min',
    'spo2_min',
    'fio2_max',
    'rr_max',

    # Ventilación y conciencia (2)
    'ventilacion_invasiva_6h',
    'gcs_min',

    # Renal (2)
    'creatinina_max',
    'diuresis_ml_kg_6h',

    # Ácido-base (3)
    'lactato_max',
    'ph_min',
    'bicarbonato_min',

    # Hepático (2)
    'bilirrubina_media',
    'gpt_max',

    # Coagulación (2)
    'tp_max',
    'plaquetas_min',

    # Hematología/inflamación (2)
    'leucocitos_min',
    'hemoglobina_min',

    # Metabólico (1)
    'glucemia_min',

    # Otro vital (1)
    'temp_min',

    # Gravedad global (1)
    'sofa_max',
]

variables_binarias = ['gender', 'ventilacion_invasiva_6h']
variables_continuas = [v for v in variables_predictoras if v not in variables_binarias]


# 1. CARGA Y FILTRADO A PRIMERA ESTANCIA POR PACIENTE

print("SIGNIFICANCIA UNIVARIANTE — SET REDUCIDO v4 (26 variables)")


df = pd.read_csv(RUTA_CSV)
print(f"\nDataset cargado:")
print(f"  Estancias totales      : {len(df)}")
print(f"  Pacientes únicos       : {df['subject_id'].nunique()}")

df = (df.sort_values(['subject_id', 'contador_estancia_uci'])
        .drop_duplicates('subject_id', keep='first')
        .reset_index(drop=True))
print(f"  Tras filtrar 1ª estancia por paciente:")
print(f"    Estancias            : {len(df)}")
print(f"    Positivos (etiq=1)   : {df[ETIQUETA].sum()} "
      f"({100*df[ETIQUETA].mean():.2f}%)")
print(f"    Negativos (etiq=0)   : {(df[ETIQUETA]==0).sum()} "
      f"({100*(df[ETIQUETA]==0).mean():.2f}%)")

# 2. PREPARACIÓN

df_test = df.copy()
df_test['gender'] = (df_test['gender'] == 'M').astype(int)

print(f"Variables a contrastar   : {len(variables_predictoras)}")
print(f"  Continuas (Mann-Whitney): {len(variables_continuas)}")
print(f"  Binarias  (Chi-cuadrado): {len(variables_binarias)}")
print()


# 3. TESTS UNIVARIANTES

print("[1/2] Aplicando tests univariantes")

filas = []

# 3a. Variables continuas — Mann-Whitney U
for var in variables_continuas:
    pos = df_test.loc[df_test[ETIQUETA] == 1, var].dropna()
    neg = df_test.loc[df_test[ETIQUETA] == 0, var].dropna()

    if len(pos) < 2 or len(neg) < 2:
        filas.append({
            'variable': var,
            'test': 'Mann-Whitney U',
            'n_pos': len(pos),
            'n_neg': len(neg),
            'mediana_pos': pos.median() if len(pos) > 0 else np.nan,
            'mediana_neg': neg.median() if len(neg) > 0 else np.nan,
            'estadistico': np.nan,
            'p_valor': np.nan,
        })
        continue

    stat, p = mannwhitneyu(pos, neg, alternative='two-sided')
    filas.append({
        'variable': var,
        'test': 'Mann-Whitney U',
        'n_pos': len(pos),
        'n_neg': len(neg),
        'mediana_pos': round(pos.median(), 4),
        'mediana_neg': round(neg.median(), 4),
        'estadistico': round(stat, 2),
        'p_valor': p,
    })

# 3b. Variables binarias — Chi-cuadrado
for var in variables_binarias:
    tabla = pd.crosstab(df_test[var], df_test[ETIQUETA])
    chi2, p, dof, _ = chi2_contingency(tabla)

    prop_pos = df_test.loc[df_test[ETIQUETA] == 1, var].mean()
    prop_neg = df_test.loc[df_test[ETIQUETA] == 0, var].mean()

    filas.append({
        'variable': var,
        'test': 'Chi-cuadrado',
        'n_pos': int((df_test[ETIQUETA] == 1).sum()),
        'n_neg': int((df_test[ETIQUETA] == 0).sum()),
        'mediana_pos': round(prop_pos, 4),  # proporción de 1s en grupo positivo
        'mediana_neg': round(prop_neg, 4),  # proporción de 1s en grupo negativo
        'estadistico': round(chi2, 2),
        'p_valor': p,
    })

resultados = pd.DataFrame(filas)
print(f"  Tests realizados        : {len(resultados)}")
print()


# 4. CORRECCIÓN POR COMPARACIONES MÚLTIPLES (FDR Benjamini-Hochberg)

print("--------------------------------------------------")
print("[2/2] Corrección por comparaciones múltiples (FDR Benjamini-Hochberg)")
print("--------------------------------------------------")

mascara_validos = resultados['p_valor'].notna()
p_validos = resultados.loc[mascara_validos, 'p_valor'].values

rechaza, p_corr, _, _ = multipletests(p_validos, method='fdr_bh', alpha=0.05)

resultados['p_valor_BH'] = np.nan
resultados['significativa_BH'] = False
resultados.loc[mascara_validos, 'p_valor_BH'] = p_corr
resultados.loc[mascara_validos, 'significativa_BH'] = rechaza

# Marca de "significancia bruta" sin corregir, para comparación
resultados['significativa_bruta'] = resultados['p_valor'] < 0.05

resultados = resultados.sort_values('p_valor').reset_index(drop=True)


# 5. RESULTADOS

print()
print("Tabla completa (ordenada por p-valor ascendente):")
print()

# Formateo de p-valores para que se lean bien
def formatea_p(p):
    if pd.isna(p):
        return 'NaN'
    if p < 1e-4:
        return f'{p:.2e}'
    return f'{p:.4f}'

tabla_print = resultados.copy()
tabla_print['p_valor'] = tabla_print['p_valor'].apply(formatea_p)
tabla_print['p_valor_BH'] = tabla_print['p_valor_BH'].apply(formatea_p)

print(tabla_print.to_string(index=False))
print()

# Resumen
n_total = len(resultados)
n_sig_brutas = resultados['significativa_bruta'].sum()
n_sig_bh = resultados['significativa_BH'].sum()

print("RESUMEN")

print(f"  Variables analizadas              : {n_total}")
print(f"  Significativas (p < 0.05 sin corregir): {n_sig_brutas}")
print(f"  Significativas (BH-FDR q < 0.05)      : {n_sig_bh}")
print()

print(f"  Significativas tras BH-FDR:")
sig = resultados[resultados['significativa_BH']]['variable'].tolist()
for v in sig:
    print(f"    - {v}")
print()

print(f"  NO significativas tras BH-FDR:")
no_sig = resultados[~resultados['significativa_BH']]['variable'].tolist()
for v in no_sig:
    print(f"    - {v}")
print()


# 6. GUARDADO

ruta_salida = os.path.join(CARPETA_TABLAS, 'significancia_univariante_v4_reducido.csv')
resultados.to_csv(ruta_salida, index=False)
print(f"Tabla guardada en: {ruta_salida}")