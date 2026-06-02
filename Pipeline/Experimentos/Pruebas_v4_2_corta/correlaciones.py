import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from statsmodels.stats.outliers_influence import variance_inflation_factor


RUTA_CSV = r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4p.csv'

CARPETA_BASE = os.path.dirname(__file__) if '__file__' in dir() else '.'
CARPETA_FIGURAS = os.path.join(CARPETA_BASE, 'figuras')
CARPETA_TABLAS = os.path.join(CARPETA_BASE, 'tablas')

os.makedirs(CARPETA_FIGURAS, exist_ok=True)
os.makedirs(CARPETA_TABLAS, exist_ok=True)


print("---------------------")
print("CORRELACIONES — v4p SET REDUCIDO")
print("Ventana: observación 0-3h / predicción 3-12h")
print("-----------------------------")

df = pd.read_csv(RUTA_CSV)
df = df.dropna(subset=['pf_max'])

print(f"\nDataset cargado:")
print(f"  Estancias totales : {len(df)}")
print(f"  Pacientes únicos  : {df['subject_id'].nunique()}")

df = (
    df.sort_values(['subject_id', 'contador_estancia_uci'])
      .drop_duplicates('subject_id', keep='first')
      .reset_index(drop=True)
)

print(f"  Tras filtrar 1ª estancia por paciente:")
print(f"    Estancias       : {len(df)}")
print(f"    Positivos       : {df['etiqueta_norad_3_12'].sum()} "
      f"({100 * df['etiqueta_norad_3_12'].mean():.2f}%)")
print()


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
    'gcs_min',
    'ventilacion_invasiva_3h',
    # Renal (2)
    'creatinina_max',
    'diuresis_ml_kg_3h',

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

variables_binarias = [
    'gender',
    'ventilacion_invasiva_3h',
]

variables_continuas = [
    v for v in variables_predictoras
    if v not in variables_binarias
]

X = df[variables_predictoras].copy()
X['gender'] = (X['gender'] == 'M').astype(int)

print(f"Variables totales   : {len(variables_predictoras)}")
print(f"  Continuas         : {len(variables_continuas)}")
print(f"  Binarias          : {len(variables_binarias)}")
print()


print("-----------------")
print("[1/4] Matriz de correlación de Spearman")
print("---------------------------")

matriz_spearman = X.corr(method='spearman')

fig, ax = plt.subplots(figsize=(12, 10))
mascara = np.triu(np.ones_like(matriz_spearman, dtype=bool), k=1)

sns.heatmap(
    matriz_spearman,
    mask=mascara,
    cmap='RdBu_r',
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    annot=True,
    fmt='.2f',
    annot_kws={'size': 7},
    cbar_kws={'shrink': 0.6, 'label': 'rho de Spearman'},
    linewidths=0.4,
    ax=ax,
)

ax.set_title(
    'Matriz de correlación de Spearman — v4p reducido',
    fontsize=13,
    pad=15
)

plt.tight_layout()

ruta_heatmap = os.path.join(
    CARPETA_FIGURAS,
    'correlacion_spearman_v4p_reducido.png'
)

plt.savefig(ruta_heatmap, dpi=150, bbox_inches='tight')
plt.close()

ruta_matriz = os.path.join(
    CARPETA_TABLAS,
    'matriz_spearman_v4p_reducido.csv'
)

matriz_spearman.round(3).to_csv(ruta_matriz)

print(f"  Heatmap guardado en : {ruta_heatmap}")
print(f"  Matriz guardada en  : {ruta_matriz}")
print()


print("-------------------")
print("[2/4] Pares de correlación")
print("----------------------")

pares = (
    matriz_spearman
    .where(np.triu(np.ones_like(matriz_spearman, dtype=bool), k=1))
    .stack()
    .reset_index()
)

pares.columns = ['variable_1', 'variable_2', 'rho_spearman']
pares['rho_abs'] = pares['rho_spearman'].abs()
pares = pares.sort_values('rho_abs', ascending=False).reset_index(drop=True)

print(f"  Total de pares        : {len(pares)}")
print(f"  Pares con |rho| > 0.7 : {(pares['rho_abs'] > 0.7).sum()}")
print(f"  Pares con |rho| > 0.5 : {(pares['rho_abs'] > 0.5).sum()}")
print(f"  Pares con |rho| > 0.3 : {(pares['rho_abs'] > 0.3).sum()}")
print(f"  Correlación media     : {pares['rho_abs'].mean():.3f}")
print(f"  Correlación mediana   : {pares['rho_abs'].median():.3f}")

print("\nTop 20 pares más correlacionados:")
print(
    pares.head(20)[['variable_1', 'variable_2', 'rho_spearman']]
    .to_string(index=False)
)

ruta_pares = os.path.join(
    CARPETA_TABLAS,
    'pares_correlacion_v4p_reducido.csv'
)

pares.drop(columns='rho_abs').to_csv(ruta_pares, index=False)

print(f"\n  Tabla guardada en: {ruta_pares}")
print()


print("-----------------")
print("[3/4] VIF")
print("--------------------------")

X_continuas = X[variables_continuas].copy()

X_std = (X_continuas - X_continuas.mean()) / X_continuas.std()
X_std_int = X_std.copy()
X_std_int.insert(0, 'intercepto', 1.0)

vif_vals = []

for i, col in enumerate(X_std_int.columns):
    if col == 'intercepto':
        continue

    try:
        vif = variance_inflation_factor(X_std_int.values, i)
    except Exception:
        vif = np.nan

    vif_vals.append({
        'variable': col,
        'vif': vif
    })

vif_df = (
    pd.DataFrame(vif_vals)
    .sort_values('vif', ascending=False)
    .reset_index(drop=True)
)

print(vif_df.round(2).to_string(index=False))

print("\nResumen por umbral:")
print(f"  VIF > 10 : {(vif_df['vif'] > 10).sum()}")
print(f"  VIF 5-10 : {((vif_df['vif'] > 5) & (vif_df['vif'] <= 10)).sum()}")
print(f"  VIF 2-5  : {((vif_df['vif'] > 2) & (vif_df['vif'] <= 5)).sum()}")
print(f"  VIF < 2  : {(vif_df['vif'] <= 2).sum()}")

ruta_vif = os.path.join(
    CARPETA_TABLAS,
    'vif_v4p_reducido.csv'
)

vif_df.to_csv(ruta_vif, index=False)

print(f"\n  VIF guardado en: {ruta_vif}")
print()


print("----------------------------------")
print("Análisis separado de variables binarias")
print("------------------------")

filas = []

for i, b1 in enumerate(variables_binarias):
    for b2 in variables_binarias[i + 1:]:
        phi = X[[b1, b2]].corr(method='pearson').iloc[0, 1]
        filas.append({
            'variable_1': b1,
            'variable_2': b2,
            'tipo': 'binaria-binaria (phi)',
            'asociacion': round(phi, 3)
        })

for b in variables_binarias:
    for c in variables_continuas:
        rho = X[[b, c]].corr(method='spearman').iloc[0, 1]

        if abs(rho) > 0.15:
            filas.append({
                'variable_1': b,
                'variable_2': c,
                'tipo': 'binaria-continua (rank-biserial)',
                'asociacion': round(rho, 3)
            })

asoc_binarias = (
    pd.DataFrame(filas)
    .sort_values(by='asociacion', key=lambda s: s.abs(), ascending=False)
    .reset_index(drop=True)
)

print(asoc_binarias.to_string(index=False))

ruta_asoc = os.path.join(
    CARPETA_TABLAS,
    'asociaciones_binarias_v4p_reducido.csv'
)

asoc_binarias.to_csv(ruta_asoc, index=False)

print(f"\n  Asociaciones binarias guardadas en: {ruta_asoc}")
print()


print("------------------")
print("[4/4] Dendrograma")
print("-------------------")

distancia = (1 - matriz_spearman.abs()).to_numpy().copy()
np.fill_diagonal(distancia, 0)

distancia_condensada = squareform(distancia, checks=False)
enlaces = linkage(distancia_condensada, method='average')

fig, ax = plt.subplots(figsize=(12, 9))

dendrogram(
    enlaces,
    labels=matriz_spearman.columns.tolist(),
    orientation='left',
    leaf_font_size=9,
    color_threshold=0.7,
    ax=ax,
)

ax.set_title(
    'Dendrograma — v4p reducido\nDistancia = 1 - |rho Spearman|',
    fontsize=12,
    pad=10
)

ax.set_xlabel('Distancia (1 - |rho|)')
ax.axvline(
    x=0.3,
    color='red',
    linestyle='--',
    linewidth=1,
    label='|rho| = 0.7'
)
ax.axvline(
    x=0.5,
    color='orange',
    linestyle='--',
    linewidth=1,
    label='|rho| = 0.5'
)

ax.legend(loc='lower right')
plt.tight_layout()

ruta_dendro = os.path.join(
    CARPETA_FIGURAS,
    'dendrograma_v4p_reducido.png'
)

plt.savefig(ruta_dendro, dpi=150, bbox_inches='tight')
plt.close()

print(f"  Dendrograma guardado en: {ruta_dendro}")
print()


print("=" * 70)
print("RESUMEN EJECUTIVO — v4p REDUCIDO")
print("=" * 70)
print(f"  Estancias analizadas        : {len(df)}")
print(f"  Variables analizadas        : {len(variables_predictoras)}")
print(f"  Pares con |rho| > 0.7       : {(pares['rho_abs'] > 0.7).sum()}")
print(f"  Pares con |rho| > 0.5       : {(pares['rho_abs'] > 0.5).sum()}")
print(f"  Variables con VIF > 10      : {(vif_df['vif'] > 10).sum()}")
print(f"  Variables con VIF 5-10      : {((vif_df['vif'] > 5) & (vif_df['vif'] <= 10)).sum()}")
print()
print("Ficheros generados:")
print(f"  Figuras : {CARPETA_FIGURAS}")
print(f"  Tablas  : {CARPETA_TABLAS}")