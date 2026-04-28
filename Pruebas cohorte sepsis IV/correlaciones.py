"""
Análisis de correlación y multicolinealidad sobre el dataset v4
(observación 0-6h, predicción 6-24h).

Cuatro análisis complementarios:

  1. Matriz de correlación de Spearman (no Pearson, por asimetría
     habitual de las variables clínicas).
  2. Pares de variables con |rho| > 0.7 (alta colinealidad).
  3. VIF (Variance Inflation Factor) sobre variables continuas
     (excluye binarias, que se reportan aparte con asociaciones).
  4. Dendrograma de agrupamiento jerárquico de variables.

Decisiones metodológicas:
  - Se utiliza el CSV winsorizado al 2-98%, que es el mismo que usa el
    modelo, así las correlaciones son las que el modelo "ve".
  - Para garantizar independencia entre observaciones, se filtra a la
    primera estancia por paciente (97.5% del dataset; 84 estancias
    excluidas de ingresos repetidos del mismo paciente).
  - Las variables binarias (gender, tiene_sepsis, ventilacion_invasiva_6h)
    se excluyen del cálculo de VIF y se analizan por separado mediante
    asociaciones específicas para variables binarias.

Salidas:
  - figuras/correlacion_spearman_v4.png
  - figuras/dendrograma_v4.png
  - tablas/pares_alta_correlacion_v4.csv
  - tablas/vif_v4.csv
  - tablas/asociaciones_binarias_v4.csv
"""

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


# -----------------------------------------------------------------------------
# CONFIGURACIÓN
# -----------------------------------------------------------------------------
RUTA_CSV = r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4.csv'

CARPETA_BASE = os.path.dirname(__file__) if '__file__' in dir() else '.'
CARPETA_FIGURAS = os.path.join(CARPETA_BASE, 'figuras')
CARPETA_TABLAS  = os.path.join(CARPETA_BASE, 'tablas')
os.makedirs(CARPETA_FIGURAS, exist_ok=True)
os.makedirs(CARPETA_TABLAS, exist_ok=True)


# -----------------------------------------------------------------------------
# 1. CARGA Y FILTRADO A PRIMERA ESTANCIA POR PACIENTE
# -----------------------------------------------------------------------------
print("=" * 70)
print("ANÁLISIS DE CORRELACIÓN Y MULTICOLINEALIDAD — DATASET v4")
print("=" * 70)

df = pd.read_csv(RUTA_CSV)
df = df.dropna(subset=['pf_max'])
print(f"\nDataset cargado:")
print(f"  Estancias totales      : {len(df)}")
print(f"  Pacientes únicos       : {df['subject_id'].nunique()}")

df = (df.sort_values(['subject_id', 'contador_estancia_uci'])
        .drop_duplicates('subject_id', keep='first')
        .reset_index(drop=True))
print(f"  Tras filtrar 1ª estancia por paciente:")
print(f"    Estancias            : {len(df)}")
print(f"    Positivos            : {df['etiqueta_norad_6_24'].sum()} "
      f"({100*df['etiqueta_norad_6_24'].mean():.2f}%)")
print()


# -----------------------------------------------------------------------------
# 2. SELECCIÓN DE VARIABLES
# -----------------------------------------------------------------------------
variables_predictoras = [
    'anchor_age', 'gender', 'contador_estancia_uci',
    'tiene_sepsis',
    'sofa_media', 'sofa_min', 'sofa_max',
    'ventilacion_invasiva_6h',
    'lactato_media', 'lactato_min', 'lactato_max',
    'creatinina_media', 'creatinina_min', 'creatinina_max',
    'plaquetas_media', 'plaquetas_min', 'plaquetas_max',
    'bilirrubina_media', 'bilirrubina_min', 'bilirrubina_max',
    'tp_media', 'tp_min', 'tp_max',
    'gpt_media', 'gpt_min', 'gpt_max',
    'got_media', 'got_min', 'got_max',
    'pao2_media', 'pao2_min', 'pao2_max',
    'ph_media', 'ph_min', 'ph_max',
    'leucocitos_media', 'leucocitos_min', 'leucocitos_max',
    'paco2_media', 'paco2_min', 'paco2_max',
    'bicarbonato_media', 'bicarbonato_min', 'bicarbonato_max',
    'glucemia_media', 'glucemia_min', 'glucemia_max',
    'hemoglobina_media', 'hemoglobina_min', 'hemoglobina_max',
    'hr_media', 'hr_min', 'hr_max',
    'rr_media', 'rr_min', 'rr_max',
    'temp_media', 'temp_min', 'temp_max',
    'spo2_media', 'spo2_min', 'spo2_max',
    'map_media', 'map_min', 'map_max',
    'fio2_media', 'fio2_min', 'fio2_max',
    'pf_media', 'pf_min', 'pf_max',
    'gcs_media', 'gcs_min', 'gcs_max',
    'peso_kg',
    'diuresis_ml_kg_6h',
]

variables_binarias = ['gender', 'tiene_sepsis', 'ventilacion_invasiva_6h']
variables_continuas = [v for v in variables_predictoras if v not in variables_binarias]

X = df[variables_predictoras].copy()
X['gender'] = (X['gender'] == 'M').astype(int)
y = df['etiqueta_norad_6_24'].astype(int)

print(f"Variables totales        : {len(variables_predictoras)}")
print(f"  Continuas              : {len(variables_continuas)}")
print(f"  Binarias               : {len(variables_binarias)}")
print()


# -----------------------------------------------------------------------------
# 3. MATRIZ DE CORRELACIÓN DE SPEARMAN
# -----------------------------------------------------------------------------
print("-" * 70)
print("[1/4] Matriz de correlación de Spearman")
print("-" * 70)

# Spearman es robusto a asimetría y outliers (típicos en variables clínicas)
matriz_spearman = X.corr(method='spearman')

# Heatmap
fig, ax = plt.subplots(figsize=(20, 18))
mascara = np.triu(np.ones_like(matriz_spearman, dtype=bool), k=1)
sns.heatmap(
    matriz_spearman,
    mask=mascara,
    cmap='RdBu_r',
    center=0,
    vmin=-1, vmax=1,
    square=True,
    annot=False,
    cbar_kws={'shrink': 0.6, 'label': 'rho de Spearman'},
    linewidths=0.3,
    ax=ax,
)
ax.set_title('Matriz de correlación de Spearman — variables predictoras (v4)',
             fontsize=14, pad=15)
plt.tight_layout()
ruta_heatmap = os.path.join(CARPETA_FIGURAS, 'correlacion_spearman_v4.png')
plt.savefig(ruta_heatmap, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Heatmap guardado en   : {ruta_heatmap}")

# Tabla completa
ruta_matriz = os.path.join(CARPETA_TABLAS, 'matriz_spearman_v4.csv')
matriz_spearman.round(3).to_csv(ruta_matriz)
print(f"  Matriz completa en    : {ruta_matriz}")
print()


# -----------------------------------------------------------------------------
# 4. PARES CON ALTA CORRELACIÓN (|rho| > 0.7)
# -----------------------------------------------------------------------------
print("-" * 70)
print("[2/4] Pares con |rho| > 0.7")
print("-" * 70)

# Extraer pares únicos del triángulo superior
pares = (matriz_spearman.where(np.triu(np.ones_like(matriz_spearman, dtype=bool), k=1))
                        .stack()
                        .reset_index())
pares.columns = ['variable_1', 'variable_2', 'rho_spearman']
pares['rho_abs'] = pares['rho_spearman'].abs()

pares_altos = (pares[pares['rho_abs'] > 0.7]
               .sort_values('rho_abs', ascending=False)
               .reset_index(drop=True))

print(f"  Pares con |rho| > 0.7 : {len(pares_altos)}")

if len(pares_altos) > 0:
    print(f"\n  Top 20 pares más correlacionados:")
    print(pares_altos.head(20)[['variable_1', 'variable_2', 'rho_spearman']]
          .to_string(index=False))

ruta_pares = os.path.join(CARPETA_TABLAS, 'pares_alta_correlacion_v4.csv')
pares_altos.drop(columns='rho_abs').to_csv(ruta_pares, index=False)
print(f"\n  Tabla guardada en     : {ruta_pares}")
print()


# -----------------------------------------------------------------------------
# 5. VIF (Variance Inflation Factor) sobre variables continuas
# -----------------------------------------------------------------------------
print("-" * 70)
print("[3/4] VIF (multicolinealidad multivariante)")
print("-" * 70)

# El VIF requiere que las variables sean continuas y que el modelo lineal
# pueda ajustarse sin singularidades. Excluimos binarias.
X_continuas = X[variables_continuas].copy()

# Estandarización suave para que el VIF sea numéricamente estable
X_std = (X_continuas - X_continuas.mean()) / X_continuas.std()

# Añadir intercepto (statsmodels lo necesita)
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
    vif_vals.append({'variable': col, 'vif': vif})

vif_df = pd.DataFrame(vif_vals).sort_values('vif', ascending=False).reset_index(drop=True)

print(f"  Variables analizadas  : {len(vif_df)}")
print(f"\n  Top 15 con mayor VIF (>10 = multicolinealidad seria):")
print(vif_df.head(15).to_string(index=False))

print(f"\n  Resumen por umbral:")
print(f"    VIF > 10           : {(vif_df['vif'] > 10).sum()} variables")
print(f"    VIF 5-10           : {((vif_df['vif'] > 5) & (vif_df['vif'] <= 10)).sum()} variables")
print(f"    VIF < 5            : {(vif_df['vif'] <= 5).sum()} variables")

ruta_vif = os.path.join(CARPETA_TABLAS, 'vif_v4.csv')
vif_df.to_csv(ruta_vif, index=False)
print(f"\n  Tabla guardada en    : {ruta_vif}")
print()


# -----------------------------------------------------------------------------
# 6. ASOCIACIONES DE VARIABLES BINARIAS (reporte separado)
# -----------------------------------------------------------------------------
print("-" * 70)
print("Análisis separado de variables binarias")
print("-" * 70)

# Para variables binarias usamos:
#  - asociación entre binarias: phi (= Pearson sobre 0/1, equivalente a chi^2 normalizado)
#  - asociación entre binaria y continua: rho de Spearman (válido como rank-biserial)

filas = []
# Entre binarias
for i, b1 in enumerate(variables_binarias):
    for b2 in variables_binarias[i+1:]:
        # Coeficiente phi (Pearson sobre 0/1)
        phi = X[[b1, b2]].corr(method='pearson').iloc[0, 1]
        filas.append({
            'variable_1': b1,
            'variable_2': b2,
            'tipo': 'binaria-binaria (phi)',
            'asociacion': round(phi, 3)
        })

# De cada binaria con cada continua: spearman (válido como rank-biserial)
for b in variables_binarias:
    for c in variables_continuas:
        rho = X[[b, c]].corr(method='spearman').iloc[0, 1]
        if abs(rho) > 0.2:  # solo reportamos las moderadas o altas, si no es ruido
            filas.append({
                'variable_1': b,
                'variable_2': c,
                'tipo': 'binaria-continua (rank-biserial)',
                'asociacion': round(rho, 3)
            })

asoc_binarias = pd.DataFrame(filas).sort_values(
    by='asociacion', key=lambda s: s.abs(), ascending=False).reset_index(drop=True)

print(f"  Asociaciones entre binarias y otras variables (filtradas |r|>0.2):")
print(asoc_binarias.to_string(index=False))

ruta_asoc = os.path.join(CARPETA_TABLAS, 'asociaciones_binarias_v4.csv')
asoc_binarias.to_csv(ruta_asoc, index=False)
print(f"\n  Tabla guardada en    : {ruta_asoc}")
print()


# -----------------------------------------------------------------------------
# 7. DENDROGRAMA DE AGRUPAMIENTO JERÁRQUICO
# -----------------------------------------------------------------------------
print("-" * 70)
print("[4/4] Dendrograma de variables (agrupamiento jerárquico)")
print("-" * 70)

# Distancia: 1 - |rho|. Usamos la matriz Spearman completa (incluye binarias).
distancia = (1 - matriz_spearman.abs()).to_numpy().copy()
# squareform requiere matriz simétrica con ceros en diagonal
np.fill_diagonal(distancia, 0)
distancia_condensada = squareform(distancia, checks=False)

# Linkage por average (más estable para datos clínicos heterogéneos)
enlaces = linkage(distancia_condensada, method='average')

# Linkage por average (más estable para datos clínicos heterogéneos)
enlaces = linkage(distancia_condensada, method='average')

# ---------------------------------------------------------------------
# EXTRA: CLUSTERS EXPLÍCITOS
# ---------------------------------------------------------------------
from scipy.cluster.hierarchy import fcluster

# Corte equivalente a |rho| >= 0.7 → distancia <= 0.3
clusters = fcluster(enlaces, t=0.3, criterion='distance')

tabla_clusters = pd.DataFrame({
    'variable': matriz_spearman.columns,
    'cluster': clusters
}).sort_values(['cluster', 'variable'])

ruta_clusters = os.path.join(CARPETA_TABLAS, 'clusters_variables_v4.csv')
tabla_clusters.to_csv(ruta_clusters, index=False)

# Resumen agrupado
resumen_clusters = (
    tabla_clusters
    .groupby('cluster')['variable']
    .apply(lambda x: ', '.join(x))
    .reset_index()
)

resumen_clusters['n_variables'] = resumen_clusters['variable'].apply(
    lambda x: len(x.split(', '))
)

ruta_resumen = os.path.join(CARPETA_TABLAS, 'resumen_clusters_v4.csv')
resumen_clusters.to_csv(ruta_resumen, index=False)

print(f"  Clusters guardados en : {ruta_clusters}")
print(f"  Resumen clusters en   : {ruta_resumen}")

fig, ax = plt.subplots(figsize=(14, 16))
dendrogram(
    enlaces,
    labels=matriz_spearman.columns.tolist(),
    orientation='left',
    leaf_font_size=8,
    color_threshold=0.5,
    ax=ax,
)
ax.set_title('Dendrograma de variables — distancia = 1 - |rho Spearman|\n'
             'Variables más cercanas (a la izquierda) → más correlacionadas',
             fontsize=12, pad=10)
ax.set_xlabel('Distancia (1 - |rho|)')
ax.axvline(x=0.3, color='red', linestyle='--', linewidth=1,
           label='|rho| = 0.7 (corte sugerido)')
ax.legend(loc='lower right')
plt.tight_layout()
ruta_dendro = os.path.join(CARPETA_FIGURAS, 'dendrograma_v4.png')
plt.savefig(ruta_dendro, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Dendrograma guardado en: {ruta_dendro}")
print()


# -----------------------------------------------------------------------------
# 8. RESUMEN EJECUTIVO
# -----------------------------------------------------------------------------
print("=" * 70)
print("RESUMEN EJECUTIVO")
print("=" * 70)
print(f"  Estancias analizadas             : {len(df)} (1ª por paciente)")
print(f"  Variables totales                : {len(variables_predictoras)}")
print(f"  Pares con |rho| > 0.7            : {len(pares_altos)}")
print(f"  Variables con VIF > 10           : {(vif_df['vif'] > 10).sum()}")
print(f"  Variables con VIF 5-10           : {((vif_df['vif'] > 5) & (vif_df['vif'] <= 10)).sum()}")
print()
print("Ficheros generados:")
print(f"  Figuras  : {CARPETA_FIGURAS}/")
print(f"  Tablas   : {CARPETA_TABLAS}/")
print()
print("LECTURA SUGERIDA PARA LA MEMORIA:")
print("  - El dendrograma es la figura principal (visualización rápida).")
print("  - La tabla de pares |rho|>0.7 documenta las redundancias por construcción.")
print("  - El VIF complementa con visión multivariante (no solo pares).")
print("  - Las asociaciones binarias se reportan aparte.")