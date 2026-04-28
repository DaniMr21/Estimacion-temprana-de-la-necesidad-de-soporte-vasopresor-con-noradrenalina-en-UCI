"""
Análisis de correlación y multicolinealidad sobre el set REDUCIDO de v4
(observación 0-6h, predicción 6-24h, 26 variables seleccionadas).
 
Replica el mismo análisis que se hizo para el set completo (76 variables)
para verificar que la reducción ha eliminado la multicolinealidad y para
documentar la estructura de correlación final del modelo.
 
Cuatro análisis complementarios:
 
  1. Matriz de correlación de Spearman (no Pearson, por asimetría
     habitual de las variables clínicas).
  2. Pares de variables con |rho| > 0.7 (debería estar prácticamente
     vacío después de la reducción).
  3. VIF (Variance Inflation Factor) sobre variables continuas
     (excluye binarias, que se reportan aparte con asociaciones).
  4. Dendrograma de agrupamiento jerárquico de variables.
 
Decisiones metodológicas (idénticas al análisis del set completo):
  - CSV winsorizado al 2-98%, mismo que usa el modelo.
  - Filtrado a primera estancia por paciente para garantizar independencia.
  - Variables binarias (gender, ventilacion_invasiva_6h) excluidas del VIF
    y reportadas mediante asociaciones específicas.
 
Salidas:
  - figuras/correlacion_spearman_v4_reducido.png
  - figuras/dendrograma_v4_reducido.png
  - tablas/matriz_spearman_v4_reducido.csv
  - tablas/pares_alta_correlacion_v4_reducido.csv
  - tablas/vif_v4_reducido.csv
  - tablas/asociaciones_binarias_v4_reducido.csv
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
print("ANÁLISIS DE CORRELACIÓN — SET REDUCIDO v4 (26 variables)")
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
# 2. SELECCIÓN DE VARIABLES (set reducido de 26)
# -----------------------------------------------------------------------------
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
    'fio2_media',
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
 
matriz_spearman = X.corr(method='spearman')
 
# Heatmap (con anotaciones porque son pocas variables)
fig, ax = plt.subplots(figsize=(14, 12))
mascara = np.triu(np.ones_like(matriz_spearman, dtype=bool), k=1)
sns.heatmap(
    matriz_spearman,
    mask=mascara,
    cmap='RdBu_r',
    center=0,
    vmin=-1, vmax=1,
    square=True,
    annot=True,
    fmt='.2f',
    annot_kws={'size': 7},
    cbar_kws={'shrink': 0.6, 'label': 'rho de Spearman'},
    linewidths=0.4,
    ax=ax,
)
ax.set_title('Matriz de correlación de Spearman — set reducido v4 (26 variables)',
             fontsize=13, pad=15)
plt.tight_layout()
ruta_heatmap = os.path.join(CARPETA_FIGURAS, 'correlacion_spearman_v4_reducido.png')
plt.savefig(ruta_heatmap, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Heatmap guardado en   : {ruta_heatmap}")
 
ruta_matriz = os.path.join(CARPETA_TABLAS, 'matriz_spearman_v4_reducido.csv')
matriz_spearman.round(3).to_csv(ruta_matriz)
print(f"  Matriz completa en    : {ruta_matriz}")
print()
 
 
# -----------------------------------------------------------------------------
# 4. PARES CON ALTA CORRELACIÓN
# -----------------------------------------------------------------------------
print("-" * 70)
print("[2/4] Pares con mayor correlación (todos, ordenados)")
print("-" * 70)
 
pares = (matriz_spearman.where(np.triu(np.ones_like(matriz_spearman, dtype=bool), k=1))
                        .stack()
                        .reset_index())
pares.columns = ['variable_1', 'variable_2', 'rho_spearman']
pares['rho_abs'] = pares['rho_spearman'].abs()
pares = pares.sort_values('rho_abs', ascending=False).reset_index(drop=True)
 
print(f"  Total de pares        : {len(pares)}")
print(f"  Pares con |rho| > 0.7 : {(pares['rho_abs'] > 0.7).sum()}")
print(f"  Pares con |rho| > 0.5 : {(pares['rho_abs'] > 0.5).sum()}")
print(f"  Pares con |rho| > 0.3 : {(pares['rho_abs'] > 0.3).sum()}")
print(f"  Correlación media     : {pares['rho_abs'].mean():.3f}")
print(f"  Correlación mediana   : {pares['rho_abs'].median():.3f}")
 
print(f"\n  Top 20 pares más correlacionados (debería ser todo modesto):")
print(pares.head(20)[['variable_1', 'variable_2', 'rho_spearman']]
      .to_string(index=False))
 
ruta_pares = os.path.join(CARPETA_TABLAS, 'pares_alta_correlacion_v4_reducido.csv')
pares.drop(columns='rho_abs').to_csv(ruta_pares, index=False)
print(f"\n  Tabla guardada en     : {ruta_pares}")
print()
 
 
# -----------------------------------------------------------------------------
# 5. VIF
# -----------------------------------------------------------------------------
print("-" * 70)
print("[3/4] VIF (multicolinealidad multivariante)")
print("-" * 70)
 
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
    vif_vals.append({'variable': col, 'vif': vif})
 
vif_df = pd.DataFrame(vif_vals).sort_values('vif', ascending=False).reset_index(drop=True)
 
print(f"  Variables analizadas  : {len(vif_df)}")
print(f"\n  Tabla completa (ordenada por VIF descendente):")
print(vif_df.round(2).to_string(index=False))
 
print(f"\n  Resumen por umbral:")
print(f"    VIF > 10           : {(vif_df['vif'] > 10).sum()} variables")
print(f"    VIF 5-10           : {((vif_df['vif'] > 5) & (vif_df['vif'] <= 10)).sum()} variables")
print(f"    VIF 2-5            : {((vif_df['vif'] > 2) & (vif_df['vif'] <= 5)).sum()} variables")
print(f"    VIF < 2            : {(vif_df['vif'] <= 2).sum()} variables")
 
ruta_vif = os.path.join(CARPETA_TABLAS, 'vif_v4_reducido.csv')
vif_df.to_csv(ruta_vif, index=False)
print(f"\n  Tabla guardada en    : {ruta_vif}")
print()
 
 
# -----------------------------------------------------------------------------
# 6. ASOCIACIONES BINARIAS
# -----------------------------------------------------------------------------
print("-" * 70)
print("Análisis separado de variables binarias")
print("-" * 70)
 
filas = []
for i, b1 in enumerate(variables_binarias):
    for b2 in variables_binarias[i+1:]:
        phi = X[[b1, b2]].corr(method='pearson').iloc[0, 1]
        filas.append({
            'variable_1': b1, 'variable_2': b2,
            'tipo': 'binaria-binaria (phi)', 'asociacion': round(phi, 3)
        })
 
for b in variables_binarias:
    for c in variables_continuas:
        rho = X[[b, c]].corr(method='spearman').iloc[0, 1]
        if abs(rho) > 0.15:
            filas.append({
                'variable_1': b, 'variable_2': c,
                'tipo': 'binaria-continua (rank-biserial)', 'asociacion': round(rho, 3)
            })
 
asoc_binarias = pd.DataFrame(filas).sort_values(
    by='asociacion', key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
 
print(f"  Asociaciones |r| > 0.15:")
print(asoc_binarias.to_string(index=False))
 
ruta_asoc = os.path.join(CARPETA_TABLAS, 'asociaciones_binarias_v4_reducido.csv')
asoc_binarias.to_csv(ruta_asoc, index=False)
print(f"\n  Tabla guardada en    : {ruta_asoc}")
print()
 
 
# -----------------------------------------------------------------------------
# 7. DENDROGRAMA
# -----------------------------------------------------------------------------
print("-" * 70)
print("[4/4] Dendrograma del set reducido")
print("-" * 70)
 
distancia = (1 - matriz_spearman.abs()).to_numpy().copy()
np.fill_diagonal(distancia, 0)
distancia_condensada = squareform(distancia, checks=False)
 
enlaces = linkage(distancia_condensada, method='average')
 
fig, ax = plt.subplots(figsize=(12, 10))
dendrogram(
    enlaces,
    labels=matriz_spearman.columns.tolist(),
    orientation='left',
    leaf_font_size=9,
    color_threshold=0.7,
    ax=ax,
)
ax.set_title('Dendrograma — set reducido v4 (26 variables)\n'
             'Distancia = 1 − |rho Spearman|',
             fontsize=12, pad=10)
ax.set_xlabel('Distancia (1 − |rho|)')
ax.axvline(x=0.3, color='red', linestyle='--', linewidth=1,
           label='|rho| = 0.7')
ax.axvline(x=0.5, color='orange', linestyle='--', linewidth=1,
           label='|rho| = 0.5')
ax.legend(loc='lower right')
plt.tight_layout()
ruta_dendro = os.path.join(CARPETA_FIGURAS, 'dendrograma_v4_reducido.png')
plt.savefig(ruta_dendro, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Dendrograma guardado en: {ruta_dendro}")
print()
 
 
# -----------------------------------------------------------------------------
# 8. RESUMEN EJECUTIVO + COMPARACIÓN CON SET COMPLETO
# -----------------------------------------------------------------------------
print("=" * 70)
print("RESUMEN EJECUTIVO — COMPARACIÓN CON SET COMPLETO")
print("=" * 70)
 
print(f"  {'Métrica':<35} {'Completo (76 vars)':<22} {'Reducido (26 vars)':<22}")
print(f"  {'-'*35} {'-'*22} {'-'*22}")
print(f"  {'Pares con |rho|>0.7':<35} {'76':<22} {(pares['rho_abs']>0.7).sum():<22}")
print(f"  {'Pares con |rho|>0.5':<35} {'~150 (estimado)':<22} {(pares['rho_abs']>0.5).sum():<22}")
print(f"  {'Variables con VIF > 10':<35} {'59':<22} {(vif_df['vif']>10).sum():<22}")
print(f"  {'Variables con VIF 5-10':<35} {'3':<22} {((vif_df['vif']>5)&(vif_df['vif']<=10)).sum():<22}")
print(f"  {'Variables con VIF < 5':<35} {'11':<22} {(vif_df['vif']<=5).sum():<22}")
print(f"  {'Correlación media entre pares':<35} {'~0.30 (estimado)':<22} {pares['rho_abs'].mean():.3f}")
print()
print("Ficheros generados:")
print(f"  Figuras  : {CARPETA_FIGURAS}/")
print(f"  Tablas   : {CARPETA_TABLAS}/")
print()
print("CONCLUSIONES:")
print("  - Si pares con |rho|>0.7 = 0: la reducción ha eliminado")
print("    completamente la colinealidad fuerte.")
print("  - Si VIF > 10 < 5 variables: la multicolinealidad multivariante")
print("    también está controlada.")
print("  - Las correlaciones que queden serán moderadas (<0.5) y")
print("    clínicamente esperables (componentes de SOFA, ácido-base, etc.).")