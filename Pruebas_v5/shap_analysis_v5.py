"""
NOTA: SHAP se calcula sobre el modelo entrenado en TODO el dataset.
"""

import warnings
warnings.filterwarnings('ignore')

import os, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# ── CONFIGURACIÓN ─────────────────────────────────────────────────────────────
RUTA_CSV = r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4.csv'
ETIQUETA = 'etiqueta_norad_6_24'

CARPETA_BASE   = os.path.dirname(__file__) if '__file__' in dir() else '.'
CARPETA_TABLAS = os.path.join(CARPETA_BASE, 'tablas')
CARPETA_FIGS   = os.path.join(CARPETA_BASE, 'figuras')
os.makedirs(CARPETA_TABLAS, exist_ok=True)
os.makedirs(CARPETA_FIGS,   exist_ok=True)

# Set final (12 variables)
VARIABLES = [
    'map_min', 'hr_media', 'lactato_max', 'diuresis_ml_kg_6h',
    'pf_min', 'spo2_min', 'rr_max', 'ph_min', 'sofa_max',
    'glucemia_min', 'temp_min'
]

# ── CARGA ─────────────────────────────────────────────────────────────────────
print("=" * 65)
print("ANÁLISIS SHAP — RF v4 (set final, 11 variables)")
print("=" * 65)

df = pd.read_csv(RUTA_CSV)
df = df.dropna(subset=['pf_max'])
X = df[VARIABLES].copy()
y = df[ETIQUETA].copy()
pid = df['subject_id'].copy()

print(f"\nDataset: N={len(X)} | Positivos={y.sum()} ({100*y.mean():.2f}%)")
print(f"Variables: {len(VARIABLES)}")

# ── BÚSQUEDA DE HIPERPARÁMETROS (1 sola vez sobre todo el dataset) ────────────
print(f"\n[1/3] Búsqueda de hiperparámetros para el RF final...")
pipeline = Pipeline([('modelo', RandomForestClassifier(random_state=42, n_jobs=-1))])
espacio = {
    'modelo__n_estimators':     [300, 500, 1000],
    'modelo__max_depth':        [10, 20, None],
    'modelo__min_samples_leaf': [1, 5],
    'modelo__max_features':     ['sqrt'],
    'modelo__class_weight':     ['balanced', 'balanced_subsample'],
}

cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
t0 = time.time()
gs = GridSearchCV(pipeline, espacio, cv=cv, scoring='roc_auc',
                  n_jobs=-1, refit=True)
gs.fit(X, y, groups=pid)
print(f"  Best params: {gs.best_params_}")
print(f"  Best CV AUC: {gs.best_score_:.4f}  ({(time.time()-t0)/60:.1f} min)")

# Modelo final ya refitted con best params sobre TODO el dataset
modelo = gs.best_estimator_.named_steps['modelo']

# ── CÁLCULO SHAP ──────────────────────────────────────────────────────────────
print(f"\n[2/3] Calculando valores SHAP (TreeExplainer)...")
t0 = time.time()
explainer = shap.TreeExplainer(modelo)

# SHAP values para la clase positiva (etiqueta=1)
shap_values_full = explainer.shap_values(X)

# Manejo de la forma de salida (varía entre versiones de SHAP)
if isinstance(shap_values_full, list):
    # Formato antiguo: lista [clase_0, clase_1]
    shap_values = shap_values_full[1]
elif shap_values_full.ndim == 3:
    # Formato nuevo: array (n_samples, n_features, n_classes)
    shap_values = shap_values_full[:, :, 1]
else:
    shap_values = shap_values_full

print(f"  SHAP shape: {shap_values.shape}  ({(time.time()-t0)/60:.1f} min)")

# ── 1. SUMMARY BAR PLOT ───────────────────────────────────────────────────────
print(f"\n[3/3] Generando figuras...")

plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, X, plot_type='bar', show=False,
                  max_display=len(VARIABLES))
plt.title(f'SHAP — Importancia global (|SHAP| medio)\n'
          f'RF v4 (12 vars, AUC CV = {gs.best_score_:.4f})', fontsize=11)
plt.tight_layout()
ruta_bar = os.path.join(CARPETA_FIGS, 'shap_summary_bar_v4.png')
plt.savefig(ruta_bar, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Bar plot      : {ruta_bar}")

# ── 2. SUMMARY BEESWARM ───────────────────────────────────────────────────────
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, X, show=False, max_display=len(VARIABLES))
plt.title(f'SHAP — Beeswarm (efecto y signo por paciente)\n'
          f'RF v4 (12 vars)', fontsize=11)
plt.tight_layout()
ruta_bee = os.path.join(CARPETA_FIGS, 'shap_summary_beeswarm_v4.png')
plt.savefig(ruta_bee, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Beeswarm      : {ruta_bee}")

# ── 3. DEPENDENCE PLOTS DE LOS TOP-3 ──────────────────────────────────────────
shap_abs_mean = np.abs(shap_values).mean(axis=0)
orden = np.argsort(-shap_abs_mean)
top3 = [VARIABLES[i] for i in orden[:3]]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, var in zip(axes, top3):
    shap.dependence_plot(var, shap_values, X, show=False, ax=ax,
                         interaction_index='auto')
    ax.set_title(f'{var}  (|SHAP| medio = {shap_abs_mean[VARIABLES.index(var)]:.3f})',
                 fontsize=10)
plt.suptitle(f'SHAP Dependence — Top 3 variables', fontsize=12, y=1.02)
plt.tight_layout()
ruta_dep = os.path.join(CARPETA_FIGS, 'shap_dependence_TOP3_v4.png')
plt.savefig(ruta_dep, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Dependence    : {ruta_dep}")

# ── TABLA DE IMPORTANCIA ──────────────────────────────────────────────────────
shap_mean_signed = shap_values.mean(axis=0)
tabla = pd.DataFrame({
    'variable':       VARIABLES,
    'shap_abs_mean':  shap_abs_mean,
    'shap_mean':      shap_mean_signed,
    'direccion':      ['↑ riesgo' if v > 0 else '↓ riesgo'
                       for v in shap_mean_signed],
}).sort_values('shap_abs_mean', ascending=False).reset_index(drop=True)

print("\n" + "=" * 65)
print("IMPORTANCIA SHAP AGREGADA")
print("=" * 65)
print(tabla.round(4).to_string(index=False))

ruta_tab = os.path.join(CARPETA_TABLAS, 'shap_importance_v4.csv')
tabla.to_csv(ruta_tab, index=False)
print(f"\n  Tabla guardada en: {ruta_tab}")

print("\n" + "=" * 65)
print("FIN")
print("=" * 65)
