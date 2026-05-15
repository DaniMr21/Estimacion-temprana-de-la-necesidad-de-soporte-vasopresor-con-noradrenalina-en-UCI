"""
Análisis de Subgrupos y Benchmark Clínico (SOFA) — Ventana Corta_3_12
======================================================================
1. Genera predicciones honestas (OOF) "DE FÁBRICA" (sin calibrador extra).
2. Evalúa métricas en subgrupos: Global, Sépticos y No Sépticos.
3. Evalúa el score SOFA aislado como baseline clínico (solo AUCs).
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.base import clone
import warnings
warnings.filterwarnings('ignore')

# ── 1. CONFIGURACIÓN ───────────────────────────────────────────────────────────
RUTA_CSV   = r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4p.csv'
RUTA_PKL   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MODELOS_ENTRENADOS\modelo_Corto_3_12_CAT.pkl')
ETIQUETA   = 'etiqueta_norad_3_12'
COLUMNA_ID = 'subject_id'
VARIABLES  = ['map_min', 'pf_min', 'sofa_max', 'tp_max']
N_SPLITS   = 5
RANDOM_SEED = 42

# ── 2. FUNCIONES DE MÉTRICAS ───────────────────────────────────────────────────
def calcular_ece(probabilidades, etiquetas, n_bins=10):
    limites = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mascara = (probabilidades >= limites[i]) & (probabilidades < limites[i + 1])
        n_bin = mascara.sum()
        if n_bin > 0:
            ece += (n_bin / len(etiquetas)) * abs(probabilidades[mascara].mean() - etiquetas[mascara].mean())
    return float(ece)

def calcular_bss(probabilidades, etiquetas):
    prevalencia = etiquetas.mean()
    bs_modelo = brier_score_loss(etiquetas, probabilidades)
    bs_clima = brier_score_loss(etiquetas, np.full_like(probabilidades, prevalencia))
    return float(1 - bs_modelo / bs_clima) if bs_clima != 0 else np.nan

def imprimir_fila(nombre, n_pacientes, auc, aupr, bss, ece):
    bss_str = f"{bss:+.4f}" if not np.isnan(bss) else "  ---  "
    ece_str = f"{ece:.4f}" if not np.isnan(ece) else "  ---  "
    print(f"{nombre:<25} | n={n_pacientes:<11} | {auc:.4f}  | {aupr:.4f}  | {bss_str:<7} | {ece_str:<7}")

# ── 3. CARGA DE DATOS ──────────────────────────────────────────────────────────
print("Cargando datos y modelo (Ventana CORTA)...")
df = pd.read_csv(RUTA_CSV)

y          = df[ETIQUETA].values.astype(int)
grupos     = df[COLUMNA_ID].values
X          = df[VARIABLES].copy()
sepsis     = df['tiene_sepsis'].values.astype(int)
sofa_score = df['sofa_max'].values  

modelo = joblib.load(RUTA_PKL)

# ── 4. GENERAR PROBABILIDADES OOF (DE FÁBRICA) ─────────────────────────────────
print(f"Generando predicciones OOF honestas ({N_SPLITS} folds)...")
cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
prob_oof = np.zeros(len(y))

for idx_train, idx_test in cv.split(X, y, grupos):
    modelo_fold = clone(modelo)
    modelo_fold.fit(X.iloc[idx_train], y[idx_train])
    # Sin calibrador extra, sacamos la probabilidad directa
    prob_oof[idx_test] = modelo_fold.predict_proba(X.iloc[idx_test])[:, 1]

# ── 5. EVALUACIÓN POR SUBGRUPOS Y BENCHMARK ────────────────────────────────────
print("\n" + "="*70)
print(f"{'SUBGRUPO / MODELO':<25} | {'N (Pacientes)':<13} | {'AUC-ROC':<7} | {'AUC-PR':<7} | {'BSS':<7} | {'ECE':<7}")
print("-" * 70)

mascaras = {
    'GLOBAL (Todos)': np.ones_like(y, dtype=bool),
    'SÉPTICOS (tiene_sepsis=1)': sepsis == 1,
    'NO SÉPTICOS (tiene_sepsis=0)': sepsis == 0
}

for nombre, mascara in mascaras.items():
    y_sub      = y[mascara]
    prob_sub   = prob_oof[mascara]
    if y_sub.sum() == 0 or y_sub.sum() == len(y_sub): continue
    imprimir_fila(nombre, mascara.sum(), roc_auc_score(y_sub, prob_sub), average_precision_score(y_sub, prob_sub), calcular_bss(prob_sub, y_sub), calcular_ece(prob_sub, y_sub))

print("-" * 70)
auc_sofa  = roc_auc_score(y, sofa_score)
imprimir_fila('SOFA (Baseline Clínico)', len(y), auc_sofa, average_precision_score(y, sofa_score), np.nan, np.nan)
print("="*70)
print(f"\n✅El Cat del Corto supera al SOFA en {(roc_auc_score(y, prob_oof) - auc_sofa)*100:.1f} puntos porcentuales de AUC-ROC.")