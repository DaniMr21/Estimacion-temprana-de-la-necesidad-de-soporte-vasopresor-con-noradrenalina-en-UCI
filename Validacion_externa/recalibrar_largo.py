"""
FASE 2: RECALIBRACIÓN EN VALIDACIÓN EXTERNA (Ventana Larga 12-48h)
==================================================================
Ajuste del intercepto y calibración (Platt) a la prevalencia local de eICU
mediante Validación Cruzada (5-folds) para evitar data leakage.
Y GUARDADO DEL MODELO FINAL RECALIBRADO.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV  # <--- Añadido para empaquetar el pkl
import warnings
warnings.filterwarnings('ignore')

# ── CONFIGURACIÓN ────────────────────────────────────────────────────────────
BASE_DIR = r'C:\Users\danie\TFG\Pruebas_v4_4'
DATA_DIR = r'C:\Users\danie\OneDrive\Escritorio\DATA'

CSV_EICU = os.path.join(DATA_DIR, 'eICU_12_48_definitivo.csv')
MODELO_PKL = os.path.join(BASE_DIR, 'MODELOS_ENTRENADOS', 'modelo_Largo_12_48_XGB.pkl')
RUTA_SALIDA = os.path.join(BASE_DIR, 'MODELOS_ENTRENADOS', 'modelo_Largo_12_48_XGB_RECALIBRADO.pkl') # <--- Ruta de salida

VARIABLES = ['temp_min', 'pf_min', 'spo2_min', 'bicarbonato_min', 'map_min', 'glucemia_min', 'sofa_max']
ETIQUETA = 'etiqueta_norad_12_48'

# ── FUNCIONES DE MÉTRICAS ────────────────────────────────────────────────────
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

# ── 1. CARGA DE DATOS Y MODELO BASE ──────────────────────────────────────────
print("Cargando datos de eICU (Largo) y modelo base de MIMIC")
df = pd.read_csv(CSV_EICU)
X = df[VARIABLES]
y = df[ETIQUETA].values.astype(int)

modelo_mimic = joblib.load(MODELO_PKL)

# Probabilidades crudas tal cual salen del modelo entrenado en Boston
prob_originales = modelo_mimic.predict_proba(X)[:, 1]

# ── 2. RECALIBRACIÓN LOCAL (PLATT EN CROSS-VALIDATION) ───────────────────────
print("Aplicando actualización local (Platt Scaling) con 5-Folds...")
prob_recalibradas = np.zeros(len(y))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Usamos Regresión Logística (Platt) teniendo como única feature la prob. original
for train_idx, test_idx in cv.split(prob_originales, y):
    lr = LogisticRegression(solver='lbfgs')
    # Entrenamos el calibrador con el 80% de eICU
    lr.fit(prob_originales[train_idx].reshape(-1, 1), y[train_idx])
    # Predecimos en el 20% restante (Totalmente limpio)
    prob_recalibradas[test_idx] = lr.predict_proba(prob_originales[test_idx].reshape(-1, 1))[:, 1]

# ── 3. CÁLCULO DE MÉTRICAS COMPARATIVAS ──────────────────────────────────────
# Original
auc_orig = roc_auc_score(y, prob_originales)
bss_orig = calcular_bss(prob_originales, y)
ece_orig = calcular_ece(prob_originales, y)

# Recalibrado
auc_recal = roc_auc_score(y, prob_recalibradas)
bss_recal = calcular_bss(prob_recalibradas, y)
ece_recal = calcular_ece(prob_recalibradas, y)

print(f"\n{'='*60}")
print(f" RESULTADOS DE LA RECALIBRACIÓN (VENTANA LARGA 12-48h)")
print(f"{'='*60}")
print(f"Métrica      | Original (MIMIC) | Recalibrado (eICU local)")
print(f"-------------|------------------|-------------------------")
print(f"AUC-ROC      | {auc_orig:.4f}           | {auc_recal:.4f}")
print(f"Brier Score  | {brier_score_loss(y, prob_originales):.4f}           | {brier_score_loss(y, prob_recalibradas):.4f}")
print(f"BSS          | {bss_orig:+.4f}          | {bss_recal:+.4f}")
print(f"ECE          | {ece_orig:.4f}           | {ece_recal:.4f}")
print(f"{'='*60}")

# ── 4. GUARDADO DEL PKL FINAL (CALIBRADOR INDEPENDIENTE) ─────────────────────
print("\n[INFO] Entrenando el calibrador definitivo con el 100% de eICU y guardándolo...")

# Entrenamos la regresión logística (Platt) con todas las predicciones de MIMIC sobre eICU
calibrador_final = LogisticRegression(solver='lbfgs')
calibrador_final.fit(prob_originales.reshape(-1, 1), y)

RUTA_CALIBRADOR = os.path.join(BASE_DIR, 'MODELOS_ENTRENADOS', 'calibrador_Largo_eICU.pkl')
joblib.dump(calibrador_final, RUTA_CALIBRADOR)

print(f"se ha guardado en: {RUTA_CALIBRADOR}")