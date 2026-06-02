import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.base import clone
import warnings
warnings.filterwarnings('ignore')

RUTA_CSV   = r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4.csv'
RUTA_PKL   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MODELOS_ENTRENADOS\modelo_Medio_6_24_XGB.pkl')
ETIQUETA   = 'etiqueta_norad_6_24'
COLUMNA_ID = 'subject_id'
VARIABLES  = ['pf_min', 'map_min', 'diuresis_ml_kg_6h', 'hr_media', 'sofa_max', 'ventilacion_invasiva_6h']
N_SPLITS   = 5
RANDOM_SEED = 42


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

print("Cargando datos y modelo...")
df = pd.read_csv(RUTA_CSV)

y          = df[ETIQUETA].values.astype(int)
grupos     = df[COLUMNA_ID].values
X          = df[VARIABLES].copy()
sepsis     = df['tiene_sepsis'].values.astype(int)
sofa_score = df['sofa_max'].values  # Score crudo para el benchmark

modelo = joblib.load(RUTA_PKL)

print(f"Generando predicciones OOF honestas ({N_SPLITS} folds)...")
cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
prob_oof_calibrada = np.zeros(len(y))

for idx_train, idx_test in cv.split(X, y, grupos):
    # 1. Entrenar XGBoost base
    modelo_fold = clone(modelo)
    modelo_fold.fit(X.iloc[idx_train], y[idx_train])
    
    prob_train = modelo_fold.predict_proba(X.iloc[idx_train])[:, 1]
    prob_test  = modelo_fold.predict_proba(X.iloc[idx_test])[:, 1]
    
    # 2. Aplicar Calibración Platt (Regresión Logística pura)
    calibrador = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
    calibrador.fit(prob_train.reshape(-1, 1), y[idx_train])
    
    prob_oof_calibrada[idx_test] = calibrador.predict_proba(prob_test.reshape(-1, 1))[:, 1]

print("\n" + "="*70)
print(f"{'SUBGRUPO / MODELO':<25} | {'N (Pacientes)':<13} | {'AUC-ROC':<7} | {'AUC-PR':<7} | {'BSS':<7} | {'ECE':<7}")
print("-" * 70)

def imprimir_fila(nombre, n_pacientes, auc, aupr, bss, ece):
    bss_str = f"{bss:+.4f}" if not np.isnan(bss) else "  ---  "
    ece_str = f"{ece:.4f}" if not np.isnan(ece) else "  ---  "
    print(f"{nombre:<25} | n={n_pacientes:<11} | {auc:.4f}  | {aupr:.4f}  | {bss_str:<7} | {ece_str:<7}")

mascaras = {
    'GLOBAL (Todos)': np.ones_like(y, dtype=bool),
    'SÉPTICOS (tiene_sepsis=1)': sepsis == 1,
    'NO SÉPTICOS (tiene_sepsis=0)': sepsis == 0
}

for nombre, mascara in mascaras.items():
    y_sub      = y[mascara]
    prob_sub   = prob_oof_calibrada[mascara]
    
    # Si en algún subgrupo no hay positivos (raro, pero por si acaso), evitamos error
    if y_sub.sum() == 0 or y_sub.sum() == len(y_sub):
        continue
        
    auc  = roc_auc_score(y_sub, prob_sub)
    aupr = average_precision_score(y_sub, prob_sub)
    bss  = calcular_bss(prob_sub, y_sub)
    ece  = calcular_ece(prob_sub, y_sub)
    
    imprimir_fila(nombre, mascara.sum(), auc, aupr, bss, ece)

print("-" * 70)


# Como el SOFA no es probabilidad, solo ses sacan AUCs
auc_sofa  = roc_auc_score(y, sofa_score)
aupr_sofa = average_precision_score(y, sofa_score)
imprimir_fila('SOFA (Baseline Clínico)', len(y), auc_sofa, aupr_sofa, np.nan, np.nan)
print("="*70)

# Comparativa final
mejora_auc = prob_oof_calibrada.sum() * 0 # Truco visual
print(f"\n✅XGBoost Calibrado supera al SOFA en {(roc_auc_score(y, prob_oof_calibrada) - auc_sofa)*100:.1f} puntos porcentuales de AUC-ROC.")