"""
Figura Final Validación Externa — Ventana Largo 12-48h
=======================================================
Usa el modelo base entrenado en MIMIC + calibrador Platt recalibrado a eICU.
Genera la figura final con ROC y curva de calibración recalibrada.

Salida:
  validacion_externa_Largo_12_48_FINAL.png
"""

import os
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    brier_score_loss, roc_curve,
)
from sklearn.calibration import calibration_curve
from scipy import stats
warnings.filterwarnings('ignore')

# ── CONFIGURACIÓN ──────────────────────────────────────────────────────────────
CARPETA_MODELOS = r'C:\Users\danie\TFG\Pruebas_v4_4\MODELOS_ENTRENADOS'
CARPETA_DATOS   = r'C:\Users\danie\OneDrive\Escritorio\DATA'
CARPETA_SALIDA  = r'C:\Users\danie\TFG\Validacion_externa'

MODELO_PKL      = 'modelo_Largo_12_48_XGB.pkl'
CALIBRADOR_PKL  = 'calibrador_Largo_eICU.pkl'
CSV_EICU        = 'eICU_12_48_definitivo.csv'
ETIQUETA        = 'etiqueta_norad_12_48'
VARIABLES       = ['temp_min', 'pf_min', 'spo2_min', 'bicarbonato_min',
                   'map_min', 'glucemia_min', 'sofa_max']

N_BOOTSTRAP = 1000
SEMILLA     = 42


# ── FUNCIONES MÉTRICAS ─────────────────────────────────────────────────────────

def calcular_ece(prob, y, n_bins=10):
    limites = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n_total = 0.0, len(y)
    for i in range(n_bins):
        m = (prob >= limites[i]) & (prob < limites[i + 1])
        if m.sum() == 0:
            continue
        ece += (m.sum() / n_total) * abs(prob[m].mean() - y[m].mean())
    return float(ece)


def calcular_bss(prob, y):
    prev = y.mean()
    bs   = brier_score_loss(y, prob)
    bs0  = brier_score_loss(y, np.full_like(prob, prev))
    return float(1 - bs / bs0) if bs0 != 0 else np.nan


def metricas_punto(y, prob):
    return {
        'auc_roc': float(roc_auc_score(y, prob)),
        'auc_pr' : float(average_precision_score(y, prob)),
        'brier'  : float(brier_score_loss(y, prob)),
        'bss'    : calcular_bss(prob, y),
        'ece'    : calcular_ece(prob, y),
    }


def bootstrap_ic95(y, prob, n_iter=N_BOOTSTRAP, semilla=SEMILLA):
    rng        = np.random.default_rng(semilla)
    resultados = {k: [] for k in ['auc_roc', 'auc_pr', 'brier', 'bss', 'ece']}
    n          = len(y)
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        p_b = prob[idx]
        if y_b.sum() == 0 or y_b.sum() == n:
            continue
        m = metricas_punto(y_b, p_b)
        for k, v in m.items():
            resultados[k].append(v)
    ic = {}
    for k, vals in resultados.items():
        arr = np.array(vals)
        ic[k] = (float(np.percentile(arr, 2.5)),
                 float(np.percentile(arr, 97.5)))
    return ic


def delong_bootstrap(y, prob_modelo, prob_sofa,
                     n_iter=N_BOOTSTRAP, semilla=SEMILLA):
    rng   = np.random.default_rng(semilla)
    n     = len(y)
    diffs = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        if y_b.sum() == 0 or y_b.sum() == n:
            continue
        diffs.append(
            roc_auc_score(y_b, prob_modelo[idx]) -
            roc_auc_score(y_b, prob_sofa[idx])
        )
    diffs = np.array(diffs)
    media = float(diffs.mean())
    z     = media / (diffs.std() + 1e-10)
    p     = float(2 * (1 - stats.norm.cdf(abs(z))))
    return media, float(z), p


def fmt_ic(val, ic):
    return f'{val:.3f} [{ic[0]:.3f}–{ic[1]:.3f}]'


def fmt_p(p):
    return '<0.001' if p < 0.001 else f'{p:.3f}'


# ── CARGA MODELO + CALIBRADOR ──────────────────────────────────────────────────
print('Cargando modelo base y calibrador...')
modelo      = joblib.load(os.path.join(CARPETA_MODELOS, MODELO_PKL))
calibrador  = joblib.load(os.path.join(CARPETA_MODELOS, CALIBRADOR_PKL))

# ── CARGA DATOS eICU ───────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(CARPETA_DATOS, CSV_EICU)).dropna(subset=VARIABLES)
y    = df[ETIQUETA].values.astype(int)
X    = df[VARIABLES].copy()
sofa = df['sofa_max'].values.astype(float)
print(f'  N={len(y):,} | Positivos={y.sum()} ({100*y.mean():.1f}%)')

# ── PROBABILIDADES: ORIGINAL Y RECALIBRADA ─────────────────────────────────────
prob_original    = modelo.predict_proba(X)[:, 1]
prob_recalibrada = calibrador.predict_proba(prob_original.reshape(-1, 1))[:, 1]

# SOFA normalizado como baseline
sofa_norm = (sofa - sofa.min()) / (sofa.max() - sofa.min() + 1e-10)

# ── MÉTRICAS ──────────────────────────────────────────────────────────────────
print('Calculando métricas y bootstrap IC95%...')
m_recal  = metricas_punto(y, prob_recalibrada)
m_orig   = metricas_punto(y, prob_original)
m_sofa   = metricas_punto(y, sofa_norm)

ic_recal = bootstrap_ic95(y, prob_recalibrada)
ic_sofa  = bootstrap_ic95(y, sofa_norm)

dif_auc, z_stat, p_delong = delong_bootstrap(y, prob_recalibrada, sofa_norm)

print(f'\n  AUC-ROC modelo recalibrado : {fmt_ic(m_recal["auc_roc"], ic_recal["auc_roc"])}')
print(f'  AUC-ROC SOFA               : {fmt_ic(m_sofa["auc_roc"], ic_sofa["auc_roc"])}')
print(f'  ΔAUC={dif_auc:+.3f}  z={z_stat:.2f}  p={fmt_p(p_delong)}')
print(f'  BSS  recalibrado : {fmt_ic(m_recal["bss"], ic_recal["bss"])}')
print(f'  ECE  recalibrado : {fmt_ic(m_recal["ece"], ic_recal["ece"])}')

# ── FIGURA ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
fig.suptitle('Validación Externa — Largo 12-48h (XGBoost + recalibración Platt eICU)',
             fontsize=12, fontweight='bold')

# Panel ROC
fpr_m, tpr_m, _ = roc_curve(y, prob_recalibrada)
fpr_s, tpr_s, _ = roc_curve(y, sofa_norm)
axes[0].plot(fpr_m, tpr_m, lw=2, color='#1f77b4',
             label=(f'Modelo recal.  AUC={m_recal["auc_roc"]:.3f} '
                    f'[{ic_recal["auc_roc"][0]:.3f}–'
                    f'{ic_recal["auc_roc"][1]:.3f}]'))
axes[0].plot(fpr_s, tpr_s, lw=2, color='#ff7f0e', linestyle='--',
             label=(f'SOFA          AUC={m_sofa["auc_roc"]:.3f} '
                    f'[{ic_sofa["auc_roc"][0]:.3f}–'
                    f'{ic_sofa["auc_roc"][1]:.3f}]'))
axes[0].plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4)
axes[0].set_xlabel('1 - Especificidad (FPR)')
axes[0].set_ylabel('Sensibilidad (TPR)')
axes[0].set_title('Curva ROC')
axes[0].legend(fontsize=8, loc='lower right')
axes[0].grid(True, alpha=0.3)
axes[0].text(0.38, 0.06, f'p DeLong = {fmt_p(p_delong)}',
             fontsize=8.5, transform=axes[0].transAxes,
             bbox=dict(boxstyle='round', facecolor='#fff9c4', alpha=0.85))

# Panel Calibración — original vs recalibrada superpuestas
frac_o, mp_o = calibration_curve(y, prob_original,    n_bins=10, strategy='quantile')
frac_r, mp_r = calibration_curve(y, prob_recalibrada, n_bins=10, strategy='quantile')

axes[1].plot(mp_o, frac_o, marker='s', lw=1.6, color='#888',
             linestyle='--', alpha=0.7,
             label=f'Original  BSS={m_orig["bss"]:.3f}  ECE={m_orig["ece"]:.3f}')
axes[1].plot(mp_r, frac_r, marker='o', lw=2, color='#1f77b4',
             label=f'Recalibrada  BSS={m_recal["bss"]:.3f}  ECE={m_recal["ece"]:.3f}')
axes[1].plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4, label='Perfecta')
axes[1].set_xlabel('Probabilidad predicha media')
axes[1].set_ylabel('Fracción de positivos')
axes[1].set_title('Curva de Calibración\nOriginal (MIMIC) vs Recalibrada (eICU)')
axes[1].legend(fontsize=7.5, loc='upper left')
axes[1].grid(True, alpha=0.3)
prob_max = max(prob_original.max(), prob_recalibrada.max())
axes[1].set_xlim(-0.01, min(prob_max + 0.05, 1.0))
axes[1].set_ylim(-0.01, 1.0)

plt.tight_layout()
ruta_fig = os.path.join(CARPETA_SALIDA, 'validacion_externa_Largo_12_48_FINAL.png')
plt.savefig(ruta_fig, dpi=180, bbox_inches='tight')
plt.close(fig)

print(f'\nFigura guardada en: {ruta_fig}')
print('[Fin]')
