import os
import pickle
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
from sklearn.isotonic import IsotonicRegression
from scipy import stats
warnings.filterwarnings('ignore')

class ModeloCalibrado:
    def __init__(self, modelo_base, calibrador, tipo_calibrador):
        self.modelo_base     = modelo_base
        self.calibrador      = calibrador
        self.tipo_calibrador = tipo_calibrador

    def predict_proba(self, X):
        prob_bruta = self.modelo_base.predict_proba(X)[:, 1]
        if isinstance(self.calibrador, IsotonicRegression):
            prob_cal = self.calibrador.predict(prob_bruta)
        else:
            prob_cal = self.calibrador.predict_proba(
                prob_bruta.reshape(-1, 1))[:, 1]
        return np.column_stack([1 - prob_cal, prob_cal])

    def predict(self, X, umbral=0.5):
        return (self.predict_proba(X)[:, 1] >= umbral).astype(int)


CARPETA_MODELOS = r'C:\Users\danie\TFG\Pruebas_v4_4\MODELOS_ENTRENADOS'
CARPETA_DATOS   = r'C:\Users\danie\OneDrive\Escritorio\Data'
CARPETA_SALIDA  = r'C:\Users\danie\TFG\Validacion_externa'

N_BOOTSTRAP = 1000
SEMILLA     = 42

VENTANAS = {
    'Corto_3_12': {
        'pkl'     : 'modelo_Corto_3_12_CAT.pkl',
        'csv'     : 'eICU_3_12_definitivo.csv',
        'etiqueta': 'etiqueta_norad_3_12',
        'vars'    : ['map_min', 'pf_min', 'sofa_max', 'tp_max'],
        'label'   : 'Corto 3-12h (CatBoost)',
    },
    'Medio_6_24': {
        'pkl'     : 'modelo_Medio_6_24_XGB_calibrado.pkl',
        'csv'     : 'eICU_6_24_definitivo.csv',
        'etiqueta': 'etiqueta_norad_6_24',
        'vars'    : ['pf_min', 'map_min', 'diuresis_ml_kg_6h',
                     'hr_media', 'sofa_max', 'ventilacion_invasiva_6h'],
        'label'   : 'Medio 6-24h (XGBoost cal.)',
    },

    'Medio_6_24_CAT': {
        'pkl'     : 'modelo_Medio_6_24_CAT.pkl',
        'csv'     : 'eICU_6_24_definitivo.csv',
        'etiqueta': 'etiqueta_norad_6_24',
        'vars'    : ['pf_min', 'rr_max', 'map_min', 'diuresis_ml_kg_6h',
                     'ventilacion_invasiva_6h', 'hr_media', 'glucemia_min'],
        'label'   : 'Medio 6-24h (CatBoost)',
    },
    'Largo_12_48': {
        'pkl'     : 'modelo_Largo_12_48_XGB.pkl',
        'csv'     : 'eICU_12_48_definitivo.csv',
        'etiqueta': 'etiqueta_norad_12_48',
        'vars'    : ['temp_min', 'pf_min', 'spo2_min', 'bicarbonato_min',
                     'map_min', 'glucemia_min', 'sofa_max'],
        'label'   : 'Largo 12-48h (XGBoost)',
    },

    'Largo_12_48_CAT': {
        'pkl'     : 'modelo_Largo_12_48_CAT.pkl',
        'csv'     : 'eICU_12_48_definitivo.csv',
        'etiqueta': 'etiqueta_norad_12_48',
        'vars'    : ['pf_min', 'bicarbonato_min', 'rr_max',
                     'diuresis_ml_kg_12h', 'temp_min', 'glucemia_min'],
        'label'   : 'Largo 12-48h (CatBoost)',
    },
}

def calcular_ece(probabilidades, etiquetas, n_bins=10):
    limites  = np.linspace(0.0, 1.0, n_bins + 1)
    ece      = 0.0
    n_total  = len(etiquetas)
    for i in range(n_bins):
        m = (probabilidades >= limites[i]) & (probabilidades < limites[i + 1])
        if m.sum() == 0:
            continue
        ece += (m.sum() / n_total) * abs(
            probabilidades[m].mean() - etiquetas[m].mean())
    return float(ece)


def calcular_bss(probabilidades, etiquetas):
    prevalencia = etiquetas.mean()
    bs  = brier_score_loss(etiquetas, probabilidades)
    bs0 = brier_score_loss(etiquetas, np.full_like(probabilidades, prevalencia))
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
        idx  = rng.integers(0, n, size=n)
        y_b  = y[idx]
        p_b  = prob[idx]
        if y_b.sum() == 0 or y_b.sum() == n:
            continue
        m = metricas_punto(y_b, p_b)
        for k, v in m.items():
            resultados[k].append(v)
    ic = {}
    for k, vals in resultados.items():
        arr    = np.array(vals)
        ic[k]  = (float(np.percentile(arr, 2.5)),
                  float(np.percentile(arr, 97.5)))
    return ic


def delong_bootstrap(y, prob_modelo, prob_sofa,
                     n_iter=N_BOOTSTRAP, semilla=SEMILLA):
    """DeLong aproximado por bootstrap — diferencia de AUCs."""
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
    diffs  = np.array(diffs)
    media  = float(diffs.mean())
    z      = media / (diffs.std() + 1e-10)
    p      = float(2 * (1 - stats.norm.cdf(abs(z))))
    return media, float(z), p


def fmt_ic(val, ic):
    return f'{val:.3f} [{ic[0]:.3f}–{ic[1]:.3f}]'


def fmt_p(p):
    return '<0.001' if p < 0.001 else f'{p:.3f}'

filas_tabla = []

for nombre_ventana, cfg in VENTANAS.items():
    print(f'\n{"="*60}')
    print(f'  {cfg["label"]}')
    print(f'{"="*60}')

    try:
        modelo = joblib.load(os.path.join(CARPETA_MODELOS, cfg['pkl']))
    except Exception:
        with open(os.path.join(CARPETA_MODELOS, cfg['pkl']), 'rb') as f:
            modelo = pickle.load(f)

    # Cargar y limpiar datos externos
    df   = pd.read_csv(os.path.join(CARPETA_DATOS, cfg['csv']))
    y    = df[cfg['etiqueta']].values.astype(int)
    X    = df[cfg['vars']].copy()
    sofa = df['sofa_max'].values.astype(float)

    print(f'  N={len(y):,} | Positivos={y.sum()} ({100*y.mean():.1f}%)')

    # Predicciones
    prob_modelo = modelo.predict_proba(X)[:, 1]
    sofa_norm   = (sofa - sofa.min()) / (sofa.max() - sofa.min() + 1e-10)

    # Métricas puntuales
    m_mod  = metricas_punto(y, prob_modelo)
    m_sofa = metricas_punto(y, sofa_norm)

    # Bootstrap IC95%
    print('  Calculando bootstrap IC95%')
    ic_mod  = bootstrap_ic95(y, prob_modelo)
    ic_sofa = bootstrap_ic95(y, sofa_norm)

    # DeLong
    dif_auc, z_stat, p_delong = delong_bootstrap(y, prob_modelo, sofa_norm)

    # Imprimir resumen
    print(f'  AUC-ROC modelo : {fmt_ic(m_mod["auc_roc"], ic_mod["auc_roc"])}')
    print(f'  AUC-ROC SOFA   : {fmt_ic(m_sofa["auc_roc"], ic_sofa["auc_roc"])}')
    print(f'  ΔAUC={dif_auc:+.3f}  z={z_stat:.2f}  p={fmt_p(p_delong)}')
    print(f'  AUC-PR         : {fmt_ic(m_mod["auc_pr"], ic_mod["auc_pr"])}')
    print(f'  Brier Score    : {fmt_ic(m_mod["brier"], ic_mod["brier"])}')
    print(f'  BSS            : {fmt_ic(m_mod["bss"], ic_mod["bss"])}')
    print(f'  ECE            : {fmt_ic(m_mod["ece"], ic_mod["ece"])}')

    filas_tabla.append({
        'Ventana'        : cfg['label'],
        'N'              : f'{len(y):,}',
        'Prevalencia (%)': f'{100*y.mean():.1f}',
        'AUC-ROC modelo' : fmt_ic(m_mod['auc_roc'],  ic_mod['auc_roc']),
        'AUC-ROC SOFA'   : fmt_ic(m_sofa['auc_roc'], ic_sofa['auc_roc']),
        'ΔAUC'           : f'{dif_auc:+.3f}',
        'p DeLong'       : fmt_p(p_delong),
        'AUC-PR'         : fmt_ic(m_mod['auc_pr'],   ic_mod['auc_pr']),
        'Brier'          : fmt_ic(m_mod['brier'],     ic_mod['brier']),
        'BSS'            : fmt_ic(m_mod['bss'],       ic_mod['bss']),
        'ECE'            : fmt_ic(m_mod['ece'],       ic_mod['ece']),
    })

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Validación Externa — {cfg["label"]}',
                 fontsize=12, fontweight='bold')

    # ROC
    fpr_m, tpr_m, _ = roc_curve(y, prob_modelo)
    fpr_s, tpr_s, _ = roc_curve(y, sofa_norm)
    axes[0].plot(fpr_m, tpr_m, lw=2, color='#1f77b4',
                 label=(f'Modelo  AUC={m_mod["auc_roc"]:.3f} '
                        f'[{ic_mod["auc_roc"][0]:.3f}–'
                        f'{ic_mod["auc_roc"][1]:.3f}]'))
    axes[0].plot(fpr_s, tpr_s, lw=2, color='#ff7f0e', linestyle='--',
                 label=(f'SOFA    AUC={m_sofa["auc_roc"]:.3f} '
                        f'[{ic_sofa["auc_roc"][0]:.3f}–'
                        f'{ic_sofa["auc_roc"][1]:.3f}]'))
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4)
    axes[0].set_xlabel('1 - Especificidad (FPR)')
    axes[0].set_ylabel('Sensibilidad (TPR)')
    axes[0].set_title('Curva ROC')
    axes[0].legend(fontsize=8, loc='lower right')
    axes[0].grid(True, alpha=0.3)
    axes[0].text(0.38, 0.06,
                 f'p DeLong = {fmt_p(p_delong)}',
                 fontsize=8.5, transform=axes[0].transAxes,
                 bbox=dict(boxstyle='round', facecolor='#fff9c4', alpha=0.8))

    # Calibración
    frac_pos, media_pred = calibration_curve(
        y, prob_modelo, n_bins=10, strategy='quantile')
    axes[1].plot(media_pred, frac_pos, marker='o', lw=2,
                 color='#1f77b4', label='Modelo (eICU)')
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4, label='Perfecta')
    axes[1].set_xlabel('Probabilidad predicha media')
    axes[1].set_ylabel('Fracción de positivos')
    axes[1].set_title(f'Curva de Calibración\n'
                      f'BSS={m_mod["bss"]:.3f}  ECE={m_mod["ece"]:.3f}')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(-0.01, min(prob_modelo.max() + 0.05, 1.0))
    axes[1].set_ylim(-0.01, 1.0)

    plt.tight_layout()
    ruta_fig = os.path.join(CARPETA_SALIDA,
                            f'validacion_externa_{nombre_ventana}.png')
    plt.savefig(ruta_fig, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'  Figura: {ruta_fig}')

df_tabla  = pd.DataFrame(filas_tabla)
ruta_csv  = os.path.join(CARPETA_SALIDA, 'validacion_externa_metricas.csv')
df_tabla.to_csv(ruta_csv, index=False, encoding='utf-8-sig')

print(f'\n{"="*60}')
print('RESUMEN FINAL')
print(f'{"="*60}')
print(df_tabla[['Ventana', 'N', 'Prevalencia (%)',
                'AUC-ROC modelo', 'AUC-ROC SOFA',
                'p DeLong', 'BSS', 'ECE']].to_string(index=False))
print(f'\nTabla guardada en: {ruta_csv}')
print('\n[Fin] Validación externa completada.')
print(f'IC95% por bootstrap ({N_BOOTSTRAP} iteraciones, semilla={SEMILLA}).')
print('DeLong test aproximado por bootstrap.')
