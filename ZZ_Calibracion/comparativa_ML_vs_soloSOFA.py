import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    roc_curve,
    precision_recall_curve,
)


# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

DIR_SCRIPT      = os.path.dirname(os.path.abspath(__file__))
CARPETA_SALIDA  = os.path.join(DIR_SCRIPT, 'comparacion')
CARPETA_FIGURAS = os.path.join(CARPETA_SALIDA, 'figuras')
CARPETA_TABLAS  = os.path.join(CARPETA_SALIDA, 'tablas')
os.makedirs(CARPETA_FIGURAS, exist_ok=True)
os.makedirs(CARPETA_TABLAS,  exist_ok=True)

# Carpeta donde evaluacion_final_v1.py guardó las tablas
CARPETA_PREDICCIONES = os.path.join(DIR_SCRIPT, 'evaluacion_final', 'tablas')


VENTANAS = {
    'v4_6_24_rf': {
        'ruta_csv': r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4.csv',
        'etiqueta': 'etiqueta_norad_6_24',
        'label_ml': 'RF 6-24h',
    },
    'v4p_3_12_catboost': {
        'ruta_csv': r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4p.csv',
        'etiqueta': 'etiqueta_norad_3_12',
        'label_ml': 'CatBoost 3-12h',
    },
    'v4l_12_48_catboost': {
        'ruta_csv': r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4l.csv',
        'etiqueta': 'etiqueta_norad_12_48',
        'label_ml': 'CatBoost 12-48h',
    },
}


# ─────────────────────────────────────────────
# MÉTRICAS
# ─────────────────────────────────────────────

def calcular_ece(y_real, probabilidad, n_bins=10):
    y_real       = np.asarray(y_real)
    probabilidad = np.asarray(probabilidad)
    cortes = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mascara = (probabilidad >= cortes[i]) & (probabilidad <= cortes[i + 1])
        else:
            mascara = (probabilidad >= cortes[i]) & (probabilidad < cortes[i + 1])
        if mascara.sum() == 0:
            continue
        ece += mascara.mean() * abs(y_real[mascara].mean() - probabilidad[mascara].mean())
    return ece


def brier_skill_score(y_real, prob_norm):
    """
    BSS = 1 - Brier / Brier_nulo.
    Brier_nulo: predecir siempre la prevalencia.
    Valores > 0 indican mejora respecto al predictor nulo.
    """
    brier      = brier_score_loss(y_real, prob_norm)
    prevalencia = y_real.mean()
    brier_nulo  = brier_score_loss(
        y_real, np.full(len(y_real), prevalencia, dtype=float),
    )
    return 1.0 - brier / brier_nulo


def normalizar_minmax(scores):
    """Lleva scores al rango [0,1]. Necesario para SOFA (rango 0-24)."""
    mn, mx = scores.min(), scores.max()
    if mx == mn:
        return np.zeros_like(scores, dtype=float)
    return (scores - mn) / (mx - mn)


def calcular_metricas(y_real, scores, nombre, es_probabilidad=True):
    """
    es_probabilidad=True  → scores ya en [0,1], no se normalizan para Brier/ECE.
    es_probabilidad=False → scores son rangos arbitrarios (SOFA): se normalizan.
    AUC y AUPRC no requieren normalización en ningún caso (son basadas en ranking).
    """
    y  = np.asarray(y_real, dtype=float)
    s  = np.asarray(scores, dtype=float)
    sn = s if es_probabilidad else normalizar_minmax(s)

    return {
        'modelo':            nombre,
        'n':                 len(y),
        'positivos':         int(y.sum()),
        'prevalencia':       round(float(y.mean()), 4),
        'auc':               round(roc_auc_score(y, s),          4),
        'auprc':             round(average_precision_score(y, s), 4),
        'brier':             round(brier_score_loss(y, sn),       4),
        'brier_skill_score': round(brier_skill_score(y, sn),      4),
        'ece':               round(calcular_ece(y, sn),           4),
    }


# ─────────────────────────────────────────────
# FIGURAS
# ─────────────────────────────────────────────

ESTILOS = {
    'ml':   {'linewidth': 2.5, 'color': '#e07b2e'},
    'sofa': {'linewidth': 2.5, 'color': '#2e86e0', 'linestyle': '--'},
}


def guardar_roc(nombre_ventana, y_real, scores_ml, scores_sofa, label_ml):
    fig, eje = plt.subplots(figsize=(6, 6))

    for scores, label, estilo in [
        (scores_ml,   label_ml, ESTILOS['ml']),
        (scores_sofa, 'SOFA',   ESTILOS['sofa']),
    ]:
        fpr, tpr, _ = roc_curve(y_real, scores)
        auc = roc_auc_score(y_real, scores)
        eje.plot(fpr, tpr, label=f'{label}  AUC={auc:.3f}', **estilo)

    eje.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
    eje.set_xlim(0, 1)
    eje.set_ylim(0, 1)
    eje.set_aspect('equal', adjustable='box')
    eje.grid(True, alpha=0.3)
    eje.set_xlabel('1 - Especificidad')
    eje.set_ylabel('Sensibilidad')
    eje.set_title(f'Curva ROC — {nombre_ventana}')
    eje.legend(loc='lower right', fontsize=10)
    plt.tight_layout()

    ruta = os.path.join(CARPETA_FIGURAS, f'roc_comparacion_{nombre_ventana}.png')
    fig.savefig(ruta, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return ruta


def guardar_pr(nombre_ventana, y_real, scores_ml, scores_sofa, label_ml):
    prevalencia = float(y_real.mean())
    max_prec    = 0.0

    fig, eje = plt.subplots(figsize=(6, 6))

    for scores, label, estilo in [
        (scores_ml,   label_ml, ESTILOS['ml']),
        (scores_sofa, 'SOFA',   ESTILOS['sofa']),
    ]:
        prec, rec, _ = precision_recall_curve(y_real, scores)
        auprc = average_precision_score(y_real, scores)
        eje.plot(rec, prec, label=f'{label}  AUPRC={auprc:.3f}', **estilo)
        max_prec = max(max_prec, prec[1:].max())

    eje.axhline(prevalencia, linestyle=':', color='gray', linewidth=1.5,
                label=f'Azar = {prevalencia:.3f}')
    eje.set_xlim(0, 1)
    eje.set_ylim(0, min(1.0, max_prec * 1.15))
    eje.grid(True, alpha=0.3)
    eje.set_xlabel('Recall / Sensibilidad')
    eje.set_ylabel('Precision / VPP')
    eje.set_title(f'Precision-Recall — {nombre_ventana}')
    eje.legend(loc='upper right', fontsize=10)
    plt.tight_layout()

    ruta = os.path.join(CARPETA_FIGURAS, f'pr_comparacion_{nombre_ventana}.png')
    fig.savefig(ruta, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return ruta


# ─────────────────────────────────────────────
# PROCESADO POR VENTANA
# ─────────────────────────────────────────────

def procesar_ventana(nombre_ventana, config):
    print(f'\n{"=" * 55}')
    print(f'  {nombre_ventana}')
    print(f'{"=" * 55}')

    # ── Cargar dataset original ──────────────────────────────────────
    df = pd.read_csv(config['ruta_csv'])
    df = df.dropna(subset=['pf_max']).reset_index(drop=True)
    # indice_original en las predicciones corresponde al índice de fila
    # del DataFrame original ANTES del reset_index. Como cargar_y_preparar
    # hace dropna y reset_index no, conservamos el índice original del df.

    if 'sofa_max' not in df.columns:
        raise ValueError(f'sofa_max no encontrado en {config["ruta_csv"]}')

    # ── Cargar predicciones del nested CV ────────────────────────────
    ruta_pred = os.path.join(
        CARPETA_PREDICCIONES, f'predicciones_{nombre_ventana}.csv',
    )
    if not os.path.exists(ruta_pred):
        raise FileNotFoundError(
            f'No se encontró el archivo de predicciones: {ruta_pred}\n'
            f'Ejecuta primero evaluacion_final_v1.py.'
        )
    predicciones = pd.read_csv(ruta_pred)

    # ── Unir con sofa_max por índice de fila ─────────────────────────
    # indice_original es el índice pandas del DataFrame original en el
    # momento en que se hizo el split en evaluacion_final_v1.py.
    sofa_por_indice = df['sofa_max']   # Series con índice = índice pandas del df

    predicciones['sofa_max'] = predicciones['indice_original'].map(sofa_por_indice)

    n_antes   = len(predicciones)
    n_sin_sofa = predicciones['sofa_max'].isna().sum()
    if n_sin_sofa > 0:
        print(f'  ⚠ {n_sin_sofa} filas sin sofa_max tras el merge — se excluyen.')
    predicciones = predicciones.dropna(subset=['sofa_max'])

    y_real      = predicciones['y_real'].values
    scores_ml   = predicciones['probabilidad_final'].values
    scores_sofa = predicciones['sofa_max'].values

    print(f'  Muestras  : {len(y_real)} ({n_antes - len(y_real)} excluidas)')
    print(f'  Positivos : {y_real.sum()} ({100 * y_real.mean():.1f}%)')
    print(f'  SOFA rango: {scores_sofa.min():.0f}–{scores_sofa.max():.0f}  '
          f'(media {scores_sofa.mean():.1f})')

    # ── Métricas ─────────────────────────────────────────────────────
    filas = [
        calcular_metricas(y_real, scores_ml,   config['label_ml'], es_probabilidad=True),
        calcular_metricas(y_real, scores_sofa, 'SOFA',             es_probabilidad=False),
    ]
    tabla = pd.DataFrame(filas)
    tabla.insert(0, 'ventana', nombre_ventana)

    print('\n  Resultados:')
    print(tabla[['modelo', 'auc', 'auprc', 'brier',
                 'brier_skill_score', 'ece']].to_string(index=False))

    ruta_tabla = os.path.join(
        CARPETA_TABLAS, f'tabla_comparacion_{nombre_ventana}.csv',
    )
    tabla.to_csv(ruta_tabla, index=False)

    # ── Figuras ───────────────────────────────────────────────────────
    ruta_roc = guardar_roc(
        nombre_ventana, y_real, scores_ml, scores_sofa, config['label_ml'],
    )
    ruta_pr = guardar_pr(
        nombre_ventana, y_real, scores_ml, scores_sofa, config['label_ml'],
    )
    print(f'\n  ROC : {ruta_roc}')
    print(f'  PR  : {ruta_pr}')
    print(f'  CSV : {ruta_tabla}')

    return tabla


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print('=' * 55)
    print('COMPETICIÓN: ML vs SOFA')
    print(f'Predicciones desde: {CARPETA_PREDICCIONES}')
    print(f'Salida en         : {CARPETA_SALIDA}')
    print('=' * 55)

    tablas = []
    for nombre, config in VENTANAS.items():
        tablas.append(procesar_ventana(nombre, config))

    tabla_global = pd.concat(tablas, ignore_index=True)
    ruta_global  = os.path.join(CARPETA_TABLAS, 'tabla_comparacion_global.csv')
    tabla_global.to_csv(ruta_global, index=False)

    print(f'\n{"=" * 55}')
    print('RESUMEN GLOBAL')
    print('=' * 55)
    print(tabla_global[
        ['ventana', 'modelo', 'auc', 'auprc', 'brier_skill_score', 'ece']
    ].to_string(index=False))
    print(f'\nTabla global: {ruta_global}')
    print('=' * 55)