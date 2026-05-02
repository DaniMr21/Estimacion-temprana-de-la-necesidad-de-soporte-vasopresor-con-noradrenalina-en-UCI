"""
Competición: modelos ML vs SOFA.

Compara el rendimiento predictivo de:
  - Modelo ML (predicciones del nested CV)
  - SOFA  (sofa_max, ya disponible en el dataset, rango 0-24)

Métricas:
  AUC, AUPRC, Brier, Brier Skill Score (vs predictor nulo), ECE.

Salida:
  comparacion/
    figuras/
      roc_comparacion_{ventana}.png
      pr_comparacion_{ventana}.png
    tablas/
      tabla_comparacion_{ventana}.csv
      tabla_comparacion_global.csv
"""

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


# CONFIGURACIÓN

DIR_SCRIPT = os.path.dirname(os.path.abspath(__file__))
CARPETA_SALIDA  = os.path.join(DIR_SCRIPT, 'comparacion')
CARPETA_FIGURAS = os.path.join(CARPETA_SALIDA, 'figuras')
CARPETA_TABLAS  = os.path.join(CARPETA_SALIDA, 'tablas')
os.makedirs(CARPETA_FIGURAS, exist_ok=True)
os.makedirs(CARPETA_TABLAS,  exist_ok=True)


# Actualiza las rutas de predicciones a las de tu última corrida
VENTANAS = {
    'v4_6_24_rf': {
        'ruta_csv':          r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4.csv',
        'ruta_predicciones': r'C:\Users\danie\TFG\Z_Calibracion\salida_calibracion_gridsearch\tablas\predicciones_calibracion_v4_6_24_rf.csv',
        'etiqueta':          'etiqueta_norad_6_24',
        'usar_calibrada':    True,
        'label_ml':          'RF 6-24h',
    },
    'v4p_3_12_catboost': {
        'ruta_csv':          r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4p.csv',
        'ruta_predicciones': r'C:\Users\danie\TFG\Z_Calibracion\salida_calibracion_gridsearch\tablas\predicciones_calibracion_v4p_3_12_catboost.csv',
        'etiqueta':          'etiqueta_norad_3_12',
        'usar_calibrada':    False,
        'label_ml':          'CatBoost 3-12h',
    },
    'v4l_12_48_catboost': {
        'ruta_csv':          r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4l.csv',
        'ruta_predicciones': r'C:\Users\danie\TFG\Z_Calibracion\salida_calibracion_gridsearch\tablas\predicciones_calibracion_v4l_12_48_catboost.csv',
        'etiqueta':          'etiqueta_norad_12_48',
        'usar_calibrada':    False,
        'label_ml':          'CatBoost 12-48h',
    },
}


# MÉTRICAS

def calcular_ece(y_real, probabilidad, n_bins=10):
    y_real = np.asarray(y_real)
    probabilidad = np.asarray(probabilidad)
    cortes = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (probabilidad >= cortes[i]) & (probabilidad <= cortes[i+1])
        else:
            mask = (probabilidad >= cortes[i]) & (probabilidad < cortes[i+1])
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(y_real[mask].mean() - probabilidad[mask].mean())
    return ece


def brier_skill_score(y_real, prob_norm):
    brier      = brier_score_loss(y_real, prob_norm)
    prevalencia = y_real.mean()
    brier_nulo  = brier_score_loss(y_real, np.full_like(y_real, prevalencia, dtype=float))
    return 1.0 - brier / brier_nulo


def normalizar(scores):
    """Lleva cualquier score al rango [0,1] para Brier y ECE."""
    mn, mx = scores.min(), scores.max()
    if mx == mn:
        return np.zeros_like(scores, dtype=float)
    return (scores - mn) / (mx - mn)


def calcular_metricas(y_real, scores, nombre):
    y  = np.asarray(y_real, dtype=float)
    s  = np.asarray(scores, dtype=float)
    sn = normalizar(s)
    return {
        'modelo':            nombre,
        'n':                 len(y),
        'positivos':         int(y.sum()),
        'prevalencia':       round(float(y.mean()), 4),
        'auc':               round(roc_auc_score(y, s),              4),
        'auprc':             round(average_precision_score(y, s),     4),
        'brier':             round(brier_score_loss(y, sn),           4),
        'brier_skill_score': round(brier_skill_score(y, sn),         4),
        'ece':               round(calcular_ece(y, sn),              4),
    }


# FIGURAS

ESTILOS = {
    'ml':   {'linewidth': 2.5, 'color': '#e07b2e'},
    'sofa': {'linewidth': 2.5, 'color': '#2e86e0', 'linestyle': '--'},
}


def guardar_roc(nombre_ventana, y_real, scores_ml, scores_sofa, label_ml):
    plt.figure(figsize=(6, 6))

    for scores, label, estilo in [
        (scores_ml,   label_ml, ESTILOS['ml']),
        (scores_sofa, 'SOFA',   ESTILOS['sofa']),
    ]:
        fpr, tpr, _ = roc_curve(y_real, scores)
        auc = roc_auc_score(y_real, scores)
        plt.plot(fpr, tpr, label=f'{label}  AUC={auc:.3f}', **estilo)

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, alpha=0.3)
    plt.xlabel('1 - Especificidad')
    plt.ylabel('Sensibilidad')
    plt.title(f'Curva ROC — {nombre_ventana}')
    plt.legend(loc='lower right', fontsize=10)
    plt.tight_layout()

    ruta = os.path.join(CARPETA_FIGURAS, f'roc_comparacion_{nombre_ventana}.png')
    plt.savefig(ruta, dpi=300, bbox_inches='tight')
    plt.close()
    return ruta


def guardar_pr(nombre_ventana, y_real, scores_ml, scores_sofa, label_ml):
    prevalencia = float(y_real.mean())
    max_prec = 0.0

    plt.figure(figsize=(6, 6))

    for scores, label, estilo in [
        (scores_ml,   label_ml, ESTILOS['ml']),
        (scores_sofa, 'SOFA',   ESTILOS['sofa']),
    ]:
        prec, rec, _ = precision_recall_curve(y_real, scores)
        auprc = average_precision_score(y_real, scores)
        plt.plot(rec, prec, label=f'{label}  AUPRC={auprc:.3f}', **estilo)
        max_prec = max(max_prec, prec[1:].max())  # ignora el punto trivial en recall=0

    plt.axhline(prevalencia, linestyle=':', color='gray',
                linewidth=1.5, label=f'Azar = {prevalencia:.3f}')
    plt.xlim(0, 1)
    plt.ylim(0, min(1.0, max_prec * 1.1))
    plt.grid(True, alpha=0.3)
    plt.xlabel('Recall / Sensibilidad')
    plt.ylabel('Precision / VPP')
    plt.title(f'Precision-Recall — {nombre_ventana}')
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()

    ruta = os.path.join(CARPETA_FIGURAS, f'pr_comparacion_{nombre_ventana}.png')
    plt.savefig(ruta, dpi=300, bbox_inches='tight')
    plt.close()
    return ruta


# PROCESADO POR VENTANA

def procesar_ventana(nombre_ventana, config):
    print(f'\n{"------------------"}')
    print(f'  {nombre_ventana}')
    print(f'{"-----------"}')

    # Dataset original → para obtener sofa_max de cada estancia
    df = pd.read_csv(config['ruta_csv'])
    df = df.dropna(subset=['pf_max']).copy()

    # Predicciones del nested CV (filas de test de los 5 folds)
    preds = pd.read_csv(config['ruta_predicciones'])

    # Merge: unir predicciones con sofa_max del dataset original
    # Las predicciones tienen subject_id + y_real
    df_join = preds.merge(
        df[['subject_id', 'sofa_max']].drop_duplicates(subset='subject_id'),
        on='subject_id',
        how='left',
    )

    n_antes  = len(preds)
    n_despues = df_join['sofa_max'].notna().sum()
    if n_despues < n_antes:
        print(f'  ⚠ {n_antes - n_despues} filas sin sofa_max tras el merge — se excluyen.')
    df_join = df_join.dropna(subset=['sofa_max'])

    y_real      = df_join['y_real'].values
    col_ml      = 'probabilidad_calibrada' if config['usar_calibrada'] else 'probabilidad_sin_calibrar'
    scores_ml   = df_join[col_ml].values
    scores_sofa = df_join['sofa_max'].values

    print(f'  Muestras: {len(y_real)}  |  Positivos: {y_real.sum()} ({100*y_real.mean():.1f}%)')
    print(f'  SOFA rango: {scores_sofa.min():.0f} – {scores_sofa.max():.0f}  '
          f'(media {scores_sofa.mean():.1f})')

    filas = [
        calcular_metricas(y_real, scores_ml,   config['label_ml']),
        calcular_metricas(y_real, scores_sofa,  'SOFA'),
    ]
    tabla = pd.DataFrame(filas)
    tabla.insert(0, 'ventana', nombre_ventana)

    ruta_tabla = os.path.join(CARPETA_TABLAS, f'tabla_comparacion_{nombre_ventana}.csv')
    tabla.to_csv(ruta_tabla, index=False)

    print('\n  Resultados:')
    print(tabla[['modelo','auc','auprc','brier','brier_skill_score','ece']].to_string(index=False))

    ruta_roc = guardar_roc(nombre_ventana, y_real, scores_ml, scores_sofa, config['label_ml'])
    ruta_pr  = guardar_pr( nombre_ventana, y_real, scores_ml, scores_sofa, config['label_ml'])

    print(f'\n  ROC: {ruta_roc}')
    print(f'  PR:  {ruta_pr}')
    print(f'  CSV: {ruta_tabla}')

    return tabla


# MAIN

if __name__ == '__main__':
    print('COMPETICIÓN: ML vs SOFA')
    print(f'Carpeta de salida: {CARPETA_SALIDA}')

    tablas = []
    for nombre, config in VENTANAS.items():
        tablas.append(procesar_ventana(nombre, config))

    global_df = pd.concat(tablas, ignore_index=True)
    ruta_global = os.path.join(CARPETA_TABLAS, 'tabla_comparacion_global.csv')
    global_df.to_csv(ruta_global, index=False)

    print(f'\n{"-------------"}')
    print('RESUMEN GLOBAL')
    print(f'{"---------------"}')
    print(global_df[['ventana','modelo','auc','auprc','brier_skill_score','ece']].to_string(index=False))
    print(f'\nTabla global guardada: {ruta_global}')