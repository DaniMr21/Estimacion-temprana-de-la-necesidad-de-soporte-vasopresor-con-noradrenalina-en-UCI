import warnings
warnings.filterwarnings('ignore')

from sklearn.base import clone

import pandas as pd
import numpy as np
import time
import joblib
import os
from collections import Counter

from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, brier_score_loss,
                             average_precision_score, log_loss)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB


# ── DIRECTORIOS ────────────────────────────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.abspath(__file__)) \
              if '__file__' in globals() else os.getcwd()
TABLAS_DIR  = os.path.join(BASE_DIR, 'TABLAS')
MODELOS_DIR = os.path.join(BASE_DIR, 'MODELOS_ENTRENADOS')

os.makedirs(TABLAS_DIR,  exist_ok=True)
os.makedirs(MODELOS_DIR, exist_ok=True)

RUTA_CSV_RESULTADOS = os.path.join(TABLAS_DIR, 'resultados_metricas_multiventana.csv')
COLUMNA_ID          = 'subject_id'


# ── CONFIGURACIÓN DE VENTANAS ──────────────────────────────────────────────────

CONFIG_VENTANAS = {

    # --- VENTANA CORTA COMENTADA PARA QUE LA IGNORE ---
    # 'Corto_3_12': {
    #     'ruta':     r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4p.csv',
    #     'etiqueta': 'etiqueta_norad_3_12',
    #     'vars': {
    #         'RF':   ['temp_min', 'rr_max'],
    #         'XGB':  ['pf_min', 'map_min', 'diuresis_ml_kg_3h', 'sofa_max'],
    #         'LGBM': ['map_min', 'pf_min', 'sofa_max'],
    #         'CAT':  ['map_min', 'pf_min', 'sofa_max', 'tp_max'],
    #         'NB':   ['map_min', 'temp_min', 'spo2_min'],
    #     }
    # },

    'Medio_6_24': {
        'ruta':     r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4.csv',
        'etiqueta': 'etiqueta_norad_6_24',
        'vars': {
            # 'LR':   ['pf_min', 'diuresis_ml_kg_6h', 'rr_max', 'sofa_max', 'hr_media', 'ventilacion_invasiva_6h'],
            # 'RF':   ['pf_min', 'map_min', 'diuresis_ml_kg_6h', 'rr_max', 'spo2_min', 'hr_media', 'glucemia_min', 'ventilacion_invasiva_6h'],
            # 'XGB':  ['pf_min', 'map_min', 'diuresis_ml_kg_6h', 'hr_media', 'sofa_max', 'ventilacion_invasiva_6h'],
            # 'LGBM': ['map_min', 'pf_min', 'diuresis_ml_kg_6h', 'spo2_min', 'ventilacion_invasiva_6h'],
            'CAT':  ['map_min', 'pf_min', 'diuresis_ml_kg_6h', 'hr_media', 'rr_max', 'glucemia_min', 'ventilacion_invasiva_6h'],
            # 'NB':   ['pf_min', 'diuresis_ml_kg_6h', 'rr_max', 'lactato_max', 'sofa_max', 'hr_media', 'ventilacion_invasiva_6h'],
        }
    },
    'Largo_12_48': {
        'ruta':     r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4l.csv',
        'etiqueta': 'etiqueta_norad_12_48',
        'vars': {
            # 'LR':   ['temp_min', 'pf_min', 'spo2_min', 'bicarbonato_min', 'rr_max', 'map_min'],
            # 'RF':   ['temp_min', 'spo2_min', 'bicarbonato_min', 'rr_max', 'map_min', 'glucemia_min', 'sofa_max', 'diuresis_ml_kg_12h', 'bilirrubina_media'],
            # 'XGB':  ['temp_min', 'pf_min', 'spo2_min', 'bicarbonato_min', 'map_min', 'glucemia_min', 'sofa_max'],
            # 'LGBM': ['temp_min', 'spo2_min', 'bicarbonato_min', 'rr_max', 'glucemia_min', 'sofa_max', 'diuresis_ml_kg_12h', 'map_min', 'pf_min'],
            'CAT':  ['pf_min', 'temp_min', 'diuresis_ml_kg_12h', 'bicarbonato_min', 'rr_max', 'glucemia_min'],
            # 'NB':   ['temp_min', 'pf_min', 'spo2_min', 'rr_max', 'map_min'],
        }
    }
}

# ── PIPELINES Y GRIDS ──────────────────────────────────────────────────────────
# IMPORTANTE: XGB y LGBM con n_jobs=1 dentro del pipeline.
# GridSearchCV ya usa n_jobs=-1 para paralelizar combinaciones.
# Combinar ambos n_jobs=-1 duplica los hilos y revienta la RAM.

ESPACIOS = {
    'LR': (
        Pipeline([
            ('escalador', RobustScaler()),
            ('modelo', LogisticRegression(
                max_iter=5000, class_weight='balanced',
                solver='liblinear', random_state=42,
            )),
        ]),
        {'modelo__C': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2,
                       5e-2, 0.1, 0.5, 1, 5, 10, 50]},
    ),
    'RF': (
        Pipeline([
            ('modelo', RandomForestClassifier(random_state=42, n_jobs=-1)),
        ]),
        {
            'modelo__n_estimators':     [300, 400, 500, 600, 700, 850, 1000],
            'modelo__max_depth':        [None, 5, 10, 20, 30],
            'modelo__min_samples_leaf': [1, 2, 5],
            'modelo__max_features':     ['sqrt', 0.3, 0.5],
            'modelo__class_weight':     ['balanced', 'balanced_subsample'],
        },
    ),
    'XGB': (
        Pipeline([
            ('modelo', XGBClassifier(
                objective='binary:logistic', eval_metric='auc',
                random_state=42,
                n_jobs=1,           # GridSearchCV ya paralela por fuera
                tree_method='hist',
                verbosity=0,
            )),
        ]),
        {
            'modelo__n_estimators':     [300, 400, 500, 600, 750, 900],
            'modelo__max_depth':        [3, 5, 7],
            'modelo__learning_rate':    [0.005, 0.01, 0.03, 0.1],
            'modelo__subsample':        [0.6, 0.8, 1.0],
            'modelo__colsample_bytree': [0.6, 0.8, 1.0],
            'modelo__reg_lambda':       [0.1, 1, 10],
            'modelo__scale_pos_weight': [1, 5, 9],
        },
    ),
    'LGBM': (
        Pipeline([
            ('modelo', LGBMClassifier(
                random_state=42, verbosity=-1,
                n_jobs=1,           # GridSearchCV ya paralela por fuera
                objective='binary',
            )),
        ]),
        {
            'modelo__n_estimators':      [300, 600, 1000],
            'modelo__num_leaves':        [15, 31, 63],
            'modelo__learning_rate':     [0.005, 0.01, 0.03, 0.1],
            'modelo__min_child_samples': [10, 30, 60],
            'modelo__reg_lambda':        [0.1, 1, 10],
            'modelo__subsample':         [0.6, 0.8, 1.0],
            'modelo__class_weight':      ['balanced', None],
        },
    ),
    'CAT': (
        Pipeline([
            ('modelo', CatBoostClassifier(
                loss_function='Logloss', eval_metric='AUC',
                random_seed=42, verbose=0,
                thread_count=1,     # GridSearchCV ya paralela por fuera
            )),
        ]),
        {
            'modelo__iterations':          [500, 1000],
            'modelo__depth':               [4, 5, 6, 7],
            'modelo__learning_rate':       [0.01, 0.03, 0.05, 0.1],
            'modelo__l2_leaf_reg':         [1, 5, 15],
            'modelo__bagging_temperature': [0, 0.5, 1],
        },
    ),
    'NB': (
        Pipeline([
            ('escalador', RobustScaler()),
            ('modelo', GaussianNB()),
        ]),
        {'modelo__var_smoothing': np.logspace(-12, -2, 30)},
    ),
}


# ── MÉTRICAS ───────────────────────────────────────────────────────────────────

def brier_skill_score(y_true, y_prob):
    prevalencia  = np.mean(y_true)
    prob_nula    = np.full_like(y_true, prevalencia, dtype=float)
    bs_modelo    = brier_score_loss(y_true, y_prob)
    bs_nulo      = brier_score_loss(y_true, prob_nula)
    return 1.0 - (bs_modelo / bs_nulo) if bs_nulo != 0 else 0.0


def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins   = np.linspace(0., 1., n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    ece    = 0.0
    for i in range(n_bins):
        mascara = (binids == i)
        if np.any(mascara):
            ece += (np.sum(mascara) / len(y_true)) * abs(
                np.mean(y_true[mascara]) - np.mean(y_prob[mascara])
            )
    return ece


# ── CARGA DE DATOS ─────────────────────────────────────────────────────────────

def preparar_datos(ruta, variables, etiqueta):
    df = pd.read_csv(ruta)
    df = df.dropna(subset=['pf_max'])   # igual que en todos los scripts anteriores
    x  = df[variables].copy()
    y  = df[etiqueta].copy()
    ids = df[COLUMNA_ID].copy()
    if 'gender' in x.columns:
        x['gender'] = (x['gender'] == 'M').astype(int)
    return x, y, ids


# ── CV ANIDADA (métricas) ──────────────────────────────────────────────────────

def ejecutar_cv(pipeline, espacio, x, y, ids):
    """
    CV anidada 5×3 para obtener métricas imparciales.
    Devuelve medias, desviaciones y los mejores parámetros de cada fold
    (para luego reentrenar el modelo final sobre todos los datos).
    """
    cv_externo = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    cv_interno = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)

    metricas  = {'AUC_ROC': [], 'AUC_PR': [], 'Brier': [],
                 'BSS': [], 'ECE': [], 'LogLoss': []}
    lista_params = []

    for indices_train, indices_test in cv_externo.split(x, y, groups=ids):
        busqueda = GridSearchCV(
            estimator=clone(pipeline),
            param_grid=espacio,
            cv=cv_interno,
            scoring='roc_auc',
            n_jobs=-1,
            refit=True,
        )
        busqueda.fit(
            x.iloc[indices_train], y.iloc[indices_train],
            groups=ids.iloc[indices_train],
        )
        probs = busqueda.predict_proba(x.iloc[indices_test])[:, 1]
        y_test = y.iloc[indices_test]

        metricas['AUC_ROC'].append(roc_auc_score(y_test, probs))
        metricas['AUC_PR'].append(average_precision_score(y_test, probs))
        metricas['Brier'].append(brier_score_loss(y_test, probs))
        metricas['BSS'].append(brier_skill_score(y_test, probs))
        metricas['ECE'].append(expected_calibration_error(y_test, probs))
        metricas['LogLoss'].append(log_loss(y_test, probs))
        lista_params.append(busqueda.best_params_)

    medias  = {k: np.mean(v) for k, v in metricas.items()}
    desviaciones = {k: np.std(v)  for k, v in metricas.items()}
    return medias, desviaciones, lista_params


# ── MODELO FINAL SOBRE TODOS LOS DATOS ────────────────────────────────────────

def entrenar_modelo_final(pipeline, espacio, x, y, ids, lista_params):
    """
    Reentrena sobre TODOS los datos con los hiperparámetros más frecuentes
    entre los 5 folds. Esto es lo que se guarda como modelo final.
    """
    # Parámetro más frecuente para cada clave
    mejores_params = {}
    for clave in lista_params[0].keys():
        valores = [p[clave] for p in lista_params]
        mejores_params[clave] = Counter(valores).most_common(1)[0][0]

    # Clonar pipeline y fijar hiperparámetros
    pipeline_final = clone(pipeline).set_params(**mejores_params)
    pipeline_final.fit(x, y)
    return pipeline_final, mejores_params


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    # Inicializar CSV si no existe
    if not os.path.exists(RUTA_CSV_RESULTADOS):
        pd.DataFrame(columns=[
            'Ventana', 'Modelo', 'n_vars',
            'AUC_ROC', 'AUC_ROC_std',
            'AUC_PR',  'AUC_PR_std',
            'Brier',   'Brier_std',
            'BSS',     'BSS_std',
            'ECE',     'ECE_std',
            'LogLoss', 'LogLoss_std',
        ]).to_csv(RUTA_CSV_RESULTADOS, index=False)

    tiempo_global = time.time()

    for nombre_ventana, conf in CONFIG_VENTANAS.items():
        print(f"\n{'='*55}")
        print(f"VENTANA: {nombre_ventana}")
        print(f"{'='*55}")

        for nombre_modelo, lista_vars in conf['vars'].items():

            # Checkpoint: omitir si ya está en el CSV
            csv_actual = pd.read_csv(RUTA_CSV_RESULTADOS)
            ya_hecho = (
                (csv_actual['Ventana'] == nombre_ventana) &
                (csv_actual['Modelo']  == nombre_modelo)
            ).any()
            if ya_hecho:
                print(f"  [{nombre_modelo}] Ya evaluado — omitido.")
                continue

            print(f"\n  [{nombre_modelo}] {len(lista_vars)} variables: {lista_vars}")
            t0 = time.time()

            try:
                x, y, ids = preparar_datos(
                    conf['ruta'], lista_vars, conf['etiqueta']
                )
                print(f"    Dataset: {x.shape} | "
                      f"Positivos: {y.sum()} ({100*y.mean():.1f}%)")

                pipeline, espacio = ESPACIOS[nombre_modelo]

                # ── CV anidada → métricas ──────────────────────────────
                medias, desviaciones, lista_params = ejecutar_cv(
                    pipeline, espacio, x, y, ids
                )

                # ── Modelo final sobre todos los datos ─────────────────
                modelo_final, params_finales = entrenar_modelo_final(
                    pipeline, espacio, x, y, ids, lista_params
                )

                # ── Guardar .pkl ───────────────────────────────────────
                ruta_pkl = os.path.join(
                    MODELOS_DIR,
                    f'modelo_{nombre_ventana}_{nombre_modelo}.pkl'
                )
                joblib.dump(modelo_final, ruta_pkl)

                # ── Guardar métricas en CSV ────────────────────────────
                fila = {
                    'Ventana': nombre_ventana,
                    'Modelo':  nombre_modelo,
                    'n_vars':  len(lista_vars),
                }
                for metrica in ['AUC_ROC', 'AUC_PR', 'Brier',
                                'BSS', 'ECE', 'LogLoss']:
                    fila[metrica]            = round(medias[metrica],      4)
                    fila[f'{metrica}_std']   = round(desviaciones[metrica], 4)

                pd.DataFrame([fila]).to_csv(
                    RUTA_CSV_RESULTADOS, mode='a', header=False, index=False
                )

                t_min = (time.time() - t0) / 60
                print(f"    AUC_ROC = {medias['AUC_ROC']:.4f} ± "
                      f"{desviaciones['AUC_ROC']:.4f}  "
                      f"| BSS = {medias['BSS']:.4f}  "
                      f"| ECE = {medias['ECE']:.4f}  "
                      f"({t_min:.1f} min)")
                print(f"    Hiperparámetros finales: {params_finales}")
                print(f"    Modelo guardado en: {ruta_pkl}")

            except Exception as error:
                print(f"    ERROR en {nombre_modelo}: {error}")

    tiempo_total = (time.time() - tiempo_global) / 3600
    print(f"\n{'='*55}")
    print(f"COMPLETADO (tiempo total: {tiempo_total:.2f} h)")
    print(f"  Métricas : {RUTA_CSV_RESULTADOS}")
    print(f"  Modelos  : {MODELOS_DIR}")
    print(f"{'='*55}")


if __name__ == '__main__':
    main()