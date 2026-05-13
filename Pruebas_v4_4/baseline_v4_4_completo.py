import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import time
import joblib
import os

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

# ── CREACIÓN DE DIRECTORIOS ──
# Obtiene la ruta exacta donde está este script
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
TABLAS_DIR = os.path.join(BASE_DIR, "TABLAS")
MODELOS_DIR = os.path.join(BASE_DIR, "MODELOS ENTRENADOS")

# Crea las carpetas si no existen
os.makedirs(TABLAS_DIR, exist_ok=True)
os.makedirs(MODELOS_DIR, exist_ok=True)

# Ruta del archivo final
FILE_CSV = os.path.join(TABLAS_DIR, 'resultados_metricas_multiventana.csv')

# ── MÉTRICAS AVANZADAS ──
def brier_skill_score(y_true, y_prob):
    prevalencia = np.mean(y_true)
    prob_nula = np.full_like(y_true, prevalencia, dtype=float)
    bs_modelo = brier_score_loss(y_true, y_prob)
    bs_nulo = brier_score_loss(y_true, prob_nula)
    return 1.0 - (bs_modelo / bs_nulo) if bs_nulo != 0 else 0.0

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0., 1., n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for i in range(n_bins):
        mask = (binids == i)
        if np.any(mask):
            ece += (np.sum(mask) / len(y_true)) * np.abs(np.mean(y_true[mask]) - np.mean(y_prob[mask]))
    return ece

# ── DICCIONARIO LIMPIO ──
CONFIG_VENTANAS = {
    'Corto_3_12': {
        'ruta': r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4p.csv',
        'etiqueta': 'etiqueta_norad_3_12',
        'vars': {
            'RF':   ['temp_min', 'rr_max'],
            'XGB':  ['pf_min', 'map_min', 'diuresis_ml_kg_3h', 'sofa_max'],
            'LGBM': ['map_min', 'pf_min', 'sofa_max'],
            'CAT':  ['map_min', 'pf_min', 'sofa_max', 'tp_max'],
            'NB':   ['map_min', 'temp_min', 'spo2_min']
        }
    },
    'Medio_6_24': {
        'ruta': r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4.csv',
        'etiqueta': 'etiqueta_norad_6_24',
        'vars': {
            'LR':   ['pf_min', 'diuresis_ml_kg_6h', 'rr_max', 'sofa_max', 'hr_media', 'ventilacion_invasiva_6h'],
            'RF':   ['pf_min', 'map_min', 'diuresis_ml_kg_6h', 'rr_max', 'spo2_min', 'hr_media', 'glucemia_min', 'ventilacion_invasiva_6h'],
            'XGB':  ['pf_min', 'map_min', 'diuresis_ml_kg_6h', 'hr_media', 'sofa_max', 'ventilacion_invasiva_6h'],
            'LGBM': ['map_min', 'pf_min', 'diuresis_ml_kg_6h', 'spo2_min', 'ventilacion_invasiva_6h'],
            'CAT':  ['map_min', 'pf_min', 'diuresis_ml_kg_6h', 'hr_media', 'rr_max', 'glucemia_min', 'ventilacion_invasiva_6h'],
            'NB':   ['pf_min', 'diuresis_ml_kg_6h', 'rr_max', 'lactato_max', 'sofa_max', 'hr_media', 'ventilacion_invasiva_6h']
        }
    },
    'Largo_12_48': {
        'ruta': r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4l.csv',
        'etiqueta': 'etiqueta_norad_12_48',
        'vars': {
            'LR':   ['temp_min', 'pf_min', 'spo2_min', 'bicarbonato_min', 'rr_max', 'map_min'],
            'RF':   ['temp_min', 'spo2_min', 'bicarbonato_min', 'rr_max', 'map_min', 'glucemia_min', 'sofa_max', 'diuresis_ml_kg_12h', 'bilirrubina_media'],
            'XGB':  ['temp_min', 'pf_min', 'spo2_min', 'bicarbonato_min', 'map_min', 'glucemia_min', 'sofa_max'],
            'LGBM': ['temp_min', 'spo2_min', 'bicarbonato_min', 'rr_max', 'glucemia_min', 'sofa_max', 'diuresis_ml_kg_12h', 'map_min', 'pf_min'],
            'CAT':  ['pf_min', 'temp_min', 'diuresis_ml_kg_12h', 'bicarbonato_min', 'rr_max', 'glucemia_min'],
            'NB':   ['temp_min', 'pf_min', 'spo2_min', 'rr_max', 'map_min']
        }
    }
}
COLUMNA_ID = 'subject_id'

# ── FUNCIONES ──
def preparar_datos(ruta, variables, etiqueta):
    df = pd.read_csv(ruta)
    X, y, ids = df[variables].copy(), df[etiqueta].copy(), df[COLUMNA_ID].copy()
    if 'gender' in X.columns: X['gender'] = (X['gender'] == 'M').astype(int)
    return X, y, ids

def ejecutar_validacion(pipeline, espacio, X, y, ids):
    cv_externo = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    cv_interno = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    res = {'AUC_ROC': [], 'AUC_PR': [], 'Brier': [], 'BSS': [], 'ECE': [], 'LogLoss': []}
    
    for train_idx, test_idx in cv_externo.split(X, y, groups=ids):
        # n_jobs=1 estrictamente para asegurar que no colapse la RAM
        busqueda = GridSearchCV(pipeline, espacio, cv=cv_interno, scoring='roc_auc', n_jobs=1, refit=True)
        busqueda.fit(X.iloc[train_idx], y.iloc[train_idx], groups=ids.iloc[train_idx])
        probs = busqueda.predict_proba(X.iloc[test_idx])[:, 1]
        y_t = y.iloc[test_idx]
        
        res['AUC_ROC'].append(roc_auc_score(y_t, probs))
        res['AUC_PR'].append(average_precision_score(y_t, probs))
        res['Brier'].append(brier_score_loss(y_t, probs))
        res['BSS'].append(brier_skill_score(y_t, probs))
        res['ECE'].append(expected_calibration_error(y_t, probs))
        res['LogLoss'].append(log_loss(y_t, probs))
        
    return {k: np.mean(v) for k, v in res.items()}, {k: np.std(v) for k, v in res.items()}, busqueda.best_estimator_

# ── MAIN ──
def main():
    # Iniciar CSV si no existe
    if not os.path.exists(FILE_CSV):
        pd.DataFrame(columns=['Ventana', 'Modelo', 'AUC_ROC', 'AUC_PR', 'Brier', 'BSS', 'ECE', 'LogLoss']).to_csv(FILE_CSV, index=False)

    espacios = {
        'LR': (Pipeline([('s', RobustScaler()), ('modelo', LogisticRegression(max_iter=5000, class_weight='balanced', solver='liblinear', random_state=42))]), {'modelo__C': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5, 1, 5, 10, 50]}),
        'RF': (Pipeline([('modelo', RandomForestClassifier(random_state=42, n_jobs=1))]), {'modelo__n_estimators': [300, 400, 500, 600, 700, 850, 1000], 'modelo__max_depth': [None, 5, 10, 20, 30], 'modelo__min_samples_leaf': [1, 2, 5], 'modelo__max_features': ['sqrt', 0.3, 0.5], 'modelo__class_weight': ['balanced', 'balanced_subsample']}),
        'XGB': (Pipeline([('modelo', XGBClassifier(objective='binary:logistic', eval_metric='auc', random_state=42, n_jobs=1, tree_method='hist'))]), {'modelo__n_estimators': [300, 400, 500, 600, 750, 900], 'modelo__max_depth': [3, 5, 7], 'modelo__learning_rate': [0.005, 0.01, 0.03, 0.1], 'modelo__subsample': [0.6, 0.8, 1.0], 'modelo__colsample_bytree': [0.6, 0.8, 1.0], 'modelo__reg_lambda': [0.1, 1, 10], 'modelo__scale_pos_weight': [1, 5, 9]}),
        'LGBM': (Pipeline([('modelo', LGBMClassifier(random_state=42, verbosity=-1, n_jobs=1, objective='binary'))]), {'modelo__n_estimators': [300, 600, 1000], 'modelo__num_leaves': [15, 31, 63], 'modelo__learning_rate': [0.005, 0.01, 0.03, 0.1], 'modelo__min_child_samples': [10, 30, 60], 'modelo__reg_lambda': [0.1, 1, 10], 'modelo__subsample': [0.6, 0.8, 1.0], 'modelo__class_weight': ['balanced', None]}),
        'CAT': (Pipeline([('modelo', CatBoostClassifier(loss_function='Logloss', eval_metric='AUC', random_seed=42, verbose=0, thread_count=1))]), {'modelo__iterations': [500, 1000], 'modelo__depth': [4, 5, 6, 7], 'modelo__learning_rate': [0.01, 0.03, 0.05, 0.1], 'modelo__l2_leaf_reg': [1, 5, 15], 'modelo__bagging_temperature': [0, 0.5, 1]}),
        'NB': (Pipeline([('s', RobustScaler()), ('modelo', GaussianNB())]), {'modelo__var_smoothing': np.logspace(-12, -2, 30)})
    }

    for vent, conf in CONFIG_VENTANAS.items():
        print(f"\n{'='*40}\nVentana: {vent}\n{'='*40}")
        for m_id, v_list in conf['vars'].items():
            
            # 1. Comprobar si ya existe en el CSV para saltarlo si el script se interrumpió
            current_csv = pd.read_csv(FILE_CSV)
            if ((current_csv['Ventana'] == vent) & (current_csv['Modelo'] == m_id)).any():
                print(f"  [OMITIDO] {m_id} ya se encuentra evaluado en el CSV.")
                continue

            # 2. Entrenar y evaluar
            X, y, ids = preparar_datos(conf['ruta'], v_list, conf['etiqueta'])
            pipe, param = espacios[m_id]
            print(f"  [ENTRENANDO] {m_id} ({len(v_list)} vars)...", end="", flush=True)
            
            try:
                m_m, m_s, m_f = ejecutar_validacion(pipe, param, X, y, ids)
                
                # 3. Guardado Inmediato del Modelo (.pkl) en subcarpeta MODELOS ENTRENADOS
                nombre_archivo = os.path.join(MODELOS_DIR, f"modelo_final_{vent}_{m_id}.pkl")
                joblib.dump(m_f, nombre_archivo)
                
                # 4. Guardado Inmediato de las Métricas (CSV) en subcarpeta TABLAS
                nueva_fila = pd.DataFrame([{'Ventana': vent, 'Modelo': m_id, **m_m}])
                nueva_fila.to_csv(FILE_CSV, mode='a', header=False, index=False)
                
                print(f" OK. Guardado.")
            except Exception as e:
                print(f" ERROR: {e}")

    print(f"\nPROCESO COMPLETADO. Revisa las carpetas '{TABLAS_DIR}' y '{MODELOS_DIR}'.")

if __name__ == "__main__":
    main()