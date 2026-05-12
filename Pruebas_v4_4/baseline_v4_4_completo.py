import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import time
import shap
import matplotlib.pyplot as plt

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

# ── 1. MÉTRICAS AVANZADAS PARA RIGOR METODOLÓGICO ─────────────────────────────

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

# ── 2. CONFIGURACIÓN DE VENTANAS (VARIABLES DEPURADAS) ────────────────────────

CONFIG_VENTANAS = {
    'Corto_3_12': {
        'ruta': r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4p.csv',
        'etiqueta': 'etiqueta_norad_3_12',
        'vars': {
            # Se eliminan gpt_max, fio2_max y creatinina_max por incoherencia 
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
            # Se eliminan gpt_max, creatinina_max, ph_min y temp_min 
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
            # Se eliminan gpt_max, ph_min y lactato_max 
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

# ── 3. FUNCIONES DE EJECUCIÓN Y EXPLICABILIDAD ────────────────────────────────

def preparar_datos(ruta, variables, etiqueta):
    df = pd.read_csv(ruta)
    if 'pf_max' in df.columns: df = df.dropna(subset=['pf_max'])
    X, y, ids = df[variables].copy(), df[etiqueta].copy(), df[COLUMNA_ID].copy()
    if 'gender' in X.columns: X['gender'] = (X['gender'] == 'M').astype(int)
    return X, y, ids

def ejecutar_validacion(nombre_modelo, pipeline, espacio, X, y, ids):
    cv_externo = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    cv_interno = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    res = {'AUC_ROC': [], 'AUC_PR': [], 'Brier': [], 'BSS': [], 'ECE': [], 'LogLoss': []}
    
    for train_idx, test_idx in cv_externo.split(X, y, groups=ids):
        busqueda = GridSearchCV(pipeline, espacio, cv=cv_interno, scoring='roc_auc', n_jobs=-1, refit=True)
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

def calcular_shap(model, X, ventana, modelo_n):
    try:
        m = model.named_steps['modelo']
        explainer = shap.TreeExplainer(m) if modelo_n not in ['LR', 'NB'] else shap.KernelExplainer(m.predict_proba, shap.sample(X, 100))
        vals = explainer.shap_values(X)
        if isinstance(vals, list): vals = vals[1]
        plt.figure(figsize=(10, 6))
        shap.summary_plot(vals, X, show=False)
        plt.title(f"SHAP - {ventana} - {modelo_n}")
        plt.savefig(f"SHAP_{ventana}_{modelo_n}.png", dpi=300)
        plt.close()
    except Exception as e: print(f"Error SHAP {modelo_n}: {e}")

# ── 4. MAIN ───────────────────────────────────────────────────────────────────

def main():
    final_res = []
    # IMPORTANTE: Reemplaza estos espacios por tus rejillas completas para el TFG final
    espacios = {
        'LR': (Pipeline([('s', RobustScaler()), ('modelo', LogisticRegression(class_weight='balanced', random_state=42))]), {'modelo__C': [0.005, 0.1, 1]}),
        'RF': (Pipeline([('modelo', RandomForestClassifier(random_state=42, n_jobs=1))]), {'modelo__n_estimators': [500], 'modelo__max_depth': [10, 20]}),
        'XGB': (Pipeline([('modelo', XGBClassifier(random_state=42, n_jobs=1, tree_method='hist'))]), {'modelo__n_estimators': [500], 'modelo__max_depth': [3, 5]}),
        'LGBM': (Pipeline([('modelo', LGBMClassifier(random_state=42, n_jobs=1))]), {'modelo__n_estimators': [500]}),
        'CAT': (Pipeline([('modelo', CatBoostClassifier(random_seed=42, verbose=0))]), {'modelo__iterations': [500]}),
        'NB': (Pipeline([('s', RobustScaler()), ('modelo', GaussianNB())]), {'modelo__var_smoothing': [1e-9, 1e-5]})
    }

    for vent, conf in CONFIG_VENTANAS.items():
        print(f"\n>>> Ventana: {vent}")
        for m_id, v_list in conf['vars'].items():
            X, y, ids = preparar_datos(conf['ruta'], v_list, conf['etiqueta'])
            pipe, param = espacios[m_id]
            print(f"  Entrenando {m_id}...", end="", flush=True)
            m_m, m_s, m_f = ejecutar_validacion(m_id, pipe, param, X, y, ids)
            final_res.append({'Ventana': vent, 'Modelo': m_id, **m_m})
            calcular_shap(m_f, X, vent, m_id)
            print(" Ok.")

    df = pd.DataFrame(final_res)
    df.to_csv('resultados_finales_tfg_depurados.csv', index=False)
    print("\nProceso completado. Resultados en CSV y gráficos SHAP generados.")

if __name__ == "__main__":
    main()