import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB


# ── CONFIGURACIÓN ──────────────────────────────────────────────────────────────

RUTA_CSV = r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4l.csv'

# Variables significativas por modelo (IC95% permutation importance excluye 0)
VARIABLES_POR_MODELO = {
    'LR': [
        'temp_min', 'pf_min', 'spo2_min', 'bicarbonato_min', 'rr_max', 'map_min'
    ],
    'RF': [
        'temp_min', 'spo2_min', 'bicarbonato_min', 'rr_max', 'map_min',
        'glucemia_min', 'sofa_max', 'lactato_max', 'diuresis_ml_kg_12h', 'bilirrubina_media'
    ],
    'XGB': [
        'temp_min', 'pf_min', 'spo2_min', 'bicarbonato_min', 'map_min', 'glucemia_min',
        'sofa_max', 'lactato_max'
    ],
    'LGBM': [
        'temp_min', 'spo2_min', 'bicarbonato_min', 'rr_max', 'glucemia_min',
         'sofa_max', 'diuresis_ml_kg_12h', 'map_min', 'pf_min'
    ],
    'CAT': [
        'pf_min', 'temp_min', 'diuresis_ml_kg_12h', 'bicarbonato_min',
        'rr_max', 'glucemia_min', 'ph_min', 'lactato_max', 
    ],
    'NB': [
        'temp_min', 'pf_min', 'spo2_min', 'rr_max', 'map_min'
    ],
}

ETIQUETA   = 'etiqueta_norad_12_48'
COLUMNA_ID = 'subject_id'


# ── CARGA Y PREPARACIÓN ────────────────────────────────────────────────────────

def cargar_datos():
    df = pd.read_csv(RUTA_CSV)
    return df


def preparar(df, variables):
    predictores = df[variables].copy()
    if 'gender' in predictores.columns:
        predictores['gender'] = (predictores['gender'] == 'M').astype(int)
    etiqueta    = df[ETIQUETA].copy()
    paciente_id = df[COLUMNA_ID].copy()
    return predictores, etiqueta, paciente_id


# ── CV ANIDADA ─────────────────────────────────────────────────────────────────

def validacion_cruzada_anidada(nombre_modelo, pipeline, espacio,
                               predictores, etiqueta, paciente_id,
                               n_jobs=-1):

    cv_externo = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    cv_interno = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)

    aucs_por_fold           = []
    mejores_params_por_fold = []
    tiempo_inicio           = time.time()

    for num_fold, (indices_train, indices_test) in enumerate(
            cv_externo.split(predictores, etiqueta, groups=paciente_id), start=1):

        x_train        = predictores.iloc[indices_train]
        x_test         = predictores.iloc[indices_test]
        y_train        = etiqueta.iloc[indices_train]
        y_test         = etiqueta.iloc[indices_test]
        pacientes_train = paciente_id.iloc[indices_train]

        busqueda = GridSearchCV(
            estimator=pipeline,
            param_grid=espacio,
            cv=cv_interno,
            scoring='roc_auc',
            n_jobs=n_jobs,
            refit=True,
        )
        busqueda.fit(x_train, y_train, groups=pacientes_train)

        probabilidades = busqueda.predict_proba(x_test)[:, 1]
        auc_fold       = roc_auc_score(y_test, probabilidades)

        aucs_por_fold.append(auc_fold)
        mejores_params_por_fold.append(busqueda.best_params_)

        print(f"  Fold {num_fold}: AUC={auc_fold:.4f}")
        print(f"    Best params: {busqueda.best_params_}")

    tiempo_min = (time.time() - tiempo_inicio) / 60
    auc_medio  = np.mean(aucs_por_fold)
    auc_desv   = np.std(aucs_por_fold)

    print(f"\n{nombre_modelo} — AUC medio: {auc_medio:.4f} ± {auc_desv:.4f}  "
          f"(tiempo: {tiempo_min:.1f} min)\n")

    return auc_medio, auc_desv, mejores_params_por_fold


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():

    df = cargar_datos()

    print("=" * 65)
    print("BASELINE v4 — VARIABLES SIGNIFICATIVAS PROPIAS POR MODELO")
    print("=" * 65)
    print(f"Estancias: {len(df)} | "
          f"Pacientes: {df[COLUMNA_ID].nunique()} | "
          f"Positivos: {df[ETIQUETA].sum()} "
          f"({100 * df[ETIQUETA].mean():.2f}%)")
    print()

    tiempo_global = time.time()
    resultados    = {}

    # ── 1. REGRESIÓN LOGÍSTICA ─────────────────────────────────────────────────
    print("─" * 50)
    print(f"Regresión Logística  ({len(VARIABLES_POR_MODELO['LR'])} variables)")
    print(f"  {VARIABLES_POR_MODELO['LR']}")

    pipeline_lr = Pipeline([
        ('escalador', RobustScaler()),
        ('modelo', LogisticRegression(
            max_iter=5000, class_weight='balanced',
            solver='liblinear', random_state=42,
        )),
    ])
    espacio_lr = [
        {'modelo__C': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2,
                       5e-2, 0.1, 0.5, 1, 5, 10, 50]}
    ]
    predictores, etiqueta, paciente_id = preparar(df, VARIABLES_POR_MODELO['LR'])
    resultados['LR'] = validacion_cruzada_anidada(
        'Regresión Logística', pipeline_lr, espacio_lr,
        predictores, etiqueta, paciente_id,
    )

    # ── 2. RANDOM FOREST ───────────────────────────────────────────────────────
    print("─" * 50)
    print(f"Random Forest  ({len(VARIABLES_POR_MODELO['RF'])} variables)")
    print(f"  {VARIABLES_POR_MODELO['RF']}")

    pipeline_rf = Pipeline([
        ('modelo', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])
    espacio_rf = {
        'modelo__n_estimators':     [300, 400, 500, 600, 700, 850, 1000],
        'modelo__max_depth':        [None, 5, 10, 20, 30],
        'modelo__min_samples_leaf': [1, 2, 5],
        'modelo__max_features':     ['sqrt', 0.3, 0.5],
        'modelo__class_weight':     ['balanced', 'balanced_subsample'],
    }
    predictores, etiqueta, paciente_id = preparar(df, VARIABLES_POR_MODELO['RF'])
    resultados['RF'] = validacion_cruzada_anidada(
        'Random Forest', pipeline_rf, espacio_rf,
        predictores, etiqueta, paciente_id,
    )

    # ── 3. XGBOOST ────────────────────────────────────────────────────────────
    print("─" * 50)
    print(f"XGBoost  ({len(VARIABLES_POR_MODELO['XGB'])} variables)")
    print(f"  {VARIABLES_POR_MODELO['XGB']}")

    pipeline_xgb = Pipeline([
        ('modelo', XGBClassifier(
            objective='binary:logistic', eval_metric='auc',
            random_state=42, n_jobs=1, tree_method='hist',
        )),
    ])
    espacio_xgb = {
        'modelo__n_estimators':     [300, 400, 500, 600, 750, 900],
        'modelo__max_depth':        [3, 5, 7],
        'modelo__learning_rate':    [0.005, 0.01, 0.03, 0.1],
        'modelo__subsample':        [0.6, 0.8, 1.0],
        'modelo__colsample_bytree': [0.6, 0.8, 1.0],
        'modelo__reg_lambda':       [0.1, 1, 10],
        'modelo__scale_pos_weight': [1, 5, 9],
    }
    predictores, etiqueta, paciente_id = preparar(df, VARIABLES_POR_MODELO['XGB'])
    resultados['XGB'] = validacion_cruzada_anidada(
        'XGBoost', pipeline_xgb, espacio_xgb,
        predictores, etiqueta, paciente_id,
    )

    # ── 4. LIGHTGBM ───────────────────────────────────────────────────────────
    print("─" * 50)
    print(f"LightGBM  ({len(VARIABLES_POR_MODELO['LGBM'])} variables)")
    print(f"  {VARIABLES_POR_MODELO['LGBM']}")

    pipeline_lgbm = Pipeline([
        ('modelo', LGBMClassifier(
            random_state=42, verbosity=-1, n_jobs=1, objective='binary',
        )),
    ])
    espacio_lgbm = {
        'modelo__n_estimators':      [300, 600, 1000],
        'modelo__num_leaves':        [15, 31, 63],
        'modelo__learning_rate':     [0.005, 0.01, 0.03, 0.1],
        'modelo__min_child_samples': [10, 30, 60],
        'modelo__reg_lambda':        [0.1, 1, 10],
        'modelo__subsample':         [0.6, 0.8, 1.0],
        'modelo__class_weight':      ['balanced', None],
    }
    predictores, etiqueta, paciente_id = preparar(df, VARIABLES_POR_MODELO['LGBM'])
    resultados['LGBM'] = validacion_cruzada_anidada(
        'LightGBM', pipeline_lgbm, espacio_lgbm,
        predictores, etiqueta, paciente_id,
    )

    # ── 5. CATBOOST ───────────────────────────────────────────────────────────
    print("─" * 50)
    print(f"CatBoost  ({len(VARIABLES_POR_MODELO['CAT'])} variables)")
    print(f"  {VARIABLES_POR_MODELO['CAT']}")

    pipeline_cat = Pipeline([
        ('modelo', CatBoostClassifier(
            loss_function='Logloss', eval_metric='AUC',
            random_seed=42, verbose=0, thread_count=-1,
        )),
    ])
    espacio_cat = {
        'modelo__iterations':          [500, 1000],
        'modelo__depth':               [4, 5, 6, 7],
        'modelo__learning_rate':       [0.01, 0.03, 0.05, 0.1],
        'modelo__l2_leaf_reg':         [1, 5, 15],
        'modelo__bagging_temperature': [0, 0.5, 1],
    }
    predictores, etiqueta, paciente_id = preparar(df, VARIABLES_POR_MODELO['CAT'])
    resultados['CAT'] = validacion_cruzada_anidada(
        'CatBoost', pipeline_cat, espacio_cat,
        predictores, etiqueta, paciente_id,
        n_jobs=1,
    )

    # ── 6. NAIVE BAYES ────────────────────────────────────────────────────────
    print("─" * 50)
    print(f"Naive Bayes  ({len(VARIABLES_POR_MODELO['NB'])} variables)")
    print(f"  {VARIABLES_POR_MODELO['NB']}")

    pipeline_nb = Pipeline([
        ('escalador', RobustScaler()),
        ('modelo', GaussianNB()),
    ])
    espacio_nb = {
        'modelo__var_smoothing': np.logspace(-12, -2, 30)
    }
    predictores, etiqueta, paciente_id = preparar(df, VARIABLES_POR_MODELO['NB'])
    resultados['NB'] = validacion_cruzada_anidada(
        'Naive Bayes', pipeline_nb, espacio_nb,
        predictores, etiqueta, paciente_id,
    )

    # ── RESUMEN FINAL ──────────────────────────────────────────────────────────
    tiempo_total_horas = (time.time() - tiempo_global) / 3600
    print()
    print("=" * 65)
    print(f"RESUMEN FINAL (tiempo total: {tiempo_total_horas:.2f} h)")
    print("=" * 65)
    print()

    ranking = sorted(resultados.items(), key=lambda item: -item[1][0])
    for clave, (auc_medio, auc_desv, _) in ranking:
        n_vars = len(VARIABLES_POR_MODELO[clave])
        print(f"  {clave:<6} ({n_vars:>2} vars)  AUC = {auc_medio:.4f} ± {auc_desv:.4f}")

if __name__ == "__main__":
    main() 