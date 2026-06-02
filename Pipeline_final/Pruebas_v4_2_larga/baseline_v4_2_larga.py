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
from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def cargar_datos():
    ruta = r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4l.csv'
    df = pd.read_csv(ruta)
    return df


def preparar(df):

    variables_predictoras = [
    # Demografía y contexto (4)
    'anchor_age',
    'gender',
    'peso_kg',
    'contador_estancia_uci',

    # Hemodinámica (2)
    'map_min',
    'hr_media',

    # Respiratorio (4)
    'pf_min',
    'spo2_min',
    #'fio2_max',   SUPRIMIDO POR ALTA CORRELACIÓN CON PF
    'rr_max',

    # Ventilación y conciencia (2)
    'ventilacion_invasiva_12h',
    'gcs_min',

    # Renal (2)
    'creatinina_max',
    'diuresis_ml_kg_12h',

    # Ácido-base (3)
    'lactato_max',
    'ph_min',
    'bicarbonato_min',

    # Hepático (2)
    'bilirrubina_media',
    'gpt_max',              

    # Coagulación (2)
    'tp_max',
    'plaquetas_min',

    # Hematología/inflamación (2)
    'leucocitos_min',       
    'hemoglobina_min',

    # Metabólico (1)
    'glucemia_min',         

    # Otro vital (1)
    'temp_min',              

    # Gravedad global (1)
    'sofa_max',              
]

    predictores = df[variables_predictoras].copy()
    etiqueta = df['etiqueta_norad_12_48'].copy()
    paciente_id = df['subject_id'].copy()

    predictores['gender'] = (predictores['gender'] == 'M').astype(int)

    return predictores, etiqueta, paciente_id


def validacion_cruzada_anidada(nombre_modelo, pipeline, espacio_hiperparametros,
                               predictores, etiqueta, paciente_id, n_jobs=-1):

    cv_externo = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    cv_interno = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)

    aucs_por_fold = []
    mejores_params_por_fold = []
    tiempo_inicio = time.time()

    for num_fold, (indices_train, indices_test) in enumerate(
            cv_externo.split(predictores, etiqueta, groups=paciente_id), start=1):

        x_train = predictores.iloc[indices_train]
        x_test = predictores.iloc[indices_test]
        y_train = etiqueta.iloc[indices_train]
        y_test = etiqueta.iloc[indices_test]
        pacientes_train = paciente_id.iloc[indices_train]

        busqueda = GridSearchCV(
            estimator=pipeline,
            param_grid=espacio_hiperparametros,
            cv=cv_interno,
            scoring='roc_auc',
            n_jobs=n_jobs,
            refit=True
        )

        busqueda.fit(x_train, y_train, groups=pacientes_train)

        probabilidades = busqueda.predict_proba(x_test)[:, 1]
        auc_fold = roc_auc_score(y_test, probabilidades)

        aucs_por_fold.append(auc_fold)
        mejores_params_por_fold.append(busqueda.best_params_)

        print(f"  Fold {num_fold}: AUC={auc_fold:.4f}")
        print(f"    Best params: {busqueda.best_params_}")

    tiempo_min = (time.time() - tiempo_inicio) / 60
    auc_medio = np.mean(aucs_por_fold)
    auc_desv = np.std(aucs_por_fold)

    print(f"\n{nombre_modelo} — AUC medio: {auc_medio:.4f} ± {auc_desv:.4f} "
          f"(tiempo: {tiempo_min:.1f} min)\n")

    return auc_medio, auc_desv, mejores_params_por_fold


def main():

    df = cargar_datos()
    predictores, etiqueta, paciente_id = preparar(df)

    print("=" * 70)
    print("BASELINE v4l — SET REDUCIDO")
    print("Ventana: observación 0-12h / predicción 12-48h")
    print("=" * 70)
    print(f"Dataset: {predictores.shape} | "
          f"Positivos: {etiqueta.sum()} ({100 * etiqueta.mean():.2f}%)")
    print(f"Pacientes únicos: {paciente_id.nunique()} | Estancias: {len(predictores)}")
    print(f"Variables predictoras: {list(predictores.columns)}\n")

    tiempo_global = time.time()
    resultados = {}

    modelos = {
        'LR': (
            'Regresión Logística',
            Pipeline([
                ('scaler', RobustScaler()),
                ('modelo', LogisticRegression(max_iter=5000, class_weight='balanced',
                                              solver='liblinear', random_state=42))
            ]),
            [{'modelo__C': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2,
                            0.1, 0.5, 1, 5, 10, 50]}],
            -1
        ),

        'RF': (
            'Random Forest',
            Pipeline([
                ('modelo', RandomForestClassifier(random_state=42, n_jobs=-1))
            ]),
            {
                'modelo__n_estimators': [300, 400, 500, 600, 700, 850, 1000],
                'modelo__max_depth': [None, 5, 10, 20, 30],
                'modelo__min_samples_leaf': [1, 2, 5],
                'modelo__max_features': ['sqrt', 0.3, 0.5],
                'modelo__class_weight': ['balanced', 'balanced_subsample'],
            },
            -1
        ),

        'XGB': (
            'XGBoost',
            Pipeline([
                ('modelo', XGBClassifier(objective='binary:logistic',
                                         eval_metric='auc',
                                         random_state=42,
                                         n_jobs=1,
                                         tree_method='hist'))
            ]),
            {
                'modelo__n_estimators': [300, 400, 500, 600, 750, 900],
                'modelo__max_depth': [3, 5, 7],
                'modelo__learning_rate': [0.005, 0.01, 0.03, 0.1],
                'modelo__subsample': [0.6, 0.8, 1.0],
                'modelo__colsample_bytree': [0.6, 0.8, 1.0],
                'modelo__reg_lambda': [0.1, 1, 10],
                'modelo__scale_pos_weight': [1, 5, 9],
            },
            -1
        ),

        'LGBM': (
            'LightGBM',
            Pipeline([
                ('modelo', LGBMClassifier(random_state=42,
                                          verbosity=-1,
                                          n_jobs=1,
                                          objective='binary'))
            ]),
            {
                'modelo__n_estimators': [300, 600, 1000],
                'modelo__num_leaves': [15, 31, 63],
                'modelo__learning_rate': [0.005, 0.01, 0.03, 0.1],
                'modelo__min_child_samples': [10, 30, 60],
                'modelo__reg_lambda': [0.1, 1, 10],
                'modelo__subsample': [0.6, 0.8, 1.0],
                'modelo__class_weight': ['balanced', None],
            },
            -1
        ),

        'CAT': (
            'CatBoost',
            Pipeline([
                ('modelo', CatBoostClassifier(loss_function='Logloss',
                                              eval_metric='AUC',
                                              random_seed=42,
                                              verbose=0,
                                              thread_count=-1))
            ]),
            {
                'modelo__iterations': [500, 1000],
                'modelo__depth': [4, 5, 6, 7],
                'modelo__learning_rate': [0.01, 0.03, 0.05, 0.1],
                'modelo__l2_leaf_reg': [1, 5, 15],
                'modelo__bagging_temperature': [0, 0.5, 1],
            },
            1
        ),

        'NB': (
            'Naive Bayes',
            Pipeline([
                ('scaler', RobustScaler()),
                ('modelo', GaussianNB())
            ]),
            {'modelo__var_smoothing': np.logspace(-12, -2, 30)},
            -1
        )
    }

    for clave, (nombre, pipeline, espacio, n_jobs) in modelos.items():
        print(nombre)
        resultados[clave] = validacion_cruzada_anidada(
            nombre, pipeline, espacio, predictores, etiqueta, paciente_id, n_jobs=n_jobs
        )

    tiempo_total_horas = (time.time() - tiempo_global) / 3600

    print("\n" + "-" * 70)
    print(f"RESUMEN FINAL — v4l REDUCIDO (tiempo total: {tiempo_total_horas:.2f} h)")
    print("-" * 70)

    ranking = sorted(resultados.items(), key=lambda item: -item[1][0])

    for nombre, (auc_medio, auc_desv, _) in ranking:
        print(f"  {nombre}  AUC = {auc_medio:.4f} ± {auc_desv:.4f}")


if __name__ == "__main__":
    main()