import warnings
warnings.filterwarnings('ignore')

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

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
    ruta = r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4.csv'
    df = pd.read_csv(ruta)
    df = df.dropna(subset=['pf_max'])
    return df


def preparar(df):

    variables_predictoras = [
        'anchor_age',
        'gender',
        'peso_kg',
        'contador_estancia_uci',

        'map_min',
        'hr_media',

        'pf_min',
        'spo2_min',
        'fio2_max',
        'rr_max',

        'ventilacion_invasiva_6h',
        'gcs_min',

        'creatinina_max',
        'diuresis_ml_kg_6h',

        'lactato_max',
        'ph_min',
        'bicarbonato_min',

        'bilirrubina_media',
        'gpt_max',

        'tp_max',
        'plaquetas_min',

        'leucocitos_min',
        'hemoglobina_min',

        'glucemia_min',

        'temp_min',
        'sofa_max'
    ]

    predictores = df[variables_predictoras].copy()
    etiqueta = df['etiqueta_norad_6_24'].copy()
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

    print(f"\n{nombre_modelo} — AUC medio: {auc_medio:.4f} ± {auc_desv:.4f}  "
          f"(tiempo: {tiempo_min:.1f} min)\n")

    return auc_medio, auc_desv, mejores_params_por_fold


def calcular_shap_rf(predictores, etiqueta, paciente_id):

    carpeta_base = os.path.dirname(__file__) if '__file__' in dir() else '.'
    carpeta_figuras = os.path.join(carpeta_base, 'figuras')
    carpeta_tablas = os.path.join(carpeta_base, 'tablas')

    os.makedirs(carpeta_figuras, exist_ok=True)
    os.makedirs(carpeta_tablas, exist_ok=True)

    print("\n" + "---------------------------")
    print("SHAP — RANDOM FOREST FINAL")
    print("---------------------------")

    modelo_rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    )

    espacio_rf = {
        'n_estimators': [300, 400, 500, 600, 700, 850, 1000],
        'max_depth': [None, 5, 10, 20, 30],
        'min_samples_leaf': [1, 2, 5],
        'max_features': ['sqrt', 0.3, 0.5],
        'class_weight': ['balanced', 'balanced_subsample'],
    }

    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)

    busqueda_final = GridSearchCV(
        estimator=modelo_rf,
        param_grid=espacio_rf,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        refit=True
    )

    busqueda_final.fit(predictores, etiqueta, groups=paciente_id)

    modelo_final = busqueda_final.best_estimator_

    print("\nMejores hiperparámetros Random Forest final:")
    print(busqueda_final.best_params_)

    explicador = shap.TreeExplainer(modelo_final)
    valores_shap = explicador.shap_values(predictores)

    if isinstance(valores_shap, list):
        valores_shap_clase_positiva = valores_shap[1]
    elif len(np.array(valores_shap).shape) == 3:
        valores_shap_clase_positiva = valores_shap[:, :, 1]
    else:
        valores_shap_clase_positiva = valores_shap

    plt.figure()
    shap.summary_plot(
        valores_shap_clase_positiva,
        predictores,
        show=False,
        max_display=20
    )
    plt.tight_layout()
    ruta_beeswarm = os.path.join(carpeta_figuras, 'shap_beeswarm_rf_v4_2.png')
    plt.savefig(ruta_beeswarm, dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure()
    shap.summary_plot(
        valores_shap_clase_positiva,
        predictores,
        plot_type='bar',
        show=False,
        max_display=20
    )
    plt.tight_layout()
    ruta_bar = os.path.join(carpeta_figuras, 'shap_bar_rf_v4_2.png')
    plt.savefig(ruta_bar, dpi=300, bbox_inches='tight')
    plt.close()

    importancia_shap = pd.DataFrame({
        'variable': predictores.columns,
        'shap_medio_absoluto': np.abs(valores_shap_clase_positiva).mean(axis=0)
    }).sort_values('shap_medio_absoluto', ascending=False)

    ruta_tabla = os.path.join(carpeta_tablas, 'importancia_shap_rf_v4_2.csv')
    importancia_shap.to_csv(ruta_tabla, index=False)

    print("\nIMPORTANCIA SHAP")
    print(importancia_shap.to_string(index=False))

    print("\nFiguras guardadas en:")
    print(f"  {ruta_beeswarm}")
    print(f"  {ruta_bar}")
    print("Tabla guardada en:")
    print(f"  {ruta_tabla}")


def main():

    df = cargar_datos()
    predictores, etiqueta, paciente_id = preparar(df)

    print("------------------")
    print("BASELINE v5")
    print("Ventana: observación 0-6h / predicción 6-24h")
    print("---------------")
    print(f"Dataset: {predictores.shape} | "
          f"Positivos: {etiqueta.sum()} ({100 * etiqueta.mean():.2f}%)")
    print(f"Pacientes únicos: {paciente_id.nunique()} | "
          f"Estancias: {len(predictores)}")
    print(f"Variables predictoras: {list(predictores.columns)}")
    print()

    tiempo_global = time.time()
    resultados = {}

    print("Regresión Logística")
    pipeline_lr = Pipeline([
        ('scaler', RobustScaler()),
        ('modelo', LogisticRegression(
            max_iter=5000,
            class_weight='balanced',
            solver='liblinear',
            random_state=42
        ))
    ])

    espacio_lr = [
        {'modelo__C': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2,
                       0.1, 0.5, 1, 5, 10, 50]}
    ]

    resultados['LR'] = validacion_cruzada_anidada(
        'Regresión Logística',
        pipeline_lr,
        espacio_lr,
        predictores,
        etiqueta,
        paciente_id
    )

    print("Random Forest")
    pipeline_rf = Pipeline([
        ('modelo', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])

    espacio_rf = {
        'modelo__n_estimators': [300, 400, 500, 600, 700, 850, 1000],
        'modelo__max_depth': [None, 5, 10, 20, 30],
        'modelo__min_samples_leaf': [1, 2, 5],
        'modelo__max_features': ['sqrt', 0.3, 0.5],
        'modelo__class_weight': ['balanced', 'balanced_subsample'],
    }

    resultados['RF'] = validacion_cruzada_anidada(
        'Random Forest',
        pipeline_rf,
        espacio_rf,
        predictores,
        etiqueta,
        paciente_id
    )

    print("XGBoost")
    pipeline_xgb = Pipeline([
        ('modelo', XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            random_state=42,
            n_jobs=1,
            tree_method='hist'
        ))
    ])

    espacio_xgb = {
        'modelo__n_estimators': [300, 400, 500, 600, 750, 900],
        'modelo__max_depth': [3, 5, 7],
        'modelo__learning_rate': [0.005, 0.01, 0.03, 0.1],
        'modelo__subsample': [0.6, 0.8, 1.0],
        'modelo__colsample_bytree': [0.6, 0.8, 1.0],
        'modelo__reg_lambda': [0.1, 1, 10],
        'modelo__scale_pos_weight': [1, 5, 9],
    }

    resultados['XGB'] = validacion_cruzada_anidada(
        'XGBoost',
        pipeline_xgb,
        espacio_xgb,
        predictores,
        etiqueta,
        paciente_id
    )

    print("LightGBM")
    pipeline_lgbm = Pipeline([
        ('modelo', LGBMClassifier(
            random_state=42,
            verbosity=-1,
            n_jobs=1,
            objective='binary'
        ))
    ])

    espacio_lgbm = {
        'modelo__n_estimators': [300, 600, 1000],
        'modelo__num_leaves': [15, 31, 63],
        'modelo__learning_rate': [0.005, 0.01, 0.03, 0.1],
        'modelo__min_child_samples': [10, 30, 60],
        'modelo__reg_lambda': [0.1, 1, 10],
        'modelo__subsample': [0.6, 0.8, 1.0],
        'modelo__class_weight': ['balanced', None],
    }

    resultados['LGBM'] = validacion_cruzada_anidada(
        'LightGBM',
        pipeline_lgbm,
        espacio_lgbm,
        predictores,
        etiqueta,
        paciente_id
    )

    print("CatBoost")
    pipeline_cat = Pipeline([
        ('modelo', CatBoostClassifier(
            loss_function='Logloss',
            eval_metric='AUC',
            random_seed=42,
            verbose=0,
            thread_count=-1
        ))
    ])

    espacio_cat = {
        'modelo__iterations': [500, 1000],
        'modelo__depth': [4, 5, 6, 7],
        'modelo__learning_rate': [0.01, 0.03, 0.05, 0.1],
        'modelo__l2_leaf_reg': [1, 5, 15],
        'modelo__bagging_temperature': [0, 0.5, 1],
    }

    resultados['CAT'] = validacion_cruzada_anidada(
        'CatBoost',
        pipeline_cat,
        espacio_cat,
        predictores,
        etiqueta,
        paciente_id,
        n_jobs=1
    )

    print("Naive Bayes")
    pipeline_nb = Pipeline([
        ('scaler', RobustScaler()),
        ('modelo', GaussianNB())
    ])

    espacio_nb = {
        'modelo__var_smoothing': np.logspace(-12, -2, 30)
    }

    resultados['NB'] = validacion_cruzada_anidada(
        'Naive Bayes',
        pipeline_nb,
        espacio_nb,
        predictores,
        etiqueta,
        paciente_id
    )

    tiempo_total_horas = (time.time() - tiempo_global) / 3600

    print("\n" + "------------")
    print(f"RESUMEN FINAL — v5 (tiempo total: {tiempo_total_horas:.2f} h)")
    print("-----------------")

    ranking = sorted(resultados.items(), key=lambda item: -item[1][0])

    for nombre, (auc_medio, auc_desv, _) in ranking:
        print(f"  {nombre}  AUC = {auc_medio:.4f} ± {auc_desv:.4f}")

    calcular_shap_rf(predictores, etiqueta)


if __name__ == "__main__":
    main()