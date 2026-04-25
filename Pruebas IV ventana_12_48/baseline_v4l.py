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
 
def cargar_datos():
    ruta = r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4l.csv'
    df = pd.read_csv(ruta)
    # Por precaución, por si quedara algún NaN en pf_max por el winsorizado.
    df = df.dropna(subset=['pf_max'])
    return df
 
def preparar(df):
    
    variables_predictoras = [
        # Demografía y contexto
        'anchor_age', 'gender', 'contador_estancia_uci',

        # Sepsis, SOFA y ventilación (ventana 0-12h)
        'tiene_sepsis',
        'sofa_media', 'sofa_min', 'sofa_max',
        'ventilacion_invasiva_12h',
 
        # Laboratorio: para cada analito se toman media, mínimo y máximo
        # dentro de las primeras 12 horas de ingreso en UCI.
        'lactato_media', 'lactato_min', 'lactato_max',
        'creatinina_media', 'creatinina_min', 'creatinina_max',
        'plaquetas_media', 'plaquetas_min', 'plaquetas_max',
        'bilirrubina_media', 'bilirrubina_min', 'bilirrubina_max',
        'tp_media', 'tp_min', 'tp_max',
        'gpt_media', 'gpt_min', 'gpt_max',
        'got_media', 'got_min', 'got_max',
        'pao2_media', 'pao2_min', 'pao2_max',
        'ph_media', 'ph_min', 'ph_max',
        'leucocitos_media', 'leucocitos_min', 'leucocitos_max',
        'paco2_media', 'paco2_min', 'paco2_max',
        'bicarbonato_media', 'bicarbonato_min', 'bicarbonato_max',
        'glucemia_media', 'glucemia_min', 'glucemia_max',
        'hemoglobina_media', 'hemoglobina_min', 'hemoglobina_max',
 
        # Constantes vitales (chartevents, también ventana 0-12h)
        'hr_media', 'hr_min', 'hr_max',
        'rr_media', 'rr_min', 'rr_max',
        'temp_media', 'temp_min', 'temp_max',
        'spo2_media', 'spo2_min', 'spo2_max',
        'map_media', 'map_min', 'map_max',
        'fio2_media', 'fio2_min', 'fio2_max',

        # Índice P/F (PaO2 / FiO2) derivado
        'pf_media', 'pf_min', 'pf_max',

        # GCS (Glasgow)
        'gcs_media', 'gcs_min', 'gcs_max',

        # Diuresis 
        'peso_kg',
        'diuresis_ml_kg_12h',
    ]
    
 
    predictores = df[variables_predictoras].copy()
 
    # 1 si el paciente inició noradrenalina entre las horas 12 y 48 desde el
    # ingreso, 0 en caso contrario.
    etiqueta = df['etiqueta_norad_12_48'].copy()
 
    # subject_id se extrae aparte: se usará como "grupo" en el CV para asegurar
    # que un mismo paciente nunca aparezca simultáneamente en train y test.
    paciente_id = df['subject_id'].copy()
 
    # Codificación binaria del género: F=0, M=1.
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
 
 
def main():
    
    df = cargar_datos()
    predictores, etiqueta, paciente_id = preparar(df)
 
    print(f"Dataset: {predictores.shape} | "
          f"Positivos: {etiqueta.sum()} ({100*etiqueta.mean():.2f}%)")
    print(f"Pacientes únicos: {paciente_id.nunique()} | "
          f"Estancias: {len(predictores)}\n")
 
    tiempo_global = time.time()
    resultados = {}
 
    # 1. REGRESIÓN LOGÍSTICA
    print("Regresión Logística")
    pipeline_regresion_logistica = Pipeline([
        ('scaler', RobustScaler()),
        ('modelo', LogisticRegression(max_iter=5000, class_weight='balanced', solver='liblinear', random_state=42))])
    
    espacio_regresion_logistica = [
        {'modelo__C': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2,
                       0.1, 0.5, 1, 5, 10, 50]}
    ]
    resultados['LR'] = validacion_cruzada_anidada(
        'Regresión Logística',
        pipeline_regresion_logistica,
        espacio_regresion_logistica,
        predictores, etiqueta, paciente_id
    )
 
    # 2. RANDOM FOREST
    print("Random Forest")
    pipeline_random_forest = Pipeline([
        ('modelo', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])
    espacio_random_forest = {
        'modelo__n_estimators': [300, 400, 500, 600, 700, 850, 1000],
        'modelo__max_depth': [None, 5, 10, 20, 30],
        'modelo__min_samples_leaf': [1, 2, 5],
        'modelo__max_features': ['sqrt', 0.3, 0.5],
        'modelo__class_weight': ['balanced', 'balanced_subsample'],
    }
    resultados['RF'] = validacion_cruzada_anidada(
        'Random Forest',
        pipeline_random_forest,
        espacio_random_forest,
        predictores, etiqueta, paciente_id
    )
 
    # 3. XGBOOST
    print("XGBoost")
    pipeline_xgboost = Pipeline([
        ('modelo', XGBClassifier(objective='binary:logistic',eval_metric='auc',random_state=42,n_jobs=1,tree_method='hist'))])
    
    espacio_xgboost = {
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
        pipeline_xgboost,
        espacio_xgboost,
        predictores, etiqueta, paciente_id
    )
 
    # 4. LIGHTGBM
    print("LightGBM")
    pipeline_lightgbm = Pipeline([
        ('modelo', LGBMClassifier(random_state=42, verbosity=-1, n_jobs=1, objective='binary'))])
    
    espacio_lightgbm = {
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
        pipeline_lightgbm,
        espacio_lightgbm,
        predictores, etiqueta, paciente_id
    )
 
    # 5. CATBOOST
    print("CatBoost")
    pipeline_catboost = Pipeline([
        ('modelo', CatBoostClassifier(loss_function='Logloss', eval_metric='AUC', random_seed=42, verbose=0, thread_count=-1))])
    
    espacio_catboost = {
        'modelo__iterations': [500, 1000],
        'modelo__depth': [4, 5, 6, 7],
        'modelo__learning_rate': [0.01, 0.03, 0.05, 0.1],
        'modelo__l2_leaf_reg': [1, 5, 15],
        'modelo__bagging_temperature': [0, 0.5, 1],
    }
    resultados['CAT'] = validacion_cruzada_anidada(
        'CatBoost',
        pipeline_catboost,
        espacio_catboost,
        predictores, etiqueta, paciente_id,
        n_jobs=1
    )
 
    # 6. NAIVE BAYES
    print("Naive Bayes")
    pipeline_naive_bayes = Pipeline([
        ('scaler', RobustScaler()),
        ('modelo', GaussianNB())
    ])
    
    espacio_naive_bayes = {
        'modelo__var_smoothing': np.logspace(-12, -2, 30)
    }
    resultados['NB'] = validacion_cruzada_anidada(
        'Naive Bayes',
        pipeline_naive_bayes,
        espacio_naive_bayes,
        predictores, etiqueta, paciente_id
    )
 
    tiempo_total_horas = (time.time() - tiempo_global) / 3600
    print("\n" + "--------------")
    print(f"RESUMEN FINAL (tiempo total: {tiempo_total_horas} h)")
    print("------------")
 
    ranking = sorted(resultados.items(), key=lambda item: -item[1][0])
    for nombre, (auc_medio, auc_desv, _) in ranking:
        print(f"  {nombre}  AUC = {auc_medio} ± {auc_desv}")
 
 
if __name__ == "__main__":
    main()
