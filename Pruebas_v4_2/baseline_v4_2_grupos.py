"""
Baseline v4 con set REDUCIDO de variables, ENTRENADO POR SUBGRUPOS.

Subgrupos analizados:
  - GLOBAL      : toda la cohorte (referencia)
  - SEPSIS      : tiene_sepsis == 1
  - NO_SEPSIS   : tiene_sepsis == 0

Hipótesis a contrastar:
  ¿El modelo predice igual de bien el inicio de noradrenalina en
  pacientes sépticos que en no sépticos? ¿Las variables relevantes
  son las mismas o cambian?

Decisiones metodológicas:
  - Mismas 26 variables que baseline_v4_reducido.py.
  - Se EXCLUYE `tiene_sepsis` del modelo cuando se entrena por subgrupo
    (porque dentro de cada subgrupo es constante, aporta cero).
  - Mismo grid de hiperparámetros que el baseline reducido global,
    para permitir comparación directa de AUCs.
  - CV anidada 5x3 agrupada por paciente.
  - class_weight='balanced' / scale_pos_weight para gestionar desbalance.
"""

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


# -----------------------------------------------------------------------------
# CONFIGURACIÓN
# -----------------------------------------------------------------------------

# Set reducido idéntico al de baseline_v4_reducido.py.
# OJO: tiene_sepsis NO se incluye aquí porque al filtrar por subgrupo es
# constante y no aporta información dentro del subgrupo.
VARIABLES_PREDICTORAS = [
    # Demografía y contexto (4)
    'anchor_age', 'gender', 'peso_kg', 'contador_estancia_uci',
    # Hemodinámica (2)
    'map_min', 'hr_media',
    # Respiratorio (4)
    'pf_min', 'spo2_min', 'fio2_media', 'rr_max',
    # Ventilación y conciencia (2)
    'ventilacion_invasiva_6h', 'gcs_min',
    # Renal (2)
    'creatinina_max', 'diuresis_ml_kg_6h',
    # Ácido-base (3)
    'lactato_max', 'ph_min', 'bicarbonato_min',
    # Hepático (2)
    'bilirrubina_media', 'gpt_media',
    # Coagulación (2)
    'tp_max', 'plaquetas_min',
    # Hematología/inflamación (2)
    'leucocitos_max', 'hemoglobina_min',
    # Otros (3)
    'glucemia_max', 'temp_max', 'sofa_media',
]

ETIQUETA = 'etiqueta_norad_6_24'


# -----------------------------------------------------------------------------
# CARGA Y PREPARACIÓN
# -----------------------------------------------------------------------------
def cargar_datos():
    ruta = r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4.csv'
    df = pd.read_csv(ruta)
    df = df.dropna(subset=['pf_max'])
    return df


def preparar_subgrupo(df, nombre_subgrupo):
    """
    Filtra el DataFrame según el subgrupo solicitado.
    Devuelve (predictores, etiqueta, paciente_id) listos para CV.
    """
    if nombre_subgrupo == 'GLOBAL':
        df_sub = df.copy()
    elif nombre_subgrupo == 'SEPSIS':
        df_sub = df[df['tiene_sepsis'] == 1].copy()
    elif nombre_subgrupo == 'NO_SEPSIS':
        df_sub = df[df['tiene_sepsis'] == 0].copy()
    else:
        raise ValueError(f"Subgrupo no reconocido: {nombre_subgrupo}")

    predictores = df_sub[VARIABLES_PREDICTORAS].copy()
    predictores['gender'] = (predictores['gender'] == 'M').astype(int)

    etiqueta = df_sub[ETIQUETA].copy()
    paciente_id = df_sub['subject_id'].copy()

    return predictores, etiqueta, paciente_id


# -----------------------------------------------------------------------------
# CV ANIDADA (igual que baseline_v4_reducido.py)
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# DEFINICIÓN DE LOS 6 MODELOS (mismo grid que el baseline global)
# -----------------------------------------------------------------------------
def definir_modelos():
    """
    Devuelve dict con (nombre_legible, pipeline, espacio_hiperparametros, n_jobs).
    """
    pipelines = {
        'LR': (
            'Regresión Logística',
            Pipeline([
                ('scaler', RobustScaler()),
                ('modelo', LogisticRegression(
                    max_iter=5000, class_weight='balanced',
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
                ('modelo', XGBClassifier(
                    objective='binary:logistic', eval_metric='auc',
                    random_state=42, n_jobs=1, tree_method='hist'))
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
                ('modelo', LGBMClassifier(
                    random_state=42, verbosity=-1, n_jobs=1, objective='binary'))
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
                ('modelo', CatBoostClassifier(
                    loss_function='Logloss', eval_metric='AUC',
                    random_seed=42, verbose=0, thread_count=-1))
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
        ),
    }
    return pipelines


# -----------------------------------------------------------------------------
# EVALUACIÓN COMPLETA DE UN SUBGRUPO
# -----------------------------------------------------------------------------
def evaluar_subgrupo(df, nombre_subgrupo):
    print("\n" + "=" * 70)
    print(f"SUBGRUPO: {nombre_subgrupo}")
    print("=" * 70)

    predictores, etiqueta, paciente_id = preparar_subgrupo(df, nombre_subgrupo)

    n_estancias = len(predictores)
    n_pacientes = paciente_id.nunique()
    n_positivos = int(etiqueta.sum())
    prevalencia = 100 * etiqueta.mean()

    print(f"  Estancias        : {n_estancias}")
    print(f"  Pacientes únicos : {n_pacientes}")
    print(f"  Positivos        : {n_positivos} ({prevalencia:.2f}%)")
    print(f"  Variables        : {predictores.shape[1]}")

    if etiqueta.nunique() < 2:
        print("\n  ERROR: el subgrupo solo tiene una clase, no se entrena.")
        return None

    if n_positivos < 30:
        print(f"\n  ADVERTENCIA: solo {n_positivos} positivos. "
              f"AUCs poco fiables, alta varianza esperable.")

    print()

    pipelines = definir_modelos()
    resultados = {}
    tiempo_subgrupo_inicio = time.time()

    for clave, (nombre_legible, pipeline, espacio, n_jobs) in pipelines.items():
        print(f"--- {nombre_legible} ({nombre_subgrupo}) ---")
        try:
            auc_medio, auc_desv, params = validacion_cruzada_anidada(
                f"{nombre_legible} [{nombre_subgrupo}]",
                pipeline, espacio,
                predictores, etiqueta, paciente_id,
                n_jobs=n_jobs
            )
            resultados[clave] = (auc_medio, auc_desv, params)
        except Exception as e:
            print(f"  ERROR en {nombre_legible}: {e}\n")
            resultados[clave] = (np.nan, np.nan, [])

    tiempo_horas = (time.time() - tiempo_subgrupo_inicio) / 3600

    print("\n" + "-" * 50)
    print(f"RESUMEN {nombre_subgrupo} (tiempo: {tiempo_horas:.2f} h)")
    print("-" * 50)
    ranking = sorted(resultados.items(),
                     key=lambda item: -item[1][0] if not np.isnan(item[1][0]) else 1)
    for nombre, (auc_medio, auc_desv, _) in ranking:
        if not np.isnan(auc_medio):
            print(f"  {nombre:5s}  AUC = {auc_medio:.4f} ± {auc_desv:.4f}")
        else:
            print(f"  {nombre:5s}  AUC = NaN (error en entrenamiento)")
    print()

    return {
        'subgrupo': nombre_subgrupo,
        'n_estancias': n_estancias,
        'n_pacientes': n_pacientes,
        'n_positivos': n_positivos,
        'prevalencia_pct': prevalencia,
        'tiempo_horas': tiempo_horas,
        'resultados': resultados,
    }


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("BASELINE v4 REDUCIDO — POR SUBGRUPOS DE SEPSIS")
    print(f"Variables: {len(VARIABLES_PREDICTORAS)} (set reducido sin tiene_sepsis)")
    print("=" * 70)

    df = cargar_datos()
    print(f"\nDataset cargado: {len(df)} estancias")
    print(f"  Sépticos      : {(df['tiene_sepsis']==1).sum()} "
          f"({100*(df['tiene_sepsis']==1).mean():.1f}%)")
    print(f"  No sépticos   : {(df['tiene_sepsis']==0).sum()} "
          f"({100*(df['tiene_sepsis']==0).mean():.1f}%)")

    tiempo_total = time.time()

    resumen = {}
    for subgrupo in ['GLOBAL', 'SEPSIS', 'NO_SEPSIS']:
        resumen[subgrupo] = evaluar_subgrupo(df, subgrupo)

    # -----------------------------------------------------------------------
    # TABLA COMPARATIVA FINAL
    # -----------------------------------------------------------------------
    horas_totales = (time.time() - tiempo_total) / 3600
    print("\n" + "=" * 70)
    print(f"TABLA COMPARATIVA FINAL (tiempo total: {horas_totales:.2f} h)")
    print("=" * 70)

    # Cabecera
    print(f"\n  {'Modelo':<8} {'GLOBAL':<22} {'SEPSIS':<22} {'NO_SEPSIS':<22}")
    print(f"  {'-'*8} {'-'*22} {'-'*22} {'-'*22}")

    modelos = ['LR', 'RF', 'XGB', 'LGBM', 'CAT', 'NB']
    for m in modelos:
        linea = f"  {m:<8}"
        for sub in ['GLOBAL', 'SEPSIS', 'NO_SEPSIS']:
            if resumen[sub] and m in resumen[sub]['resultados']:
                auc, sd, _ = resumen[sub]['resultados'][m]
                if not np.isnan(auc):
                    linea += f" {auc:.4f} ± {sd:.4f}     "
                else:
                    linea += f" {'NaN':<22}"
            else:
                linea += f" {'-':<22}"
        print(linea)

    print()
    print("INFO POBLACIONAL:")
    for sub in ['GLOBAL', 'SEPSIS', 'NO_SEPSIS']:
        r = resumen[sub]
        if r:
            print(f"  {sub:<10} N={r['n_estancias']:<5} "
                  f"Pos={r['n_positivos']:<4} "
                  f"Prev={r['prevalencia_pct']:.2f}%")
    print()
    print("INTERPRETACIÓN PREVISTA:")
    print("  - Si AUC SEPSIS ≈ AUC NO_SEPSIS: el modelo captura señal universal.")
    print("  - Si AUC SEPSIS > AUC NO_SEPSIS: la fisiología séptica es más predecible.")
    print("  - Si AUC NO_SEPSIS > AUC SEPSIS: deterioro no séptico tiene patrón más claro.")
    print("  - Desviaciones altas (±0.04-0.05) en SEPSIS son esperables")
    print("    por menor tamaño muestral (~993 vs ~2288 estancias).")


if __name__ == "__main__":
    main()