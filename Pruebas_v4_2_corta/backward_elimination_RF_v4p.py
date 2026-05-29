import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

#CONFIGURACIÓN
RUTA_CSV     = r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4p.csv'
ETIQUETA     = 'etiqueta_norad_3_12'
UMBRAL_CAIDA = 0.005

CARPETA_BASE   = os.path.dirname(__file__) if '__file__' in dir() else '.'
CARPETA_TABLAS = os.path.join(CARPETA_BASE, 'tablas')
os.makedirs(CARPETA_TABLAS, exist_ok=True)

ORDEN_ELIMINACION = [
    'peso_kg',
    'glucemia_min',
    'hemoglobina_min',
    'creatinina_max',
    'ventilacion_invasiva_3h',
    'leucocitos_min',
    'plaquetas_min',
    'gender',
    'contador_estancia_uci',
    'bicarbonato_min',
    'tp_max',
    'anchor_age',
    'fio2_max',
    'hr_media',
    'lactato_max',
    'spo2_min',
    'ph_min',
    'rr_max',
    'diuresis_ml_kg_3h',
]

VARIABLES_INICIO = [
    'anchor_age', 'gender', 'peso_kg', 'contador_estancia_uci',
    'map_min', 'hr_media',
    'pf_min', 'spo2_min', 'fio2_max', 'rr_max',
    'ventilacion_invasiva_3h', 'gcs_min',
    'creatinina_max', 'diuresis_ml_kg_3h',
    'lactato_max', 'ph_min', 'bicarbonato_min',
    'bilirrubina_media',
    'tp_max', 'plaquetas_min',
    'leucocitos_min', 'hemoglobina_min',
    'glucemia_min', 'temp_min', 'sofa_max',
]
# 25 variables (sin gpt_max)

#PIPELINE Y GRID REDUCIDO (idéntico a ventana principal)
pipeline = Pipeline([
    ('modelo', RandomForestClassifier(random_state=42, n_jobs=-1))
])
espacio = {
    'modelo__n_estimators':     [300, 500, 1000],
    'modelo__max_depth':        [10, 20, None],
    'modelo__min_samples_leaf': [1, 5],
    'modelo__max_features':     ['sqrt'],
    'modelo__class_weight':     ['balanced', 'balanced_subsample'],
}
# 36 combinaciones

#FUNCIONES
def cargar_datos():
    df = pd.read_csv(RUTA_CSV)
    return df


def preparar(df, variables):
    predictores = df[variables].copy()
    etiqueta    = df[ETIQUETA].copy()
    paciente_id = df['subject_id'].copy()

    if 'gender' in predictores.columns:
        predictores['gender'] = predictores['gender'].map({'M': 1, 'F': 0}).astype(int)

    return predictores, etiqueta, paciente_id


def entrenar_rf(predictores, etiqueta, paciente_id):
    cv_ext = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    cv_int = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    aucs = []
    for idx_tr, idx_te in cv_ext.split(predictores, etiqueta, groups=paciente_id):
        x_tr, x_te = predictores.iloc[idx_tr], predictores.iloc[idx_te]
        y_tr, y_te = etiqueta.iloc[idx_tr],    etiqueta.iloc[idx_te]
        pid_tr = paciente_id.iloc[idx_tr]
        gs = GridSearchCV(pipeline, espacio, cv=cv_int,
                          scoring='roc_auc', n_jobs=-1, refit=True)
        gs.fit(x_tr, y_tr, groups=pid_tr)
        aucs.append(roc_auc_score(y_te, gs.predict_proba(x_te)[:, 1]))
    return np.mean(aucs), np.std(aucs)


# MAIN
def main():
    print("=" * 65)
    print("BACKWARD ELIMINATION — RF v4p CORTA (sin gpt_max)")
    print("=" * 65)
    print(f"Variables de partida : {len(VARIABLES_INICIO)}")
    print(f"Variables a probar   : {len(ORDEN_ELIMINACION)}")
    print(f"Umbral de parada     : caída AUC > {UMBRAL_CAIDA}\n")

    df = cargar_datos()
    variables_actuales = VARIABLES_INICIO.copy()
    historial = []
    t_global = time.time()

    print(f"[Referencia] {len(variables_actuales)} variables")
    t0 = time.time()
    pred, et, pid = preparar(df, variables_actuales)
    auc_ref, std_ref = entrenar_rf(pred, et, pid)
    print(f"  AUC = {auc_ref:.4f} ± {std_ref:.4f}  ({(time.time()-t0)/60:.1f} min)\n")
    historial.append({'paso': 0, 'variable_eliminada': 'ninguna (referencia)',
                      'n_variables': len(variables_actuales),
                      'auc': auc_ref, 'std': std_ref,
                      'delta': 0.0, 'decision': 'referencia'})
    auc_anterior = auc_ref

    for paso, var in enumerate(ORDEN_ELIMINACION, start=1):
        if var not in variables_actuales:
            print(f"[Paso {paso:2d}] '{var}' ya no está en el set, saltando.\n")
            continue

        vars_prueba = [v for v in variables_actuales if v != var]
        print(f"[Paso {paso:2d}] Eliminando '{var}' → {len(vars_prueba)} variables")

        t0 = time.time()
        pred, et, pid = preparar(df, vars_prueba)
        auc_nuevo, std_nuevo = entrenar_rf(pred, et, pid)
        delta = auc_nuevo - auc_anterior
        t_min = (time.time() - t0) / 60

        print(f"  AUC = {auc_nuevo:.4f} ± {std_nuevo:.4f}  "
              f"Δ = {delta:+.4f}  ({t_min:.1f} min)")

        if delta >= -UMBRAL_CAIDA:
            decision = 'ELIMINAR'
            variables_actuales = vars_prueba
            auc_anterior = auc_nuevo
            print(f"  → ELIMINAR\n")
        else:
            decision = 'CONSERVAR'
            print(f"  → CONSERVAR: cae {abs(delta):.4f} > {UMBRAL_CAIDA}\n")

        historial.append({'paso': paso, 'variable_eliminada': var,
                          'n_variables': len(vars_prueba),
                          'auc': auc_nuevo, 'std': std_nuevo,
                          'delta': delta, 'decision': decision})

    t_total = (time.time() - t_global) / 60
    print("=" * 65)
    print(f"RESUMEN FINAL (tiempo total: {t_total:.1f} min)")
    print("=" * 65)
    df_hist = pd.DataFrame(historial)
    print(df_hist[['paso','variable_eliminada','n_variables',
                   'auc','std','delta','decision']].to_string(index=False))

    eliminadas  = [h['variable_eliminada'] for h in historial if h['decision']=='ELIMINAR']
    conservadas = [h['variable_eliminada'] for h in historial if h['decision']=='CONSERVAR']

    print(f"\nEliminadas ({len(eliminadas)}): {eliminadas}")
    print(f"Conservadas no-sig ({len(conservadas)}): {conservadas}")
    print(f"\nSet final ({len(variables_actuales)} vars):")
    for v in variables_actuales: print(f"  - {v}")
    print(f"\nAUC referencia : {auc_ref:.4f} ± {std_ref:.4f}")
    print(f"AUC final      : {auc_anterior:.4f}")
    print(f"Δ total        : {auc_anterior - auc_ref:+.4f}")

    ruta = os.path.join(CARPETA_TABLAS, 'backward_elimination_RF_v4p.csv')
    df_hist.to_csv(ruta, index=False)
    print(f"\nGuardado en: {ruta}")


if __name__ == "__main__":
    main()
