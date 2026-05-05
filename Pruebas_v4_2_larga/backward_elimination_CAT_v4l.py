"""
Backward elimination — CatBoost v4l LARGA (ventana 0-12h / predicción 12-48h).

Parte de las variables originales SIN gpt_max y elimina una a una las no
significativas (IC95% incluye 0 en permutation importance), empezando por
la menos importante.

Orden de eliminación estimado desde la figura de permutation importance
(de menos a más importante entre las no significativas):
  1.  peso_kg               (más negativa)
  2.  hemoglobina_min
  3.  creatinina_max
  4.  plaquetas_min
  5.  contador_estancia_uci
  6.  leucocitos_min
  7.  gender
  8.  ph_min
  9.  anchor_age
  10. tp_max
  11. ventilacion_invasiva_12h
  12. gcs_min
  13. lactato_max
  14. hr_media

Grid CatBoost reducido: 48 combinaciones vs 240 originales (~5x más rápido).
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
import os

# ── CONFIGURACIÓN ──────────────────────────────────────────────────────────────
RUTA_CSV     = r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4l.csv'
ETIQUETA     = 'etiqueta_norad_12_48'
UMBRAL_CAIDA = 0.005

CARPETA_BASE   = os.path.dirname(__file__) if '__file__' in dir() else '.'
CARPETA_TABLAS = os.path.join(CARPETA_BASE, 'tablas')
os.makedirs(CARPETA_TABLAS, exist_ok=True)

ORDEN_ELIMINACION = [
    'peso_kg',
    'hemoglobina_min',
    'creatinina_max',
    'plaquetas_min',
    'contador_estancia_uci',
    'leucocitos_min',
    'gender',
    'ph_min',
    'anchor_age',
    'tp_max',
    'ventilacion_invasiva_12h',
    'gcs_min',
    'lactato_max',
    'hr_media',
]

VARIABLES_INICIO = [
    'anchor_age', 'gender', 'peso_kg', 'contador_estancia_uci',
    'map_min', 'hr_media',
    'pf_min', 'spo2_min', 'rr_max',
    'ventilacion_invasiva_12h', 'gcs_min',
    'creatinina_max', 'diuresis_ml_kg_12h',
    'lactato_max', 'ph_min', 'bicarbonato_min',
    'bilirrubina_media',
    'tp_max', 'plaquetas_min',
    'leucocitos_min', 'hemoglobina_min',
    'glucemia_min', 'temp_min', 'sofa_max',
]
# 24 variables (sin gpt_max, sin fio2_max que ya se quitó en v4l)

# ── PIPELINE Y GRID REDUCIDO ───────────────────────────────────────────────────
pipeline = Pipeline([
    ('modelo', CatBoostClassifier(
        loss_function='Logloss', eval_metric='AUC',
        random_seed=42, verbose=0, thread_count=-1
    ))
])
espacio = {
    'modelo__iterations':          [500, 1000],
    'modelo__depth':               [4, 6],
    'modelo__learning_rate':       [0.01, 0.05, 0.1],
    'modelo__l2_leaf_reg':         [1, 5],
    'modelo__bagging_temperature': [0, 1],
}
# 2*2*3*2*2 = 48 combinaciones


# ── FUNCIONES ─────────────────────────────────────────────────────────────────
def cargar_datos():
    df = pd.read_csv(RUTA_CSV)
    df = df.dropna(subset=['pf_max'])
    return df


def preparar(df, variables):
    predictores = df[variables].copy()
    etiqueta    = df[ETIQUETA].copy()
    paciente_id = df['subject_id'].copy()
    predictores['gender'] = (predictores['gender'] == 'M').astype(int)
    return predictores, etiqueta, paciente_id


def entrenar_cat(predictores, etiqueta, paciente_id):
    cv_ext = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    cv_int = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    aucs = []
    for idx_tr, idx_te in cv_ext.split(predictores, etiqueta, groups=paciente_id):
        x_tr, x_te = predictores.iloc[idx_tr], predictores.iloc[idx_te]
        y_tr, y_te = etiqueta.iloc[idx_tr],    etiqueta.iloc[idx_te]
        pid_tr = paciente_id.iloc[idx_tr]
        gs = GridSearchCV(pipeline, espacio, cv=cv_int,
                          scoring='roc_auc', n_jobs=1, refit=True)
        gs.fit(x_tr, y_tr, groups=pid_tr)
        aucs.append(roc_auc_score(y_te, gs.predict_proba(x_te)[:, 1]))
    return np.mean(aucs), np.std(aucs)


# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("BACKWARD ELIMINATION — CatBoost v4l LARGA (sin gpt_max)")
    print("=" * 65)
    print(f"Variables de partida : {len(VARIABLES_INICIO)}")
    print(f"Variables a probar   : {len(ORDEN_ELIMINACION)}")
    print(f"Umbral de parada     : caída AUC > {UMBRAL_CAIDA}\n")

    df = cargar_datos()
    variables_actuales = VARIABLES_INICIO.copy()
    historial = []
    t_global = time.time()

    # Modelo de referencia
    print(f"[Referencia] {len(variables_actuales)} variables")
    t0 = time.time()
    pred, et, pid = preparar(df, variables_actuales)
    auc_ref, std_ref = entrenar_cat(pred, et, pid)
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
        auc_nuevo, std_nuevo = entrenar_cat(pred, et, pid)
        delta = auc_nuevo - auc_anterior
        t_min = (time.time() - t0) / 60

        print(f"  AUC = {auc_nuevo:.4f} ± {std_nuevo:.4f}  "
              f"Δ = {delta:+.4f}  ({t_min:.1f} min)")

        if delta >= -UMBRAL_CAIDA:
            decision = 'ELIMINAR'
            variables_actuales = vars_prueba
            auc_anterior = auc_nuevo
            print(f"  → ELIMINAR: AUC se mantiene.\n")
        else:
            decision = 'CONSERVAR'
            print(f"  → CONSERVAR: AUC baja {abs(delta):.4f} > umbral.\n")

        historial.append({'paso': paso, 'variable_eliminada': var,
                          'n_variables': len(vars_prueba),
                          'auc': auc_nuevo, 'std': std_nuevo,
                          'delta': delta, 'decision': decision})

    # Resumen
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
    print(f"\nSet final ({len(variables_actuales)} vars): {variables_actuales}")
    print(f"\nAUC referencia : {auc_ref:.4f} ± {std_ref:.4f}")
    print(f"AUC final      : {auc_anterior:.4f}")
    print(f"Δ total        : {auc_anterior - auc_ref:+.4f}")

    ruta = os.path.join(CARPETA_TABLAS, 'backward_elimination_CAT_v4l.csv')
    df_hist.to_csv(ruta, index=False)
    print(f"\nGuardado en: {ruta}")


if __name__ == "__main__":
    main()
