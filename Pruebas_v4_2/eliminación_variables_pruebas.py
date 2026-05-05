"""
Backward elimination — RF v4 sin gpt_max.

Elimina variables no significativas (IC95% incluye 0 en permutation importance)
una a una, empezando por la menos importante, y compara el AUC en cada paso.

Orden de eliminación (de menos a más importante según caída AUC):
  1.  plaquetas_min        (-0.27 pp)
  2.  creatinina_max       (-0.12 pp)
  3.  bilirrubina_media    (-0.05 pp)
  4.  contador_estancia_uci(-0.01 pp)
  5.  fio2_max             ( 0.09 pp)
  6.  tp_max               ( 0.10 pp)
  7.  peso_kg              ( 0.11 pp)
  8.  hemoglobina_min      ( 0.16 pp)
  9.  gcs_min              ( 0.19 pp)
  10. ventilacion_invasiva_6h(0.19 pp)
  11. anchor_age           ( 0.28 pp)
  12. bicarbonato_min      ( 0.35 pp)
  13. sofa_max             ( 0.37 pp)
  14. lactato_max          ( 0.46 pp)
  15. temp_min             ( 0.55 pp)

Criterio de parada: si el AUC cae > 0.005 respecto al modelo anterior,
se para y la variable eliminada en ese paso se considera el límite.

Grid reducido (36 combinaciones, ~3 min por modelo).
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier


# ── CONFIGURACIÓN ──────────────────────────────────────────────────────────────
RUTA_CSV = r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4.csv'

# Orden de eliminación: de menos a más importante
ORDEN_ELIMINACION = [
    'plaquetas_min',
    'creatinina_max',
    'bilirrubina_media',
    'contador_estancia_uci',
    'fio2_max',
    'tp_max',
    'peso_kg',
    'hemoglobina_min',
    'gcs_min',
    'ventilacion_invasiva_6h',
    'anchor_age',
    'bicarbonato_min',
    'sofa_max',
    'lactato_max',
    'temp_min',
]

# Variables de partida (26 originales - gpt_max ya eliminada)
VARIABLES_INICIO = [
    'anchor_age', 'gender', 'peso_kg', 'contador_estancia_uci',
    'map_min', 'hr_media',
    'pf_min', 'spo2_min', 'fio2_max', 'rr_max',
    'ventilacion_invasiva_6h', 'gcs_min',
    'creatinina_max', 'diuresis_ml_kg_6h',
    'lactato_max', 'ph_min', 'bicarbonato_min',
    'bilirrubina_media',
    'tp_max', 'plaquetas_min',
    'leucocitos_min', 'hemoglobina_min',
    'glucemia_min', 'temp_min', 'sofa_max',
]
# 25 variables (sin gpt_max)

UMBRAL_CAIDA = 0.005  # si AUC baja más de esto, parar


# ── FUNCIONES ─────────────────────────────────────────────────────────────────
def cargar_datos():
    df = pd.read_csv(RUTA_CSV)
    df = df.dropna(subset=['pf_max'])
    return df


def preparar(df, variables):
    predictores = df[variables].copy()
    etiqueta = df['etiqueta_norad_6_24'].copy()
    paciente_id = df['subject_id'].copy()
    predictores['gender'] = (predictores['gender'] == 'M').astype(int)
    return predictores, etiqueta, paciente_id


def entrenar_rf(predictores, etiqueta, paciente_id):
    cv_ext = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    cv_int = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)

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


# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("BACKWARD ELIMINATION — RF v4 (sin gpt_max)")
    print("=" * 65)
    print(f"Variables de partida : {len(VARIABLES_INICIO)}")
    print(f"Variables a probar   : {len(ORDEN_ELIMINACION)}")
    print(f"Umbral de parada     : caída AUC > {UMBRAL_CAIDA}\n")

    df = cargar_datos()
    variables_actuales = VARIABLES_INICIO.copy()
    historial = []
    t_global = time.time()

    # Modelo de referencia (25 variables sin gpt_max)
    print(f"[Referencia] {len(variables_actuales)} variables (sin gpt_max)")
    t0 = time.time()
    pred, et, pid = preparar(df, variables_actuales)
    auc_ref, std_ref = entrenar_rf(pred, et, pid)
    print(f"  AUC = {auc_ref:.4f} ± {std_ref:.4f}  ({(time.time()-t0)/60:.1f} min)\n")

    historial.append({
        'paso': 0,
        'variable_eliminada': 'ninguna (referencia)',
        'n_variables': len(variables_actuales),
        'auc': auc_ref,
        'std': std_ref,
        'delta': 0.0,
        'decision': 'referencia'
    })

    auc_anterior = auc_ref

    for paso, var_eliminar in enumerate(ORDEN_ELIMINACION, start=1):

        variables_prueba = [v for v in variables_actuales if v != var_eliminar]

        print(f"[Paso {paso:2d}] Eliminando '{var_eliminar}' "
              f"→ {len(variables_prueba)} variables")

        t0 = time.time()
        pred, et, pid = preparar(df, variables_prueba)
        auc_nuevo, std_nuevo = entrenar_rf(pred, et, pid)
        delta = auc_nuevo - auc_anterior
        t_min = (time.time() - t0) / 60

        print(f"  AUC = {auc_nuevo:.4f} ± {std_nuevo:.4f}  "
              f"Δ = {delta:+.4f}  ({t_min:.1f} min)")

        if delta >= -UMBRAL_CAIDA:
            decision = 'ELIMINAR'
            variables_actuales = variables_prueba
            auc_anterior = auc_nuevo
            print(f"  → {decision}: AUC se mantiene. "
                  f"Se elimina '{var_eliminar}'.\n")
        else:
            decision = 'CONSERVAR'
            print(f"  → {decision}: AUC baja {abs(delta):.4f} > umbral "
                  f"{UMBRAL_CAIDA}. Se conserva '{var_eliminar}'.\n")

        historial.append({
            'paso': paso,
            'variable_eliminada': var_eliminar,
            'n_variables': len(variables_prueba),
            'auc': auc_nuevo,
            'std': std_nuevo,
            'delta': delta,
            'decision': decision
        })

    # ── RESUMEN FINAL ──────────────────────────────────────────────────────────
    t_total = (time.time() - t_global) / 60
    print("=" * 65)
    print(f"RESUMEN FINAL (tiempo total: {t_total:.1f} min)")
    print("=" * 65)

    df_hist = pd.DataFrame(historial)
    print(df_hist[['paso','variable_eliminada','n_variables',
                   'auc','std','delta','decision']].to_string(index=False))

    eliminadas = [h['variable_eliminada'] for h in historial
                  if h['decision'] == 'ELIMINAR']
    conservadas_no_sig = [h['variable_eliminada'] for h in historial
                          if h['decision'] == 'CONSERVAR']

    print(f"\nVariables eliminadas ({len(eliminadas)}):")
    for v in eliminadas:
        print(f"  - {v}")

    print(f"\nVariables no significativas que se conservan ({len(conservadas_no_sig)}):")
    for v in conservadas_no_sig:
        print(f"  - {v}")

    print(f"\nSet final: {len(variables_actuales)} variables")
    print(f"Variables finales: {variables_actuales}")

    print(f"\nAUC referencia (25 vars sin gpt_max): {auc_ref:.4f} ± {std_ref:.4f}")
    print(f"AUC modelo final ({len(variables_actuales)} vars)   : "
          f"{auc_anterior:.4f} ± "
          f"{[h['std'] for h in historial if h['auc']==auc_anterior][0]:.4f}")
    print(f"Diferencia total                      : "
          f"{auc_anterior - auc_ref:+.4f}")

    df_hist.to_csv('tablas/backward_elimination_RF_v4.csv', index=False)
    print(f"\nHistorial guardado en: tablas/backward_elimination_RF_v4.csv")


if __name__ == "__main__":
    main()