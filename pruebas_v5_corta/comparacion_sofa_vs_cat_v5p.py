"""
Comparación SOFA solo vs modelo CatBoost final — v4p (3-12h)

Compara:
  A) SOFA como predictor único (score clínico estándar)
  B) Modelo CatBoost con el set final de 10 variables

Salidas:
  - tablas/comparacion_sofa_vs_cat_v4p.csv
"""

import warnings
warnings.filterwarnings('ignore')

import os, time
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

# ── CONFIGURACIÓN ─────────────────────────────────────────────────────────────
RUTA_CSV = r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4p.csv'
ETIQUETA = 'etiqueta_norad_3_12'

CARPETA_BASE   = os.path.dirname(__file__) if '__file__' in dir() else '.'
CARPETA_TABLAS = os.path.join(CARPETA_BASE, 'tablas')
os.makedirs(CARPETA_TABLAS, exist_ok=True)

VARIABLES_CAT = [
    'map_min', 'hr_media', 'pf_min', 'spo2_min', 'rr_max',
    'diuresis_ml_kg_3h', 'lactato_max', 'ph_min',
    'temp_min', 'sofa_max',
]

pipeline_cat = Pipeline([
    ('modelo', CatBoostClassifier(
        loss_function='Logloss', eval_metric='AUC',
        random_seed=42, verbose=0, thread_count=-1
    ))
])
espacio_cat = {
    'modelo__iterations':          [500, 1000],
    'modelo__depth':               [4, 6],
    'modelo__learning_rate':       [0.01, 0.05],
    'modelo__l2_leaf_reg':         [1, 5],
    'modelo__bagging_temperature': [0],
}


def main():
    print("=" * 65)
    print("COMPARACIÓN SOFA vs CatBoost — v4p (3-12h)")
    print("=" * 65)

    df = pd.read_csv(RUTA_CSV)
    df = df.dropna(subset=['pf_max'])

    y   = df[ETIQUETA].copy()
    pid = df['subject_id'].copy()

    print(f"\nDataset: N={len(df)} | "
          f"Positivos={y.sum()} ({100*y.mean():.2f}%)\n")

    cv_ext = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    cv_int = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)

    aucs_sofa = []
    aucs_cat  = []

    for nf, (idx_tr, idx_te) in enumerate(
            cv_ext.split(df, y, groups=pid), start=1):

        y_te = y.iloc[idx_te]

        # ── A. SOFA SOLO ──────────────────────────────────────────────────
        sofa_te = df['sofa_max'].iloc[idx_te].fillna(df['sofa_max'].median())
        auc_sofa = roc_auc_score(y_te, sofa_te)
        aucs_sofa.append(auc_sofa)

        # ── B. CatBoost ───────────────────────────────────────────────────
        X_full = df[VARIABLES_CAT].copy()
        x_tr = X_full.iloc[idx_tr]; x_te = X_full.iloc[idx_te]
        y_tr = y.iloc[idx_tr]
        pid_tr = pid.iloc[idx_tr]

        gs = GridSearchCV(pipeline_cat, espacio_cat, cv=cv_int,
                          scoring='roc_auc', n_jobs=1, refit=True)
        gs.fit(x_tr, y_tr, groups=pid_tr)
        proba = gs.predict_proba(x_te)[:, 1]
        auc_cat = roc_auc_score(y_te, proba)
        aucs_cat.append(auc_cat)

        print(f"  Fold {nf}: SOFA={auc_sofa:.4f}  |  CAT={auc_cat:.4f}  "
              f"| Δ={auc_cat-auc_sofa:+.4f}")

    auc_s_m, auc_s_s = np.mean(aucs_sofa), np.std(aucs_sofa)
    auc_c_m, auc_c_s = np.mean(aucs_cat),  np.std(aucs_cat)
    mejora = auc_c_m - auc_s_m

    print(f"\n{'='*65}")
    print(f"RESULTADOS FINALES — v4p (3-12h)")
    print(f"{'='*65}")
    print(f"  SOFA solo (1 var)       : AUC = {auc_s_m:.4f} ± {auc_s_s:.4f}")
    print(f"  CatBoost (10 vars)      : AUC = {auc_c_m:.4f} ± {auc_c_s:.4f}")
    print(f"  Mejora absoluta         : {mejora:+.4f} (+{mejora*100:.2f} pp)")
    print(f"  Mejora relativa         : {mejora/auc_s_m*100:+.1f}%")

    print(f"\n  AUC por fold:")
    print(f"    Fold   SOFA     CAT      Δ")
    for i, (s, c) in enumerate(zip(aucs_sofa, aucs_cat), start=1):
        print(f"    {i}      {s:.4f}   {c:.4f}   {c-s:+.4f}")

    df_res = pd.DataFrame({
        'fold':      list(range(1,6)) + ['media', 'std'],
        'auc_sofa':  aucs_sofa + [auc_s_m, auc_s_s],
        'auc_cat':   aucs_cat  + [auc_c_m, auc_c_s],
        'delta':     [c-s for c,s in zip(aucs_cat,aucs_sofa)] + [mejora, np.nan],
    })
    ruta = os.path.join(CARPETA_TABLAS, 'comparacion_sofa_vs_cat_v4p.csv')
    df_res.to_csv(ruta, index=False)
    print(f"\n  Guardado: {ruta}")


if __name__ == "__main__":
    main()
