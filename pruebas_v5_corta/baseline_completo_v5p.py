"""
BASELINE COMPLETO — v4p (3-12h)
6 algoritmos × 3 subgrupos (Global, Sépticos, No Sépticos)
Set final v4p: 10 variables.

Grids recortados respecto al baseline original.
"""

import warnings
warnings.filterwarnings('ignore')

import os, time
import pandas as pd
import numpy as np

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

# ── CONFIGURACIÓN ─────────────────────────────────────────────────────────────
RUTA_CSV = r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4p.csv'
ETIQUETA = 'etiqueta_norad_3_12'

CARPETA_BASE   = os.path.dirname(__file__) if '__file__' in dir() else '.'
CARPETA_TABLAS = os.path.join(CARPETA_BASE, 'tablas')
os.makedirs(CARPETA_TABLAS, exist_ok=True)

VARIABLES = [
    'map_min', 'hr_media', 'pf_min', 'spo2_min', 'rr_max',
    'diuresis_ml_kg_3h', 'lactato_max', 'ph_min',
    'temp_min', 'sofa_max',
]


# ── PIPELINES Y GRIDS RECORTADOS ──────────────────────────────────────────────
def construir_pipelines_y_grids():
    pipelines = {}
    espacios  = {}

    pipelines['LR'] = Pipeline([
        ('scaler', RobustScaler()),
        ('modelo', LogisticRegression(max_iter=5000, class_weight='balanced',
                                      solver='liblinear', random_state=42))
    ])
    espacios['LR'] = {'modelo__C': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2,
                                     0.1, 0.5, 1, 5, 10, 50]}

    pipelines['RF'] = Pipeline([
        ('modelo', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])
    espacios['RF'] = {
        'modelo__n_estimators':     [300, 500, 1000],
        'modelo__max_depth':        [10, 20, None],
        'modelo__min_samples_leaf': [1, 5],
        'modelo__max_features':     ['sqrt', 0.3],
        'modelo__class_weight':     ['balanced', 'balanced_subsample'],
    }

    pipelines['XGB'] = Pipeline([
        ('modelo', XGBClassifier(objective='binary:logistic', eval_metric='auc',
                                 random_state=42, n_jobs=1, tree_method='hist'))
    ])
    espacios['XGB'] = {
        'modelo__n_estimators':     [300, 600, 900],
        'modelo__max_depth':        [3, 5, 7],
        'modelo__learning_rate':    [0.01, 0.05],
        'modelo__subsample':        [0.6, 1.0],
        'modelo__colsample_bytree': [0.6, 1.0],
        'modelo__reg_lambda':       [0.1, 1],
        'modelo__scale_pos_weight': [1],
    }

    pipelines['LGBM'] = Pipeline([
        ('modelo', LGBMClassifier(random_state=42, verbosity=-1, n_jobs=1,
                                  objective='binary'))
    ])
    espacios['LGBM'] = {
        'modelo__n_estimators':      [300, 600, 1000],
        'modelo__num_leaves':        [15, 31],
        'modelo__learning_rate':     [0.01, 0.05],
        'modelo__min_child_samples': [10, 30],
        'modelo__reg_lambda':        [0.1, 1],
        'modelo__subsample':         [0.6],
        'modelo__class_weight':      ['balanced', None],
    }

    pipelines['CAT'] = Pipeline([
        ('modelo', CatBoostClassifier(loss_function='Logloss', eval_metric='AUC',
                                      random_seed=42, verbose=0, thread_count=-1))
    ])
    espacios['CAT'] = {
        'modelo__iterations':          [500, 1000],
        'modelo__depth':               [4, 6],
        'modelo__learning_rate':       [0.01, 0.05],
        'modelo__l2_leaf_reg':         [1, 5],
        'modelo__bagging_temperature': [0],
    }

    pipelines['NB'] = Pipeline([
        ('scaler', RobustScaler()),
        ('modelo', GaussianNB())
    ])
    espacios['NB'] = {'modelo__var_smoothing': np.logspace(-12, -2, 30)}

    return pipelines, espacios


def cv_anidado(nombre, pipeline, espacio, X, y, pid, n_jobs=-1):
    cv_ext = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    cv_int = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    aucs = []
    t0 = time.time()
    for nf, (idx_tr, idx_te) in enumerate(cv_ext.split(X, y, groups=pid), start=1):
        x_tr, x_te = X.iloc[idx_tr], X.iloc[idx_te]
        y_tr, y_te = y.iloc[idx_tr], y.iloc[idx_te]
        pid_tr     = pid.iloc[idx_tr]
        gs = GridSearchCV(pipeline, espacio, cv=cv_int,
                          scoring='roc_auc', n_jobs=n_jobs, refit=True)
        gs.fit(x_tr, y_tr, groups=pid_tr)
        proba = gs.predict_proba(x_te)[:, 1]
        auc = roc_auc_score(y_te, proba)
        aucs.append(auc)
        print(f"    Fold {nf}: AUC={auc:.4f}")
    t_min = (time.time() - t0) / 60
    auc_m, auc_s = np.mean(aucs), np.std(aucs)
    print(f"    {nombre} → AUC = {auc_m:.4f} ± {auc_s:.4f}  ({t_min:.1f} min)")
    return auc_m, auc_s, t_min


def entrenar_subgrupo(nombre_subgrupo, X, y, pid):
    print(f"\n{'─'*72}")
    print(f"SUBGRUPO: {nombre_subgrupo}")
    print(f"{'─'*72}")
    print(f"  N={len(X)} | Positivos={y.sum()} ({100*y.mean():.2f}%) | "
          f"Pacientes únicos={pid.nunique()}")

    if y.sum() < 30:
        print(f"  ⚠ Muy pocos positivos ({y.sum()}). Saltando este subgrupo.")
        return None

    pipelines, espacios = construir_pipelines_y_grids()
    resultados = {}

    orden = ['LR', 'NB', 'RF', 'LGBM', 'CAT', 'XGB']
    for nombre in orden:
        print(f"\n  [{nombre}]  combinaciones grid: "
              f"{np.prod([len(v) for v in espacios[nombre].values()])}")
        n_jobs = 1 if nombre == 'CAT' else -1
        try:
            auc_m, auc_s, t_min = cv_anidado(
                nombre, pipelines[nombre], espacios[nombre],
                X, y, pid, n_jobs=n_jobs
            )
            resultados[nombre] = {'auc': auc_m, 'std': auc_s, 'tiempo_min': t_min,
                                  'subgrupo': nombre_subgrupo}
        except Exception as e:
            print(f"    ⚠ Error en {nombre}: {e}")
            resultados[nombre] = {'auc': np.nan, 'std': np.nan, 'tiempo_min': np.nan,
                                  'subgrupo': nombre_subgrupo}

    print(f"\n  RANKING — {nombre_subgrupo}:")
    rk = sorted(resultados.items(),
                key=lambda x: (-x[1]['auc'] if not np.isnan(x[1]['auc']) else 1))
    for nombre, r in rk:
        print(f"    {nombre:6s}  AUC = {r['auc']:.4f} ± {r['std']:.4f}")

    return resultados


def main():
    print("=" * 72)
    print("BASELINE COMPLETO — v4p (3-12h)  |  6 algoritmos × 3 subgrupos")
    print("=" * 72)
    print(f"Set de variables ({len(VARIABLES)}):")
    for v in VARIABLES:
        print(f"  - {v}")

    df = pd.read_csv(RUTA_CSV)
    df = df.dropna(subset=['pf_max'])
    df = df.dropna(subset=VARIABLES + [ETIQUETA, 'tiene_sepsis', 'subject_id'])
    df = df.reset_index(drop=True)

    print(f"\nDataset total: N={len(df)} | "
          f"Positivos={df[ETIQUETA].sum()} ({100*df[ETIQUETA].mean():.2f}%)")
    print(f"Sépticos    : {(df['tiene_sepsis']==1).sum()}")
    print(f"No sépticos : {(df['tiene_sepsis']==0).sum()}")

    t_global = time.time()
    todos = []

    Xg = df[VARIABLES].copy()
    yg = df[ETIQUETA].copy()
    pidg = df['subject_id'].copy()
    res_g = entrenar_subgrupo('GLOBAL', Xg, yg, pidg)
    if res_g:
        for alg, r in res_g.items():
            todos.append({'subgrupo':'Global', 'algoritmo':alg, **r})

    df_sep = df[df['tiene_sepsis'] == 1].reset_index(drop=True)
    Xs = df_sep[VARIABLES].copy()
    ys = df_sep[ETIQUETA].copy()
    pids = df_sep['subject_id'].copy()
    res_s = entrenar_subgrupo('SÉPTICOS', Xs, ys, pids)
    if res_s:
        for alg, r in res_s.items():
            todos.append({'subgrupo':'Sépticos', 'algoritmo':alg, **r})

    df_nsep = df[df['tiene_sepsis'] == 0].reset_index(drop=True)
    Xn = df_nsep[VARIABLES].copy()
    yn = df_nsep[ETIQUETA].copy()
    pidn = df_nsep['subject_id'].copy()
    res_n = entrenar_subgrupo('NO SÉPTICOS', Xn, yn, pidn)
    if res_n:
        for alg, r in res_n.items():
            todos.append({'subgrupo':'No sépticos', 'algoritmo':alg, **r})

    t_total_h = (time.time() - t_global) / 3600
    print(f"\n{'='*72}")
    print(f"RESUMEN GLOBAL — v4p  (tiempo total: {t_total_h:.2f} h)")
    print(f"{'='*72}")

    df_res = pd.DataFrame(todos)
    if len(df_res):
        df_res = df_res.sort_values(['subgrupo','auc'], ascending=[True,False])
        print(df_res[['subgrupo','algoritmo','auc','std','tiempo_min']]
              .round(4).to_string(index=False))

        print(f"\n  GANADORES:")
        for sub in df_res['subgrupo'].unique():
            sub_df = df_res[df_res['subgrupo']==sub]
            ganador = sub_df.iloc[0]
            print(f"    {sub:14s}  →  {ganador['algoritmo']:6s} "
                  f"AUC={ganador['auc']:.4f} ± {ganador['std']:.4f}")

        ruta = os.path.join(CARPETA_TABLAS, 'baseline_completo_v4p.csv')
        df_res.to_csv(ruta, index=False)
        print(f"\n  Guardado: {ruta}")


if __name__ == "__main__":
    main()
