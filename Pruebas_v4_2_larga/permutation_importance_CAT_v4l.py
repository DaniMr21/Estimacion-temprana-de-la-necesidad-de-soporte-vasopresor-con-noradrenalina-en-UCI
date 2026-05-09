"""
Permutation Importance — CatBoost, ventana LARGA v4l (25 variables).
Observación: 0-12h | Predicción: 12-48h
Etiqueta: etiqueta_norad_12_48

Salidas:
  - tablas/permutation_importance_CAT_v4l.csv
  - figuras/permutation_importance_CAT_v4l.png
"""

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
from catboost import CatBoostClassifier

# ── CONFIGURACIÓN ──────────────────────────────────────────────────────────────
RUTA_CSV   = r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4l.csv'
ETIQUETA   = 'etiqueta_norad_12_48'
N_PERM     = 50
RANDOM_STATE = 42

CARPETA_BASE   = os.path.dirname(__file__) if '__file__' in dir() else '.'
CARPETA_TABLAS = os.path.join(CARPETA_BASE, 'tablas')
CARPETA_FIGURAS= os.path.join(CARPETA_BASE, 'figuras')
os.makedirs(CARPETA_TABLAS,  exist_ok=True)
os.makedirs(CARPETA_FIGURAS, exist_ok=True)

variables_predictoras = [
    'anchor_age', 'gender', 'peso_kg', 'contador_estancia_uci',
    'map_min', 'hr_media',
    'pf_min', 'spo2_min', 'rr_max',              # sin fio2_max
    'ventilacion_invasiva_12h', 'gcs_min',        # 12h
    'creatinina_max', 'diuresis_ml_kg_12h',       # 12h
    'lactato_max', 'ph_min', 'bicarbonato_min',
    'bilirrubina_media', 'gpt_max',
    'tp_max', 'plaquetas_min',
    'leucocitos_min', 'hemoglobina_min',
    'glucemia_min', 'temp_min', 'sofa_max',
]

# ── CARGA Y PREPARACIÓN ────────────────────────────────────────────────────────
def cargar_datos():
    df = pd.read_csv(RUTA_CSV)
    df = df.dropna(subset=['pf_max'])
    return df

def preparar(df):
    predictores = df[variables_predictoras].copy()
    predictores['gender'] = (predictores['gender'] == 'M').astype(int)
    etiqueta    = df[ETIQUETA].copy()
    paciente_id = df['subject_id'].copy()
    return predictores, etiqueta, paciente_id

# ── PIPELINE Y GRID (idéntico a baseline_v4_2_larga.py) ───────────────────────
pipeline = Pipeline([
    ('modelo', CatBoostClassifier(
        loss_function='Logloss', eval_metric='AUC',
        random_seed=RANDOM_STATE, verbose=0, thread_count=-1
    ))
])
espacio = {
    'modelo__iterations':         [500, 1000],
    'modelo__depth':              [4, 5, 6, 7],
    'modelo__learning_rate':      [0.01, 0.03, 0.05, 0.1],
    'modelo__l2_leaf_reg':        [1, 5, 15],
    'modelo__bagging_temperature':[0, 0.5, 1],
}

# ── CV ANIDADO + PERMUTATION IMPORTANCE ───────────────────────────────────────
def run_cv(predictores, etiqueta, paciente_id):
    cv_ext = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_int = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    nombres = predictores.columns.tolist()
    n_vars  = len(nombres)

    aucs, params_list = [], []
    imp_medias = np.zeros((5, n_vars))
    todas_perm = []
    t0 = time.time()

    for nf, (idx_tr, idx_te) in enumerate(
            cv_ext.split(predictores, etiqueta, groups=paciente_id), start=1):

        x_tr, x_te = predictores.iloc[idx_tr], predictores.iloc[idx_te]
        y_tr, y_te = etiqueta.iloc[idx_tr],    etiqueta.iloc[idx_te]
        pid_tr = paciente_id.iloc[idx_tr]

        gs = GridSearchCV(pipeline, espacio, cv=cv_int,
                          scoring='roc_auc', n_jobs=1, refit=True)
        gs.fit(x_tr, y_tr, groups=pid_tr)

        proba = gs.predict_proba(x_te)[:, 1]
        auc   = roc_auc_score(y_te, proba)
        aucs.append(auc)
        params_list.append(gs.best_params_)

        print(f"  Fold {nf}: AUC={auc:.4f}  params={gs.best_params_}")
        print(f"    Calculando permutation importance ({N_PERM} permutaciones)...")

        res = permutation_importance(
            gs.best_estimator_, x_te, y_te,
            scoring='roc_auc', n_repeats=N_PERM,
            random_state=RANDOM_STATE, n_jobs=-1
        )
        imp_medias[nf-1, :] = res.importances_mean
        todas_perm.append(res.importances)

        top5 = np.argsort(-res.importances_mean)[:5]
        print("    Top-5:")
        for i in top5:
            print(f"      {nombres[i]:<28} {res.importances_mean[i]:.4f}")
        print()

    t_min = (time.time() - t0) / 60
    auc_m, auc_s = np.mean(aucs), np.std(aucs)
    print(f"CatBoost — AUC medio: {auc_m:.4f} ± {auc_s:.4f}  ({t_min:.1f} min)")
    return dict(aucs=aucs, auc_medio=auc_m, auc_desv=auc_s,
                imp_medias=imp_medias, todas_perm=todas_perm, nombres=nombres)

# ── TABLA RESUMEN ──────────────────────────────────────────────────────────────
def tabla_resumen(res):
    nombres, imp_folds, todas = res['nombres'], res['imp_medias'], res['todas_perm']
    filas = []
    for i, nom in enumerate(nombres):
        vals  = imp_folds[:, i]
        media = vals.mean(); desv = vals.std()
        ic_inf = np.percentile(vals, 2.5); ic_sup = np.percentile(vals, 97.5)
        perm_concat = np.concatenate([m[i, :] for m in todas])
        p_emp = np.mean(perm_concat <= 0)
        filas.append(dict(variable=nom,
                          caida_AUC_pp=media*100, caida_AUC_desv=desv,
                          caida_AUC_IC_inf=ic_inf, caida_AUC_IC_sup=ic_sup,
                          p_valor_empirico=p_emp,
                          IC_excluye_cero=(ic_inf > 0),
                          p_emp_menor_005=(p_emp < 0.05)))
    return pd.DataFrame(filas).sort_values('caida_AUC_pp', ascending=False).reset_index(drop=True)

# ── FIGURA ─────────────────────────────────────────────────────────────────────
def graficar(df_res, ruta, auc_m):
    df_p = df_res.sort_values('caida_AUC_pp').copy()
    colores = ['#2ca02c' if r else '#d62728' for r in df_p['IC_excluye_cero']]
    fig, ax = plt.subplots(figsize=(10, 9))
    pos = np.arange(len(df_p))
    ax.barh(pos, df_p['caida_AUC_pp'], xerr=df_p['caida_AUC_desv']*100,
            color=colores, edgecolor='black', linewidth=0.5, alpha=0.85, capsize=3)
    ax.set_yticks(pos); ax.set_yticklabels(df_p['variable'])
    ax.set_xlabel('Caída de AUC (pp) al permutar la variable')
    ax.set_title(f'Permutation Importance — CatBoost v4l LARGA (25 vars)\n'
                 f'AUC base = {auc_m:.4f} | 5 folds | {N_PERM} permutaciones/fold')
    ax.axvline(0, color='black', linewidth=0.8); ax.grid(axis='x', alpha=0.3)
    ax.legend(handles=[
        Patch(facecolor='#2ca02c', edgecolor='black', label='IC95% excluye 0'),
        Patch(facecolor='#d62728', edgecolor='black', label='IC95% incluye 0'),
    ], loc='lower right')
    plt.tight_layout()
    plt.savefig(ruta, dpi=150, bbox_inches='tight'); plt.close()

# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    print("─" * 60)
    print("PERMUTATION IMPORTANCE — CatBoost v4l LARGA")
    print("─" * 60)
    df = cargar_datos()
    predictores, etiqueta, paciente_id = preparar(df)
    print(f"Dataset: {predictores.shape} | "
          f"Positivos: {etiqueta.sum()} ({100*etiqueta.mean():.2f}%)\n")

    t_global = time.time()
    res = run_cv(predictores, etiqueta, paciente_id)

    df_res = tabla_resumen(res)

    print("\nTabla de permutation importance:")
    tp = df_res.copy()
    for c in ['caida_AUC_pp','caida_AUC_desv','caida_AUC_IC_inf',
              'caida_AUC_IC_sup','p_valor_empirico']:
        tp[c] = tp[c].round(4)
    print(tp.to_string(index=False))

    n_imp = df_res['IC_excluye_cero'].sum()
    print(f"\nImportantes (IC95% excluye 0): {n_imp} / {len(df_res)}")
    print("Importantes:")
    for _, f in df_res[df_res['IC_excluye_cero']].iterrows():
        print(f"  - {f['variable']:<28} {f['caida_AUC_pp']:.2f} pp")
    print("No importantes:")
    for _, f in df_res[~df_res['IC_excluye_cero']].iterrows():
        print(f"  - {f['variable']:<28} {f['caida_AUC_pp']:.2f} pp")

    ruta_tab = os.path.join(CARPETA_TABLAS,  'permutation_importance_CAT_v4l.csv')
    ruta_fig = os.path.join(CARPETA_FIGURAS, 'permutation_importance_CAT_v4l.png')
    df_res.to_csv(ruta_tab, index=False)
    graficar(df_res, ruta_fig, res['auc_medio'])
    print(f"\nTabla  : {ruta_tab}")
    print(f"Figura : {ruta_fig}")
    print(f"Tiempo total: {(time.time()-t_global)/3600:.2f} h")

if __name__ == "__main__":
    main()
