import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier


# ── VARIABLES POR VENTANA ─────────────────────────────────────────────────────

VARS_V4P = ['map_min', 'hr_media', 'pf_min', 'spo2_min', 'rr_max', 'diuresis_ml_kg_3h', 'creatinina_max', 'lactato_max', 'ph_min', 'temp_min', 'sofa_max']


VENTANAS = [
    {
        'nombre':    'v4p  (3-12h)',
        'ruta':      r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4p.csv',
        'etiqueta':  'etiqueta_norad_3_12',
        'variables': VARS_V4P,
        'auc_ref':   0.6823,
    }
]

# ── PIPELINE Y GRID ───────────────────────────────────────────────────────────
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


# ── FUNCIÓN CV ────────────────────────────────────────────────────────────────
def entrenar_rf(predictores, etiqueta, paciente_id):
    cv_ext = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    cv_int = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)

    aucs = []

    for idx_tr, idx_te in cv_ext.split(predictores, etiqueta, groups=paciente_id):
        x_tr, x_te = predictores.iloc[idx_tr], predictores.iloc[idx_te]
        y_tr, y_te = etiqueta.iloc[idx_tr], etiqueta.iloc[idx_te]
        pid_tr = paciente_id.iloc[idx_tr]

        gs = GridSearchCV(
            pipeline,
            espacio,
            cv=cv_int,
            scoring='roc_auc',
            n_jobs=-1,
            refit=True
        )

        gs.fit(x_tr, y_tr, groups=pid_tr)

        aucs.append(
            roc_auc_score(y_te, gs.predict_proba(x_te)[:, 1])
        )

        print(f"    Fold AUC={aucs[-1]:.4f}  params={gs.best_params_}")

    return np.mean(aucs), np.std(aucs)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("VALIDACIÓN SET TRABAJO — RF v4p CORTA")
    print("=" * 65)

    t_global = time.time()
    resultados = []

    for v in VENTANAS:
        print(f"\n{'─'*65}")
        print(f"Ventana: {v['nombre']}  |  {len(v['variables'])} variables")
        print(f"{'─'*65}")

        df = pd.read_csv(v['ruta'])
        df = df.dropna(subset=['pf_max'])

        predictores = df[v['variables']].copy()
        etiqueta    = df[v['etiqueta']].copy()
        paciente_id = df['subject_id'].copy()

        print(f"  N={len(df)} | Positivos={etiqueta.sum()} "
              f"({100*etiqueta.mean():.2f}%)")

        t0 = time.time()
        auc_m, auc_s = entrenar_rf(predictores, etiqueta, paciente_id)
        t_min = (time.time() - t0) / 60

        delta = auc_m - v['auc_ref']

        print(f"\n  AUC set trabajo: {auc_m:.4f} ± {auc_s:.4f}")
        print(f"  AUC referencia : {v['auc_ref']:.4f}")
        print(f"  Δ              : {delta:+.4f}")
        print(f"  Tiempo         : {t_min:.1f} min")

        resultados.append({
            'ventana':    v['nombre'],
            'n_vars':     len(v['variables']),
            'auc_final':  auc_m,
            'std_final':  auc_s,
            'auc_ref':    v['auc_ref'],
            'delta':      delta,
        })

    t_total = (time.time() - t_global) / 60

    print(f"\n{'='*65}")
    print(f"RESUMEN COMPARATIVO (tiempo total: {t_total:.1f} min)")
    print(f"{'='*65}")

    print(f"  {'Ventana':<15} {'Vars':<6} {'AUC final':<14} {'AUC ref':<12} {'Δ'}")
    print(f"  {'-'*58}")

    for r in resultados:
        signo = '✓' if r['delta'] >= -0.005 else '!'

        print(f"  {r['ventana']:<15} {r['n_vars']:<6} "
              f"{r['auc_final']:.4f}±{r['std_final']:.4f}  "
              f"{r['auc_ref']:.4f}       {r['delta']:+.4f}  {signo}")

    print(f"\n  ✓ = AUC se mantiene o baja ≤0.005")
    print(f"  ! = AUC baja >0.005")


if __name__ == "__main__":
    main()
