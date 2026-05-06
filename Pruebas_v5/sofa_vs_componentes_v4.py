"""
SOFA vs componentes individuales — RF v4 (ventana 6-24h).

Compara dos sets de variables con RF y grid reducido:

  Modelo A: set final v4 CON sofa_max (15 variables)
  Modelo B: reemplaza sofa_max por sus 4 componentes (18 variables)
            bilirrubina_media, plaquetas_min, creatinina_max, gcs_min

Componentes del SOFA presentes en los datos:
  - Respiratorio  : pf_min (ya en ambos modelos, no se duplica)
  - Cardiovascular: map_min (ya en ambos modelos, no se duplica)
  - Hepático      : bilirrubina_media  ← nuevo en B
  - Coagulación   : plaquetas_min      ← nuevo en B
  - Renal         : creatinina_max     ← nuevo en B
  - Neurológico   : gcs_min            ← nuevo en B

Salidas (nombres únicos para no sobreescribir análisis anteriores):
  - tablas/sofa_vs_componentes_v4.csv
  - (no figura: solo tabla comparativa)
"""

import warnings
warnings.filterwarnings('ignore')

import os, time
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# ── CONFIGURACIÓN ─────────────────────────────────────────────────────────────
RUTA_CSV = r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4.csv'
ETIQUETA = 'etiqueta_norad_6_24'

CARPETA_BASE   = os.path.dirname(__file__) if '__file__' in dir() else '.'
CARPETA_TABLAS = os.path.join(CARPETA_BASE, 'tablas')
os.makedirs(CARPETA_TABLAS, exist_ok=True)

# Modelo A — set final con sofa_max (15 vars)
VARS_A = [
    'map_min', 'hr_media', 'lactato_max', 'diuresis_ml_kg_6h',
    'pf_min', 'spo2_min', 'rr_max',
    'ph_min',
    'sofa_max', 'hemoglobina_min',
    'glucemia_min', 'temp_min' 
]

# Modelo B — sofa_max reemplazado por sus 4 componentes (18 vars)
VARS_B = [
    'map_min', 'hr_media', 'lactato_max', 'diuresis_ml_kg_6h',
    'pf_min', 'spo2_min', 'rr_max',
    'ph_min',
    'bilirrubina_media',                 # componente hepático
    'plaquetas_min',                     # componente coagulación
    'creatinina_max',                    # componente renal
    'gcs_min',                           # componente neurológico
    'hemoglobina_min',
    'glucemia_min', 'temp_min',
]

# ── PIPELINE Y GRID REDUCIDO ──────────────────────────────────────────────────
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

# ── FUNCIÓN CV ────────────────────────────────────────────────────────────────
def entrenar_rf(predictores, etiqueta, paciente_id, nombre):
    cv_ext = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    cv_int = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    aucs = []
    t0 = time.time()

    for nf, (idx_tr, idx_te) in enumerate(
            cv_ext.split(predictores, etiqueta, groups=paciente_id), start=1):
        x_tr, x_te = predictores.iloc[idx_tr], predictores.iloc[idx_te]
        y_tr, y_te = etiqueta.iloc[idx_tr],    etiqueta.iloc[idx_te]
        pid_tr = paciente_id.iloc[idx_tr]

        gs = GridSearchCV(pipeline, espacio, cv=cv_int,
                          scoring='roc_auc', n_jobs=-1, refit=True)
        gs.fit(x_tr, y_tr, groups=pid_tr)

        auc = roc_auc_score(y_te, gs.predict_proba(x_te)[:, 1])
        aucs.append(auc)
        print(f"    Fold {nf}: AUC={auc:.4f}  params={gs.best_params_}")

    t_min = (time.time() - t0) / 60
    auc_m, auc_s = np.mean(aucs), np.std(aucs)
    print(f"\n  {nombre} — AUC: {auc_m:.4f} ± {auc_s:.4f}  ({t_min:.1f} min)\n")
    return auc_m, auc_s, aucs


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("SOFA vs COMPONENTES INDIVIDUALES — RF v4 (6-24h)")
    print("=" * 65)

    df = pd.read_csv(RUTA_CSV)
    df = df.dropna(subset=['pf_max'])
    etiqueta   = df[ETIQUETA].copy()
    paciente_id = df['subject_id'].copy()

    print(f"\nDataset: N={len(df)} | "
          f"Positivos={etiqueta.sum()} ({100*etiqueta.mean():.2f}%)\n")

    t_global = time.time()

    # ── MODELO A: con sofa_max ─────────────────────────────────────────────
    print(f"{'─'*65}")
    print(f"MODELO A — CON sofa_max ({len(VARS_A)} variables)")
    print(f"{'─'*65}")
    pred_A = df[VARS_A].copy()
    auc_A, std_A, aucs_A = entrenar_rf(pred_A, etiqueta, paciente_id,
                                       'Modelo A (sofa_max)')

    # ── MODELO B: con componentes ──────────────────────────────────────────
    print(f"{'─'*65}")
    print(f"MODELO B — COMPONENTES de SOFA ({len(VARS_B)} variables)")
    print(f"  (bilirrubina_media, plaquetas_min, creatinina_max, gcs_min)")
    print(f"{'─'*65}")
    pred_B = df[VARS_B].copy()
    auc_B, std_B, aucs_B = entrenar_rf(pred_B, etiqueta, paciente_id,
                                       'Modelo B (componentes)')

    # ── COMPARACIÓN ───────────────────────────────────────────────────────
    diff = auc_B - auc_A
    t_total = (time.time() - t_global) / 60

    print("=" * 65)
    print(f"COMPARACIÓN FINAL (tiempo total: {t_total:.1f} min)")
    print("=" * 65)
    print(f"\n  Modelo A — sofa_max      ({len(VARS_A)} vars): "
          f"AUC = {auc_A:.4f} ± {std_A:.4f}")
    print(f"  Modelo B — componentes   ({len(VARS_B)} vars): "
          f"AUC = {auc_B:.4f} ± {std_B:.4f}")
    print(f"  Diferencia (B − A)                         : {diff:+.4f}")

    print(f"\n  AUC por fold:")
    print(f"    Fold   A (sofa_max)   B (componentes)   Δ")
    for i, (a, b) in enumerate(zip(aucs_A, aucs_B), start=1):
        print(f"    {i}      {a:.4f}        {b:.4f}           {b-a:+.4f}")

    print(f"\n  Conclusión clínico-estadística:")
    if auc_B >= auc_A - 0.002:
        print(f"    → Los COMPONENTES individuales funcionan igual o mejor.")
        print(f"      Usar componentes da más interpretabilidad sin coste en AUC.")
        print(f"      RECOMENDACIÓN: sustituir sofa_max por sus componentes.")
    elif auc_B >= auc_A - 0.005:
        print(f"    → Diferencia marginal ({diff:+.4f}). Ambas opciones son defensibles.")
        print(f"      RECOMENDACIÓN: conservar sofa_max por parsimonia.")
    else:
        print(f"    → sofa_max supera a sus componentes por {abs(diff):.4f}.")
        print(f"      RECOMENDACIÓN: mantener sofa_max. La agregación del SOFA")
        print(f"      captura información que sus componentes individuales no.")

    # ── GUARDADO ──────────────────────────────────────────────────────────
    resultados = pd.DataFrame({
        'fold':           list(range(1, 6)),
        'auc_modelo_A':   aucs_A,
        'auc_modelo_B':   aucs_B,
        'delta':          [b-a for a,b in zip(aucs_A, aucs_B)],
    })
    resumen = pd.DataFrame([
        {'modelo': 'A_sofa_max',       'n_vars': len(VARS_A),
         'auc': auc_A, 'std': std_A, 'variables': str(VARS_A)},
        {'modelo': 'B_componentes',    'n_vars': len(VARS_B),
         'auc': auc_B, 'std': std_B, 'variables': str(VARS_B)},
    ])
    ruta = os.path.join(CARPETA_TABLAS, 'sofa_vs_componentes_v4.csv')
    pd.concat([resumen, resultados], axis=0).to_csv(ruta, index=False)
    print(f"\n  Tabla guardada en: {ruta}")


if __name__ == "__main__":
    main()
