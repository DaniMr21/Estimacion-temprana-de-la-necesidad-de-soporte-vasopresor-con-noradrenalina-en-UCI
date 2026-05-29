import warnings
warnings.filterwarnings('ignore')
 
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
 
 
def cargar_datos():
    ruta = r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4.csv'
    df = pd.read_csv(ruta)
    df = df.dropna(subset=['pf_max'])
    return df
 
 
VARIABLES_BASE = [
    # Demografía y contexto (4)
    'anchor_age', 'gender', 'peso_kg', 'contador_estancia_uci',
    # Hemodinámica (2)
    'map_min', 'hr_media',
    # Respiratorio (4)
    'pf_min', 'spo2_min', 'fio2_max', 'rr_max',
    # Ventilación y conciencia (2)
    'ventilacion_invasiva_6h', 'gcs_min',
    # Renal (2)
    'creatinina_max', 'diuresis_ml_kg_6h',
    # Ácido-base (3)
    'lactato_max', 'ph_min', 'bicarbonato_min',
    # Hepático (2)  — gpt_max se añade/quita según el modelo
    'bilirrubina_media',
    # Coagulación (2)
    'tp_max', 'plaquetas_min',
    # Hematología/inflamación (2)
    'leucocitos_min', 'hemoglobina_min',
    # Metabólico (1)
    'glucemia_min',
    # Otro vital (1)
    'temp_min',
    # Gravedad global (1)
    'sofa_max',
]
 
 
def preparar(df, incluir_gpt):
    if incluir_gpt:
        # gpt_max insertada justo después de bilirrubina_media (mantener orden lógico)
        variables = VARIABLES_BASE.copy()
        idx = variables.index('bilirrubina_media') + 1
        variables.insert(idx, 'gpt_max')
    else:
        variables = VARIABLES_BASE.copy()
 
    predictores = df[variables].copy()
    etiqueta = df['etiqueta_norad_6_24'].copy()
    paciente_id = df['subject_id'].copy()
    predictores['gender'] = (predictores['gender'] == 'M').astype(int)
    return predictores, etiqueta, paciente_id
 
 
def validacion_cruzada_anidada(nombre_modelo, pipeline, espacio,
                               predictores, etiqueta, paciente_id, n_jobs=-1):
    cv_externo = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    cv_interno = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
 
    aucs, params_list = [], []
    t0 = time.time()
 
    for nf, (idx_tr, idx_te) in enumerate(
            cv_externo.split(predictores, etiqueta, groups=paciente_id), start=1):
 
        x_tr, x_te = predictores.iloc[idx_tr], predictores.iloc[idx_te]
        y_tr, y_te = etiqueta.iloc[idx_tr],    etiqueta.iloc[idx_te]
        pid_tr = paciente_id.iloc[idx_tr]
 
        gs = GridSearchCV(pipeline, espacio, cv=cv_interno,
                          scoring='roc_auc', n_jobs=n_jobs, refit=True)
        gs.fit(x_tr, y_tr, groups=pid_tr)
 
        proba = gs.predict_proba(x_te)[:, 1]
        auc = roc_auc_score(y_te, proba)
        aucs.append(auc)
        params_list.append(gs.best_params_)
 
        print(f"  Fold {nf}: AUC={auc:.4f}")
        print(f"    Best params: {gs.best_params_}")
 
    t_min = (time.time() - t0) / 60
    auc_m, auc_s = np.mean(aucs), np.std(aucs)
    print(f"\n{nombre_modelo} — AUC medio: {auc_m:.4f} ± {auc_s:.4f}  "
          f"(tiempo: {t_min:.1f} min)\n")
    return auc_m, auc_s, aucs, params_list
 
 
def main():
    df = cargar_datos()
 
    # Grid REDUCIDO basado en best_params del baseline original
    pipeline_rf = Pipeline([
        ('modelo', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])
    espacio_rf = {
        'modelo__n_estimators':     [300, 500, 1000],
        'modelo__max_depth':        [10, 20, None],
        'modelo__min_samples_leaf': [1, 5],
        'modelo__max_features':     ['sqrt'],
        'modelo__class_weight':     ['balanced', 'balanced_subsample'],
    }
    # 3*3*2*1*2 = 36 combinaciones
 
    t_global = time.time()
 
    # ── MODELO A: CON gpt_max (26 variables) ─────────────────────────────────
    print("=" * 60)
    print("MODELO A — RF CON gpt_max (26 variables)")
    print("=" * 60)
    pred_A, et_A, pid_A = preparar(df, incluir_gpt=True)
    print(f"Dataset: {pred_A.shape} | "
          f"Positivos: {et_A.sum()} ({100*et_A.mean():.2f}%)\n")
 
    auc_A, std_A, aucs_A, _ = validacion_cruzada_anidada(
        'RF CON gpt_max', pipeline_rf, espacio_rf,
        pred_A, et_A, pid_A
    )
 
    # ── MODELO B: SIN gpt_max (25 variables) ─────────────────────────────────
    print("=" * 60)
    print("MODELO B — RF SIN gpt_max (25 variables)")
    print("=" * 60)
    pred_B, et_B, pid_B = preparar(df, incluir_gpt=False)
    print(f"Dataset: {pred_B.shape} | "
          f"Positivos: {et_B.sum()} ({100*et_B.mean():.2f}%)\n")
 
    auc_B, std_B, aucs_B, _ = validacion_cruzada_anidada(
        'RF SIN gpt_max', pipeline_rf, espacio_rf,
        pred_B, et_B, pid_B
    )
 
    # ── COMPARACIÓN FINAL ────────────────────────────────────────────────────
    diff = auc_B - auc_A
    t_total_min = (time.time() - t_global) / 60
 
    print("=" * 60)
    print(f"COMPARACIÓN FINAL (tiempo total: {t_total_min:.1f} min)")
    print("=" * 60)
    print(f"  Modelo A — CON gpt_max (26 vars): AUC = {auc_A:.4f} ± {std_A:.4f}")
    print(f"  Modelo B — SIN gpt_max (25 vars): AUC = {auc_B:.4f} ± {std_B:.4f}")
    print(f"  Diferencia (B − A)              : {diff:+.4f}")
    print()
 
    print("  AUC por fold:")
    print(f"    Fold      A (con gpt)    B (sin gpt)    Δ")
    for i, (a, b) in enumerate(zip(aucs_A, aucs_B), start=1):
        print(f"    {i}         {a:.4f}         {b:.4f}        {b-a:+.4f}")
    print()
 
    print("  Conclusión:")
    if abs(diff) < 0.002:
        print("    → AUC prácticamente idéntico. ELIMINACIÓN JUSTIFICADA")
        print("      (variable redundante o ruido).")
    elif diff >= 0:
        print("    → AUC mejora sin gpt_max. ELIMINACIÓN CLARAMENTE JUSTIFICADA")
        print("      (la variable estaba introduciendo ruido).")
    elif diff > -0.005:
        print("    → AUC baja ligeramente. ELIMINACIÓN DEFENDIBLE por")
        print("      interpretabilidad clínica (trade-off mínimo).")
    else:
        print("    → AUC baja apreciablemente. Decisión más delicada:")
        print("      considerar mantener gpt_max o documentar el trade-off")
        print("      entre interpretabilidad y rendimiento.")
 
 
if __name__ == "__main__":
    main()