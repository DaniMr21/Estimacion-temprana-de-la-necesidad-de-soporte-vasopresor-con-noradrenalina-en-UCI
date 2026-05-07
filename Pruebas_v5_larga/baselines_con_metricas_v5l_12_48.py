"""
MÉTRICAS FINALES — CatBoost global — 12_48h

Calcula, con validación cruzada externa agrupada por paciente:
  - ROC-AUC
  - PR-AUC / Average Precision
  - sensibilidad
  - especificidad
  - PPV / precisión
  - NPV
  - F1
  - matriz de confusión
  - curva ROC
  - curva precisión-recall

Importante:
  - Solo usa el subgrupo GLOBAL.
  - Mantiene StratifiedGroupKFold por subject_id.
  - El umbral se elige dentro de cada fold usando solo entrenamiento
    mediante Youden sobre predicciones out-of-fold internas.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
)
from catboost import CatBoostClassifier


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────
RUTA_CSV = r"C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4l.csv"
ETIQUETA = "etiqueta_norad_12_48"
NOMBRE_VENTANA = "12_48h"

CARPETA_BASE = os.path.dirname(__file__) if "__file__" in globals() else "."
CARPETA_SALIDAS = os.path.join(CARPETA_BASE, "metricas_catboost", NOMBRE_VENTANA)
CARPETA_TABLAS = os.path.join(CARPETA_SALIDAS, "tablas")
CARPETA_FIGURAS = os.path.join(CARPETA_SALIDAS, "figuras")
os.makedirs(CARPETA_TABLAS, exist_ok=True)
os.makedirs(CARPETA_FIGURAS, exist_ok=True)

VARIABLES = [
    'map_min', 'hr_media', 'pf_min', 'spo2_min', 'rr_max',
    'ventilacion_invasiva_12h', 'diuresis_ml_kg_12h',
    'lactato_max', 'bicarbonato_min', 'bilirrubina_media',
    'gcs_min', 'glucemia_min', 'temp_min', 'sofa_max',
]

GRID_CATBOOST = {
    "modelo__iterations": [500, 1000],
    "modelo__depth": [4, 6],
    "modelo__learning_rate": [0.01, 0.05],
    "modelo__l2_leaf_reg": [1, 5],
    "modelo__bagging_temperature": [0],
}

RANDOM_STATE = 42
N_SPLITS_EXT = 5
N_SPLITS_INT = 3


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES
# ─────────────────────────────────────────────────────────────────────────────
def construir_pipeline_catboost():
    return Pipeline([
        ("modelo", CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=RANDOM_STATE,
            verbose=0,
            thread_count=-1,
            allow_writing_files=False,
        ))
    ])


def calcular_metricas_binarias(y_real, y_prob, umbral):
    y_pred = (y_prob >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_real, y_pred, labels=[0, 1]).ravel()

    sensibilidad = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    especificidad = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan

    return {
        "umbral": umbral,
        "roc_auc": roc_auc_score(y_real, y_prob),
        "pr_auc": average_precision_score(y_real, y_prob),
        "sensibilidad": sensibilidad,
        "especificidad": especificidad,
        "precision_ppv": ppv,
        "npv": npv,
        "f1": f1_score(y_real, y_pred, zero_division=0),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def elegir_umbral_youden(y_real, y_prob):
    fpr, tpr, thresholds = roc_curve(y_real, y_prob)
    youden = tpr - fpr
    idx = int(np.argmax(youden))
    return float(thresholds[idx])


def predicciones_oof_internas(modelo_base, X, y, grupos, n_splits=N_SPLITS_INT):
    """
    Genera probabilidades out-of-fold en el entrenamiento del fold externo.
    Se usa solo para elegir el umbral sin mirar el fold externo de test.
    """
    cv_int = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    probas = np.zeros(len(y), dtype=float)

    for idx_tr, idx_val in cv_int.split(X, y, groups=grupos):
        x_tr_int = X.iloc[idx_tr]
        x_val_int = X.iloc[idx_val]
        y_tr_int = y.iloc[idx_tr]

        modelo = clone(modelo_base)
        modelo.fit(x_tr_int, y_tr_int)
        probas[idx_val] = modelo.predict_proba(x_val_int)[:, 1]

    return probas


def guardar_curvas(y_real, y_prob):
    # Curva ROC
    fpr, tpr, _ = roc_curve(y_real, y_prob)
    auc_roc = roc_auc_score(y_real, y_prob)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc_roc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Azar")
    plt.xlabel("1 - especificidad")
    plt.ylabel("sensibilidad")
    plt.title(f"Curva ROC — CatBoost global — {NOMBRE_VENTANA}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    ruta_roc = os.path.join(CARPETA_FIGURAS, f"roc_catboost_global_{NOMBRE_VENTANA}.png")
    plt.savefig(ruta_roc, dpi=300)
    plt.close()

    # Curva precisión-recall
    precision, recall, _ = precision_recall_curve(y_real, y_prob)
    pr_auc = average_precision_score(y_real, y_prob)
    prevalencia = np.mean(y_real)

    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.3f}")
    plt.axhline(prevalencia, linestyle="--", label=f"Prevalencia = {prevalencia:.3f}")
    plt.xlabel("sensibilidad / recall")
    plt.ylabel("precisión / PPV")
    plt.title(f"Curva precisión-recall — CatBoost global — {NOMBRE_VENTANA}")
    plt.legend(loc="upper right")
    plt.tight_layout()
    ruta_pr = os.path.join(CARPETA_FIGURAS, f"pr_catboost_global_{NOMBRE_VENTANA}.png")
    plt.savefig(ruta_pr, dpi=300)
    plt.close()

    return ruta_roc, ruta_pr


def guardar_matriz_confusion(y_real, y_pred):
    cm = confusion_matrix(y_real, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No vasopresor", "Vasopresor"],
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(values_format="d", ax=ax)
    ax.set_title(f"Matriz de confusión — CatBoost global — {NOMBRE_VENTANA}")
    plt.tight_layout()
    ruta_cm = os.path.join(CARPETA_FIGURAS, f"matriz_confusion_catboost_global_{NOMBRE_VENTANA}.png")
    plt.savefig(ruta_cm, dpi=300)
    plt.close()
    return ruta_cm


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print(f"MÉTRICAS FINALES — CATBOOST GLOBAL — {NOMBRE_VENTANA}")
    print("=" * 80)

    df = pd.read_csv(RUTA_CSV)

    # Mantengo el mismo filtro que usabas en los baselines.
    if "pf_max" in df.columns:
        df = df.dropna(subset=["pf_max"])

    columnas_necesarias = VARIABLES + [ETIQUETA, "subject_id"]
    columnas_faltantes = [c for c in columnas_necesarias if c not in df.columns]
    if columnas_faltantes:
        raise ValueError(f"Faltan columnas en el CSV: {columnas_faltantes}")

    df = df.dropna(subset=columnas_necesarias).reset_index(drop=True)

    X = df[VARIABLES].copy()
    y = df[ETIQUETA].astype(int).copy()
    grupos = df["subject_id"].copy()

    print(f"N = {len(df)}")
    print(f"Positivos = {int(y.sum())} ({100 * y.mean():.2f}%)")
    print(f"Pacientes únicos = {grupos.nunique()}")
    print(f"Variables = {len(VARIABLES)}")

    cv_ext = StratifiedGroupKFold(n_splits=N_SPLITS_EXT, shuffle=True, random_state=RANDOM_STATE)
    cv_int = StratifiedGroupKFold(n_splits=N_SPLITS_INT, shuffle=True, random_state=RANDOM_STATE)

    pipeline = construir_pipeline_catboost()

    resultados_folds = []
    predicciones = []

    for fold, (idx_tr, idx_te) in enumerate(cv_ext.split(X, y, groups=grupos), start=1):
        print(f"\nFold externo {fold}/{N_SPLITS_EXT}")

        x_tr = X.iloc[idx_tr]
        x_te = X.iloc[idx_te]
        y_tr = y.iloc[idx_tr]
        y_te = y.iloc[idx_te]
        grupos_tr = grupos.iloc[idx_tr]
        grupos_te = grupos.iloc[idx_te]

        busqueda = GridSearchCV(
            estimator=pipeline,
            param_grid=GRID_CATBOOST,
            scoring="roc_auc",
            cv=cv_int,
            n_jobs=1,
            refit=True,
        )
        busqueda.fit(x_tr, y_tr, groups=grupos_tr)
        mejor_modelo = busqueda.best_estimator_

        # Umbral elegido sin mirar el test externo.
        probas_tr_oof = predicciones_oof_internas(mejor_modelo, x_tr, y_tr, grupos_tr)
        umbral = elegir_umbral_youden(y_tr, probas_tr_oof)

        probas_te = mejor_modelo.predict_proba(x_te)[:, 1]
        y_pred_te = (probas_te >= umbral).astype(int)

        metricas = calcular_metricas_binarias(y_te, probas_te, umbral)
        metricas.update({
            "fold": fold,
            "n_test": len(y_te),
            "positivos_test": int(y_te.sum()),
            "mejores_parametros": str(busqueda.best_params_),
        })
        resultados_folds.append(metricas)

        df_pred_fold = pd.DataFrame({
            "fold": fold,
            "subject_id": grupos_te.values,
            "y_real": y_te.values,
            "probabilidad": probas_te,
            "umbral_fold": umbral,
            "y_pred": y_pred_te,
        })
        predicciones.append(df_pred_fold)

        print(
            f"  ROC-AUC={metricas['roc_auc']:.4f} | "
            f"PR-AUC={metricas['pr_auc']:.4f} | "
            f"Sens={metricas['sensibilidad']:.4f} | "
            f"Esp={metricas['especificidad']:.4f} | "
            f"PPV={metricas['precision_ppv']:.4f} | "
            f"NPV={metricas['npv']:.4f} | "
            f"F1={metricas['f1']:.4f} | "
            f"umbral={umbral:.4f}"
        )

    df_folds = pd.DataFrame(resultados_folds)
    df_pred = pd.concat(predicciones, ignore_index=True)

    # Métricas globales OOF usando el umbral específico de cada fold ya aplicado.
    y_real_global = df_pred["y_real"].values
    y_prob_global = df_pred["probabilidad"].values
    y_pred_global = df_pred["y_pred"].values

    tn, fp, fn, tp = confusion_matrix(y_real_global, y_pred_global, labels=[0, 1]).ravel()
    sensibilidad = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    especificidad = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan

    resumen_global = pd.DataFrame([{
        "ventana": NOMBRE_VENTANA,
        "modelo": "CatBoost global",
        "n": len(y_real_global),
        "positivos": int(np.sum(y_real_global)),
        "prevalencia": float(np.mean(y_real_global)),
        "roc_auc": roc_auc_score(y_real_global, y_prob_global),
        "pr_auc": average_precision_score(y_real_global, y_prob_global),
        "sensibilidad": sensibilidad,
        "especificidad": especificidad,
        "precision_ppv": ppv,
        "npv": npv,
        "f1": f1_score(y_real_global, y_pred_global, zero_division=0),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "umbral_medio": float(df_pred["umbral_fold"].mean()),
    }])

    ruta_folds = os.path.join(CARPETA_TABLAS, f"metricas_folds_catboost_global_{NOMBRE_VENTANA}.csv")
    ruta_resumen = os.path.join(CARPETA_TABLAS, f"metricas_resumen_catboost_global_{NOMBRE_VENTANA}.csv")
    ruta_pred = os.path.join(CARPETA_TABLAS, f"predicciones_oof_catboost_global_{NOMBRE_VENTANA}.csv")

    df_folds.to_csv(ruta_folds, index=False)
    resumen_global.to_csv(ruta_resumen, index=False)
    df_pred.to_csv(ruta_pred, index=False)

    ruta_roc, ruta_pr = guardar_curvas(y_real_global, y_prob_global)
    ruta_cm = guardar_matriz_confusion(y_real_global, y_pred_global)

    print("\n" + "=" * 80)
    print("RESUMEN GLOBAL OUT-OF-FOLD")
    print("=" * 80)
    print(resumen_global.round(4).to_string(index=False))

    print("\nArchivos guardados:")
    print(f"  - {ruta_resumen}")
    print(f"  - {ruta_folds}")
    print(f"  - {ruta_pred}")
    print(f"  - {ruta_roc}")
    print(f"  - {ruta_pr}")
    print(f"  - {ruta_cm}")


if __name__ == "__main__":
    main()
