
import os
import time
import copy
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_curve,
    precision_recall_curve,
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from catboost import CatBoostClassifier


# =============================================================================
# CONFIGURACION GENERAL
# =============================================================================

DIR_SCRIPT = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
CARPETA_SALIDA = os.path.join(DIR_SCRIPT, "calibracion_catboost_final_v5")
CARPETA_TABLAS = os.path.join(CARPETA_SALIDA, "tablas")
CARPETA_FIGURAS = os.path.join(CARPETA_SALIDA, "figuras")
os.makedirs(CARPETA_TABLAS, exist_ok=True)
os.makedirs(CARPETA_FIGURAS, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS_EXTERNOS = 5
N_SPLITS_INTERNOS = 3

PROPORCIONES_CALIBRACION = [0.20]
METODOS_CALIBRACION = ["sigmoid", "isotonic"]

N_BINS_CALIBRACION = 10
BINS_ECE_SENSIBILIDAD = [5, 10, 15]
N_BINS_ECE = 10
ESTRATEGIA_BINS = "quantile"


# =============================================================================
# VARIABLES FINALES: MISMAS QUE EN baselines_con_metricas_v5*.py
# =============================================================================

VARIABLES_V5P_3_12 = [
    "map_min", "hr_media", "pf_min", "spo2_min", "rr_max",
    "diuresis_ml_kg_3h", "lactato_max", "ph_min", "temp_min", "sofa_max",
]

VARIABLES_V5_6_24 = [
    "map_min", "hr_media", "lactato_max", "diuresis_ml_kg_6h",
    "pf_min", "spo2_min", "rr_max", "ph_min",
    "sofa_max", "hemoglobina_min", "glucemia_min", "temp_min",
]

VARIABLES_V5L_12_48 = [
    "map_min", "hr_media", "pf_min", "spo2_min", "rr_max",
    "ventilacion_invasiva_12h", "diuresis_ml_kg_12h",
    "lactato_max", "bicarbonato_min", "bilirrubina_media",
    "gcs_min", "glucemia_min", "temp_min", "sofa_max",
]

GRID_CATBOOST = {
    "modelo__iterations": [500, 1000],
    "modelo__depth": [4, 6],
    "modelo__learning_rate": [0.01, 0.05],
    "modelo__l2_leaf_reg": [1, 5],
    "modelo__bagging_temperature": [0],
}

VENTANAS = {
    "v5p_3_12_catboost": {
        "ruta_csv": r"C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4p.csv",
        "variables": VARIABLES_V5P_3_12,
        "etiqueta": "etiqueta_norad_3_12",
    },
    "v5_6_24_catboost": {
        "ruta_csv": r"C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4.csv",
        "variables": VARIABLES_V5_6_24,
        "etiqueta": "etiqueta_norad_6_24",
    },
    "v5l_12_48_catboost": {
        "ruta_csv": r"C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4l.csv",
        "variables": VARIABLES_V5L_12_48,
        "etiqueta": "etiqueta_norad_12_48",
    },
}


# =============================================================================
# MODELO Y CARGA
# =============================================================================

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


def cargar_y_preparar(config):
    df = pd.read_csv(config["ruta_csv"])

    if "pf_max" in df.columns:
        df = df.dropna(subset=["pf_max"]).copy()

    columnas_necesarias = config["variables"] + [config["etiqueta"], "subject_id"]
    columnas_faltantes = [c for c in columnas_necesarias if c not in df.columns]
    if columnas_faltantes:
        raise ValueError(f"Faltan columnas en el CSV: {columnas_faltantes}")

    df = df.dropna(subset=columnas_necesarias).reset_index(drop=True)

    predictores = df[config["variables"]].copy()
    etiqueta = df[config["etiqueta"]].astype(int).copy()
    paciente_id = df["subject_id"].copy()

    return predictores, etiqueta, paciente_id


# =============================================================================
# DIVISION MODELO/CALIBRACION SIN FUGA POR PACIENTE
# =============================================================================

def dividir_modelo_calibracion_por_paciente(predictores, etiqueta, paciente_id, proporcion):
    """
    Divide dentro del entrenamiento externo a nivel de paciente.
    La estratificacion usa la etiqueta maxima por paciente.
    """
    pacientes = pd.DataFrame({
        "subject_id": paciente_id.values,
        "etiqueta": etiqueta.values,
    }).groupby("subject_id")["etiqueta"].max().reset_index()

    pacientes_modelo, pacientes_calibracion = train_test_split(
        pacientes["subject_id"],
        test_size=proporcion,
        random_state=RANDOM_STATE,
        stratify=pacientes["etiqueta"],
    )

    mascara_modelo = paciente_id.isin(pacientes_modelo)
    mascara_calibracion = paciente_id.isin(pacientes_calibracion)

    return (
        predictores.loc[mascara_modelo].copy(),
        etiqueta.loc[mascara_modelo].copy(),
        paciente_id.loc[mascara_modelo].copy(),
        predictores.loc[mascara_calibracion].copy(),
        etiqueta.loc[mascara_calibracion].copy(),
    )


# =============================================================================
# GRIDSEARCH Y CALIBRACION
# =============================================================================

def ejecutar_gridsearch_catboost(x_modelo, y_modelo, grupos_modelo):
    cv_interno = StratifiedGroupKFold(
        n_splits=N_SPLITS_INTERNOS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    busqueda = GridSearchCV(
        estimator=construir_pipeline_catboost(),
        param_grid=GRID_CATBOOST,
        scoring="roc_auc",
        cv=cv_interno,
        n_jobs=1,       # CatBoost paraleliza internamente
        refit=True,
    )
    busqueda.fit(x_modelo, y_modelo, groups=grupos_modelo)
    return busqueda.best_estimator_, busqueda.best_params_, busqueda.best_score_


def crear_calibrador_prefit(modelo_entrenado, metodo):
    """
    Compatible con versiones antiguas y nuevas de scikit-learn.
    """
    try:
        from sklearn.frozen import FrozenEstimator
        calibrador = CalibratedClassifierCV(
            estimator=FrozenEstimator(modelo_entrenado),
            method=metodo,
        )
    except Exception:
        calibrador = CalibratedClassifierCV(
            estimator=modelo_entrenado,
            method=metodo,
            cv="prefit",
        )
    return calibrador


def ajustar_y_evaluar_calibradores(x_entreno, y_entreno, grupos_entreno):
    """
    Prueba proporcion x metodo dentro del entrenamiento externo.
    Selecciona por menor Brier en el subconjunto de calibracion.
    """
    resultados = []

    for proporcion in PROPORCIONES_CALIBRACION:
        x_mod, y_mod, grupos_mod, x_cal, y_cal = dividir_modelo_calibracion_por_paciente(
            x_entreno, y_entreno, grupos_entreno, proporcion,
        )

        modelo_sin_calibrar, mejores_params, mejor_auc_interno = ejecutar_gridsearch_catboost(
            x_mod, y_mod, grupos_mod,
        )

        prob_cal_sin = modelo_sin_calibrar.predict_proba(x_cal)[:, 1]
        metricas_sin_cal = calcular_metricas_probabilisticas(y_cal, prob_cal_sin)

        for metodo in METODOS_CALIBRACION:
            try:
                calibrador = crear_calibrador_prefit(copy.deepcopy(modelo_sin_calibrar), metodo)
                calibrador.fit(x_cal, y_cal)
                prob_cal_calibrada = calibrador.predict_proba(x_cal)[:, 1]
                metricas_cal = calcular_metricas_probabilisticas(y_cal, prob_cal_calibrada)

                resultados.append({
                    "proporcion_calibracion": proporcion,
                    "metodo_calibracion": metodo,
                    "n_modelo": len(x_mod),
                    "positivos_modelo": int(y_mod.sum()),
                    "n_calibracion": len(x_cal),
                    "positivos_calibracion": int(y_cal.sum()),
                    "mejor_auc_grid_interno": mejor_auc_interno,
                    "mejores_parametros": str(mejores_params),
                    "brier_sin_calibrar_en_cal": metricas_sin_cal["brier"],
                    "brier_calibrado_en_cal": metricas_cal["brier"],
                    "logloss_sin_calibrar_en_cal": metricas_sin_cal["logloss"],
                    "logloss_calibrado_en_cal": metricas_cal["logloss"],
                    "ece_sin_calibrar_en_cal": metricas_sin_cal["ece"],
                    "ece_calibrado_en_cal": metricas_cal["ece"],
                    "ece_5_bins_sin_calibrar_en_cal": metricas_sin_cal["ece_5_bins"],
                    "ece_5_bins_calibrado_en_cal": metricas_cal["ece_5_bins"],
                    "ece_10_bins_sin_calibrar_en_cal": metricas_sin_cal["ece_10_bins"],
                    "ece_10_bins_calibrado_en_cal": metricas_cal["ece_10_bins"],
                    "ece_15_bins_sin_calibrar_en_cal": metricas_sin_cal["ece_15_bins"],
                    "ece_15_bins_calibrado_en_cal": metricas_cal["ece_15_bins"],
                    "modelo_sin_calibrar": modelo_sin_calibrar,
                    "calibrador": calibrador,
                    "mejores_params_dict": mejores_params,
                })
            except Exception as error:
                print(f"    ERROR calibrando: proporcion={proporcion}, metodo={metodo}: {error}")

    if not resultados:
        raise RuntimeError("No funciono ningun calibrador.")

    ganador = min(resultados, key=lambda d: d["brier_calibrado_en_cal"])

    tabla = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ["modelo_sin_calibrar", "calibrador", "mejores_params_dict"]}
        for r in resultados
    ]).sort_values("brier_calibrado_en_cal").reset_index(drop=True)

    return ganador, tabla


# =============================================================================
# METRICAS
# =============================================================================

def calcular_ece(y_real, probabilidad, n_bins=N_BINS_ECE):
    """
    Expected Calibration Error con bins uniformes en [0, 1].

    Nota: se usa para evaluacion/sensibilidad, no para seleccionar el calibrador.
    La seleccion del calibrador se hace por Brier en el subconjunto de calibracion.
    """
    y_real = np.asarray(y_real)
    probabilidad = np.asarray(probabilidad)

    cortes = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        if i == n_bins - 1:
            mascara = (probabilidad >= cortes[i]) & (probabilidad <= cortes[i + 1])
        else:
            mascara = (probabilidad >= cortes[i]) & (probabilidad < cortes[i + 1])

        if mascara.sum() == 0:
            continue

        confianza_media = probabilidad[mascara].mean()
        frecuencia_observada = y_real[mascara].mean()
        peso = mascara.mean()
        ece += peso * abs(frecuencia_observada - confianza_media)

    return float(ece)


def calcular_ece_sensibilidad(y_real, probabilidad):
    """Devuelve ECE con varios numeros de bins para analisis de sensibilidad."""
    return {
        f"ece_{n_bins}_bins": calcular_ece(y_real, probabilidad, n_bins=n_bins)
        for n_bins in BINS_ECE_SENSIBILIDAD
    }


def calcular_metricas_probabilisticas(y_real, probabilidad):
    metricas = {
        "auc": roc_auc_score(y_real, probabilidad),
        "auprc": average_precision_score(y_real, probabilidad),
        "brier": brier_score_loss(y_real, probabilidad),
        "logloss": log_loss(y_real, probabilidad, labels=[0, 1]),
        "ece": calcular_ece(y_real, probabilidad),
    }
    metricas.update(calcular_ece_sensibilidad(y_real, probabilidad))
    return metricas


# =============================================================================
# FIGURAS Y TABLAS
# =============================================================================

def guardar_tabla_calibracion_deciles(nombre_ventana, predicciones):
    df = predicciones.copy()

    df["decil_calibrado"] = pd.qcut(
        df["probabilidad_calibrada"],
        q=10,
        labels=False,
        duplicates="drop",
    ) + 1

    tabla = df.groupby("decil_calibrado").agg(
        n=("y_real", "size"),
        positivos=("y_real", "sum"),
        frecuencia_observada=("y_real", "mean"),
        probabilidad_media_sin_calibrar=("probabilidad_sin_calibrar", "mean"),
        probabilidad_media_calibrada=("probabilidad_calibrada", "mean"),
    ).reset_index()

    ruta = os.path.join(CARPETA_TABLAS, f"tabla_calibracion_deciles_{nombre_ventana}.csv")
    tabla.to_csv(ruta, index=False)
    return ruta


def guardar_curva_calibracion(nombre_ventana, predicciones):
    y_real = predicciones["y_real"]
    prob_sin = predicciones["probabilidad_sin_calibrar"]
    prob_cal = predicciones["probabilidad_calibrada"]

    frac_sin, media_sin = calibration_curve(
        y_real, prob_sin, n_bins=N_BINS_CALIBRACION, strategy=ESTRATEGIA_BINS,
    )
    frac_cal, media_cal = calibration_curve(
        y_real, prob_cal, n_bins=N_BINS_CALIBRACION, strategy=ESTRATEGIA_BINS,
    )

    limite = min(1.0, max(prob_sin.max(), prob_cal.max(), frac_sin.max(), frac_cal.max()) * 1.10)
    limite = max(limite, 0.20)

    plt.figure(figsize=(6, 6))
    plt.plot([0, limite], [0, limite], linestyle="--", linewidth=2, label="Perfecta")
    plt.plot(media_sin, frac_sin, marker="o", linewidth=2, label="Sin calibrar")
    plt.plot(media_cal, frac_cal, marker="o", linewidth=2, label="Calibrada")
    plt.xlim(0, limite)
    plt.ylim(0, limite)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True, alpha=0.3)
    plt.xlabel("Probabilidad predicha")
    plt.ylabel("Fraccion observada de positivos")
    plt.title(f"Curva de calibracion — {nombre_ventana}")
    plt.legend(loc="upper left")
    plt.tight_layout()

    ruta = os.path.join(CARPETA_FIGURAS, f"calibracion_{nombre_ventana}.png")
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close()
    return ruta


def guardar_histograma_probabilidades(nombre_ventana, predicciones):
    prob_sin = predicciones["probabilidad_sin_calibrar"]
    prob_cal = predicciones["probabilidad_calibrada"]
    limite = min(1.0, max(prob_sin.max(), prob_cal.max()) * 1.05)
    limite = max(limite, 0.20)
    bins = np.linspace(0, limite, 25)

    plt.figure(figsize=(7, 5))
    plt.hist(prob_sin, bins=bins, alpha=0.6, label="Sin calibrar")
    plt.hist(prob_cal, bins=bins, alpha=0.6, label="Calibrada")
    plt.xlim(0, limite)
    plt.xlabel("Probabilidad predicha")
    plt.ylabel("Frecuencia")
    plt.title(f"Histograma de probabilidades — {nombre_ventana}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    ruta = os.path.join(CARPETA_FIGURAS, f"histograma_probabilidades_{nombre_ventana}.png")
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close()
    return ruta


def guardar_curva_roc(nombre_ventana, predicciones):
    y_real = predicciones["y_real"]
    prob_sin = predicciones["probabilidad_sin_calibrar"]
    prob_cal = predicciones["probabilidad_calibrada"]

    fpr_sin, tpr_sin, _ = roc_curve(y_real, prob_sin)
    fpr_cal, tpr_cal, _ = roc_curve(y_real, prob_cal)
    auc_sin = roc_auc_score(y_real, prob_sin)
    auc_cal = roc_auc_score(y_real, prob_cal)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr_sin, tpr_sin, linewidth=2, label=f"Sin calibrar AUC={auc_sin:.3f}")
    plt.plot(fpr_cal, tpr_cal, linewidth=2, label=f"Calibrada AUC={auc_cal:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Azar")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True, alpha=0.3)
    plt.xlabel("1 - Especificidad")
    plt.ylabel("Sensibilidad")
    plt.title(f"Curva ROC — {nombre_ventana}")
    plt.legend(loc="lower right")
    plt.tight_layout()

    ruta = os.path.join(CARPETA_FIGURAS, f"roc_{nombre_ventana}.png")
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close()
    return ruta


def guardar_curva_precision_recall(nombre_ventana, predicciones):
    y_real = predicciones["y_real"]
    prob_sin = predicciones["probabilidad_sin_calibrar"]
    prob_cal = predicciones["probabilidad_calibrada"]

    precision_sin, recall_sin, _ = precision_recall_curve(y_real, prob_sin)
    precision_cal, recall_cal, _ = precision_recall_curve(y_real, prob_cal)
    auprc_sin = average_precision_score(y_real, prob_sin)
    auprc_cal = average_precision_score(y_real, prob_cal)
    prevalencia = y_real.mean()

    plt.figure(figsize=(6, 6))
    plt.plot(recall_sin, precision_sin, linewidth=2, label=f"Sin calibrar AUPRC={auprc_sin:.3f}")
    plt.plot(recall_cal, precision_cal, linewidth=2, label=f"Calibrada AUPRC={auprc_cal:.3f}")
    plt.axhline(prevalencia, linestyle="--", linewidth=1, label=f"Azar={prevalencia:.3f}")
    plt.xlim(0, 1)
    plt.ylim(0, max(precision_sin.max(), precision_cal.max()) * 1.05)
    plt.grid(True, alpha=0.3)
    plt.xlabel("Recall / sensibilidad")
    plt.ylabel("Precision / PPV")
    plt.title(f"Curva Precision-Recall — {nombre_ventana}")
    plt.legend(loc="upper right")
    plt.tight_layout()

    ruta = os.path.join(CARPETA_FIGURAS, f"precision_recall_{nombre_ventana}.png")
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close()
    return ruta


# =============================================================================
# PROCESO PRINCIPAL POR VENTANA
# =============================================================================

def calibrar_ventana(nombre_ventana, config):
    print("\n" + "=" * 80)
    print(f"CALIBRACION FINAL CATBOOST — {nombre_ventana}")
    print("=" * 80)

    predictores, etiqueta, paciente_id = cargar_y_preparar(config)
    print(f"Estancias: {len(predictores)}")
    print(f"Positivos: {int(etiqueta.sum())} ({100 * etiqueta.mean():.2f}%)")
    print(f"Pacientes unicos: {paciente_id.nunique()}")
    print(f"Variables: {predictores.shape[1]}")

    cv_externo = StratifiedGroupKFold(
        n_splits=N_SPLITS_EXTERNOS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    predicciones_todos = []
    metricas_todos = []
    parametros_todos = []
    seleccion_todos = []

    for fold, (idx_entreno, idx_test) in enumerate(
        cv_externo.split(predictores, etiqueta, groups=paciente_id), start=1
    ):
        print("\n" + "-" * 80)
        print(f"Fold externo {fold}/{N_SPLITS_EXTERNOS}")
        t0 = time.time()

        x_entreno = predictores.iloc[idx_entreno].copy()
        y_entreno = etiqueta.iloc[idx_entreno].copy()
        grupos_entreno = paciente_id.iloc[idx_entreno].copy()

        x_test = predictores.iloc[idx_test].copy()
        y_test = etiqueta.iloc[idx_test].copy()
        grupos_test = paciente_id.iloc[idx_test].copy()

        ganador, tabla_seleccion = ajustar_y_evaluar_calibradores(
            x_entreno, y_entreno, grupos_entreno,
        )

        modelo_sin_calibrar = ganador["modelo_sin_calibrar"]
        calibrador = ganador["calibrador"]

        prob_sin = modelo_sin_calibrar.predict_proba(x_test)[:, 1]
        prob_cal = calibrador.predict_proba(x_test)[:, 1]

        metricas_sin = calcular_metricas_probabilisticas(y_test, prob_sin)
        metricas_cal = calcular_metricas_probabilisticas(y_test, prob_cal)

        metricas_fold = {
            "ventana": nombre_ventana,
            "fold": fold,
            "n_test": len(y_test),
            "positivos_test": int(y_test.sum()),
            "prevalencia_test": float(y_test.mean()),
            "proporcion_calibracion": ganador["proporcion_calibracion"],
            "metodo_calibracion": ganador["metodo_calibracion"],
            "n_modelo": ganador["n_modelo"],
            "positivos_modelo": ganador["positivos_modelo"],
            "n_calibracion": ganador["n_calibracion"],
            "positivos_calibracion": ganador["positivos_calibracion"],
            "mejor_auc_grid_interno": ganador["mejor_auc_grid_interno"],
            "auc_sin_calibrar": metricas_sin["auc"],
            "auc_calibrado": metricas_cal["auc"],
            "auprc_sin_calibrar": metricas_sin["auprc"],
            "auprc_calibrado": metricas_cal["auprc"],
            "brier_sin_calibrar": metricas_sin["brier"],
            "brier_calibrado": metricas_cal["brier"],
            "logloss_sin_calibrar": metricas_sin["logloss"],
            "logloss_calibrado": metricas_cal["logloss"],
            "ece_sin_calibrar": metricas_sin["ece"],
            "ece_calibrado": metricas_cal["ece"],
            "ece_5_bins_sin_calibrar": metricas_sin["ece_5_bins"],
            "ece_5_bins_calibrado": metricas_cal["ece_5_bins"],
            "ece_10_bins_sin_calibrar": metricas_sin["ece_10_bins"],
            "ece_10_bins_calibrado": metricas_cal["ece_10_bins"],
            "ece_15_bins_sin_calibrar": metricas_sin["ece_15_bins"],
            "ece_15_bins_calibrado": metricas_cal["ece_15_bins"],
            "tiempo_min": (time.time() - t0) / 60,
        }
        metricas_todos.append(metricas_fold)

        parametros_fold = {
            "ventana": nombre_ventana,
            "fold": fold,
            "mejor_auc_grid_interno": ganador["mejor_auc_grid_interno"],
        }
        parametros_fold.update(ganador["mejores_params_dict"])
        parametros_todos.append(parametros_fold)

        tabla_seleccion.insert(0, "ventana", nombre_ventana)
        tabla_seleccion.insert(1, "fold", fold)
        seleccion_todos.append(tabla_seleccion)

        pred_fold = pd.DataFrame({
            "ventana": nombre_ventana,
            "fold": fold,
            "subject_id": grupos_test.values,
            "y_real": y_test.values,
            "probabilidad_sin_calibrar": prob_sin,
            "probabilidad_calibrada": prob_cal,
            "metodo_calibracion": ganador["metodo_calibracion"],
            "proporcion_calibracion": ganador["proporcion_calibracion"],
        })
        predicciones_todos.append(pred_fold)

        print(f"  Calibrador: {ganador['metodo_calibracion']} | prop={ganador['proporcion_calibracion']:.0%}")
        print(f"  AUC sin/cal:    {metricas_sin['auc']:.4f} / {metricas_cal['auc']:.4f}")
        print(f"  AUPRC sin/cal:  {metricas_sin['auprc']:.4f} / {metricas_cal['auprc']:.4f}")
        print(f"  Brier sin/cal:  {metricas_sin['brier']:.5f} / {metricas_cal['brier']:.5f}")
        print(f"  ECE 10 bins sin/cal: {metricas_sin['ece_10_bins']:.5f} / {metricas_cal['ece_10_bins']:.5f}")
        print(f"  Tiempo fold:    {metricas_fold['tiempo_min']:.2f} min")

    predicciones = pd.concat(predicciones_todos, ignore_index=True)
    metricas = pd.DataFrame(metricas_todos)
    parametros = pd.DataFrame(parametros_todos)
    seleccion = pd.concat(seleccion_todos, ignore_index=True)

    # Resumen por folds
    columnas_resumen = [
        "mejor_auc_grid_interno",
        "auc_sin_calibrar", "auc_calibrado",
        "auprc_sin_calibrar", "auprc_calibrado",
        "brier_sin_calibrar", "brier_calibrado",
        "logloss_sin_calibrar", "logloss_calibrado",
        "ece_sin_calibrar", "ece_calibrado",
        "ece_5_bins_sin_calibrar", "ece_5_bins_calibrado",
        "ece_10_bins_sin_calibrar", "ece_10_bins_calibrado",
        "ece_15_bins_sin_calibrar", "ece_15_bins_calibrado",
    ]
    resumen = metricas[columnas_resumen].agg(["mean", "std", "min", "max"]).T.reset_index()
    resumen.columns = ["metrica", "media", "desviacion_estandar", "min", "max"]
    resumen.insert(0, "ventana", nombre_ventana)

    # Metricas OOF agregadas sobre todas las predicciones externas
    metricas_oof_sin = calcular_metricas_probabilisticas(
        predicciones["y_real"], predicciones["probabilidad_sin_calibrar"],
    )
    metricas_oof_cal = calcular_metricas_probabilisticas(
        predicciones["y_real"], predicciones["probabilidad_calibrada"],
    )
    resumen_oof = pd.DataFrame([{
        "ventana": nombre_ventana,
        "n": len(predicciones),
        "positivos": int(predicciones["y_real"].sum()),
        "prevalencia": float(predicciones["y_real"].mean()),
        "auc_sin_calibrar_oof": metricas_oof_sin["auc"],
        "auc_calibrado_oof": metricas_oof_cal["auc"],
        "auprc_sin_calibrar_oof": metricas_oof_sin["auprc"],
        "auprc_calibrado_oof": metricas_oof_cal["auprc"],
        "brier_sin_calibrar_oof": metricas_oof_sin["brier"],
        "brier_calibrado_oof": metricas_oof_cal["brier"],
        "logloss_sin_calibrar_oof": metricas_oof_sin["logloss"],
        "logloss_calibrado_oof": metricas_oof_cal["logloss"],
        "ece_sin_calibrar_oof": metricas_oof_sin["ece"],
        "ece_calibrado_oof": metricas_oof_cal["ece"],
        "ece_5_bins_sin_calibrar_oof": metricas_oof_sin["ece_5_bins"],
        "ece_5_bins_calibrado_oof": metricas_oof_cal["ece_5_bins"],
        "ece_10_bins_sin_calibrar_oof": metricas_oof_sin["ece_10_bins"],
        "ece_10_bins_calibrado_oof": metricas_oof_cal["ece_10_bins"],
        "ece_15_bins_sin_calibrar_oof": metricas_oof_sin["ece_15_bins"],
        "ece_15_bins_calibrado_oof": metricas_oof_cal["ece_15_bins"],
    }])

    sensibilidad_ece = []
    for n_bins in BINS_ECE_SENSIBILIDAD:
        sensibilidad_ece.append({
            "ventana": nombre_ventana,
            "n_bins": n_bins,
            "ece_sin_calibrar_oof": calcular_ece(
                predicciones["y_real"], predicciones["probabilidad_sin_calibrar"], n_bins=n_bins,
            ),
            "ece_calibrado_oof": calcular_ece(
                predicciones["y_real"], predicciones["probabilidad_calibrada"], n_bins=n_bins,
            ),
        })
    sensibilidad_ece = pd.DataFrame(sensibilidad_ece)


    # Guardar tablas
    ruta_predicciones = os.path.join(CARPETA_TABLAS, f"predicciones_calibracion_{nombre_ventana}.csv")
    ruta_metricas = os.path.join(CARPETA_TABLAS, f"metricas_calibracion_{nombre_ventana}.csv")
    ruta_resumen = os.path.join(CARPETA_TABLAS, f"resumen_calibracion_{nombre_ventana}.csv")
    ruta_resumen_oof = os.path.join(CARPETA_TABLAS, f"resumen_oof_calibracion_{nombre_ventana}.csv")
    ruta_parametros = os.path.join(CARPETA_TABLAS, f"mejores_parametros_{nombre_ventana}.csv")
    ruta_seleccion = os.path.join(CARPETA_TABLAS, f"seleccion_calibrador_{nombre_ventana}.csv")
    ruta_sensibilidad_ece = os.path.join(CARPETA_TABLAS, f"sensibilidad_ece_{nombre_ventana}.csv")

    predicciones.to_csv(ruta_predicciones, index=False)
    metricas.to_csv(ruta_metricas, index=False)
    resumen.to_csv(ruta_resumen, index=False)
    resumen_oof.to_csv(ruta_resumen_oof, index=False)
    parametros.to_csv(ruta_parametros, index=False)
    seleccion.to_csv(ruta_seleccion, index=False)
    sensibilidad_ece.to_csv(ruta_sensibilidad_ece, index=False)
    ruta_deciles = guardar_tabla_calibracion_deciles(nombre_ventana, predicciones)

    # Guardar figuras
    ruta_calibracion = guardar_curva_calibracion(nombre_ventana, predicciones)
    ruta_histograma = guardar_histograma_probabilidades(nombre_ventana, predicciones)
    ruta_roc = guardar_curva_roc(nombre_ventana, predicciones)
    ruta_pr = guardar_curva_precision_recall(nombre_ventana, predicciones)

    print("\nResumen por folds:")
    print(resumen.round(5).to_string(index=False))
    print("\nResumen OOF:")
    print(resumen_oof.round(5).to_string(index=False))
    print("\nSensibilidad ECE OOF:")
    print(sensibilidad_ece.round(5).to_string(index=False))

    print("\nArchivos guardados:")
    print(f"  Predicciones:       {ruta_predicciones}")
    print(f"  Metricas fold:      {ruta_metricas}")
    print(f"  Resumen folds:      {ruta_resumen}")
    print(f"  Resumen OOF:        {ruta_resumen_oof}")
    print(f"  Parametros:         {ruta_parametros}")
    print(f"  Seleccion cal.:     {ruta_seleccion}")
    print(f"  Sensibilidad ECE:   {ruta_sensibilidad_ece}")
    print(f"  Tabla deciles:      {ruta_deciles}")
    print(f"  Calibracion:        {ruta_calibracion}")
    print(f"  Histograma:         {ruta_histograma}")
    print(f"  ROC:                {ruta_roc}")
    print(f"  Precision-Recall:   {ruta_pr}")

    return metricas, resumen, resumen_oof, parametros, sensibilidad_ece


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("CALIBRACION FINAL — CATBOOST GLOBAL — v5")
    print("=" * 80)
    print(f"Salida: {CARPETA_SALIDA}")
    print(f"Folds externos: {N_SPLITS_EXTERNOS}")
    print(f"Folds internos: {N_SPLITS_INTERNOS}")
    print(f"Proporciones calibracion: {PROPORCIONES_CALIBRACION}")
    print(f"Metodos calibracion: {METODOS_CALIBRACION}")
    print(f"Bins ECE sensibilidad: {BINS_ECE_SENSIBILIDAD}")

    metricas_globales = []
    resumenes_globales = []
    resumenes_oof_globales = []
    parametros_globales = []
    sensibilidad_ece_global = []

    for nombre, config in VENTANAS.items():
        metricas, resumen, resumen_oof, parametros, sensibilidad_ece = calibrar_ventana(nombre, config)
        metricas_globales.append(metricas)
        resumenes_globales.append(resumen)
        resumenes_oof_globales.append(resumen_oof)
        parametros_globales.append(parametros)
        sensibilidad_ece_global.append(sensibilidad_ece)

    metricas_globales = pd.concat(metricas_globales, ignore_index=True)
    resumenes_globales = pd.concat(resumenes_globales, ignore_index=True)
    resumenes_oof_globales = pd.concat(resumenes_oof_globales, ignore_index=True)
    parametros_globales = pd.concat(parametros_globales, ignore_index=True)
    sensibilidad_ece_global = pd.concat(sensibilidad_ece_global, ignore_index=True)

    ruta_metricas_globales = os.path.join(CARPETA_TABLAS, "metricas_calibracion_todas_las_ventanas.csv")
    ruta_resumen_global = os.path.join(CARPETA_TABLAS, "resumen_calibracion_todas_las_ventanas.csv")
    ruta_resumen_oof_global = os.path.join(CARPETA_TABLAS, "resumen_oof_calibracion_todas_las_ventanas.csv")
    ruta_parametros_globales = os.path.join(CARPETA_TABLAS, "mejores_parametros_todas_las_ventanas.csv")
    ruta_sensibilidad_ece_global = os.path.join(CARPETA_TABLAS, "sensibilidad_ece_todas_las_ventanas.csv")

    metricas_globales.to_csv(ruta_metricas_globales, index=False)
    resumenes_globales.to_csv(ruta_resumen_global, index=False)
    resumenes_oof_globales.to_csv(ruta_resumen_oof_global, index=False)
    parametros_globales.to_csv(ruta_parametros_globales, index=False)
    sensibilidad_ece_global.to_csv(ruta_sensibilidad_ece_global, index=False)

    print("\n" + "=" * 80)
    print("CALIBRACION COMPLETADA")
    print("=" * 80)
    print(f"Metricas globales: {ruta_metricas_globales}")
    print(f"Resumen folds:     {ruta_resumen_global}")
    print(f"Resumen OOF:       {ruta_resumen_oof_global}")
    print(f"Parametros:        {ruta_parametros_globales}")
    print(f"Sensibilidad ECE:  {ruta_sensibilidad_ece_global}")
