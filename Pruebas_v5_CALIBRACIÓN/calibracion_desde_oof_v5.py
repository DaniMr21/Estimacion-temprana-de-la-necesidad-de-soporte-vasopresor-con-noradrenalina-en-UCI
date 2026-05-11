"""
ANALISIS DE CALIBRACION DESDE PREDICCIONES OOF — CatBoost global — v5
======================================================================

Lee los ficheros de predicciones out-of-fold (OOF) ya generados por
baselines_con_metricas_v5*.py y calcula métricas de calibración directamente.

NO reentrana el modelo. NO aplica calibrador post-hoc.
Analiza la calibración nativa de CatBoost sobre las probabilidades OOF.

Métricas calculadas:
  - Brier Score
  - Brier Skill Score (BSS) — referencia: modelo ingenuo con prevalencia
  - Test de Hosmer-Lemeshow (HL) con g=10 grupos por deciles de riesgo
  - ECE (Expected Calibration Error) con 10 bins por cuantiles
  - Curva de calibración (reliability diagram) con IC bootstrap
  - Histograma de probabilidades predichas

Salidas:
  calibracion_desde_oof_v5/
    tablas/
      metricas_calibracion_oof.csv   <- tabla resumen por ventana
      tabla_deciles_{ventana}.csv  <- tabla HL por grupos
    figuras/
      calibracion_{ventana}.png
      histograma_{ventana}.png
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from scipy import stats
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve


# =============================================================================
# CONFIGURACION
# =============================================================================

DIR_SCRIPT = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."

CARPETA_SALIDA  = os.path.join(DIR_SCRIPT, "calibracion_desde_oof_v5")
CARPETA_TABLAS  = os.path.join(CARPETA_SALIDA, "tablas")
CARPETA_FIGURAS = os.path.join(CARPETA_SALIDA, "figuras")
os.makedirs(CARPETA_TABLAS,  exist_ok=True)
os.makedirs(CARPETA_FIGURAS, exist_ok=True)

# Ficheros OOF generados por baselines_con_metricas_v5*.py
VENTANAS = {
    "6_24h": {
        "ruta_oof": r"C:\Users\danie\TFG\Pruebas_v5\metricas_catboost\6_24h\tablas\predicciones_oof_catboost_global_6_24h.csv",
        "col_probabilidad": "probabilidad",
        "etiqueta_grafica": "v4 — 6-24h",
    },
    "3_12h": {
        "ruta_oof": r"C:\Users\danie\TFG\pruebas_v5_corta\metricas_catboost\3_12h\tablas\predicciones_oof_catboost_global_3_12h.csv",
        "col_probabilidad": "probabilidad",
        "etiqueta_grafica": "v4p — 3-12h",
    },
    "12_48h": {
        "ruta_oof": r"C:\Users\danie\TFG\Pruebas_v5_larga\metricas_catboost\12_48h\tablas\predicciones_oof_catboost_global_12_48h.csv",
        "col_probabilidad": "probabilidad",
        "etiqueta_grafica": "v4l — 12-48h",
    },
}

N_GRUPOS_HL    = 10    # grupos para el test de Hosmer-Lemeshow
N_BINS_CURVA   = 10    # bins para la curva de calibración
N_BOOTSTRAP    = 1000  # iteraciones bootstrap para IC de la curva
SEMILLA        = 42
np.random.seed(SEMILLA)


# =============================================================================
# TEST DE HOSMER-LEMESHOW
# =============================================================================

def test_hosmer_lemeshow(y_real, probabilidad, n_grupos=10):
    
    y_real      = np.asarray(y_real)
    probabilidad = np.asarray(probabilidad)

    # Ordenar por probabilidad predicha y asignar grupos por cuantiles
    orden          = np.argsort(probabilidad)
    y_real_ord     = y_real[orden]
    prob_ord       = probabilidad[orden]
    indices_grupo  = np.array_split(np.arange(len(y_real)), n_grupos)

    estadistico_hl = 0.0
    tabla_grupos   = []

    for idx in indices_grupo:
        n_j  = len(idx)
        o_j  = y_real_ord[idx].sum()            # observados positivos
        e_j  = prob_ord[idx].sum()              # esperados = suma de probabilidades
        pi_j = e_j / n_j                        # probabilidad media del grupo

        if e_j == 0 or n_j == 0:
            continue

        # Varianza: n_j * pi_j * (1 - pi_j)
        varianza = n_j * pi_j * (1.0 - pi_j)
        if varianza == 0:
            continue

        estadistico_hl += (o_j - e_j) ** 2 / varianza

        tabla_grupos.append({
            "n_grupo":               n_j,
            "observados_positivos":  int(o_j),
            "esperados_positivos":   round(e_j, 3),
            "prob_media_predicha":   round(pi_j, 4),
            "prob_media_observada":  round(float(y_real_ord[idx].mean()), 4),
        })

    grados_libertad = n_grupos - 2
    p_valor         = 1.0 - stats.chi2.cdf(estadistico_hl, df=grados_libertad)

    return float(estadistico_hl), grados_libertad, float(p_valor), pd.DataFrame(tabla_grupos)


# =============================================================================
# BRIER SCORE Y BSS
# =============================================================================

def calcular_brier_y_bss(y_real, probabilidad):
    
    y_real       = np.asarray(y_real, dtype=float)
    prevalencia  = y_real.mean()
    brier        = brier_score_loss(y_real, probabilidad)
    brier_ref    = prevalencia * (1.0 - prevalencia)
    bss          = 1.0 - brier / brier_ref if brier_ref > 0 else np.nan
    return float(brier), float(bss), float(brier_ref)


# =============================================================================
# ECE
# =============================================================================

def calcular_ece(y_real, probabilidad, n_bins=N_BINS_CURVA):
    """
    Expected Calibration Error por cuantiles de probabilidad.
    ECE = suma_j ( |n_j/N| * |frec_observada_j - prob_media_j| )
    """
    y_real       = np.asarray(y_real, dtype=float)
    probabilidad = np.asarray(probabilidad, dtype=float)
    cortes       = np.quantile(probabilidad, np.linspace(0, 1, n_bins + 1))
    cortes       = np.unique(cortes)

    ece = 0.0
    for i in range(len(cortes) - 1):
        if i < len(cortes) - 2:
            mascara = (probabilidad >= cortes[i]) & (probabilidad < cortes[i + 1])
        else:
            mascara = (probabilidad >= cortes[i]) & (probabilidad <= cortes[i + 1])
        if mascara.sum() == 0:
            continue
        peso             = mascara.sum() / len(probabilidad)
        prob_media       = probabilidad[mascara].mean()
        frec_observada   = y_real[mascara].mean()
        ece             += peso * abs(frec_observada - prob_media)

    return float(ece)


# =============================================================================
# CURVA DE CALIBRACION CON IC BOOTSTRAP
# =============================================================================

def calcular_curva_con_bootstrap(y_real, probabilidad, n_bins=N_BINS_CURVA, n_iter=N_BOOTSTRAP):
    """
    Calcula la curva de calibración central más banda de IC 95% por bootstrap
    (remuestreo con reemplazamiento).
    """
    y_real       = np.asarray(y_real, dtype=float)
    probabilidad = np.asarray(probabilidad, dtype=float)
    n            = len(y_real)

    # Curva central
    frac_obs_central, prob_media_central = calibration_curve(
        y_real, probabilidad, n_bins=n_bins, strategy="quantile"
    )

    # Bootstrap
    curvas_bootstrap = []
    for _ in range(n_iter):
        idx  = np.random.choice(n, size=n, replace=True)
        try:
            frac_b, prob_b = calibration_curve(
                y_real[idx], probabilidad[idx], n_bins=n_bins, strategy="quantile"
            )
            if len(frac_b) == len(frac_obs_central):
                curvas_bootstrap.append(frac_b)
        except Exception:
            continue

    if curvas_bootstrap:
        matriz_bootstrap = np.array(curvas_bootstrap)
        ic_inf = np.percentile(matriz_bootstrap, 2.5,  axis=0)
        ic_sup = np.percentile(matriz_bootstrap, 97.5, axis=0)
    else:
        ic_inf = ic_sup = frac_obs_central

    return prob_media_central, frac_obs_central, ic_inf, ic_sup


# =============================================================================
# FIGURAS
# =============================================================================

def guardar_curva_calibracion(nombre_ventana, etiqueta_grafica, y_real, probabilidad):
    """
    Curva de calibración con IC 95% bootstrap y línea de referencia perfecta.
    """
    prob_media, frac_obs, ic_inf, ic_sup = calcular_curva_con_bootstrap(
        y_real, probabilidad
    )
    prevalencia = float(np.asarray(y_real).mean())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="steelblue", linewidth=1.5,
            label="Calibración perfecta")
    ax.axhline(prevalencia, linestyle=":", color="gray", linewidth=1.0,
               label=f"Prevalencia ({prevalencia:.3f})")
    ax.fill_between(prob_media, ic_inf, ic_sup, alpha=0.20, color="darkorange",
                    label="IC 95% bootstrap")
    ax.plot(prob_media, frac_obs, marker="o", color="darkorange", linewidth=2,
            markersize=6, label="CatBoost (OOF)")

    # Solo mostrar la región donde hay predicciones
    limite = min(1.0, max(probabilidad) * 1.10)
    limite = max(limite, 0.25)
    ax.set_xlim(0, limite)
    ax.set_ylim(0, min(1.0, max(frac_obs) * 1.30))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Probabilidad predicha", fontsize=12)
    ax.set_ylabel("Fracción observada de positivos", fontsize=12)
    ax.set_title(f"Curva de calibración — {etiqueta_grafica}", fontsize=13)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    ruta = os.path.join(CARPETA_FIGURAS, f"calibracion_{nombre_ventana}.png")
    fig.tight_layout()
    fig.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return ruta


def guardar_histograma(nombre_ventana, etiqueta_grafica, y_real, probabilidad):
    """
    Histograma de probabilidades predichas separado por clase real.
    """
    prob_pos = probabilidad[y_real == 1]
    prob_neg = probabilidad[y_real == 0]
    limite   = min(1.0, max(probabilidad) * 1.05)
    bins     = np.linspace(0, limite, 30)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(prob_neg, bins=bins, alpha=0.6, color="steelblue",  label="No vasopresor (0)")
    ax.hist(prob_pos, bins=bins, alpha=0.6, color="darkorange", label="Vasopresor (1)")
    ax.set_xlabel("Probabilidad predicha", fontsize=12)
    ax.set_ylabel("Frecuencia", fontsize=12)
    ax.set_title(f"Distribución de probabilidades — {etiqueta_grafica}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ruta = os.path.join(CARPETA_FIGURAS, f"histograma_{nombre_ventana}.png")
    fig.tight_layout()
    fig.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return ruta


# =============================================================================
# PROCESO POR VENTANA
# =============================================================================

def analizar_calibracion(nombre_ventana, config):
    print("\n" + "=" * 70)
    print(f"CALIBRACION OOF — {nombre_ventana}")
    print("=" * 70)

    # Carga
    df           = pd.read_csv(config["ruta_oof"])
    y_real       = df["y_real"].values.astype(float)
    col_prob     = config.get("col_probabilidad", "probabilidad")
    probabilidad = df[col_prob].values.astype(float)
    n            = len(y_real)
    n_positivos  = int(y_real.sum())
    prevalencia  = float(y_real.mean())

    print(f"  N total:     {n}")
    print(f"  Positivos:   {n_positivos} ({100*prevalencia:.2f}%)")

    # --- Brier y BSS ---
    brier, bss, brier_ref = calcular_brier_y_bss(y_real, probabilidad)
    print(f"  Brier Score: {brier:.5f}  (ref trivial={brier_ref:.5f})")
    print(f"  BSS:         {bss:.4f}")

    # --- ECE ---
    ece = calcular_ece(y_real, probabilidad)
    print(f"  ECE (10 bins cuantiles): {ece:.5f}")

    # --- Hosmer-Lemeshow ---
    hl_stat, hl_gl, hl_p, tabla_grupos = test_hosmer_lemeshow(
        y_real, probabilidad, n_grupos=N_GRUPOS_HL
    )
    print(f"  Hosmer-Lemeshow: chi2={hl_stat:.3f}, gl={hl_gl}, p={hl_p:.4f}")
    if hl_p >= 0.05:
        print("    → No se rechaza H0: calibración aceptable (p ≥ 0.05)")
    else:
        print("    → Se rechaza H0: posible descalibración (p < 0.05)")
        print("      (con N grande el test HL es muy sensible; revisar curva visualmente)")

    # --- Figuras ---
    ruta_curva      = guardar_curva_calibracion(
        nombre_ventana, config["etiqueta_grafica"], y_real, probabilidad
    )
    ruta_histograma = guardar_histograma(
        nombre_ventana, config["etiqueta_grafica"], y_real, probabilidad
    )
    print(f"  Curva:        {ruta_curva}")
    print(f"  Histograma:   {ruta_histograma}")

    # --- Tabla HL por grupos ---
    tabla_grupos.insert(0, "ventana", nombre_ventana)
    ruta_deciles = os.path.join(CARPETA_TABLAS, f"tabla_deciles_{nombre_ventana}.csv")
    tabla_grupos.to_csv(ruta_deciles, index=False)
    print(f"  Tabla deciles: {ruta_deciles}")
    print(tabla_grupos.drop(columns="ventana").to_string(index=False))

    return {
        "ventana":        nombre_ventana,
        "n":              n,
        "positivos":      n_positivos,
        "prevalencia":    round(prevalencia, 4),
        "brier":          round(brier, 5),
        "brier_ref":      round(brier_ref, 5),
        "bss":            round(bss, 4),
        "ece":            round(ece, 5),
        "hl_chi2":        round(hl_stat, 3),
        "hl_gl":          hl_gl,
        "hl_p_valor":     round(hl_p, 4),
        "hl_interpretacion": ("No rechaza H0" if hl_p >= 0.05 else "Rechaza H0"),
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ANALISIS DE CALIBRACION DESDE PREDICCIONES OOF — v5")
    print("=" * 70)
    print(f"Salida: {CARPETA_SALIDA}")
    print(f"Grupos HL: {N_GRUPOS_HL} | Bootstrap: {N_BOOTSTRAP} iter | Semilla: {SEMILLA}")

    filas_resumen = []

    for nombre, config in VENTANAS.items():
        fila = analizar_calibracion(nombre, config)
        filas_resumen.append(fila)

    # Tabla resumen global
    resumen = pd.DataFrame(filas_resumen)
    ruta_resumen = os.path.join(CARPETA_TABLAS, "metricas_calibracion_oof.csv")
    resumen.to_csv(ruta_resumen, index=False)

    print("\n" + "=" * 70)
    print("RESUMEN FINAL DE CALIBRACION")
    print("=" * 70)
    print(resumen.to_string(index=False))
    print(f"\nGuardado en: {ruta_resumen}")
    print("\nListo. Sin reentrenamiento, sin calibrador externo, métricas limpias.")
