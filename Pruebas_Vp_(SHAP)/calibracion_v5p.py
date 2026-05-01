import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    average_precision_score,
    precision_recall_curve,
)

# CONFIGURACIÓN
RUTA_CSV = r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4p.csv'

CARPETA_BASE = os.path.dirname(__file__) if '__file__' in dir() else '.'
CARPETA_FIGURAS = os.path.join(CARPETA_BASE, 'figuras')
CARPETA_TABLAS = os.path.join(CARPETA_BASE, 'tablas')
os.makedirs(CARPETA_FIGURAS, exist_ok=True)
os.makedirs(CARPETA_TABLAS, exist_ok=True)

# Hiperparámetros del RF — los mismos que usaste para SHAP
HIPERPARAMETROS_RF = dict(
    n_estimators=500,
    max_depth=10,
    min_samples_leaf=5,
    max_features=0.3,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
)

# Variables predictoras (set reducido v4p — 26 variables)
VARIABLES_PREDICTORAS = [
    'anchor_age', 'gender', 'peso_kg', 'contador_estancia_uci',
    'map_min', 'hr_media',
    'pf_min', 'spo2_min', 'fio2_max', 'rr_max',
    'ventilacion_invasiva_3h', 'gcs_min',
    'creatinina_max', 'diuresis_ml_kg_3h',
    'lactato_max', 'ph_min', 'bicarbonato_min',
    'bilirrubina_media', 'gpt_max',
    'tp_max', 'plaquetas_min',
    'leucocitos_min', 'hemoglobina_min',
    'glucemia_min',
    'temp_min',
    'sofa_max',
]

# 1. CARGA Y PREPARACIÓN
def cargar_y_preparar():
    df = pd.read_csv(RUTA_CSV)
    df = df.dropna(subset=['pf_max'])

    predictores = df[VARIABLES_PREDICTORAS].copy()
    predictores['gender'] = (predictores['gender'] == 'M').astype(int)

    # Ojo: etiqueta 3-12h
    etiqueta = df['etiqueta_norad_3_12'].astype(int).copy()
    paciente_id = df['subject_id'].copy()

    return predictores, etiqueta, paciente_id


# 2. PROBABILIDADES OUT-OF-FOLD
def calcular_probabilidades_oof(predictores, etiqueta, paciente_id):
    cv_externo = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    probabilidades_oof = np.zeros(len(predictores))
    fold_asignado = np.full(len(predictores), -1, dtype=int)

    for num_fold, (idx_train, idx_test) in enumerate(
            cv_externo.split(predictores, etiqueta, groups=paciente_id), start=1):

        modelo = RandomForestClassifier(**HIPERPARAMETROS_RF)
        # Aquí ya no da error porque 'etiqueta' entra como pd.Series
        modelo.fit(predictores.iloc[idx_train], etiqueta.iloc[idx_train])
        probabilidades_oof[idx_test] = modelo.predict_proba(predictores.iloc[idx_test])[:, 1]
        fold_asignado[idx_test] = num_fold

        auc_fold = roc_auc_score(etiqueta.iloc[idx_test], probabilidades_oof[idx_test])
        print(f"  Fold {num_fold}: AUC out-of-fold = {auc_fold:.4f}")

    auc_global = roc_auc_score(etiqueta, probabilidades_oof)
    print(f"\nAUC global out-of-fold (concatenado): {auc_global:.4f}")

    return probabilidades_oof, fold_asignado


# 3. MÉTRICAS DE CALIBRACIÓN
def calcular_metricas_calibracion(y_real, p_predicha, etiqueta_modelo='RF'):
    brier = brier_score_loss(y_real, p_predicha)
    p_clip = np.clip(p_predicha, 1e-6, 1 - 1e-6)
    logit_p = np.log(p_clip / (1 - p_clip))

    lr = LogisticRegression(C=1e9, solver='lbfgs') 
    lr.fit(logit_p.reshape(-1, 1), y_real)

    intercept = float(lr.intercept_[0])
    slope = float(lr.coef_[0, 0])

    auc = roc_auc_score(y_real, p_predicha)
    auprc = average_precision_score(y_real, p_predicha)

    return {
        'modelo': etiqueta_modelo,
        'AUC': round(auc, 4),
        'AUPRC': round(auprc, 4),
        'Brier': round(brier, 4),
        'calibration_intercept': round(intercept, 4),
        'calibration_slope': round(slope, 4),
    }

# 4. CALIBRACIÓN POSTHOC (Platt e Isotonic)
def calibrar_platt(probs_train, y_train, probs_test):
    p_clip = np.clip(probs_train, 1e-6, 1 - 1e-6)
    logit_p = np.log(p_clip / (1 - p_clip)).reshape(-1, 1)

    lr = LogisticRegression(C=1e9, solver='lbfgs')
    lr.fit(logit_p, y_train)

    p_test_clip = np.clip(probs_test, 1e-6, 1 - 1e-6)
    logit_test = np.log(p_test_clip / (1 - p_test_clip)).reshape(-1, 1)
    return lr.predict_proba(logit_test)[:, 1]

def calibrar_isotonic(probs_train, y_train, probs_test):
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(probs_train, y_train)
    return iso.predict(probs_test)

def aplicar_calibracion_oof(probs_oof, y, fold_asignado, metodo):
    probs_calibradas = np.zeros_like(probs_oof)

    for fold_test in range(1, 6):
        idx_test = (fold_asignado == fold_test)
        idx_train = ~idx_test

        if metodo == 'platt':
            probs_calibradas[idx_test] = calibrar_platt(
                probs_oof[idx_train], y[idx_train], probs_oof[idx_test]
            )
        elif metodo == 'isotonic':
            probs_calibradas[idx_test] = calibrar_isotonic(
                probs_oof[idx_train], y[idx_train], probs_oof[idx_test]
            )

    return probs_calibradas


# 5. DECISION CURVE ANALYSIS
def decision_curve_analysis(y_real, p_predicha, umbrales=None):
    if umbrales is None:
        umbrales = np.arange(0.01, 0.51, 0.01)

    n = len(y_real)
    prevalencia = y_real.mean()

    beneficio_modelo = []
    beneficio_tratar_todos = []

    for umbral in umbrales:
        predicho_positivo = (p_predicha >= umbral)
        tp = ((predicho_positivo) & (y_real == 1)).sum()
        fp = ((predicho_positivo) & (y_real == 0)).sum()

        bn_modelo = (tp / n) - (fp / n) * (umbral / (1 - umbral))
        beneficio_modelo.append(bn_modelo)

        bn_todos = prevalencia - (1 - prevalencia) * (umbral / (1 - umbral))
        beneficio_tratar_todos.append(bn_todos)

    return {
        'umbrales': umbrales,
        'beneficio_modelo': np.array(beneficio_modelo),
        'beneficio_tratar_todos': np.array(beneficio_tratar_todos),
        'beneficio_tratar_nadie': np.zeros(len(umbrales)),
    }


# 6. FIGURAS
def figura_calibracion(y_real, probs_dict, ruta):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Calibración perfecta')

    colores = {'Sin calibrar': 'tab:red', 'Platt': 'tab:blue', 'Isotonic': 'tab:green'}

    for nombre, probs in probs_dict.items():
        prob_observada, prob_predicha_bin = calibration_curve(
            y_real, probs, n_bins=10, strategy='quantile'
        )
        ax.plot(prob_predicha_bin, prob_observada,
                marker='o', linewidth=2, label=nombre, color=colores.get(nombre, 'gray'))

    ax.set_xlabel('Probabilidad predicha (media en bin)', fontsize=12)
    ax.set_ylabel('Frecuencia observada del evento', fontsize=12)
    ax.set_title('Reliability diagram — RF v5p (out-of-fold)', fontsize=13)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    plt.tight_layout()
    plt.savefig(ruta, dpi=200, bbox_inches='tight')
    plt.close()

def figura_histograma_probabilidades(y_real, p_predicha, ruta):
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, 1, 30)
    ax.hist(p_predicha[y_real == 0], bins=bins, alpha=0.6, label='Negativos', color='tab:blue')
    ax.hist(p_predicha[y_real == 1], bins=bins, alpha=0.6, label='Positivos', color='tab:red')

    ax.set_xlabel('Probabilidad predicha', fontsize=12)
    ax.set_ylabel('Número de estancias', fontsize=12)
    ax.set_title('Distribución de probabilidades predichas por clase real (v5p)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(ruta, dpi=200, bbox_inches='tight')
    plt.close()

def figura_auprc(y_real, probs_dict, ruta):
    fig, ax = plt.subplots(figsize=(8, 7))
    prevalencia = y_real.mean()
    ax.axhline(y=prevalencia, color='gray', linestyle='--', label=f'Azar (prev = {prevalencia:.3f})')

    colores = {'Sin calibrar': 'tab:red', 'Platt': 'tab:blue', 'Isotonic': 'tab:green'}

    for nombre, probs in probs_dict.items():
        precision, recall, _ = precision_recall_curve(y_real, probs)
        auprc = average_precision_score(y_real, probs)
        ax.plot(recall, precision, linewidth=2,
                label=f'{nombre} (AUPRC = {auprc:.3f})', color=colores.get(nombre, 'gray'))

    ax.set_xlabel('Recall (sensibilidad)', fontsize=12)
    ax.set_ylabel('Precisión (VPP)', fontsize=12)
    ax.set_title('Curva Precisión-Recall — RF v5p (out-of-fold)', fontsize=13)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(ruta, dpi=200, bbox_inches='tight')
    plt.close()

def figura_dca(dca_resultado, ruta):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(dca_resultado['umbrales'], dca_resultado['beneficio_modelo'],
            linewidth=2.5, color='tab:blue', label='Modelo RF (Platt)')
    ax.plot(dca_resultado['umbrales'], dca_resultado['beneficio_tratar_todos'],
            linewidth=1.5, color='tab:gray', linestyle='--', label='Tratar a todos')
    ax.plot(dca_resultado['umbrales'], dca_resultado['beneficio_tratar_nadie'],
            linewidth=1.5, color='black', linestyle=':', label='No tratar a nadie')

    ax.set_xlabel('Umbral de probabilidad', fontsize=12)
    ax.set_ylabel('Beneficio neto', fontsize=12)
    ax.set_title('Decision Curve Analysis — RF v5p (out-of-fold)', fontsize=13)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlim([0, 0.5])

    plt.tight_layout()
    plt.savefig(ruta, dpi=200, bbox_inches='tight')
    plt.close()


# 7. MAIN
def main():
    print("--------------------------------------")
    print("CALIBRACIÓN + AUPRC + DCA — RF v5p (CORTA)")
    print("-------------------------")
    print()

    print("[1/6] Cargando datos...")
    predictores, etiqueta, paciente_id = cargar_y_preparar()
    print(f"  Estancias: {len(predictores)}")
    print(f"  Positivos: {etiqueta.sum()} ({100 * etiqueta.mean():.2f}%)")
    print(f"  Pacientes únicos: {paciente_id.nunique()}")
    print()

    print("[2/6] Calculando probabilidades out-of-fold (CV 5 folds)...")
    probs_oof, fold_asignado = calcular_probabilidades_oof(
        predictores, etiqueta, paciente_id # <-- CORREGIDO (.values quitado)
    )
    print()

    print("[3/6] Aplicando calibración Platt (sigmoid)...")
    probs_platt = aplicar_calibracion_oof(
        probs_oof, etiqueta.values, fold_asignado, metodo='platt'
    )
    print()

    print("[4/6] Aplicando calibración Isotonic...")
    probs_iso = aplicar_calibracion_oof(
        probs_oof, etiqueta.values, fold_asignado, metodo='isotonic'
    )
    print()

    print("[5/6] Calculando métricas...")
    metricas = []
    for nombre, probs in [
        ('RF sin calibrar', probs_oof),
        ('RF + Platt', probs_platt),
        ('RF + Isotonic', probs_iso),
    ]:
        m = calcular_metricas_calibracion(etiqueta.values, probs, nombre)
        metricas.append(m)
        print(f"  {nombre}:")
        for k, v in m.items():
            if k != 'modelo':
                print(f"    {k} = {v}")

    df_metricas = pd.DataFrame(metricas)
    ruta_tabla_metricas = os.path.join(CARPETA_TABLAS, 'metricas_calibracion_v5p.csv')
    df_metricas.to_csv(ruta_tabla_metricas, index=False)
    print(f"\nTabla métricas guardada en: {ruta_tabla_metricas}")
    print()

    df_probs = pd.DataFrame({
        'fold': fold_asignado,
        'etiqueta_real': etiqueta.values,
        'prob_sin_calibrar': probs_oof,
        'prob_platt': probs_platt,
        'prob_isotonic': probs_iso,
    })
    ruta_probs = os.path.join(CARPETA_TABLAS, 'probabilidades_oof_v5p.csv')
    df_probs.to_csv(ruta_probs, index=False)
    print()

    print("[6/6] Generando figuras...")

    probs_dict = {
        'Sin calibrar': probs_oof,
        'Platt': probs_platt,
        'Isotonic': probs_iso,
    }
    
    ruta = os.path.join(CARPETA_FIGURAS, 'calibracion_curva_v5p.png')
    figura_calibracion(etiqueta.values, probs_dict, ruta)

    ruta = os.path.join(CARPETA_FIGURAS, 'calibracion_histograma_v5p.png')
    figura_histograma_probabilidades(etiqueta.values, probs_oof, ruta)

    ruta = os.path.join(CARPETA_FIGURAS, 'auprc_v5p.png')
    figura_auprc(etiqueta.values, probs_dict, ruta)

    # Nota: He fijado probs_platt como la probabilidad para el DCA
    dca = decision_curve_analysis(etiqueta.values, probs_platt)
    ruta = os.path.join(CARPETA_FIGURAS, 'dca_v5p.png')
    figura_dca(dca, ruta)

    print()
    print("-----------------------")
    print("RESUMEN v5p")
    print("----------------------------")
    print(df_metricas.to_string(index=False))

if __name__ == '__main__':
    main()