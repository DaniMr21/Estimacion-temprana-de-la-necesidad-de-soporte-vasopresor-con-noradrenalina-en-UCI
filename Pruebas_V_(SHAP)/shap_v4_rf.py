"""
SHAP del modelo final v4 — RANDOM FOREST.
RF es el modelo ganador en la ventana principal (6-24h, AUC 0.7077).

Pasos:
  1. Carga el dataset v4 winsorizado.
  2. Grid search del RF con CV agrupado por paciente.
  3. Entrena el RF final con los mejores hiperparámetros.
  4. Calcula SHAP global (Tree SHAP exacto).
  5. Guarda figuras (bar + beeswarm) y tabla de importancias.
"""

import warnings
warnings.filterwarnings('ignore')

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier


# CONFIGURACIÓN

RUTA_CSV = r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4.csv'

CARPETA_BASE = os.path.dirname(__file__) if '__file__' in dir() else '.'
CARPETA_FIGURAS = os.path.join(CARPETA_BASE, 'figuras')
CARPETA_TABLAS = os.path.join(CARPETA_BASE, 'tablas')
os.makedirs(CARPETA_FIGURAS, exist_ok=True)
os.makedirs(CARPETA_TABLAS, exist_ok=True)


VARIABLES_PREDICTORAS = [
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
    # Hepático (2)
    'bilirrubina_media', 'gpt_max',
    # Coagulación (2)
    'tp_max', 'plaquetas_min',
    # Hematología (2)
    'leucocitos_min', 'hemoglobina_min',
    # Metabólico (1)
    'glucemia_min',
    # Otro vital (1)
    'temp_min',
    # Gravedad global (1)
    'sofa_max',
]

# CARGA Y PREPARACIÓN

def cargar_y_preparar():
    df = pd.read_csv(RUTA_CSV)
    df = df.dropna(subset=['pf_max'])

    predictores = df[VARIABLES_PREDICTORAS].copy()
    predictores['gender'] = (predictores['gender'] == 'M').astype(int)

    etiqueta = df['etiqueta_norad_6_24'].astype(int).copy()
    paciente_id = df['subject_id'].copy()

    return predictores, etiqueta, paciente_id


# GRID SEARCH + ENTRENAMIENTO + SHAP

def main():
    print("=" * 70)
    print("SHAP v4 — RANDOM FOREST (modelo ganador 6-24h)")
    print("=" * 70)

    print("\n[1/4] Cargando datos...")
    predictores, etiqueta, paciente_id = cargar_y_preparar()
    print(f"  Estancias: {len(predictores)}")
    print(f"  Positivos: {etiqueta.sum()} ({100 * etiqueta.mean():.2f}%)")
    print(f"  Pacientes únicos: {paciente_id.nunique()}")
    print(f"  Variables: {len(VARIABLES_PREDICTORAS)}")

    print("\n[2/4] Grid search Random Forest (CV agrupado, 3 folds internos)...")
    tiempo_inicio = time.time()

    modelo_rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    espacio_rf = {
        'n_estimators': [300, 400, 500, 600, 700, 850, 1000],
        'max_depth': [None, 5, 10, 20, 30],
        'min_samples_leaf': [1, 2, 5],
        'max_features': ['sqrt', 0.3, 0.5],
        'class_weight': ['balanced', 'balanced_subsample'],
    }

    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)

    busqueda = GridSearchCV(
        estimator=modelo_rf,
        param_grid=espacio_rf,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        refit=True,
    )

    busqueda.fit(predictores, etiqueta, groups=paciente_id)

    tiempo_grid_min = (time.time() - tiempo_inicio) / 60

    print(f"\n  Tiempo de grid search: {tiempo_grid_min:.1f} min")
    print(f"  Mejor AUC en CV: {busqueda.best_score_:.4f}")
    print(f"  Mejores hiperparámetros:")
    for k, v in busqueda.best_params_.items():
        print(f"    {k} = {v}")

    modelo_final = busqueda.best_estimator_

    print("\n[3/4] Calculando SHAP (Tree SHAP exacto)...")
    explicador = shap.TreeExplainer(modelo_final)
    valores_shap = explicador.shap_values(predictores)

    # Manejo robusto del formato según versión de SHAP
    if isinstance(valores_shap, list):
        shap_clase_positiva = valores_shap[1]
    elif len(np.array(valores_shap).shape) == 3:
        shap_clase_positiva = valores_shap[:, :, 1]
    else:
        shap_clase_positiva = valores_shap

    print(f"  Forma de los valores SHAP: {shap_clase_positiva.shape}")

    print("\n[4/4] Generando figuras y tabla...")

    # BAR PLOT
    plt.figure()
    shap.summary_plot(
        shap_clase_positiva, predictores,
        plot_type='bar', show=False, max_display=26
    )
    plt.tight_layout()
    ruta_bar = os.path.join(CARPETA_FIGURAS, 'shap_bar_rf_v4_2.png')
    plt.savefig(ruta_bar, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Bar plot     : {ruta_bar}")

    # BEESWARM
    plt.figure()
    shap.summary_plot(
        shap_clase_positiva, predictores,
        show=False, max_display=26
    )
    plt.tight_layout()
    ruta_beeswarm = os.path.join(CARPETA_FIGURAS, 'shap_beeswarm_rf_v4_2.png')
    plt.savefig(ruta_beeswarm, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Beeswarm     : {ruta_beeswarm}")

    # TABLA
    importancia_shap = pd.DataFrame({
        'variable': predictores.columns,
        'shap_medio_absoluto': np.abs(shap_clase_positiva).mean(axis=0),
    }).sort_values('shap_medio_absoluto', ascending=False).reset_index(drop=True)

    ruta_tabla = os.path.join(CARPETA_TABLAS, 'importancia_shap_rf_v4_2.csv')
    importancia_shap.to_csv(ruta_tabla, index=False)
    print(f"  Tabla        : {ruta_tabla}")

    print("\n" + "--------------------------")
    print("IMPORTANCIA SHAP (ordenada)")
    print("-------------------")
    print(importancia_shap.to_string(index=False))


if __name__ == '__main__':
    main()
