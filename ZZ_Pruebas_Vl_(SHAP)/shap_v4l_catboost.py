"""
SHAP del modelo final v4l — CATBOOST.
CatBoost es el modelo ganador en la ventana larga (12-48h, AUC 0.6530).

Pasos:
  1. Carga el dataset v4l winsorizado.
  2. Grid search del CatBoost con CV agrupado por paciente.
  3. Entrena el CatBoost final con los mejores hiperparámetros.
  4. Calcula SHAP global (Tree SHAP).
  5. Guarda figuras (bar + beeswarm) y tabla de importancias.

Nota: en v4l se suprimió fio2_max por alta correlación con pf_min,
quedan 25 variables predictoras en lugar de 26.
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
from catboost import CatBoostClassifier


# CONFIGURACIÓN

RUTA_CSV = r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4l.csv'

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
    # Respiratorio (3) — fio2_max suprimido por alta correlación con pf en v4l
    'pf_min', 'spo2_min', 'rr_max',
    # Ventilación y conciencia (2)
    'ventilacion_invasiva_12h', 'gcs_min',
    # Renal (2)
    'creatinina_max', 'diuresis_ml_kg_12h',
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

    etiqueta = df['etiqueta_norad_12_48'].astype(int).copy()
    paciente_id = df['subject_id'].copy()

    return predictores, etiqueta, paciente_id


# GRID SEARCH + ENTRENAMIENTO + SHAP

def main():
    print("----------------")
    print("SHAP v4l — CATBOOST (modelo ganador 12-48h)")
    print("--------------")

    print("\n[1/4] Cargando datos...")
    predictores, etiqueta, paciente_id = cargar_y_preparar()
    print(f"  Estancias: {len(predictores)}")
    print(f"  Positivos: {etiqueta.sum()} ({100 * etiqueta.mean():.2f}%)")
    print(f"  Pacientes únicos: {paciente_id.nunique()}")
    print(f"  Variables: {len(VARIABLES_PREDICTORAS)}")

    print("\n[2/4] Grid search CatBoost (CV agrupado, 3 folds internos)...")
    tiempo_inicio = time.time()

    modelo_cat = CatBoostClassifier(
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42,
        verbose=0,
        thread_count=-1,
    )

    espacio_cat = {
        'iterations': [500, 1000],
        'depth': [4, 5, 6, 7],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'l2_leaf_reg': [1, 5, 15],
        'bagging_temperature': [0, 0.5, 1],
    }

    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)

    busqueda = GridSearchCV(
        estimator=modelo_cat,
        param_grid=espacio_cat,
        cv=cv,
        scoring='roc_auc',
        n_jobs=1,
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

    print("\n[3/4] Calculando SHAP (Tree SHAP)...")
    explicador = shap.TreeExplainer(modelo_final)
    valores_shap = explicador.shap_values(predictores)

    if isinstance(valores_shap, list):
        shap_clase_positiva = valores_shap[1]
    elif len(np.array(valores_shap).shape) == 3:
        shap_clase_positiva = valores_shap[:, :, 1]
    else:
        shap_clase_positiva = valores_shap

    print(f"  Forma de los valores SHAP: {shap_clase_positiva.shape}")

    print("\n[4/4] Generando figuras y tabla...")

    plt.figure()
    shap.summary_plot(
        shap_clase_positiva, predictores,
        plot_type='bar', show=False, max_display=25
    )
    plt.tight_layout()
    ruta_bar = os.path.join(CARPETA_FIGURAS, 'shap_bar_catboost_v4_2_larga.png')
    plt.savefig(ruta_bar, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Bar plot     : {ruta_bar}")

    plt.figure()
    shap.summary_plot(
        shap_clase_positiva, predictores,
        show=False, max_display=25
    )
    plt.tight_layout()
    ruta_beeswarm = os.path.join(CARPETA_FIGURAS, 'shap_beeswarm_catboost_v4_2_larga.png')
    plt.savefig(ruta_beeswarm, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Beeswarm     : {ruta_beeswarm}")

    importancia_shap = pd.DataFrame({
        'variable': predictores.columns,
        'shap_medio_absoluto': np.abs(shap_clase_positiva).mean(axis=0),
    }).sort_values('shap_medio_absoluto', ascending=False).reset_index(drop=True)

    ruta_tabla = os.path.join(CARPETA_TABLAS, 'importancia_shap_catboost_v4_2_larga.csv')
    importancia_shap.to_csv(ruta_tabla, index=False)
    print(f"  Tabla        : {ruta_tabla}")

    print("\n" + "-------------------------")
    print("IMPORTANCIA SHAP (ordenada)")
    print("-----------------------")
    print(importancia_shap.to_string(index=False))


if __name__ == '__main__':
    main()
