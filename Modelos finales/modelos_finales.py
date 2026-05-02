"""
SHAP de los tres modelos finales — hiperparámetros fijos.

Entrena cada modelo sobre el conjunto completo con los hiperparámetros
elegidos tras el análisis de corridas previas (sin GridSearch).
Calcula Tree SHAP, genera bar plot + beeswarm y guarda tabla CSV.
Guarda también el modelo entrenado en pickle para la validación externa.

Salida (carpeta al lado del script):
  shap/
    figuras/
      shap_bar_v4_6_24_rf.png
      shap_beeswarm_v4_6_24_rf.png
      shap_bar_v4p_3_12_catboost.png
      ...
    tablas/
      importancia_shap_v4_6_24_rf.csv
      ...
    modelos/
      modelo_shap_v4_6_24_rf.pkl        ← RF base (sin calibrar, para SHAP)
      modelo_shap_v4p_3_12_catboost.pkl
      modelo_shap_v4l_12_48_catboost.pkl
"""

import warnings
warnings.filterwarnings('ignore')

import os
import copy
import pickle
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier


# CONFIGURACIÓN

DIR_SCRIPT = os.path.dirname(os.path.abspath(__file__))
CARPETA_FIGURAS = os.path.join(DIR_SCRIPT, 'shap', 'figuras')
CARPETA_TABLAS  = os.path.join(DIR_SCRIPT, 'shap', 'tablas')
CARPETA_MODELOS = os.path.join(DIR_SCRIPT, 'shap', 'modelos')

for carpeta in [CARPETA_FIGURAS, CARPETA_TABLAS, CARPETA_MODELOS]:
    os.makedirs(carpeta, exist_ok=True)

RANDOM_STATE = 42


# VARIABLES

VARIABLES_V4_6_24 = [
    'anchor_age', 'gender', 'peso_kg', 'contador_estancia_uci',
    'map_min', 'hr_media',
    'pf_min', 'spo2_min', 'fio2_max', 'rr_max',
    'ventilacion_invasiva_6h', 'gcs_min',
    'creatinina_max', 'diuresis_ml_kg_6h',
    'lactato_max', 'ph_min', 'bicarbonato_min',
    'bilirrubina_media', 'gpt_max',
    'tp_max', 'plaquetas_min',
    'leucocitos_min', 'hemoglobina_min',
    'glucemia_min', 'temp_min', 'sofa_max',
]

VARIABLES_V4P_3_12 = [
    'anchor_age', 'gender', 'peso_kg', 'contador_estancia_uci',
    'map_min', 'hr_media',
    'pf_min', 'spo2_min', 'fio2_max', 'rr_max',
    'ventilacion_invasiva_3h', 'gcs_min',
    'creatinina_max', 'diuresis_ml_kg_3h',
    'lactato_max', 'ph_min', 'bicarbonato_min',
    'bilirrubina_media', 'gpt_max',
    'tp_max', 'plaquetas_min',
    'leucocitos_min', 'hemoglobina_min',
    'glucemia_min', 'temp_min', 'sofa_max',
]

VARIABLES_V4L_12_48 = [
    'anchor_age', 'gender', 'peso_kg', 'contador_estancia_uci',
    'map_min', 'hr_media',
    'pf_min', 'spo2_min', 'rr_max',          # sin fio2_max (correlación alta)
    'ventilacion_invasiva_12h', 'gcs_min',
    'creatinina_max', 'diuresis_ml_kg_12h',
    'lactato_max', 'ph_min', 'bicarbonato_min',
    'bilirrubina_media', 'gpt_max',
    'tp_max', 'plaquetas_min',
    'leucocitos_min', 'hemoglobina_min',
    'glucemia_min', 'temp_min', 'sofa_max',
]


# VENTANAS
# Hiperparámetros fijos elegidos tras las corridas de calibración + SHAP previas.
# RF:       balanced_subsample, sqrt, leaf=5, depth=None, 400 árboles.
# CatBoost: lr=0.01, iter=500 (conservador, siempre elegido por el grid).

VENTANAS = {
    'v4_6_24_rf': {
        'ruta_csv':  r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4.csv',
        'variables': VARIABLES_V4_6_24,
        'etiqueta':  'etiqueta_norad_6_24',
        'modelo': RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=5,
            max_features='sqrt',
            class_weight='balanced_subsample',
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        'max_display': 26,
    },

    'v4p_3_12_catboost': {
        'ruta_csv':  r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4p.csv',
        'variables': VARIABLES_V4P_3_12,
        'etiqueta':  'etiqueta_norad_3_12',
        'modelo': CatBoostClassifier(
            iterations=500,
            depth=5,
            learning_rate=0.01,
            l2_leaf_reg=5,
            bagging_temperature=0,
            loss_function='Logloss',
            eval_metric='AUC',
            random_seed=RANDOM_STATE,
            verbose=0,
            thread_count=-1,
        ),
        'max_display': 26,
    },

    'v4l_12_48_catboost': {
        'ruta_csv':  r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4l.csv',
        'variables': VARIABLES_V4L_12_48,
        'etiqueta':  'etiqueta_norad_12_48',
        'modelo': CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.01,
            l2_leaf_reg=5,
            bagging_temperature=0,
            loss_function='Logloss',
            eval_metric='AUC',
            random_seed=RANDOM_STATE,
            verbose=0,
            thread_count=-1,
        ),
        'max_display': 25,
    },
}


# FUNCIONES

def cargar_y_preparar(config):
    df = pd.read_csv(config['ruta_csv'])
    df = df.dropna(subset=['pf_max']).copy()

    x = df[config['variables']].copy()
    x['gender'] = (x['gender'] == 'M').astype(int)

    y = df[config['etiqueta']].astype(int).copy()
    paciente_id = df['subject_id'].copy()

    return x, y, paciente_id


def extraer_shap_positiva(valores_shap):
    """Devuelve siempre la matriz SHAP de la clase positiva (n_muestras, n_features)."""
    arr = np.array(valores_shap)
    if isinstance(valores_shap, list):
        return valores_shap[1]
    if arr.ndim == 3:          # (n, features, clases)
        return arr[:, :, 1]
    return valores_shap        # ya es (n, features)


def guardar_figuras(nombre, shap_vals, predictores, max_display):
    # Bar plot
    plt.figure()
    shap.summary_plot(shap_vals, predictores, plot_type='bar',
                      show=False, max_display=max_display)
    plt.tight_layout()
    ruta = os.path.join(CARPETA_FIGURAS, f'shap_bar_{nombre}.png')
    plt.savefig(ruta, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Bar plot  : {ruta}')

    # Beeswarm
    plt.figure()
    shap.summary_plot(shap_vals, predictores,
                      show=False, max_display=max_display)
    plt.tight_layout()
    ruta = os.path.join(CARPETA_FIGURAS, f'shap_beeswarm_{nombre}.png')
    plt.savefig(ruta, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Beeswarm  : {ruta}')


def guardar_tabla(nombre, shap_vals, predictores):
    tabla = pd.DataFrame({
        'variable': predictores.columns,
        'shap_medio_absoluto': np.abs(shap_vals).mean(axis=0),
    }).sort_values('shap_medio_absoluto', ascending=False).reset_index(drop=True)

    ruta = os.path.join(CARPETA_TABLAS, f'importancia_shap_{nombre}.csv')
    tabla.to_csv(ruta, index=False)
    print(f'  Tabla     : {ruta}')

    return tabla


def guardar_pkl(nombre, modelo, config, n_total, prevalencia):
    obj = {
        'modelo': modelo,
        'variables': config['variables'],
        'etiqueta': config['etiqueta'],
        'n_total': n_total,
        'prevalencia': prevalencia,
        'hiperparametros': modelo.get_params(),
        'fecha': datetime.datetime.now().isoformat(),
    }
    ruta = os.path.join(CARPETA_MODELOS, f'modelo_shap_{nombre}.pkl')
    with open(ruta, 'wb') as f:
        pickle.dump(obj, f)
    print(f'  Pickle    : {ruta}')


def procesar_ventana(nombre, config):
    print(f'\n{"=" * 60}')
    print(f'  {nombre}')
    print(f'{"=" * 60}')

    print('\n[1/4] Cargando datos...')
    x, y, paciente_id = cargar_y_preparar(config)
    print(f'  Estancias: {len(x)}')
    print(f'  Positivos: {y.sum()} ({100 * y.mean():.2f}%)')
    print(f'  Pacientes únicos: {paciente_id.nunique()}')
    print(f'  Variables: {x.shape[1]}')
    print(f'  Hiperparámetros: {config["modelo"].get_params()}')

    print('\n[2/4] Entrenando modelo (hiperparámetros fijos)...')
    modelo = copy.deepcopy(config['modelo'])
    modelo.fit(x, y)
    print('  Listo.')

    print('\n[3/4] Calculando SHAP (Tree SHAP)...')
    explicador = shap.TreeExplainer(modelo)
    shap_vals = extraer_shap_positiva(explicador.shap_values(x))
    print(f'  Forma valores SHAP: {shap_vals.shape}')

    print('\n[4/4] Guardando figuras, tabla y pickle...')
    guardar_figuras(nombre, shap_vals, x, config['max_display'])
    tabla = guardar_tabla(nombre, shap_vals, x)
    guardar_pkl(nombre, modelo, config, len(x), float(y.mean()))

    print('\nIMPORTANCIA SHAP (ordenada):')
    print(tabla.to_string(index=False))


# MAIN

if __name__ == '__main__':
    print(f'Carpeta de salida: {os.path.join(DIR_SCRIPT, "shap")}')
    print(f'Fecha: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}')

    for nombre, config in VENTANAS.items():
        procesar_ventana(nombre, config)

    print(f'\n{"=" * 60}')
    print('SHAP COMPLETADO — tres modelos procesados.')
    print(f'{"=" * 60}')