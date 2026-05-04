"""
Modelos finales — GridSearch real + calibración óptima + SHAP.

Flujo por modelo:

  RF (v4_6_24) — SE CALIBRA:
    1. División de todos los datos al 80/20 a nivel de paciente.
       El 20 % se reserva para calibración y NO participa en el GridSearch.
    2. GridSearchCV con CV interna agrupada (3 folds) sobre el 80 %.
    3. Se comparan sigmoid e isotonic sobre el 20 % (criterio: Brier score).
    4. Pickle final: CalibratedClassifierCV con el método ganador.
    5. SHAP: el mejor modelo (hiperparámetros ya conocidos) se reentrena
       sobre TODOS los datos y se pasa a TreeExplainer.

  CatBoost (v4p_3_12, v4l_12_48) — NO SE CALIBRA:
    1. GridSearchCV con CV interna agrupada (3 folds) sobre TODOS los datos.
    2. Pickle final: CatBoostClassifier con los mejores hiperparámetros.
    3. SHAP: el mismo modelo entrenado en todos los datos.

Nota sobre el GridSearch anterior (calibracion_gridsearch_tres_modelos_v4.py):
  Ese script estimaba el rendimiento dentro de una validación cruzada externa.
  Cada modelo entrenaba con ~64 % de los datos y se descartaba.
  Los hiperparámetros óptimos con el 64 % no tienen por qué coincidir
  con los óptimos al entrenar con el 80 % o el 100 %. Por eso se repite
  el GridSearch aquí sobre los datos reales de entrenamiento final.

Salida:
  modelos_finales/
    modelos/
      modelo_final_v4_6_24_rf.pkl          <- RF calibrado
      modelo_shap_v4_6_24_rf.pkl           <- RF base en todos los datos (solo SHAP)
      modelo_final_v4p_3_12_catboost.pkl   <- CatBoost sin calibrar
      modelo_final_v4l_12_48_catboost.pkl  <- CatBoost sin calibrar
    shap/
      figuras/  shap_bar_<ventana>.png  /  shap_beeswarm_<ventana>.png
      tablas/   importancia_shap_<ventana>.csv
    calibracion/
      resultado_seleccion_calibrador_v4_6_24_rf.csv
    gridsearch/
      mejores_parametros_finales_<ventana>.csv
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import (
    StratifiedGroupKFold,
    GridSearchCV,
    train_test_split,
)
from catboost import CatBoostClassifier


# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

DIR_SCRIPT = os.path.dirname(os.path.abspath(__file__))

CARPETA_MODELOS     = os.path.join(DIR_SCRIPT, 'modelos_finales', 'modelos')
CARPETA_FIGURAS     = os.path.join(DIR_SCRIPT, 'modelos_finales', 'shap', 'figuras')
CARPETA_TABLAS      = os.path.join(DIR_SCRIPT, 'modelos_finales', 'shap', 'tablas')
CARPETA_CALIBRACION = os.path.join(DIR_SCRIPT, 'modelos_finales', 'calibracion')
CARPETA_GRIDSEARCH  = os.path.join(DIR_SCRIPT, 'modelos_finales', 'gridsearch')

for carpeta in [CARPETA_MODELOS, CARPETA_FIGURAS, CARPETA_TABLAS,
                CARPETA_CALIBRACION, CARPETA_GRIDSEARCH]:
    os.makedirs(carpeta, exist_ok=True)

RANDOM_STATE             = 42
N_SPLITS_INTERNOS        = 3
PROPORCIONES_CALIBRACION = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
METODOS_CALIBRACION      = ['sigmoid', 'isotonic']
# Por cada proporción se ejecuta un GridSearch completo sobre el subconjunto
# modelo y se calibra sobre el subconjunto de calibración.
# Gana la combinación proporción × método con menor Brier score.


# ─────────────────────────────────────────────
# VARIABLES
# ─────────────────────────────────────────────

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
    'pf_min', 'spo2_min', 'rr_max',          # sin fio2_max (correlación alta con spo2)
    'ventilacion_invasiva_12h', 'gcs_min',
    'creatinina_max', 'diuresis_ml_kg_12h',
    'lactato_max', 'ph_min', 'bicarbonato_min',
    'bilirrubina_media', 'gpt_max',
    'tp_max', 'plaquetas_min',
    'leucocitos_min', 'hemoglobina_min',
    'glucemia_min', 'temp_min', 'sofa_max',
]


# ─────────────────────────────────────────────
# VENTANAS
# Los espacios de búsqueda son los mismos que en
# calibracion_gridsearch_tres_modelos_v4.py.
# calibrar=True  → división 80/20, GridSearch en 80 %, calibración en 20 %.
# calibrar=False → GridSearch en todos los datos, sin calibración.
# ─────────────────────────────────────────────

VENTANAS = {

    'v4_6_24_rf': {
        'ruta_csv':    r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4.csv',
        'variables':   VARIABLES_V4_6_24,
        'etiqueta':    'etiqueta_norad_6_24',
        'calibrar':    True,
        'max_display': 26,
        'n_jobs_grid': -1,
        'modelo': RandomForestClassifier(
            # class_weight y max_features fijados por ser ganadores en
            # prácticamente todos los folds del análisis previo.
            class_weight='balanced_subsample',
            max_features='sqrt',
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        'espacio': {
            # 3 × 3 × 3 = 27 combinaciones × 3 folds = 81 entrenamientos
            'n_estimators':     [300, 400, 500],
            'max_depth':        [None, 10, 20],
            'min_samples_leaf': [1, 2, 5],
        },
    },

    'v4p_3_12_catboost': {
        'ruta_csv':    r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4p.csv',
        'variables':   VARIABLES_V4P_3_12,
        'etiqueta':    'etiqueta_norad_3_12',
        'calibrar':    False,
        'max_display': 26,
        'n_jobs_grid': 1,
        'modelo': CatBoostClassifier(
            # learning_rate, iterations y bagging_temperature fijados por
            # ser constantes en todos los folds del análisis previo.
            iterations=500,
            learning_rate=0.01,
            bagging_temperature=0,
            loss_function='Logloss',
            eval_metric='AUC',
            random_seed=RANDOM_STATE,
            verbose=0,
            thread_count=-1,
        ),
        'espacio': {
            # 4 × 3 = 12 combinaciones × 3 folds = 36 entrenamientos
            'depth':       [4, 5, 6, 7],
            'l2_leaf_reg': [1, 5, 15],
        },
    },

    'v4l_12_48_catboost': {
        'ruta_csv':    r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4l.csv',
        'variables':   VARIABLES_V4L_12_48,
        'etiqueta':    'etiqueta_norad_12_48',
        'calibrar':    False,
        'max_display': 25,
        'n_jobs_grid': 1,
        'modelo': CatBoostClassifier(
            iterations=500,
            learning_rate=0.01,
            bagging_temperature=0,
            loss_function='Logloss',
            eval_metric='AUC',
            random_seed=RANDOM_STATE,
            verbose=0,
            thread_count=-1,
        ),
        'espacio': {
            'depth':       [4, 5, 6, 7],
            'l2_leaf_reg': [1, 5, 15],
        },
    },
}


# ─────────────────────────────────────────────
# FUNCIONES: CARGA
# ─────────────────────────────────────────────

def cargar_y_preparar(config):
    df = pd.read_csv(config['ruta_csv'])
    df = df.dropna(subset=['pf_max']).copy()

    x = df[config['variables']].copy()
    x['gender'] = (x['gender'] == 'M').astype(int)

    y           = df[config['etiqueta']].astype(int).copy()
    paciente_id = df['subject_id'].copy()

    return x, y, paciente_id


# ─────────────────────────────────────────────
# FUNCIONES: DIVISIÓN A NIVEL DE PACIENTE
# ─────────────────────────────────────────────

def dividir_por_paciente(x, y, paciente_id, proporcion_calibracion):
    """
    Divide x e y en subconjunto modelo y subconjunto calibración
    a nivel de paciente, estratificado por etiqueta máxima del paciente.
    Un mismo paciente no puede aparecer en los dos grupos.
    """
    etiqueta_paciente = pd.DataFrame({
        'subject_id': paciente_id.values,
        'etiqueta':   y.values,
    }).groupby('subject_id')['etiqueta'].max().reset_index()

    pacientes_modelo, pacientes_calibracion = train_test_split(
        etiqueta_paciente['subject_id'],
        test_size=proporcion_calibracion,
        random_state=RANDOM_STATE,
        stratify=etiqueta_paciente['etiqueta'],
    )

    mascara_modelo      = paciente_id.isin(pacientes_modelo)
    mascara_calibracion = paciente_id.isin(pacientes_calibracion)

    return (
        x.loc[mascara_modelo].copy(),
        y.loc[mascara_modelo].copy(),
        paciente_id.loc[mascara_modelo].copy(),
        x.loc[mascara_calibracion].copy(),
        y.loc[mascara_calibracion].copy(),
    )


# ─────────────────────────────────────────────
# FUNCIONES: GRIDSEARCH
# ─────────────────────────────────────────────

def ejecutar_gridsearch(nombre, config, x_entreno, y_entreno, grupos_entreno):
    """
    GridSearchCV con CV interna agrupada por paciente.
    Mismo diseño que en calibracion_gridsearch_tres_modelos_v4.py.
    """
    cv_interno = StratifiedGroupKFold(
        n_splits=N_SPLITS_INTERNOS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    busqueda = GridSearchCV(
        estimator=copy.deepcopy(config['modelo']),
        param_grid=config['espacio'],
        cv=cv_interno,
        scoring='roc_auc',
        n_jobs=config['n_jobs_grid'],
        refit=True,
    )

    busqueda.fit(x_entreno, y_entreno, groups=grupos_entreno)

    print(f'  Mejor AUC interno : {busqueda.best_score_:.4f}')
    print('  Mejores hiperparámetros:')
    for clave, valor in busqueda.best_params_.items():
        print(f'    {clave} = {valor}')

    # Guardar tabla de resultados del GridSearch
    tabla_gs = pd.DataFrame(busqueda.cv_results_)[
        ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
    ].sort_values('rank_test_score')
    ruta_gs = os.path.join(
        CARPETA_GRIDSEARCH,
        f'mejores_parametros_finales_{nombre}.csv',
    )
    tabla_gs.to_csv(ruta_gs, index=False)
    print(f'  Resultados GridSearch: {ruta_gs}')

    return busqueda.best_estimator_, busqueda.best_params_, busqueda.best_score_


# ─────────────────────────────────────────────
# FUNCIONES: CALIBRACIÓN
# ─────────────────────────────────────────────

def calibrar_prefit(modelo_entrenado, metodo, x_cal, y_cal):
    """
    Calibra un modelo ya entrenado. Compatible con scikit-learn antiguo
    (cv='prefit') y nuevo (FrozenEstimator).
    """
    try:
        from sklearn.frozen import FrozenEstimator
        cal = CalibratedClassifierCV(
            estimator=FrozenEstimator(modelo_entrenado),
            method=metodo,
        )
    except ImportError:
        cal = CalibratedClassifierCV(
            estimator=modelo_entrenado,
            method=metodo,
            cv='prefit',
        )
    cal.fit(x_cal, y_cal)
    return cal


# ─────────────────────────────────────────────
# FUNCIONES: SHAP
# ─────────────────────────────────────────────

def extraer_shap_positiva(valores_shap):
    arr = np.array(valores_shap)
    if isinstance(valores_shap, list):
        return valores_shap[1]
    if arr.ndim == 3:
        return arr[:, :, 1]
    return valores_shap


def calcular_y_guardar_shap(nombre, modelo_base, x, max_display):
    """
    Tree SHAP sobre el modelo base entrenado en TODOS los datos.
    TreeExplainer no es compatible con CalibratedClassifierCV.
    """
    print('  Calculando Tree SHAP...')
    explicador = shap.TreeExplainer(modelo_base)
    shap_vals  = extraer_shap_positiva(explicador.shap_values(x))
    print(f'  Forma matriz SHAP: {shap_vals.shape}')

    # Bar plot
    plt.figure()
    shap.summary_plot(shap_vals, x, plot_type='bar',
                      show=False, max_display=max_display)
    plt.tight_layout()
    ruta_bar = os.path.join(CARPETA_FIGURAS, f'shap_bar_{nombre}.png')
    plt.savefig(ruta_bar, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Bar plot  : {ruta_bar}')

    # Beeswarm
    plt.figure()
    shap.summary_plot(shap_vals, x, show=False, max_display=max_display)
    plt.tight_layout()
    ruta_beeswarm = os.path.join(CARPETA_FIGURAS, f'shap_beeswarm_{nombre}.png')
    plt.savefig(ruta_beeswarm, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Beeswarm  : {ruta_beeswarm}')

    # Tabla
    tabla = pd.DataFrame({
        'variable':            x.columns,
        'shap_medio_absoluto': np.abs(shap_vals).mean(axis=0),
    }).sort_values('shap_medio_absoluto', ascending=False).reset_index(drop=True)
    ruta_tabla = os.path.join(CARPETA_TABLAS, f'importancia_shap_{nombre}.csv')
    tabla.to_csv(ruta_tabla, index=False)
    print(f'  Tabla SHAP: {ruta_tabla}')

    print('\n  Importancia SHAP (ordenada):')
    print(tabla.to_string(index=False))


# ─────────────────────────────────────────────
# FUNCIONES: GUARDADO DE PICKLES
# ─────────────────────────────────────────────

def guardar_pickle(nombre_archivo, contenido):
    ruta = os.path.join(CARPETA_MODELOS, nombre_archivo)
    with open(ruta, 'wb') as f:
        pickle.dump(contenido, f)
    print(f'  Pickle guardado: {ruta}')


# ─────────────────────────────────────────────
# PROCESO PRINCIPAL POR VENTANA
# ─────────────────────────────────────────────

def procesar_ventana(nombre, config):
    print(f'\n{"=" * 65}')
    print(f'  {nombre}')
    print(f'{"=" * 65}')

    # ── 1. Cargar datos ──────────────────────────────────────────────
    print('\n[1] Cargando datos...')
    x, y, paciente_id = cargar_y_preparar(config)
    n_total     = len(x)
    prevalencia = float(y.mean())
    print(f'  Estancias       : {n_total}')
    print(f'  Positivos       : {y.sum()} ({100 * prevalencia:.2f}%)')
    print(f'  Pacientes únicos: {paciente_id.nunique()}')
    print(f'  Variables       : {x.shape[1]}')

    if config['calibrar']:
        # ── RF: para cada proporción → GridSearch + calibración ──────
        # Gana la combinación proporción × método con menor Brier score.

        n_combinaciones = len(PROPORCIONES_CALIBRACION) * len(METODOS_CALIBRACION)
        print(f'\n[2] Búsqueda sobre {n_combinaciones} combinaciones '
              f'({len(PROPORCIONES_CALIBRACION)} proporciones × '
              f'{len(METODOS_CALIBRACION)} métodos)...')

        resultados_busqueda = []

        for proporcion in PROPORCIONES_CALIBRACION:
            print(f'\n  -- Proporción {proporcion:.0%} --')
            x_modelo, y_modelo, grupos_modelo, x_calibracion, y_calibracion = \
                dividir_por_paciente(x, y, paciente_id, proporcion)
            print(f'  Modelo      : {len(x_modelo)} filas | {y_modelo.sum()} positivos')
            print(f'  Calibración : {len(x_calibracion)} filas | {y_calibracion.sum()} positivos')

            print(f'  GridSearchCV interno ({N_SPLITS_INTERNOS} folds)...')
            mejor_modelo_prop, mejores_params_prop, mejor_auc_prop = ejecutar_gridsearch(
                f'{nombre}_prop{int(proporcion * 100)}',
                config, x_modelo, y_modelo, grupos_modelo,
            )

            for metodo in METODOS_CALIBRACION:
                try:
                    calibrador = calibrar_prefit(
                        copy.deepcopy(mejor_modelo_prop), metodo,
                        x_calibracion, y_calibracion,
                    )
                    prob = calibrador.predict_proba(x_calibracion)[:, 1]
                    brier = brier_score_loss(y_calibracion, prob)
                    resultados_busqueda.append({
                        'proporcion':      proporcion,
                        'metodo':          metodo,
                        'brier_score':     brier,
                        'mejor_auc_grid':  mejor_auc_prop,
                        'mejores_params':  mejores_params_prop,
                        'calibrador':      calibrador,
                        'modelo_base':     mejor_modelo_prop,
                        'n_modelo':        len(x_modelo),
                        'n_calibracion':   len(x_calibracion),
                    })
                    print(f'    {metodo:10s}  →  Brier = {brier:.5f}')
                except Exception as error:
                    print(f'    {metodo:10s}  →  ERROR: {error}')

        if not resultados_busqueda:
            raise RuntimeError(f'Ninguna combinación funcionó para {nombre}.')

        ganador = min(resultados_busqueda, key=lambda d: d['brier_score'])
        print(f'\n  → GANADOR: proporcion={ganador["proporcion"]:.0%}  '
              f'metodo={ganador["metodo"]}  '
              f'Brier={ganador["brier_score"]:.5f}')

        # Guardar tabla comparativa completa
        tabla_comparacion = pd.DataFrame([
            {
                'proporcion':    r['proporcion'],
                'metodo':        r['metodo'],
                'n_modelo':      r['n_modelo'],
                'n_calibracion': r['n_calibracion'],
                'mejor_auc_grid': r['mejor_auc_grid'],
                'brier_score':   r['brier_score'],
            }
            for r in resultados_busqueda
        ]).sort_values('brier_score').reset_index(drop=True)

        ruta_comparacion = os.path.join(
            CARPETA_CALIBRACION,
            f'resultado_seleccion_calibrador_{nombre}.csv',
        )
        tabla_comparacion.to_csv(ruta_comparacion, index=False)
        print(f'\n  Tabla comparación completa: {ruta_comparacion}')
        print(tabla_comparacion.to_string(index=False))

        # Pickle modelo final (calibrado, para validación externa)
        guardar_pickle(
            f'modelo_final_{nombre}.pkl',
            {
                'modelo':                  ganador['calibrador'],
                'variables':               config['variables'],
                'etiqueta':                config['etiqueta'],
                'n_total':                 n_total,
                'prevalencia':             prevalencia,
                'mejores_hiperparametros': ganador['mejores_params'],
                'mejor_auc_grid_interno':  ganador['mejor_auc_grid'],
                'calibrado':               True,
                'proporcion_calibracion':  ganador['proporcion'],
                'metodo_calibracion':      ganador['metodo'],
                'brier_calibracion':       ganador['brier_score'],
                'n_modelo':                ganador['n_modelo'],
                'n_calibracion':           ganador['n_calibracion'],
                'fecha':                   datetime.datetime.now().isoformat(),
            }
        )

        # SHAP: reentrenar el modelo base ganador en TODOS los datos
        print(f'\n[3] Reentrenando modelo base ganador en todos los datos (para SHAP)...')
        modelo_base_todos = copy.deepcopy(config['modelo'])
        modelo_base_todos.set_params(**ganador['mejores_params'])
        modelo_base_todos.fit(x, y)
        print('  Listo.')

        print('\n[4] SHAP sobre modelo base (todos los datos)...')
        calcular_y_guardar_shap(nombre, modelo_base_todos, x, config['max_display'])

        guardar_pickle(
            f'modelo_shap_{nombre}.pkl',
            {
                'modelo':                  modelo_base_todos,
                'variables':               config['variables'],
                'etiqueta':                config['etiqueta'],
                'n_total':                 n_total,
                'prevalencia':             prevalencia,
                'mejores_hiperparametros': ganador['mejores_params'],
                'calibrado':               False,
                'fecha':                   datetime.datetime.now().isoformat(),
                'nota':                    'Modelo base en todos los datos. Solo para SHAP.',
            }
        )

    else:
        # ── CatBoost: GridSearch en todos los datos, sin calibración ─

        print('\n[2] GridSearchCV interno sobre todos los datos...')
        mejor_modelo, mejores_params, mejor_auc = ejecutar_gridsearch(
            nombre, config, x, y, paciente_id,
        )

        # Pickle modelo final (sin calibrar, para validación externa y SHAP)
        guardar_pickle(
            f'modelo_final_{nombre}.pkl',
            {
                'modelo':                mejor_modelo,
                'variables':             config['variables'],
                'etiqueta':              config['etiqueta'],
                'n_total':               n_total,
                'prevalencia':           prevalencia,
                'mejores_hiperparametros': mejores_params,
                'mejor_auc_grid_interno':  mejor_auc,
                'calibrado':             False,
                'metodo_calibracion':    None,
                'fecha':                 datetime.datetime.now().isoformat(),
                'nota': (
                    'CatBoost optimiza Logloss internamente. '
                    'La calibración empeora métricas (confirmado en análisis previo).'
                ),
            }
        )

        print('\n[3] SHAP sobre modelo final (todos los datos)...')
        calcular_y_guardar_shap(nombre, mejor_modelo, x, config['max_display'])

        # Para CatBoost el modelo_final y el modelo_shap son el mismo objeto.
        # Se guarda también con el nombre modelo_shap para consistencia.
        guardar_pickle(
            f'modelo_shap_{nombre}.pkl',
            {
                'modelo':                mejor_modelo,
                'variables':             config['variables'],
                'etiqueta':              config['etiqueta'],
                'n_total':               n_total,
                'prevalencia':           prevalencia,
                'mejores_hiperparametros': mejores_params,
                'calibrado':             False,
                'fecha':                 datetime.datetime.now().isoformat(),
                'nota':                  'Mismo modelo que modelo_final. Sin calibrar.',
            }
        )

    print(f'\n  Ventana {nombre} completada.')


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print('=' * 65)
    print('MODELOS FINALES — GridSearch real + calibración óptima + SHAP')
    print(f'Fecha             : {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print(f'Carpeta base      : {os.path.join(DIR_SCRIPT, "modelos_finales")}')
    print(f'Folds internos    : {N_SPLITS_INTERNOS}')
    print(f'Proporciones cal. : {PROPORCIONES_CALIBRACION}  (solo RF)')
    print(f'Métodos calibrac. : {METODOS_CALIBRACION}  (criterio: Brier score)')
    print('=' * 65)

    for nombre, config in VENTANAS.items():
        procesar_ventana(nombre, config)

    print('\n' + '=' * 65)
    print('PROCESO COMPLETADO.')
    print('Modelos validación externa → modelos_finales/modelos/modelo_final_*.pkl')
    print('Modelos para SHAP         → modelos_finales/modelos/modelo_shap_*.pkl')
    print('Figuras SHAP              → modelos_finales/shap/figuras/')
    print('Tablas SHAP               → modelos_finales/shap/tablas/')
    print('Resultados GridSearch     → modelos_finales/gridsearch/')
    print('Comparación calibradores  → modelos_finales/calibracion/')
    print('=' * 65)