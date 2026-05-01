"""
Calibración metodológicamente sólida de modelos v4/v4p/v4l.

Diseño:
  - Validación externa agrupada por paciente.
  - En cada fold externo:
      1. Se separa TEST externo.
      2. Dentro del entrenamiento externo se separa:
          - subconjunto modelo
          - subconjunto calibración
      3. En el subconjunto modelo se hace GridSearchCV con CV agrupada.
      4. Se entrena el mejor modelo.
      5. Se calibra el mejor modelo en el subconjunto de calibración.
      6. Se evalúa en TEST externo.
  - Se guardan métricas, predicciones, resumen, curva de calibración,
    histograma, ROC y Precision-Recall.

Notas:
  - No se fijan hiperparámetros finales a mano.
  - La calibración se aplica después de seleccionar hiperparámetros.
  - La calibración NO debe aumentar necesariamente el AUC.
"""

import warnings
warnings.filterwarnings('ignore')

import os
import time
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_curve,
    precision_recall_curve,
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier


# CONFIGURACIÓN

DIR_SCRIPT = os.path.dirname(os.path.abspath(__file__))
CARPETA_SALIDA = os.path.join(DIR_SCRIPT, 'salida_calibracion_gridsearch')
CARPETA_FIGURAS = os.path.join(CARPETA_SALIDA, 'figuras')
CARPETA_TABLAS = os.path.join(CARPETA_SALIDA, 'tablas')
os.makedirs(CARPETA_FIGURAS, exist_ok=True)
os.makedirs(CARPETA_TABLAS, exist_ok=True)

N_SPLITS_EXTERNOS = 5
N_SPLITS_INTERNOS = 3
TEST_SIZE_CALIBRACION = 0.20
RANDOM_STATE = 42

METODO_CALIBRACION = 'sigmoid'

# Solo visualización
N_BINS_CALIBRACION = 5
ESTRATEGIA_BINS = 'quantile'


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
    'glucemia_min',
    'temp_min',
    'sofa_max',
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
    'glucemia_min',
    'temp_min',
    'sofa_max',
]

VARIABLES_V4L_12_48 = [
    'anchor_age', 'gender', 'peso_kg', 'contador_estancia_uci',
    'map_min', 'hr_media',
    'pf_min', 'spo2_min', 'rr_max',
    'ventilacion_invasiva_12h', 'gcs_min',
    'creatinina_max', 'diuresis_ml_kg_12h',
    'lactato_max', 'ph_min', 'bicarbonato_min',
    'bilirrubina_media', 'gpt_max',
    'tp_max', 'plaquetas_min',
    'leucocitos_min', 'hemoglobina_min',
    'glucemia_min',
    'temp_min',
    'sofa_max',
]


VENTANAS = {
    'v4_6_24_rf': {
        'ruta_csv': r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4.csv',
        'variables': VARIABLES_V4_6_24,
        'etiqueta': 'etiqueta_norad_6_24',
        'modelo': RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        'espacio': {
            'n_estimators': [300, 400, 500, 600, 700, 850, 1000],
            'max_depth': [None, 5, 10, 20, 30],
            'min_samples_leaf': [1, 2, 5],
            'max_features': ['sqrt', 0.3, 0.5],
            'class_weight': ['balanced', 'balanced_subsample'],
        },
        'n_jobs_grid': -1,
    },

    'v4p_3_12_catboost': {
        'ruta_csv': r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4p.csv',
        'variables': VARIABLES_V4P_3_12,
        'etiqueta': 'etiqueta_norad_3_12',
        'modelo': CatBoostClassifier(
            loss_function='Logloss',
            eval_metric='AUC',
            random_seed=RANDOM_STATE,
            verbose=0,
            thread_count=-1,
        ),
        'espacio': {
            'iterations': [500, 1000],
            'depth': [4, 5, 6, 7],
            'learning_rate': [0.01, 0.03, 0.05, 0.1],
            'l2_leaf_reg': [1, 5, 15],
            'bagging_temperature': [0, 0.5, 1],
        },
        # CatBoost ya paraleliza internamente.
        'n_jobs_grid': 1,
    },

    'v4l_12_48_catboost': {
        'ruta_csv': r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4l.csv',
        'variables': VARIABLES_V4L_12_48,
        'etiqueta': 'etiqueta_norad_12_48',
        'modelo': CatBoostClassifier(
            loss_function='Logloss',
            eval_metric='AUC',
            random_seed=RANDOM_STATE,
            verbose=0,
            thread_count=-1,
        ),
        'espacio': {
            'iterations': [500, 1000],
            'depth': [4, 5, 6, 7],
            'learning_rate': [0.01, 0.03, 0.05, 0.1],
            'l2_leaf_reg': [1, 5, 15],
            'bagging_temperature': [0, 0.5, 1],
        },
        # CatBoost ya paraleliza internamente.
        'n_jobs_grid': 1,
    },
}


# CARGA Y PREPARACIÓN

def cargar_y_preparar(config):
    df = pd.read_csv(config['ruta_csv'])
    df = df.dropna(subset=['pf_max']).copy()

    predictores = df[config['variables']].copy()
    predictores['gender'] = (predictores['gender'] == 'M').astype(int)

    etiqueta = df[config['etiqueta']].astype(int).copy()
    paciente_id = df['subject_id'].copy()

    return predictores, etiqueta, paciente_id


def dividir_entrenamiento_calibracion(predictores, etiqueta, paciente_id):
    pacientes = pd.DataFrame({
        'subject_id': paciente_id.values,
        'etiqueta': etiqueta.values,
    }).groupby('subject_id')['etiqueta'].max().reset_index()

    pacientes_modelo, pacientes_calibracion = train_test_split(
        pacientes['subject_id'],
        test_size=TEST_SIZE_CALIBRACION,
        random_state=RANDOM_STATE,
        stratify=pacientes['etiqueta'],
    )

    mascara_modelo = paciente_id.isin(pacientes_modelo)
    mascara_calibracion = paciente_id.isin(pacientes_calibracion)

    return mascara_modelo, mascara_calibracion


def crear_calibrador_prefit(modelo_entrenado):
    try:
        from sklearn.frozen import FrozenEstimator
        calibrador = CalibratedClassifierCV(
            estimator=FrozenEstimator(modelo_entrenado),
            method=METODO_CALIBRACION,
        )
    except Exception:
        calibrador = CalibratedClassifierCV(
            estimator=modelo_entrenado,
            method=METODO_CALIBRACION,
            cv='prefit',
        )

    return calibrador


def calcular_ece(y_real, probabilidad, n_bins=10):
    y_real = np.asarray(y_real)
    probabilidad = np.asarray(probabilidad)

    cortes = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        limite_inferior = cortes[i]
        limite_superior = cortes[i + 1]

        if i == n_bins - 1:
            mascara = (probabilidad >= limite_inferior) & (probabilidad <= limite_superior)
        else:
            mascara = (probabilidad >= limite_inferior) & (probabilidad < limite_superior)

        if mascara.sum() == 0:
            continue

        confianza_media = probabilidad[mascara].mean()
        frecuencia_observada = y_real[mascara].mean()
        peso = mascara.mean()

        ece += peso * abs(frecuencia_observada - confianza_media)

    return ece


def calcular_metricas(y_real, probabilidad):
    return {
        'auc': roc_auc_score(y_real, probabilidad),
        'auprc': average_precision_score(y_real, probabilidad),
        'brier': brier_score_loss(y_real, probabilidad),
        'logloss': log_loss(y_real, probabilidad, labels=[0, 1]),
        'ece': calcular_ece(y_real, probabilidad, n_bins=10),
    }


# FIGURAS Y TABLAS

def guardar_tabla_calibracion(nombre_ventana, predicciones):
    df = predicciones.copy()
    df['decil_calibrado'] = pd.qcut(
        df['probabilidad_calibrada'],
        q=10,
        labels=False,
        duplicates='drop',
    ) + 1

    tabla = df.groupby('decil_calibrado').agg(
        n=('y_real', 'size'),
        positivos=('y_real', 'sum'),
        probabilidad_media_calibrada=('probabilidad_calibrada', 'mean'),
        probabilidad_media_sin_calibrar=('probabilidad_sin_calibrar', 'mean'),
        frecuencia_observada=('y_real', 'mean'),
    ).reset_index()

    ruta = os.path.join(CARPETA_TABLAS, f'tabla_calibracion_deciles_{nombre_ventana}.csv')
    tabla.to_csv(ruta, index=False)

    return ruta


def guardar_curva_calibracion(nombre_ventana, predicciones):
    frac_pos_sin, media_pred_sin = calibration_curve(
        predicciones['y_real'],
        predicciones['probabilidad_sin_calibrar'],
        n_bins=N_BINS_CALIBRACION,
        strategy=ESTRATEGIA_BINS,
    )
    frac_pos_cal, media_pred_cal = calibration_curve(
        predicciones['y_real'],
        predicciones['probabilidad_calibrada'],
        n_bins=N_BINS_CALIBRACION,
        strategy=ESTRATEGIA_BINS,
    )

    prob_max = max(
        predicciones['probabilidad_sin_calibrar'].max(),
        predicciones['probabilidad_calibrada'].max(),
        frac_pos_sin.max() if len(frac_pos_sin) else 0,
        frac_pos_cal.max() if len(frac_pos_cal) else 0,
    )
    limite = min(1.0, prob_max * 1.1)

    plt.figure(figsize=(6, 6))
    plt.plot([0, limite], [0, limite], linestyle='--', label='Perfecta', linewidth=2)
    plt.plot(media_pred_sin, frac_pos_sin, marker='o', label='Sin calibrar', linewidth=2)
    plt.plot(media_pred_cal, frac_pos_cal, marker='o', label='Calibrada', linewidth=2)

    plt.xlim(0, limite)
    plt.ylim(0, limite)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Probabilidad predicha')
    plt.ylabel('Fracción observada de positivos')
    plt.title(f'Curva de calibración - {nombre_ventana}')
    plt.legend(loc='upper left')
    plt.tight_layout()

    ruta = os.path.join(CARPETA_FIGURAS, f'calibracion_{nombre_ventana}.png')
    plt.savefig(ruta, dpi=300, bbox_inches='tight')
    plt.close()

    return ruta


def guardar_histograma_probabilidades(nombre_ventana, predicciones):
    prob_max = max(
        predicciones['probabilidad_sin_calibrar'].max(),
        predicciones['probabilidad_calibrada'].max(),
    )
    limite = min(1.0, prob_max * 1.05)
    bins = np.linspace(0, limite, 25)

    plt.figure(figsize=(7, 5))
    plt.hist(predicciones['probabilidad_sin_calibrar'], bins=bins, alpha=0.6, label='Sin calibrar')
    plt.hist(predicciones['probabilidad_calibrada'], bins=bins, alpha=0.6, label='Calibrada')
    plt.xlim(0, limite)
    plt.xlabel('Probabilidad predicha')
    plt.ylabel('Frecuencia')
    plt.title(f'Histograma de probabilidades - {nombre_ventana}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    ruta = os.path.join(CARPETA_FIGURAS, f'histograma_probabilidades_{nombre_ventana}.png')
    plt.savefig(ruta, dpi=300, bbox_inches='tight')
    plt.close()

    return ruta


def guardar_curva_roc(nombre_ventana, predicciones):
    fpr_sin, tpr_sin, _ = roc_curve(
        predicciones['y_real'],
        predicciones['probabilidad_sin_calibrar'],
    )
    fpr_cal, tpr_cal, _ = roc_curve(
        predicciones['y_real'],
        predicciones['probabilidad_calibrada'],
    )

    auc_sin = roc_auc_score(predicciones['y_real'], predicciones['probabilidad_sin_calibrar'])
    auc_cal = roc_auc_score(predicciones['y_real'], predicciones['probabilidad_calibrada'])

    plt.figure(figsize=(6, 6))
    plt.plot(fpr_sin, tpr_sin, label=f'Sin calibrar AUC={auc_sin:.3f}', linewidth=2)
    plt.plot(fpr_cal, tpr_cal, label=f'Calibrada AUC={auc_cal:.3f}', linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, alpha=0.3)
    plt.xlabel('1 - Especificidad')
    plt.ylabel('Sensibilidad')
    plt.title(f'Curva ROC - {nombre_ventana}')
    plt.legend(loc='lower right')
    plt.tight_layout()

    ruta = os.path.join(CARPETA_FIGURAS, f'roc_{nombre_ventana}.png')
    plt.savefig(ruta, dpi=300, bbox_inches='tight')
    plt.close()

    return ruta


def guardar_curva_precision_recall(nombre_ventana, predicciones):
    precision_sin, recall_sin, _ = precision_recall_curve(
        predicciones['y_real'],
        predicciones['probabilidad_sin_calibrar'],
    )
    precision_cal, recall_cal, _ = precision_recall_curve(
        predicciones['y_real'],
        predicciones['probabilidad_calibrada'],
    )

    auprc_sin = average_precision_score(
        predicciones['y_real'],
        predicciones['probabilidad_sin_calibrar'],
    )
    auprc_cal = average_precision_score(
        predicciones['y_real'],
        predicciones['probabilidad_calibrada'],
    )

    prevalencia = predicciones['y_real'].mean()

    plt.figure(figsize=(6, 6))
    plt.plot(recall_sin, precision_sin, label=f'Sin calibrar AUPRC={auprc_sin:.3f}', linewidth=2)
    plt.plot(recall_cal, precision_cal, label=f'Calibrada AUPRC={auprc_cal:.3f}', linewidth=2)
    plt.axhline(prevalencia, linestyle='--', color='gray', label=f'Azar = {prevalencia:.3f}')
    plt.xlim(0, 1)
    plt.ylim(0, max(precision_sin.max(), precision_cal.max()) * 1.05)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Recall / Sensibilidad')
    plt.ylabel('Precision / VPP')
    plt.title(f'Curva Precision-Recall - {nombre_ventana}')
    plt.legend(loc='upper right')
    plt.tight_layout()

    ruta = os.path.join(CARPETA_FIGURAS, f'precision_recall_{nombre_ventana}.png')
    plt.savefig(ruta, dpi=300, bbox_inches='tight')
    plt.close()

    return ruta


# ENTRENAMIENTO + GRID + CALIBRACIÓN

def calibrar_ventana(nombre_ventana, config):
    print('-------------')
    print(f'CALIBRACIÓN CON GRIDSEARCH — {nombre_ventana}')
    print('------------')

    print('\n[1/5] Cargando datos...')
    predictores, etiqueta, paciente_id = cargar_y_preparar(config)

    print(f'  Estancias: {len(predictores)}')
    print(f'  Positivos: {etiqueta.sum()} ({100 * etiqueta.mean():.2f}%)')
    print(f'  Pacientes únicos: {paciente_id.nunique()}')
    print(f'  Variables: {predictores.shape[1]}')

    cv_externo = StratifiedGroupKFold(
        n_splits=N_SPLITS_EXTERNOS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    predicciones_todos = []
    metricas_todos = []
    mejores_parametros_todos = []

    print(f'\n[2/5] Validación externa agrupada + GridSearchCV + calibración ({METODO_CALIBRACION})...')

    for fold, (idx_entreno_ext, idx_test_ext) in enumerate(
        cv_externo.split(predictores, etiqueta, groups=paciente_id),
        start=1,
    ):
        print('\n' + '-' * 70)
        print(f'Fold externo {fold}/{N_SPLITS_EXTERNOS}')
        tiempo_inicio = time.time()

        x_entreno_ext = predictores.iloc[idx_entreno_ext].copy()
        y_entreno_ext = etiqueta.iloc[idx_entreno_ext].copy()
        grupos_entreno_ext = paciente_id.iloc[idx_entreno_ext].copy()

        x_test = predictores.iloc[idx_test_ext].copy()
        y_test = etiqueta.iloc[idx_test_ext].copy()
        grupos_test = paciente_id.iloc[idx_test_ext].copy()

        mascara_modelo, mascara_calibracion = dividir_entrenamiento_calibracion(
            x_entreno_ext,
            y_entreno_ext,
            grupos_entreno_ext,
        )

        x_modelo = x_entreno_ext.loc[mascara_modelo].copy()
        y_modelo = y_entreno_ext.loc[mascara_modelo].copy()
        grupos_modelo = grupos_entreno_ext.loc[mascara_modelo].copy()

        x_calibracion = x_entreno_ext.loc[mascara_calibracion].copy()
        y_calibracion = y_entreno_ext.loc[mascara_calibracion].copy()

        print(f'  Modelo/grid:  {len(x_modelo)} filas | {y_modelo.sum()} positivos')
        print(f'  Calibración:  {len(x_calibracion)} filas | {y_calibracion.sum()} positivos')
        print(f'  Test externo: {len(x_test)} filas | {y_test.sum()} positivos')

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

        print('  Ejecutando GridSearchCV interno...')
        busqueda.fit(x_modelo, y_modelo, groups=grupos_modelo)

        modelo_sin_calibrar = busqueda.best_estimator_

        print(f'  Mejor AUC grid interno: {busqueda.best_score_:.4f}')
        print('  Mejores hiperparámetros:')
        for clave, valor in busqueda.best_params_.items():
            print(f'    {clave} = {valor}')

        calibrador = crear_calibrador_prefit(modelo_sin_calibrar)
        calibrador.fit(x_calibracion, y_calibracion)

        probabilidad_sin_calibrar = modelo_sin_calibrar.predict_proba(x_test)[:, 1]
        probabilidad_calibrada = calibrador.predict_proba(x_test)[:, 1]

        metricas_sin = calcular_metricas(y_test, probabilidad_sin_calibrar)
        metricas_cal = calcular_metricas(y_test, probabilidad_calibrada)

        metricas_fold = {
            'ventana': nombre_ventana,
            'fold': fold,
            'n_modelo': len(x_modelo),
            'positivos_modelo': int(y_modelo.sum()),
            'n_calibracion': len(x_calibracion),
            'positivos_calibracion': int(y_calibracion.sum()),
            'n_test': len(y_test),
            'positivos_test': int(y_test.sum()),
            'prevalencia_test': float(y_test.mean()),
            'mejor_auc_grid_interno': busqueda.best_score_,
            'auc_sin_calibrar': metricas_sin['auc'],
            'auc_calibrado': metricas_cal['auc'],
            'auprc_sin_calibrar': metricas_sin['auprc'],
            'auprc_calibrado': metricas_cal['auprc'],
            'brier_sin_calibrar': metricas_sin['brier'],
            'brier_calibrado': metricas_cal['brier'],
            'logloss_sin_calibrar': metricas_sin['logloss'],
            'logloss_calibrado': metricas_cal['logloss'],
            'ece_sin_calibrar': metricas_sin['ece'],
            'ece_calibrado': metricas_cal['ece'],
            'tiempo_min': (time.time() - tiempo_inicio) / 60,
        }
        metricas_todos.append(metricas_fold)

        parametros_fold = {
            'ventana': nombre_ventana,
            'fold': fold,
            'mejor_auc_grid_interno': busqueda.best_score_,
        }
        parametros_fold.update(busqueda.best_params_)
        mejores_parametros_todos.append(parametros_fold)

        pred_fold = pd.DataFrame({
            'ventana': nombre_ventana,
            'fold': fold,
            'subject_id': grupos_test.values,
            'y_real': y_test.values,
            'probabilidad_sin_calibrar': probabilidad_sin_calibrar,
            'probabilidad_calibrada': probabilidad_calibrada,
        })
        predicciones_todos.append(pred_fold)

        print(f'  AUC sin calibrar: {metricas_fold["auc_sin_calibrar"]:.4f}')
        print(f'  AUC calibrado:    {metricas_fold["auc_calibrado"]:.4f}')
        print(f'  AUPRC sin cal.:   {metricas_fold["auprc_sin_calibrar"]:.4f}')
        print(f'  AUPRC calibrado:  {metricas_fold["auprc_calibrado"]:.4f}')
        print(f'  Brier sin cal.:   {metricas_fold["brier_sin_calibrar"]:.4f}')
        print(f'  Brier calibrado:  {metricas_fold["brier_calibrado"]:.4f}')
        print(f'  LogLoss sin cal.: {metricas_fold["logloss_sin_calibrar"]:.4f}')
        print(f'  LogLoss calibr.:  {metricas_fold["logloss_calibrado"]:.4f}')
        print(f'  ECE sin cal.:     {metricas_fold["ece_sin_calibrar"]:.4f}')
        print(f'  ECE calibrado:    {metricas_fold["ece_calibrado"]:.4f}')
        print(f'  Tiempo fold:      {metricas_fold["tiempo_min"]:.2f} min')

    print('\n[3/5] Guardando tablas...')

    predicciones = pd.concat(predicciones_todos, ignore_index=True)
    metricas = pd.DataFrame(metricas_todos)
    mejores_parametros = pd.DataFrame(mejores_parametros_todos)

    ruta_predicciones = os.path.join(CARPETA_TABLAS, f'predicciones_calibracion_{nombre_ventana}.csv')
    ruta_metricas = os.path.join(CARPETA_TABLAS, f'metricas_calibracion_{nombre_ventana}.csv')
    ruta_parametros = os.path.join(CARPETA_TABLAS, f'mejores_parametros_{nombre_ventana}.csv')

    predicciones.to_csv(ruta_predicciones, index=False)
    metricas.to_csv(ruta_metricas, index=False)
    mejores_parametros.to_csv(ruta_parametros, index=False)

    columnas_resumen = [
        'mejor_auc_grid_interno',
        'auc_sin_calibrar', 'auc_calibrado',
        'auprc_sin_calibrar', 'auprc_calibrado',
        'brier_sin_calibrar', 'brier_calibrado',
        'logloss_sin_calibrar', 'logloss_calibrado',
        'ece_sin_calibrar', 'ece_calibrado',
    ]
    resumen = metricas[columnas_resumen].agg(['mean', 'std']).T.reset_index()
    resumen.columns = ['metrica', 'media', 'desviacion_estandar']

    ruta_resumen = os.path.join(CARPETA_TABLAS, f'resumen_calibracion_{nombre_ventana}.csv')
    ruta_tabla_deciles = guardar_tabla_calibracion(nombre_ventana, predicciones)
    resumen.to_csv(ruta_resumen, index=False)

    print('\n[4/5] Generando figuras...')
    ruta_calibracion = guardar_curva_calibracion(nombre_ventana, predicciones)
    ruta_histograma = guardar_histograma_probabilidades(nombre_ventana, predicciones)
    ruta_roc = guardar_curva_roc(nombre_ventana, predicciones)
    ruta_pr = guardar_curva_precision_recall(nombre_ventana, predicciones)

    print('\n[5/5] Resumen')
    print(resumen.to_string(index=False))

    print('\nArchivos guardados:')
    print(f'  Predicciones:       {ruta_predicciones}')
    print(f'  Métricas por fold:  {ruta_metricas}')
    print(f'  Mejores params:     {ruta_parametros}')
    print(f'  Resumen:            {ruta_resumen}')
    print(f'  Tabla deciles:      {ruta_tabla_deciles}')
    print(f'  Calibración:        {ruta_calibracion}')
    print(f'  Histograma:         {ruta_histograma}')
    print(f'  ROC:                {ruta_roc}')
    print(f'  Precision-Recall:   {ruta_pr}')

    return metricas, predicciones, mejores_parametros


# MAIN

if __name__ == '__main__':
    print('=' * 70)
    print('CALIBRACIÓN CON GRIDSEARCH DE MODELOS v4/v4p/v4l')
    print('=' * 70)
    print(f'Carpeta de salida: {CARPETA_SALIDA}')
    print(f'Método de calibración: {METODO_CALIBRACION}')
    print(f'Test size calibración: {TEST_SIZE_CALIBRACION}')
    print()

    metricas_globales = []
    parametros_globales = []

    for nombre_ventana, config in VENTANAS.items():
        metricas, _, mejores_parametros = calibrar_ventana(nombre_ventana, config)
        metricas_globales.append(metricas)
        parametros_globales.append(mejores_parametros)

    metricas_globales = pd.concat(metricas_globales, ignore_index=True)
    parametros_globales = pd.concat(parametros_globales, ignore_index=True)

    ruta_metricas_globales = os.path.join(CARPETA_TABLAS, 'metricas_calibracion_todas_las_ventanas.csv')
    ruta_parametros_globales = os.path.join(CARPETA_TABLAS, 'mejores_parametros_todas_las_ventanas.csv')

    metricas_globales.to_csv(ruta_metricas_globales, index=False)
    parametros_globales.to_csv(ruta_parametros_globales, index=False)

    print('\n' + '----------------')
    print('CALIBRACIÓN COMPLETADA')
    print(f'Métricas globales:      {ruta_metricas_globales}')
    print(f'Mejores params globales:{ruta_parametros_globales}')
    print('-------------')
