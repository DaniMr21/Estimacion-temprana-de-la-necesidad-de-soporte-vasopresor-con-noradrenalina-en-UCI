"""
Búsqueda automática de configuración de calibración para CatBoost v4p (3-12h).

Objetivo:
  - Mantener fijo el modelo CatBoost final de la ventana corta.
  - Probar automáticamente combinaciones de:
      * tamaño del subconjunto de calibración
      * método de calibración
  - Evaluar con validación externa agrupada por paciente.
  - Elegir por Brier score y LogLoss.
  - Guardar tablas y figuras.

Nota:
  - N_BINS_CALIBRACION y ESTRATEGIA_BINS afectan a la figura, no al modelo.
  - No se optimizan automáticamente para evitar "optimizar la gráfica".
"""

import warnings
warnings.filterwarnings('ignore')

import os
import time
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold, train_test_split
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


# CONFIGURACIÓN

RUTA_CSV = r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4p.csv'

DIR_SCRIPT = os.path.dirname(os.path.abspath(__file__))
CARPETA_SALIDA = os.path.join(DIR_SCRIPT, 'salida_calibracion_catboost_v4p_grid')
CARPETA_FIGURAS = os.path.join(CARPETA_SALIDA, 'figuras')
CARPETA_TABLAS = os.path.join(CARPETA_SALIDA, 'tablas')
os.makedirs(CARPETA_FIGURAS, exist_ok=True)
os.makedirs(CARPETA_TABLAS, exist_ok=True)

N_SPLITS_EXTERNOS = 5
RANDOM_STATE = 42

TEST_SIZE_CALIBRACION_GRID = [0.20, 0.25, 0.30]
METODO_CALIBRACION_GRID = ['sigmoid', 'isotonic']

# Solo visualización
N_BINS_CALIBRACION = 5
ESTRATEGIA_BINS = 'quantile'


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


# Hiperparámetros conservadores del grid previo de la ventana corta.
# En tus salidas previas, CatBoost corto tendía a elegir:
# iterations=500, learning_rate=0.01, bagging_temperature=0,
# con depth entre 4-6 y l2_leaf_reg entre 1-15.
# Aquí se usa la configuración ganadora original del SHAP v4p:
# depth=4, l2_leaf_reg=15.
MODELO_BASE = CatBoostClassifier(
    iterations=500,
    depth=4,
    learning_rate=0.01,
    l2_leaf_reg=15,
    bagging_temperature=0,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=42,
    verbose=0,
    thread_count=-1,
)


# FUNCIONES

def cargar_y_preparar():
    df = pd.read_csv(RUTA_CSV)
    df = df.dropna(subset=['pf_max']).copy()

    predictores = df[VARIABLES_PREDICTORAS].copy()
    predictores['gender'] = (predictores['gender'] == 'M').astype(int)

    etiqueta = df['etiqueta_norad_3_12'].astype(int).copy()
    paciente_id = df['subject_id'].copy()

    return predictores, etiqueta, paciente_id


def dividir_entrenamiento_calibracion(predictores, etiqueta, paciente_id, test_size_calibracion):
    pacientes = pd.DataFrame({
        'subject_id': paciente_id.values,
        'etiqueta': etiqueta.values,
    }).groupby('subject_id')['etiqueta'].max().reset_index()

    pacientes_entreno, pacientes_calibracion = train_test_split(
        pacientes['subject_id'],
        test_size=test_size_calibracion,
        random_state=RANDOM_STATE,
        stratify=pacientes['etiqueta'],
    )

    mascara_entreno = paciente_id.isin(pacientes_entreno)
    mascara_calibracion = paciente_id.isin(pacientes_calibracion)

    return mascara_entreno, mascara_calibracion


def crear_calibrador_prefit(modelo_entrenado, metodo_calibracion):
    try:
        from sklearn.frozen import FrozenEstimator
        calibrador = CalibratedClassifierCV(
            estimator=FrozenEstimator(modelo_entrenado),
            method=metodo_calibracion,
        )
    except Exception:
        calibrador = CalibratedClassifierCV(
            estimator=modelo_entrenado,
            method=metodo_calibracion,
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


def evaluar_configuracion(predictores, etiqueta, paciente_id, test_size_calibracion, metodo_calibracion):
    cv_externo = StratifiedGroupKFold(
        n_splits=N_SPLITS_EXTERNOS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    predicciones_todos = []
    metricas_todos = []

    for fold, (idx_entreno_ext, idx_test_ext) in enumerate(
        cv_externo.split(predictores, etiqueta, groups=paciente_id),
        start=1,
    ):
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
            test_size_calibracion,
        )

        x_modelo = x_entreno_ext.loc[mascara_modelo].copy()
        y_modelo = y_entreno_ext.loc[mascara_modelo].copy()

        x_calibracion = x_entreno_ext.loc[mascara_calibracion].copy()
        y_calibracion = y_entreno_ext.loc[mascara_calibracion].copy()

        modelo_sin_calibrar = copy.deepcopy(MODELO_BASE)
        modelo_sin_calibrar.fit(x_modelo, y_modelo)

        calibrador = crear_calibrador_prefit(modelo_sin_calibrar, metodo_calibracion)
        calibrador.fit(x_calibracion, y_calibracion)

        probabilidad_sin_calibrar = modelo_sin_calibrar.predict_proba(x_test)[:, 1]
        probabilidad_calibrada = calibrador.predict_proba(x_test)[:, 1]

        metricas_sin = calcular_metricas(y_test, probabilidad_sin_calibrar)
        metricas_cal = calcular_metricas(y_test, probabilidad_calibrada)

        metricas_fold = {
            'test_size_calibracion': test_size_calibracion,
            'metodo_calibracion': metodo_calibracion,
            'fold': fold,
            'n_modelo': len(x_modelo),
            'positivos_modelo': int(y_modelo.sum()),
            'n_calibracion': len(x_calibracion),
            'positivos_calibracion': int(y_calibracion.sum()),
            'n_test': len(y_test),
            'positivos_test': int(y_test.sum()),
            'prevalencia_test': float(y_test.mean()),
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

        pred_fold = pd.DataFrame({
            'test_size_calibracion': test_size_calibracion,
            'metodo_calibracion': metodo_calibracion,
            'fold': fold,
            'subject_id': grupos_test.values,
            'y_real': y_test.values,
            'probabilidad_sin_calibrar': probabilidad_sin_calibrar,
            'probabilidad_calibrada': probabilidad_calibrada,
        })
        predicciones_todos.append(pred_fold)

        print(
            f'    Fold {fold}/{N_SPLITS_EXTERNOS} | '
            f'AUC {metricas_fold["auc_calibrado"]:.4f} | '
            f'AUPRC {metricas_fold["auprc_calibrado"]:.4f} | '
            f'Brier {metricas_fold["brier_calibrado"]:.4f} | '
            f'LogLoss {metricas_fold["logloss_calibrado"]:.4f} | '
            f'ECE {metricas_fold["ece_calibrado"]:.4f}'
        )

    metricas = pd.DataFrame(metricas_todos)
    predicciones = pd.concat(predicciones_todos, ignore_index=True)

    return metricas, predicciones


def resumir_metricas(metricas):
    columnas = [
        'auc_sin_calibrar', 'auc_calibrado',
        'auprc_sin_calibrar', 'auprc_calibrado',
        'brier_sin_calibrar', 'brier_calibrado',
        'logloss_sin_calibrar', 'logloss_calibrado',
        'ece_sin_calibrar', 'ece_calibrado',
    ]

    resumen = metricas.groupby(
        ['test_size_calibracion', 'metodo_calibracion']
    )[columnas].agg(['mean', 'std'])

    resumen.columns = [
        f'{metrica}_{estadistico}'
        for metrica, estadistico in resumen.columns
    ]
    resumen = resumen.reset_index()

    resumen = resumen.sort_values(
        by=['brier_calibrado_mean', 'logloss_calibrado_mean', 'ece_calibrado_mean'],
        ascending=True,
    ).reset_index(drop=True)

    resumen['ranking'] = np.arange(1, len(resumen) + 1)

    return resumen


def filtrar_predicciones_mejor_configuracion(predicciones, mejor_config):
    mascara = (
        (predicciones['test_size_calibracion'] == mejor_config['test_size_calibracion']) &
        (predicciones['metodo_calibracion'] == mejor_config['metodo_calibracion'])
    )
    return predicciones.loc[mascara].copy()


def guardar_tabla_calibracion(nombre, predicciones):
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

    ruta = os.path.join(CARPETA_TABLAS, f'tabla_calibracion_deciles_{nombre}.csv')
    tabla.to_csv(ruta, index=False)

    return ruta


def guardar_curva_calibracion(nombre, predicciones):
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
    plt.title(f'Curva de calibración - {nombre}')
    plt.legend(loc='upper left')
    plt.tight_layout()

    ruta = os.path.join(CARPETA_FIGURAS, f'calibracion_{nombre}.png')
    plt.savefig(ruta, dpi=300, bbox_inches='tight')
    plt.close()

    return ruta


def guardar_histograma_probabilidades(nombre, predicciones):
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
    plt.title(f'Histograma de probabilidades - {nombre}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    ruta = os.path.join(CARPETA_FIGURAS, f'histograma_probabilidades_{nombre}.png')
    plt.savefig(ruta, dpi=300, bbox_inches='tight')
    plt.close()

    return ruta


def guardar_curva_roc(nombre, predicciones):
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
    plt.title(f'Curva ROC - {nombre}')
    plt.legend(loc='lower right')
    plt.tight_layout()

    ruta = os.path.join(CARPETA_FIGURAS, f'roc_{nombre}.png')
    plt.savefig(ruta, dpi=300, bbox_inches='tight')
    plt.close()

    return ruta


def guardar_curva_precision_recall(nombre, predicciones):
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
    plt.title(f'Curva Precision-Recall - {nombre}')
    plt.legend(loc='upper right')
    plt.tight_layout()

    ruta = os.path.join(CARPETA_FIGURAS, f'precision_recall_{nombre}.png')
    plt.savefig(ruta, dpi=300, bbox_inches='tight')
    plt.close()

    return ruta


def main():
    print('=' * 70)
    print('BÚSQUEDA DE CALIBRACIÓN — CATBOOST v4p 3-12h')
    print('=' * 70)

    print('\n[1/5] Cargando datos...')
    predictores, etiqueta, paciente_id = cargar_y_preparar()

    print(f'  Estancias: {len(predictores)}')
    print(f'  Positivos: {etiqueta.sum()} ({100 * etiqueta.mean():.2f}%)')
    print(f'  Pacientes únicos: {paciente_id.nunique()}')
    print(f'  Variables: {predictores.shape[1]}')
    print(f'  Modelo base: {MODELO_BASE.get_params()}')

    metricas_total = []
    predicciones_total = []

    print('\n[2/5] Probando configuraciones...')
    for test_size_calibracion in TEST_SIZE_CALIBRACION_GRID:
        for metodo_calibracion in METODO_CALIBRACION_GRID:
            print('\n' + '-' * 70)
            print(f'Test size calibración: {test_size_calibracion}')
            print(f'Método calibración:    {metodo_calibracion}')

            metricas, predicciones = evaluar_configuracion(
                predictores=predictores,
                etiqueta=etiqueta,
                paciente_id=paciente_id,
                test_size_calibracion=test_size_calibracion,
                metodo_calibracion=metodo_calibracion,
            )

            metricas_total.append(metricas)
            predicciones_total.append(predicciones)

    print('\n[3/5] Guardando tablas globales...')
    metricas_total = pd.concat(metricas_total, ignore_index=True)
    predicciones_total = pd.concat(predicciones_total, ignore_index=True)
    resumen = resumir_metricas(metricas_total)

    ruta_metricas = os.path.join(CARPETA_TABLAS, 'metricas_catboost_v4p_grid_calibracion.csv')
    ruta_predicciones = os.path.join(CARPETA_TABLAS, 'predicciones_catboost_v4p_grid_calibracion.csv')
    ruta_resumen = os.path.join(CARPETA_TABLAS, 'resumen_catboost_v4p_grid_calibracion.csv')

    metricas_total.to_csv(ruta_metricas, index=False)
    predicciones_total.to_csv(ruta_predicciones, index=False)
    resumen.to_csv(ruta_resumen, index=False)

    print('\nResumen ordenado por Brier calibrado, LogLoss calibrado y ECE calibrado:')
    columnas_mostrar = [
        'ranking',
        'test_size_calibracion',
        'metodo_calibracion',
        'auc_calibrado_mean',
        'auprc_calibrado_mean',
        'brier_calibrado_mean',
        'logloss_calibrado_mean',
        'ece_calibrado_mean',
        'brier_sin_calibrar_mean',
        'logloss_sin_calibrar_mean',
        'ece_sin_calibrar_mean',
    ]
    print(resumen[columnas_mostrar].to_string(index=False))

    mejor_config = resumen.iloc[0].to_dict()
    nombre_mejor = (
        f'catboost_v4p_3_12_'
        f'{mejor_config["metodo_calibracion"]}_'
        f'cal{str(mejor_config["test_size_calibracion"]).replace(".", "")}'
    )

    print('\n[4/5] Mejor configuración')
    print(f'  Test size calibración: {mejor_config["test_size_calibracion"]}')
    print(f'  Método calibración:    {mejor_config["metodo_calibracion"]}')
    print(f'  Brier calibrado:       {mejor_config["brier_calibrado_mean"]:.4f}')
    print(f'  LogLoss calibrado:     {mejor_config["logloss_calibrado_mean"]:.4f}')
    print(f'  ECE calibrado:         {mejor_config["ece_calibrado_mean"]:.4f}')

    predicciones_mejor = filtrar_predicciones_mejor_configuracion(
        predicciones_total,
        mejor_config,
    )

    ruta_predicciones_mejor = os.path.join(CARPETA_TABLAS, f'predicciones_{nombre_mejor}.csv')
    predicciones_mejor.to_csv(ruta_predicciones_mejor, index=False)

    ruta_tabla_deciles = guardar_tabla_calibracion(nombre_mejor, predicciones_mejor)

    print('\n[5/5] Generando figuras de la mejor configuración...')
    ruta_calibracion = guardar_curva_calibracion(nombre_mejor, predicciones_mejor)
    ruta_histograma = guardar_histograma_probabilidades(nombre_mejor, predicciones_mejor)
    ruta_roc = guardar_curva_roc(nombre_mejor, predicciones_mejor)
    ruta_pr = guardar_curva_precision_recall(nombre_mejor, predicciones_mejor)

    print('\nArchivos guardados:')
    print(f'  Métricas grid:       {ruta_metricas}')
    print(f'  Predicciones grid:   {ruta_predicciones}')
    print(f'  Resumen grid:        {ruta_resumen}')
    print(f'  Predicciones mejor:  {ruta_predicciones_mejor}')
    print(f'  Tabla deciles:       {ruta_tabla_deciles}')
    print(f'  Calibración:         {ruta_calibracion}')
    print(f'  Histograma:          {ruta_histograma}')
    print(f'  ROC:                 {ruta_roc}')
    print(f'  Precision-Recall:    {ruta_pr}')
    print('\nCALIBRACIÓN CATBOOST v4p COMPLETADA')


if __name__ == '__main__':
    main()
